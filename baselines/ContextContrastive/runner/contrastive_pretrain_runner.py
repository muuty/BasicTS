"""
Stage 1 Runner: Contrastive Pre-training with Context-Aware Encoder.

Handles:
- Loading adjacency matrix for spatial encoding
- Passing edge_index to the model
- Contrastive loss computation
"""
from typing import Dict, Optional

import torch
import numpy as np

from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.utils import load_adj


class ContrastivePretrainRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner for Stage 1 contrastive pre-training.

    Config requirements:
        CFG.MODEL.PARAM: dict with 'use_spatial' flag
        CFG.GRAPH (optional): dict with 'adj_path' for spatial encoding
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        # Load edge_index for spatial encoding
        self.edge_index = None
        use_spatial = cfg.get('MODEL', {}).get('PARAM', {}).get('use_spatial', False)

        if use_spatial:
            graph_cfg = cfg.get('GRAPH', {})
            adj_path = graph_cfg.get('adj_path')
            if adj_path:
                self.edge_index = self._load_edge_index(adj_path)
                self.logger.info(f"Loaded edge_index with {self.edge_index.shape[1]} edges")
            else:
                self.logger.warning("use_spatial=True but no adj_path provided in GRAPH config")

    def _load_edge_index(self, adj_path: str) -> torch.Tensor:
        """Load adjacency matrix and convert to edge_index format."""
        adj_mx, _ = load_adj(adj_path, 'doubletransition')

        if isinstance(adj_mx, list):
            adj_mx = adj_mx[0]

        if isinstance(adj_mx, np.ndarray):
            adj_mx = torch.from_numpy(adj_mx).float()

        # Get edge indices where adj > 0
        edge_index = (adj_mx > 0).nonzero(as_tuple=False).T.contiguous()
        return edge_index

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Forward pass for contrastive pre-training.
        """
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Prepare edge_index
        edge_index = None
        if self.edge_index is not None:
            edge_index = self.edge_index.to(history_data.device)

        # Forward pass
        model_return = self.model(
            history_data=history_data,
            future_data=future_data_4_dec,
            edge_index=edge_index,
            batch_seen=iter_num,
            epoch=epoch,
            train=train
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)
        return model_return
