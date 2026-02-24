"""
Enhance Model: Wrapper for pre-trained GPT-ST encoder with downstream predictor.

This module provides a backbone-agnostic design that can work with any
downstream predictor model.
"""

from typing import Dict, Optional, Tuple, Type

import torch
import torch.nn as nn

from .gptst_arch import GPTSTModel
from .modules import Fusion


class EnhanceModel(nn.Module):
    """Enhanced model combining pre-trained GPT-ST encoder with downstream predictor.

    This wrapper model:
    1. Uses pre-trained GPT-ST encoder to generate representations
    2. Fuses pre-trained representations with input embeddings
    3. Passes fused representations to downstream predictor

    The design is backbone-agnostic - any downstream model can be used as predictor.
    """

    def __init__(
        self,
        # GPT-ST encoder parameters
        num_nodes: int,
        input_base_dim: int = 1,
        input_extra_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 1,
        horizon: int = 12,
        embed_dim: int = 16,
        embed_dim_spa: int = 8,
        HS: int = 4,
        HT: int = 4,
        HT_Tem: int = 4,
        num_route: int = 3,
        # Mode and paths
        mode: str = 'eval',
        load_pretrain_path: Optional[str] = None,
        freeze_pretrain: bool = True,
        # Downstream predictor
        backbone_class: Optional[Type[nn.Module]] = None,
        backbone_params: Optional[Dict] = None,
        # Device
        device: str = 'cuda:0',
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph
            input_base_dim: Number of input features
            input_extra_dim: Number of extra temporal features
            hidden_dim: Hidden dimension of GPT-ST encoder
            output_dim: Output dimension
            horizon: Sequence length
            embed_dim: Embedding dimension for temporal features
            embed_dim_spa: Embedding dimension for spatial features
            HS: Number of spatial hyperedge heads
            HT: Number of temporal hyperedge heads
            HT_Tem: Number of temporal hypergraph heads
            num_route: Number of capsule routing iterations
            mode: 'eval' (use pretrained) or 'ori' (no pretrain)
            load_pretrain_path: Path to pre-trained model checkpoint
            freeze_pretrain: Whether to freeze pre-trained parameters
            backbone_class: Class of downstream predictor model
            backbone_params: Parameters for downstream predictor
            device: Device to use
        """
        super(EnhanceModel, self).__init__()

        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.input_extra_dim = input_extra_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.mode = mode
        self.load_pretrain_path = load_pretrain_path
        self.freeze_pretrain = freeze_pretrain
        self.device = device

        # Determine input dimension for predictor
        if mode == 'ori':
            self.predictor_input_dim = input_base_dim
        else:
            self.predictor_input_dim = hidden_dim

        # Build pretrained model for eval mode
        if mode == 'eval':
            self.pretrain_model = GPTSTModel(
                num_nodes=num_nodes,
                input_base_dim=input_base_dim,
                input_extra_dim=input_extra_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                horizon=horizon,
                embed_dim=embed_dim,
                embed_dim_spa=embed_dim_spa,
                HS=HS,
                HT=HT,
                HT_Tem=HT_Tem,
                num_route=num_route,
                mode='eval',  # Set to eval mode (no masking)
                device=device,
            )

            # Load pretrained weights if provided
            if load_pretrain_path is not None:
                self.load_pretrained_model(load_pretrain_path)

            # Freeze pretrained parameters if specified
            if freeze_pretrain:
                for param in self.pretrain_model.parameters():
                    param.requires_grad = False

            # Fusion layer
            self.fusion = Fusion(hidden_dim)

            # Linear projection for input
            self.lin_input = nn.Linear(input_base_dim, hidden_dim)

        # Build downstream predictor
        if backbone_class is not None and backbone_params is not None:
            # Update backbone params with correct input/output dimensions
            predictor_params = backbone_params.copy()

            # Handle common parameter names for input dimension
            for key in ['input_dim', 'in_dim', 'dim_in', 'c_in', 'input_channels']:
                if key in predictor_params:
                    predictor_params[key] = self.predictor_input_dim

            # Handle common parameter names for output dimension
            for key in ['output_dim', 'out_dim', 'dim_out', 'c_out', 'output_channels']:
                if key in predictor_params:
                    predictor_params[key] = output_dim

            self.predictor = backbone_class(**predictor_params)
        else:
            # Default: simple linear predictor
            self.predictor = nn.Linear(self.predictor_input_dim * horizon, output_dim * horizon)
            self._use_default_predictor = True

    def load_pretrained_model(self, path: str):
        """Load pre-trained model weights.

        Args:
            path: Path to checkpoint file
        """
        state_dict = torch.load(path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.pretrain_model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained model from {path}")

    def forward_pretrain(self, source: torch.Tensor, label: Optional[torch.Tensor] = None,
                         batch_seen: Optional[int] = None) -> torch.Tensor:
        """Forward pass using pre-trained encoder.

        Args:
            source: Input data [B, T, N, D]
            label: Target labels (optional, for compatibility)
            batch_seen: Number of batches seen (optional)

        Returns:
            prediction: Model predictions [B, T, N, output_dim]
        """
        # Get pretrained representations (no masking in eval mode)
        with torch.no_grad() if self.freeze_pretrain else torch.enable_grad():
            x_pretrain_flow = self.pretrain_model.get_encoder_output(source)

        # Project input to hidden dimension
        x_input = self.lin_input(source[..., :self.input_base_dim])

        # Fuse pretrained and input representations
        pretrain_eb = self.fusion(x_pretrain_flow, x_input)

        # Pass through predictor
        if hasattr(self, '_use_default_predictor') and self._use_default_predictor:
            # Default linear predictor
            batch_size = pretrain_eb.shape[0]
            pretrain_eb = pretrain_eb.transpose(1, 2)  # [B, N, T, D]
            pretrain_eb = pretrain_eb.reshape(batch_size, self.num_node, -1)
            x_predic = self.predictor(pretrain_eb)
            x_predic = x_predic.reshape(batch_size, self.num_node, self.horizon, self.output_dim)
            x_predic = x_predic.transpose(1, 2)  # [B, T, N, output_dim]
        else:
            x_predic = self.predictor(pretrain_eb)

        return x_predic

    def forward_ori(self, source: torch.Tensor, label: Optional[torch.Tensor] = None,
                    batch_seen: Optional[int] = None) -> torch.Tensor:
        """Forward pass without pre-training (baseline mode).

        Args:
            source: Input data [B, T, N, D]
            label: Target labels (optional)
            batch_seen: Number of batches seen (optional)

        Returns:
            prediction: Model predictions [B, T, N, output_dim]
        """
        x_input = source[..., :self.input_base_dim]

        if hasattr(self, '_use_default_predictor') and self._use_default_predictor:
            batch_size = x_input.shape[0]
            x_input = x_input.transpose(1, 2)  # [B, N, T, D]
            x_input = x_input.reshape(batch_size, self.num_node, -1)
            x_predic = self.predictor(x_input)
            x_predic = x_predic.reshape(batch_size, self.num_node, self.horizon, self.output_dim)
            x_predic = x_predic.transpose(1, 2)  # [B, T, N, output_dim]
        else:
            x_predic = self.predictor(x_input)

        return x_predic

    def forward(self, history_data: torch.Tensor, future_data: Optional[torch.Tensor] = None,
                batch_seen: Optional[int] = None, epoch: Optional[int] = None,
                train: bool = True, **kwargs) -> torch.Tensor:
        """
        Forward pass compatible with BasiCTS runner.

        Args:
            history_data: Input history data [B, T, N, D]
            future_data: Future data (optional, not used in this model)
            batch_seen: Number of batches seen
            epoch: Current epoch
            train: Whether in training mode

        Returns:
            prediction: Model predictions [B, T, N, output_dim]
        """
        if self.mode == 'ori':
            return self.forward_ori(history_data, future_data, batch_seen)
        else:
            return self.forward_pretrain(history_data, future_data, batch_seen)


class GPTSTForForecasting(nn.Module):
    """GPT-ST model adapted for direct forecasting in BasiCTS.

    This is a simpler wrapper that uses GPT-ST encoder directly
    with a prediction head, without requiring a separate backbone.
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        input_dim: int = 3,
        output_dim: int = 1,
        input_base_dim: int = 1,
        hidden_dim: int = 64,
        embed_dim: int = 16,
        embed_dim_spa: int = 8,
        HS: int = 4,
        HT: int = 4,
        HT_Tem: int = 4,
        num_route: int = 3,
        mode: str = 'pretrain',
        mask_ratio: float = 0.3,
        ada_mask_ratio: float = 1.0,
        ada_type: str = 'all',
        change_epoch: int = 10,
        epochs: int = 100,
        scaler_zeros: float = 0.0,
        device: str = 'cuda:0',
        pretrained_path: str = None,
        freeze_encoder: bool = False,
    ):
        """
        Args:
            num_nodes: Number of nodes
            in_steps: Input sequence length
            out_steps: Output sequence length
            input_dim: Total input dimension (including temporal features)
            output_dim: Output dimension per node
            input_base_dim: Base input dimension (e.g., 1 for flow only)
            hidden_dim: Hidden dimension
            embed_dim: Temporal embedding dimension
            embed_dim_spa: Spatial embedding dimension
            HS: Number of spatial hyperedge heads
            HT: Number of temporal hyperedge heads
            HT_Tem: Temporal hypergraph heads
            num_route: Capsule routing iterations
            mode: 'pretrain' or 'eval'
            mask_ratio: Masking ratio for pretraining
            ada_mask_ratio: Adaptive mask ratio
            ada_type: Adaptive masking type
            change_epoch: Epoch to switch to adaptive masking
            epochs: Total epochs
            scaler_zeros: Value for masked positions
            device: Device string
            pretrained_path: Path to pretrained checkpoint (optional)
            freeze_encoder: Whether to freeze encoder weights during finetuning
        """
        super(GPTSTForForecasting, self).__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_base_dim = input_base_dim
        self.hidden_dim = hidden_dim
        self.mode = mode

        # GPT-ST model
        self.gptst = GPTSTModel(
            num_nodes=num_nodes,
            input_base_dim=input_base_dim,
            input_extra_dim=input_dim - input_base_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=in_steps,
            embed_dim=embed_dim,
            embed_dim_spa=embed_dim_spa,
            HS=HS,
            HT=HT,
            HT_Tem=HT_Tem,
            num_route=num_route,
            mode=mode,
            scaler_zeros=scaler_zeros,
            mask_ratio=mask_ratio,
            ada_mask_ratio=ada_mask_ratio,
            ada_type=ada_type,
            change_epoch=change_epoch,
            epochs=epochs,
            device=device,
        )

        # Prediction head (encoder output -> forecast)
        self.pred_head = nn.Sequential(
            nn.Linear(in_steps * hidden_dim, out_steps * hidden_dim),
            nn.ReLU(),
            nn.Linear(out_steps * hidden_dim, out_steps * output_dim),
        )

        # Load pretrained weights if provided
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained weights from checkpoint."""
        import glob
        import os

        # Handle glob patterns
        if '*' in pretrained_path:
            matches = glob.glob(pretrained_path)
            if matches:
                pretrained_path = sorted(matches)[-1]  # Use latest
            else:
                print(f"Warning: No pretrained checkpoint found matching {pretrained_path}")
                return

        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Load with partial matching (ignore missing/extra keys)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained parameters")
        else:
            print(f"Warning: Pretrained checkpoint not found: {pretrained_path}")

    def _freeze_encoder(self):
        """Freeze encoder (gptst) parameters."""
        for param in self.gptst.parameters():
            param.requires_grad = False
        print("Encoder frozen for fine-tuning")

    def forward(self, history_data: torch.Tensor, future_data: Optional[torch.Tensor] = None,
                batch_seen: Optional[int] = None, epoch: Optional[int] = None,
                train: bool = True, **kwargs) -> Dict:
        """
        Forward pass for BasiCTS.

        Args:
            history_data: [B, T, N, D]
            future_data: [B, T, N, D] (target)
            batch_seen: Batch index
            epoch: Current epoch
            train: Training mode flag

        Returns:
            Dict with 'prediction' and optionally pretraining outputs
        """
        batch_size = history_data.shape[0]

        if self.mode == 'pretrain' and train:
            # Pre-training mode: return reconstruction and auxiliary outputs
            flow_out, flow_decode, mask, probability, HS = self.gptst(
                history_data, future_data, batch_seen, epoch
            )

            return {
                'prediction': flow_out,  # Reconstruction
                'reconstruction': flow_out,
                'mask': mask,
                'probability': probability,
                'hidden_states': HS,
            }
        else:
            # Inference mode: get encoder output and predict
            encoder_out = self.gptst.get_encoder_output(history_data)

            # Reshape and project to output
            # encoder_out: [B, T, N, hidden_dim]
            encoder_out = encoder_out.transpose(1, 2)  # [B, N, T, hidden_dim]
            encoder_out = encoder_out.reshape(batch_size, self.num_nodes, -1)  # [B, N, T*hidden_dim]

            pred = self.pred_head(encoder_out)  # [B, N, out_steps*output_dim]
            pred = pred.reshape(batch_size, self.num_nodes, self.out_steps, self.output_dim)
            pred = pred.transpose(1, 2)  # [B, out_steps, N, output_dim]

            return pred
