from typing import Dict, Optional

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class PatternOnlyRunner(SimpleTimeSeriesForecastingRunner):
    """Runner that removes per-node mean from flow before model forward.

    Additive centering in Z-score space:
    - Input flow is centered by subtracting per-node mean over the input window.
    - Model predicts centered (pattern-only) output.
    - Per-node mean is added back to prediction before postprocessing.

    This tests the hypothesis: if scale information is removed during training,
    does the model learn more transferable temporal patterns across years?
    """

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # === Pattern-only: center flow (channel 0) by per-node mean ===
        # input_flow_mean: (B, 1, N) — per-node mean flow in Z-score space
        input_flow_mean = history_data[:, :, :, 0].mean(dim=1, keepdim=True)

        history_centered = history_data.clone()
        history_centered[:, :, :, 0] = history_data[:, :, :, 0] - input_flow_mean

        # Forward pass with centered input
        model_return = self.model(
            history_data=history_centered, future_data=future_data_4_dec,
            batch_seen=iter_num, epoch=epoch, train=train
        )

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        # De-center: add back per-node mean
        # prediction: (B, T_out, N, 1), input_flow_mean: (B, 1, N) → unsqueeze to (B, 1, N, 1)
        model_return['prediction'] = model_return['prediction'] + input_flow_mean.unsqueeze(-1)

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes]

        model_return = self.postprocessing(model_return)

        return model_return
