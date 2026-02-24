from typing import Dict, Optional

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class InstanceNormRunner(SimpleTimeSeriesForecastingRunner):
    """Runner that removes both per-node mean AND std from flow before model forward.

    Instance normalization in Z-score space:
    - Input flow is centered (subtract mean) and scaled (divide by std) per-node.
    - Model predicts fully normalized (pure pattern) output.
    - Per-node std and mean are restored to prediction before postprocessing.

    This isolates pure temporal pattern drift by removing BOTH:
    - Additive scale drift (mean shift)
    - Multiplicative scale drift (variance change)

    Comparison:
    - Baseline: no centering → captures all drift
    - PatternOnly: subtract mean → removes additive scale
    - InstanceNorm: subtract mean + divide std → removes all scale, only pattern remains
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

        # === Instance norm: center and scale flow (channel 0) per-node ===
        flow = history_data[:, :, :, 0]  # (B, T, N)
        input_flow_mean = flow.mean(dim=1, keepdim=True)   # (B, 1, N)
        input_flow_std = flow.std(dim=1, keepdim=True) + 1e-5  # (B, 1, N), eps for stability

        history_normed = history_data.clone()
        history_normed[:, :, :, 0] = (flow - input_flow_mean) / input_flow_std

        # Forward pass with instance-normalized input
        model_return = self.model(
            history_data=history_normed, future_data=future_data_4_dec,
            batch_seen=iter_num, epoch=epoch, train=train
        )

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        # De-normalize: multiply back std, add back mean
        # prediction: (B, T_out, N, 1), stats: (B, 1, N) → unsqueeze to (B, 1, N, 1)
        model_return['prediction'] = model_return['prediction'] * input_flow_std.unsqueeze(-1) + input_flow_mean.unsqueeze(-1)

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes]

        model_return = self.postprocessing(model_return)

        return model_return
