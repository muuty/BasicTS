from typing import Dict, Optional

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class DecomposedRunner(SimpleTimeSeriesForecastingRunner):
    """Runner for DecomposedSTAEformer.

    Computes auxiliary targets (mu_y, sigma_y) from the Z-scored future data
    so the loss function can supervise the scale head.
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

        model_return = self.model(
            history_data=history_data, future_data=future_data_4_dec,
            batch_seen=iter_num, epoch=epoch, train=train
        )

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}

        # Auxiliary targets from Z-scored future data (before postprocessing)
        target_flow_z = self.select_target_features(future_data)[:, :, :, 0]  # (B, T_out, N)
        model_return['mu_y_target'] = target_flow_z.mean(dim=1)              # (B, N)
        model_return['sigma_y_target'] = target_flow_z.std(dim=1) + 1e-5     # (B, N)

        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes]

        model_return = self.postprocessing(model_return)

        return model_return
