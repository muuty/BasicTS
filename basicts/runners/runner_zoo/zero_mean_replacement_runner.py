from typing import Dict

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class ZeroMeanReplacementRunner(SimpleTimeSeriesForecastingRunner):
    """Runner that replaces zero flow values with training mean before z-score normalization.

    Motivation: Z-score normalization turns zero flow into a distinctive negative value,
    causing spatial attention to over-attend failing sensors by 2-6x. Replacing zeros
    with the training mean makes them z-score=0, eliminating the distinctive signal
    and normalizing attention patterns.

    Only replaces zeros in INPUT history data (not targets).
    Targets are handled by masked_mae (NULL_VAL=0) which already excludes zeros.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        # Flow mean from training data.
        # scalar for norm_each_channel=False, (1, N) for norm_each_channel=True
        self.flow_mean = self.scaler.mean.clone()

    def preprocessing(self, input_data: Dict) -> Dict:
        # Replace zero flow with training mean BEFORE z-score normalization
        flow = input_data['inputs'][..., 0]  # [B, L, N]
        zero_mask = (flow == 0)
        if zero_mask.any():
            mean_val = self.flow_mean.to(flow.device)
            # torch.where broadcasts: scalar or (1,N) → (B,L,N) naturally
            input_data['inputs'][..., 0] = torch.where(zero_mask, mean_val, flow)

        # Normal preprocessing (z-score transform on both inputs and target)
        return super().preprocessing(input_data)
