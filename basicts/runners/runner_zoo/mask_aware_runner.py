from typing import Dict, Optional

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class MaskAwareRunner(SimpleTimeSeriesForecastingRunner):
    """Runner that extracts target mask from data and passes it to loss/metrics.

    Extracts a binary mask channel from the target data before feature selection,
    and includes it as 'target_mask' in the forward return dict. This allows
    mask-aware loss functions to distinguish between real zero targets and
    missing-data zeros.

    Config:
        CFG.MODEL.MASK_CHANNEL_IDX: int - index of the mask channel in the
            original data (before feature selection). Default: 3.
            For SAN_BERNARDINO_MASK 8ch: [flow, occ, speed, mask_f, mask_o, mask_s, tod, dow]
            → mask_f is at index 3.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.mask_channel_idx = cfg['MODEL'].get('MASK_CHANNEL_IDX', 3)

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None,
                train: bool = True, **kwargs) -> Dict:
        # Extract mask from raw target data before any processing.
        # Scaler only touches channel 0 (flow), so mask channel is unaffected.
        target_mask = data['target'][:, :, :, self.mask_channel_idx:self.mask_channel_idx + 1].clone()

        # Normal forward pass (preprocessing, model, postprocessing)
        result = super().forward(data, epoch, iter_num, train, **kwargs)

        # Add mask to return dict (metric_forward auto-passes it if loss signature accepts it)
        result['target_mask'] = target_mask.to(result['prediction'].device)

        return result
