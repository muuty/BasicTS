"""MICL Pretraining Runner.

Handles the SSL pretraining loop. Overrides forward() to:
1. Only use history data (no future target needed for SSL)
2. Log collapse monitoring metrics (z_std)

Overrides test() to skip robustness metrics (which assume 4D forecasting tensors).
"""

from typing import Dict, Optional

import torch

from basicts.runners import SimpleTimeSeriesForecastingRunner


class MICLPretrainRunner(SimpleTimeSeriesForecastingRunner):

    def forward(self, data: Dict, epoch: Optional[int] = None,
                iter_num: Optional[int] = None, train: bool = True,
                **kwargs) -> Dict:
        data = self.preprocessing(data)

        history_data = data['inputs']
        history_data = self.to_running_device(history_data)
        history_data = self.select_input_features(history_data)

        model_return = self.model(
            history_data=history_data,
            epoch=epoch,
            train=train,
        )

        # Collapse monitoring: std of representations across batch dim
        with torch.no_grad():
            z = model_return['target']  # [B, N, D]
            z_std = z.std(dim=0).mean().item()
            if iter_num is not None and iter_num % 100 == 0:
                print(f"[collapse monitor] z_std={z_std:.4f}")

        return model_return

    @torch.no_grad()
    def test(self, train_epoch: Optional[int] = None,
             save_metrics: bool = False, save_results: bool = False) -> Dict:
        """Simplified test for SSL pretraining (skip robustness metrics)."""

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, data in enumerate(self.test_data_loader):
            forward_return = self.forward(data=data, epoch=None, iter_num=None, train=False)

            loss = self.metric_forward(self.loss, forward_return)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.logger.info(f"Test SSL_Loss: {avg_loss:.6f}")

        # Save as test_metrics.json for checkpoint selection
        if save_metrics:
            import json
            import os
            metrics = {'overall': {'SSL_Loss': avg_loss}}
            path = os.path.join(self.ckpt_save_dir, 'test_metrics.json')
            with open(path, 'w') as f:
                json.dump(metrics, f, indent=4)

        return {'SSL_Loss': avg_loss}
