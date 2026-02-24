#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from easytorch.utils import master_only

from .jy_independent_learning_runner import IndependentLearningRunner


class ClientWiseIndependentLearningRunner(IndependentLearningRunner):
    """
    - IndependentLearningRunner + client-wise best tracking
    - Partition/ckpt meta는 base가 저장/검증
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        self._client_best_metric: List[Optional[float]] = [None for _ in self._il_client_nodes_list]
        self._client_best_epoch: List[Optional[int]] = [None for _ in self._il_client_nodes_list]
        self._client_val_sum: List[float] = [0.0 for _ in self._il_client_nodes_list]
        self._client_val_weight: List[float] = [0.0 for _ in self._il_client_nodes_list]

        self._client_best_dir = os.path.join(self.ckpt_save_dir, "client_best")
        os.makedirs(self._client_best_dir, exist_ok=True)

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        if iter_index == 0:
            self._client_val_sum = [0.0 for _ in self._il_client_nodes_list]
            self._client_val_weight = [0.0 for _ in self._il_client_nodes_list]

        forward_return = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)

        # overall meters
        loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return["target"])
        self.update_epoch_meter("val/loss", loss.item(), weight)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"val/{metric_name}", metric_item.item(), weight)

        # per-client accumulate
        metric_func = self.loss if self.target_metrics == "loss" else self.metrics[self.target_metrics]

        pred = forward_return["prediction"]
        target = forward_return["target"]
        device = pred.device

        for ci, nodes in enumerate(self._il_client_nodes_list):
            idx = torch.tensor(nodes, device=device, dtype=torch.long)
            client_pred = pred.index_select(2, idx)
            client_tgt = target.index_select(2, idx)

            mval = self.metric_forward(metric_func, {"prediction": client_pred, "target": client_tgt})
            w = self._get_metric_weight(client_tgt)
            self._client_val_sum[ci] += float(mval.item()) * float(w)
            self._client_val_weight[ci] += float(w)

    def _reduce_client_stats_ddp(self, device: torch.device):
        if not (dist.is_available() and dist.is_initialized()):
            return
        for ci in range(len(self._il_client_nodes_list)):
            t = torch.tensor([self._client_val_sum[ci], self._client_val_weight[ci]], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self._client_val_sum[ci] = float(t[0].item())
            self._client_val_weight[ci] = float(t[1].item())

    # ✅ master_only 제거 (DDP deadlock 방지). 저장만 rank0가 수행
    def on_validating_end(self, train_epoch: Optional[int]):
        if train_epoch is None:
            return

        m = self.model.module if hasattr(self.model, "module") else self.model
        device = next(m.parameters()).device if any(p.requires_grad for p in m.parameters()) else torch.device("cpu")
        self._reduce_client_stats_ddp(device)

        is_master = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        if not is_master:
            return

        greater_best = (self.metrics_best != "min")

        for ci in range(len(self._il_client_nodes_list)):
            if self._client_val_weight[ci] <= 0:
                continue
            avg = self._client_val_sum[ci] / self._client_val_weight[ci]
            prev = self._client_best_metric[ci]
            improved = (prev is None) or ((avg > prev) if greater_best else (avg < prev))
            if improved:
                self._client_best_metric[ci] = float(avg)
                self._client_best_epoch[ci] = int(train_epoch)
                self._save_client_best(ci, train_epoch, avg)

        with open(os.path.join(self._client_best_dir, "client_best_summary.txt"), "w") as f:
            for ci in range(len(self._il_client_nodes_list)):
                f.write(f"client={ci}\tbest_epoch={self._client_best_epoch[ci]}\tbest_{self.target_metrics}={self._client_best_metric[ci]}\n")

    def _save_client_best(self, client_idx: int, epoch: int, metric_val: float):
        m = self.model.module if hasattr(self.model, "module") else self.model
        sub_model = m.client_models[client_idx]
        save_path = os.path.join(self._client_best_dir, f"client_{client_idx}_best.pt")
        payload = {
            "client_idx": client_idx,
            "epoch": epoch,
            "metric": float(metric_val),
            "state_dict": {k: v.detach().cpu() for k, v in sub_model.state_dict().items()},
        }
        torch.save(payload, save_path)

    def load_client_best_models(self):
        m = self.model.module if hasattr(self.model, "module") else self.model
        for ci in range(len(m.client_models)):
            path = os.path.join(self._client_best_dir, f"client_{ci}_best.pt")
            if not os.path.exists(path):
                continue
            ckpt = torch.load(path, map_location="cpu")
            m.client_models[ci].load_state_dict(ckpt["state_dict"], strict=True)

    @torch.no_grad()
    @master_only
    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, save_results: bool = False):
        self.load_client_best_models()
        return super().test(train_epoch=train_epoch, save_metrics=save_metrics, save_results=save_results)

    @master_only
    def on_training_end(self, cfg: Dict, train_epoch: int):
        # client-best로 평가
        if hasattr(cfg, "TEST"):
            self.logger.info("Evaluating client-wise best models on the test set.")
            self.load_client_best_models()
            self.test_pipeline(cfg=cfg, train_epoch=train_epoch, save_metrics=True, save_results=self.save_results)

    def _get_method_name(self) -> str:
        return 'independent_clientwise'
