#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from .jy_partitioned_runner_base import PartitionedRunnerBase
from .jy_independent_ensemble import IndependentClientEnsemble


class IndependentLearningRunner(PartitionedRunnerBase):
    """
    - PartitionedRunnerBase가 partition을 1회 계산/저장/검증/ckpt 저장까지 담당
    - 여기서는 define_model + client-wise backward만 담당
    """

    def define_model(self, cfg: Dict) -> nn.Module:
        base_model_class = cfg["MODEL"].get("CLASS", None) or cfg["MODEL"].get("ARCH", None)
        if base_model_class is None:
            raise ValueError("cfg['MODEL']['CLASS'] or cfg['MODEL']['ARCH'] must be provided.")

        base_model_params = cfg["MODEL"].get("PARAM", {})
        base_model_params = dict(base_model_params) if not isinstance(base_model_params, dict) else dict(base_model_params)

        output_dim = base_model_params.get("output_dim", None) or cfg["DATASET"].get("OUTPUT_DIM", None)
        if output_dim is None:
            raise ValueError("Need output_dim: MODEL.PARAM.output_dim or cfg['DATASET']['OUTPUT_DIM']")
        output_dim = int(output_dim)

        # global num_nodes (ensemble 내부에서 client별로 덮어씀)
        base_model_params["num_nodes"] = self._il_total_nodes

        return IndependentClientEnsemble(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            client_nodes_list=self._il_client_nodes_list,
            subgraph_adj_list=self._il_subgraph_adj_list,
            full_adj=self._il_full_adj,
            output_dim=output_dim,
        )

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> Optional[torch.Tensor]:
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        pred = forward_return["prediction"]
        target = forward_return["target"]

        self.optim.zero_grad()

        # ✅ Global loss: full prediction vs full target
        loss = self.metric_forward(self.loss, {"prediction": pred, "target": target})
        loss.backward()
        self.optim.step()
        total_loss = loss

        # losses = []

        # device = pred.device
        # for nodes in self._il_client_nodes_list:
        #     idx = torch.tensor(nodes, device=device, dtype=torch.long)
        #     client_pred = pred.index_select(2, idx)
        #     client_tgt = target.index_select(2, idx)
        #     losses.append(self.metric_forward(self.loss, {"prediction": client_pred, "target": client_tgt}))

        # torch.autograd.backward(tensors=losses)
        # self.optim.step()

        # # logging (overall)
        # total_loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(target)
        self.update_epoch_meter("train/loss", total_loss.item(), weight)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item(), weight)

        return None

    def _get_method_name(self) -> str:
        return 'independent'
