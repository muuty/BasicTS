#!/usr/bin/env python3
from __future__ import annotations

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from easytorch.utils import master_only

from ..node_partition import setup_split_learning_nodes
from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


def _exact_same_nodes(saved: List[int], cur: List[int]) -> bool:
    return list(map(int, saved)) == list(map(int, cur))


class PartitionedRunnerBase(SimpleTimeSeriesForecastingRunner):
    """
    공통 기능:
    - __init__에서 partition을 1회 계산하여 scaler/model이 동일 split 사용
    - ckpt에 partition meta 저장 + resume 시 정확 비교(노드 id까지)
    """

    def __init__(self, cfg: Dict):
        # ---- IL params (override 반영된 cfg 기준) ----
        il = cfg.get("IL", {})
        self._il_num_clients = int(il["NUM_CLIENTS"])
        self._il_grouping_method = str(il.get("GROUPING_METHOD", "random")).lower()
        self._il_partition_seed = il.get("PARTITION_SEED", None)

        self._il_imbalance = il.get("IMBALANCE", None)
        self._il_imbalance_alpha = il.get("IMBALANCE_ALPHA", None)
        self._il_min_nodes = int(il.get("MIN_NODES_PER_CLIENT", 1))

        # ---- dataset meta ----
        base_model_params = cfg.get("MODEL", {}).get("PARAM", {})
        total_nodes = cfg.get("DATASET", {}).get("NUM_NODES", None) or base_model_params.get("num_nodes", None)
        if total_nodes is None:
            raise ValueError("Need total num_nodes: cfg['DATASET']['NUM_NODES'] or MODEL.PARAM.num_nodes")
        self._il_total_nodes = int(total_nodes)

        ds = cfg.get("DATASET", {})
        self._il_adj_matrix = ds.get("ADJ_MX", None)
        self._il_dataset_name = ds.get("NAME", None)
        self._il_load_adj_func = ds.get("LOAD_ADJ_FUNC", None)
        self._il_metadata = ds.get("METADATA", None)

        # ---- partition 계산 1회 ----
        self._il_client_nodes_list, self._il_subgraph_adj_list, self._il_node_partitioner = setup_split_learning_nodes(
            num_nodes=self._il_total_nodes,
            num_clients=self._il_num_clients,
            grouping_method=self._il_grouping_method,
            adj_matrix=self._il_adj_matrix,
            dataset_name=self._il_dataset_name,
            load_adj_func=self._il_load_adj_func,
            imbalance=self._il_imbalance,
            imbalance_alpha=self._il_imbalance_alpha,
            min_nodes_per_client=self._il_min_nodes,
            random_seed=self._il_partition_seed,
        )

        # full adj (split/gru server용)
        self._il_full_adj = None
        if self._il_node_partitioner is not None and getattr(self._il_node_partitioner, "adj_matrix", None) is not None:
            self._il_full_adj = self._il_node_partitioner.adj_matrix
        elif self._il_adj_matrix is not None:
            self._il_full_adj = self._il_adj_matrix

        super().__init__(cfg)

    # ---- scaler injection (PartitionedZScoreScaler 등) ----
    def build_scaler(self, cfg: Dict):
        if "SCALER" not in cfg:
            return None

        scaler_cls = cfg["SCALER"]["TYPE"]
        params = dict(cfg["SCALER"]["PARAM"])

        sig = inspect.signature(scaler_cls.__init__).parameters
        if "client_nodes_list" in sig and "client_nodes_list" not in params:
            params["client_nodes_list"] = self._il_client_nodes_list

        return scaler_cls(**params)

    # ---- helpers ----
    def _get_client_nodes_list(self) -> List[List[int]]:
        return self._il_client_nodes_list

    def _partition_meta_for_ckpt(self) -> Dict[str, Any]:
        return {
            "il_client_nodes_list": self._il_client_nodes_list,
            "il_num_clients": len(self._il_client_nodes_list),
            "il_grouping_method": self._il_grouping_method,
            "il_partition_seed": self._il_partition_seed,
            "il_min_nodes_per_client": self._il_min_nodes,
            "il_imbalance": self._il_imbalance,
            "il_imbalance_alpha": self._il_imbalance_alpha,
        }

    def _validate_partition_meta(self, ckpt: Dict[str, Any]):
        if "il_client_nodes_list" not in ckpt:
            self.logger.warning("⚠️ checkpoint has no il_client_nodes_list; cannot validate grouping strictly.")
            return

        saved = ckpt["il_client_nodes_list"]
        cur = self._il_client_nodes_list

        if len(saved) != len(cur):
            raise RuntimeError(
                f"Grouping mismatch: saved clients={len(saved)} vs current clients={len(cur)}"
            )

        for ci, (s, c) in enumerate(zip(saved, cur)):
            if not _exact_same_nodes(s, c):
                raise RuntimeError(
                    f"Grouping mismatch at client {ci}: node ids differ.\n"
                    f"saved={s}\ncurrent={c}"
                )

    # ---- extension points (HierSplit 등 추가 meta) ----
    def _extra_ckpt_meta(self) -> Dict[str, Any]:
        return {}

    def _validate_extra_ckpt_meta(self, ckpt: Dict[str, Any]):
        return

    # ---- common save/load (single-optimizer runners) ----
    @master_only
    def save_model(self, epoch: int):
        from easytorch.core.checkpoint import save_ckpt, backup_last_ckpt, clear_ckpt

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        ckpt_dict: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metrics": self.best_metrics,
            **self._partition_meta_for_ckpt(),
            **self._extra_ckpt_meta(),
        }
        if getattr(self, "optim", None) is not None:
            ckpt_dict["optim_state_dict"] = self.optim.state_dict()

        # backup last
        if epoch > 1:
            last_ckpt_path = self.get_ckpt_path(epoch - 1)
            if os.path.exists(last_ckpt_path):
                backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt_dict, ckpt_path, self.logger)

        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    def load_model_resume(self, strict: bool = True):
        from easytorch.core.checkpoint import load_ckpt

        try:
            ckpt = load_ckpt(self.ckpt_save_dir, logger=self.logger)

            # validate meta
            self._validate_partition_meta(ckpt)
            self._validate_extra_ckpt_meta(ckpt)

            # load model
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(ckpt["model_state_dict"], strict=strict)
            else:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)

            # load optim
            if getattr(self, "optim", None) is not None and ckpt.get("optim_state_dict", None) is not None:
                self.optim.load_state_dict(ckpt["optim_state_dict"])

            self.start_epoch = ckpt["epoch"]
            if ckpt.get("best_metrics") is not None:
                self.best_metrics = ckpt["best_metrics"]

            if self.scheduler is not None:
                self.scheduler.last_epoch = ckpt["epoch"]

            if ckpt.get("early_stopping_completed", False):
                self.early_stopping_completed = True
                self.logger.info("Early stopping was already completed. Skipping training.")
            else:
                self.logger.info(f"Resume training from epoch {self.start_epoch}")

        except (FileNotFoundError, IndexError):
            self.logger.info("No checkpoint found, start training from scratch.")
