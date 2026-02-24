#!/usr/bin/env python3
"""
Federated Learning Runner (JY version) - Efficient Implementation

- ClientWiseIndependentLearningRunner 기반
- IndependentLearningRunner처럼 한 번에 forward 후 클라이언트별 backward
- Round 끝에 aggregation으로 파라미터 동기화
- FedAvg, FedProx 지원

Usage:
    CFG.RUNNER = FederatedLearningRunner
    CFG.FL = EasyDict()
    CFG.FL.NUM_ROUNDS = 50
    CFG.FL.AGGREGATOR = "fedavg"  # or "fedprox"
    CFG.FL.FEDPROX_MU = 0.01      # for fedprox only
"""
from __future__ import annotations

import copy
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .jy_independent_learning_runner_clientwise import ClientWiseIndependentLearningRunner


# ============================================================================
# Aggregators (FedAvg, FedProx)
# ============================================================================

class BaseAggregator(ABC):
    """Base Aggregator for Federated Learning"""
    
    # 노드 수에 따라 달라지는 파라미터 (aggregation에서 제외)
    NODE_SPECIFIC_PARAMS = ['adaptive_embedding', 'node_emb', 'tc1_ln', 'tc2_ln']
    
    def __init__(self):
        self.total_communication_bytes = 0
    
    @abstractmethod
    def aggregate(
        self,
        client_params: Dict[int, OrderedDict],
        client_weights: Dict[int, float],
        global_params: Optional[OrderedDict],
    ) -> Tuple[OrderedDict, Dict]:
        pass
    
    def get_proximal_loss(self, model: nn.Module, global_params: OrderedDict) -> torch.Tensor:
        return torch.tensor(0.0)
    
    def _is_node_specific_param(self, key: str) -> bool:
        return any(skip in key for skip in self.NODE_SPECIFIC_PARAMS)
    
    def _compute_communication_bytes(self, params: OrderedDict, num_clients: int) -> int:
        total_bytes = 0
        for key, value in params.items():
            if not self._is_node_specific_param(key):
                total_bytes += value.numel() * value.element_size()
        return total_bytes * num_clients * 2


class FedAvgAggregator(BaseAggregator):
    """FedAvg: Federated Averaging"""
    
    def aggregate(
        self,
        client_params: Dict[int, OrderedDict],
        client_weights: Dict[int, float],
        global_params: Optional[OrderedDict],
    ) -> Tuple[OrderedDict, Dict]:
        total_weight = sum(client_weights.values())
        normalized_weights = {cid: w / total_weight for cid, w in client_weights.items()}
        
        first_client_id = list(client_params.keys())[0]
        averaged_params = copy.deepcopy(client_params[first_client_id])
        
        for key in averaged_params.keys():
            if self._is_node_specific_param(key):
                continue
            
            averaged_params[key] = torch.zeros_like(averaged_params[key], dtype=torch.float32)
            
            for client_id, params in client_params.items():
                if key in params and params[key].shape == averaged_params[key].shape:
                    averaged_params[key] += normalized_weights[client_id] * params[key].float()
        
        num_clients = len(client_params)
        comm_bytes = self._compute_communication_bytes(averaged_params, num_clients)
        self.total_communication_bytes += comm_bytes
        
        return averaged_params, {'total_communication_bytes': comm_bytes}


class FedProxAggregator(FedAvgAggregator):
    """FedProx: FedAvg + Proximal Term"""
    
    def __init__(self, mu: float = 0.01):
        super().__init__()
        self.mu = mu
    
    def get_proximal_loss(self, model: nn.Module, global_params: OrderedDict) -> torch.Tensor:
        if self.mu == 0.0 or global_params is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        proximal_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if not param.requires_grad or self._is_node_specific_param(name):
                continue
            if name in global_params:
                global_param = global_params[name].to(param.device)
                if global_param.shape == param.shape:
                    proximal_loss += torch.sum((param - global_param) ** 2)
        
        return (self.mu / 2.0) * proximal_loss


def get_aggregator(name: str, **kwargs) -> BaseAggregator:
    name = name.lower()
    if name == "fedavg":
        return FedAvgAggregator()
    elif name == "fedprox":
        return FedProxAggregator(**kwargs)
    else:
        raise ValueError(f"Unknown aggregator: {name}")


# ============================================================================
# Federated Learning Runner (Efficient Version)
# ============================================================================

class FederatedLearningRunner(ClientWiseIndependentLearningRunner):
    """
    Federated Learning Runner - Efficient Implementation
    
    - IndependentLearningRunner처럼 한 번에 forward
    - 클라이언트별 backward (gradient가 자동으로 분리됨)
    - Round 끝에 aggregation
    """
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        # FL 설정
        fl_cfg = cfg.get("FL", {})
        self.aggregator_type = fl_cfg.get("AGGREGATOR", "fedavg")
        aggregator_kwargs = {"mu": fl_cfg.get("FEDPROX_MU", 0.01)} if self.aggregator_type == "fedprox" else {}
        self.aggregator = get_aggregator(self.aggregator_type, **aggregator_kwargs)
        
        # 통신 비용 추적
        self.total_communication_bytes = 0
        
        # 글로벌 파라미터 (FedProx용)
        self._global_params: Optional[OrderedDict] = None
        
        # 클라이언트별 optimizer (init_training에서 설정)
        self._client_optimizers: List[torch.optim.Optimizer] = []
        self._client_schedulers: List[Optional[Any]] = []
        
        self.logger.info(f"🌐 FederatedLearningRunner initialized (Efficient)")
        self.logger.info(f"   - Aggregator: {self.aggregator_type}")
    
    def _setup_client_optimizers(self, cfg: Dict):
        """클라이언트별 optimizer 설정"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        self._client_optimizers = []
        self._client_schedulers = []
        
        for client_model in model.client_models:
            optim_cfg = cfg['TRAIN']['OPTIM']
            optimizer = getattr(torch.optim, optim_cfg['TYPE'])(
                client_model.parameters(), **optim_cfg['PARAM']
            )
            self._client_optimizers.append(optimizer)
            
            if 'LR_SCHEDULER' in cfg['TRAIN']:
                sched_cfg = cfg['TRAIN']['LR_SCHEDULER']
                scheduler = getattr(torch.optim.lr_scheduler, sched_cfg['TYPE'])(
                    optimizer, **sched_cfg['PARAM']
                )
                self._client_schedulers.append(scheduler)
            else:
                self._client_schedulers.append(None)
    
    def init_training(self, cfg: Dict):
        """Initialize training"""
        super().init_training(cfg)
        self._setup_client_optimizers(cfg)
        
        # Resume 시 optimizer state 복원
        if getattr(self, '_pending_optim_states', None) is not None:
            for i, state in enumerate(self._pending_optim_states):
                if i < len(self._client_optimizers) and state is not None:
                    self._client_optimizers[i].load_state_dict(state)
            self._pending_optim_states = None
        
        if getattr(self, '_pending_scheduler_states', None) is not None:
            for i, state in enumerate(self._pending_scheduler_states):
                if i < len(self._client_schedulers) and state is not None and self._client_schedulers[i] is not None:
                    self._client_schedulers[i].load_state_dict(state)
            self._pending_scheduler_states = None
    
    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> Optional[torch.Tensor]:
        """
        IndependentLearningRunner처럼 전체 output에 대해 한 번에 loss 계산
        - FedAvg: 전체 loss 한 번 계산
        - FedProx: 전체 loss + 각 클라이언트별 proximal term
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        
        # 1. 전체 forward (IndependentClientEnsemble)
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        pred = forward_return["prediction"]  # [B, T, N, C]
        target = forward_return["target"]
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        num_clients = len(model.client_models)
        
        # 2. Zero grad for all client optimizers
        for optimizer in self._client_optimizers:
            optimizer.zero_grad()
        
        # 3. 전체 prediction에 대해 loss 계산 (IndependentLearning과 동일)
        loss = self.metric_forward(self.loss, {"prediction": pred, "target": target})
        
        # 4. FedProx: 각 클라이언트별 proximal term 추가
        if self.aggregator_type == "fedprox" and self._global_params is not None:
            # 노드 수 기반 가중치 (aggregation과 동일한 스케일)
            total_nodes = 0
            client_node_counts = []
            for ci in range(num_clients):
                n = len(model.get_client_nodes(ci))
                client_node_counts.append(n)
                total_nodes += n

            # weighted mean prox
            for ci in range(num_clients):
                wi = client_node_counts[ci] / max(total_nodes, 1)
                proximal_loss = self.aggregator.get_proximal_loss(
                    model.client_models[ci], self._global_params
                )
                loss = loss + wi * proximal_loss
        
        # 5. Backward (한 번만)
        loss.backward()
        
        # 6. 각 client optimizer step
        for optimizer in self._client_optimizers:
            optimizer.step()
        
        # 7. 로깅
        weight = self._get_metric_weight(target)
        self.update_epoch_meter("train/loss", loss.item(), weight)
        
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item(), weight)
        
        return None  # 부모가 backward 호출하지 않도록
    
    def on_epoch_end(self, epoch: int):
        """Epoch 끝에 aggregation 수행"""
        # Aggregation
        comm_info = self._aggregate_and_distribute()
        self.total_communication_bytes += comm_info['total_communication_bytes']
        
        # LR scheduler step
        for scheduler in self._client_schedulers:
            if scheduler is not None:
                scheduler.step()
        
        if self._client_schedulers and self._client_schedulers[0] is not None:
            self.update_epoch_meter('train/lr', self._client_schedulers[0].get_last_lr()[0])
        
        # 부모 호출 (validation, test, save 등)
        super().on_epoch_end(epoch)
    
    def _aggregate_and_distribute(self) -> Dict:
        """클라이언트 파라미터 집계 및 배포"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        num_clients = len(model.client_models)
        
        # 파라미터 수집
        client_params = {}
        client_weights = {}
        total_nodes = 0
        
        for ci in range(num_clients):
            client_params[ci] = copy.deepcopy(model.client_models[ci].state_dict())
            num_nodes = len(model.get_client_nodes(ci))
            client_weights[ci] = float(num_nodes)
            total_nodes += num_nodes
        
        for ci in client_weights:
            client_weights[ci] /= total_nodes
        
        # Aggregation
        averaged_params, comm_info = self.aggregator.aggregate(
            client_params=client_params,
            client_weights=client_weights,
            global_params=self._global_params,
        )
        
        # 글로벌 파라미터 저장
        self._global_params = copy.deepcopy(averaged_params)
        
        # 배포
        for ci in range(num_clients):
            current_params = model.client_models[ci].state_dict()
            for key in averaged_params.keys():
                if self.aggregator._is_node_specific_param(key):
                    continue
                if key in current_params and current_params[key].shape == averaged_params[key].shape:
                    current_params[key] = averaged_params[key].clone()
            model.client_models[ci].load_state_dict(current_params, strict=False)
        
        return comm_info
    
    def _save_test_metrics(self):
        """Save test metrics with communication stats"""
        metrics_results = {}
        metrics_results['overall'] = {
            k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()
        }
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i+1}'] = {
                k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()
            }
        
        metrics_results['communication_cost'] = {
            'total_communication_bytes': self.total_communication_bytes,
            'total_communication_mb': self.total_communication_bytes / (1024 * 1024),
            'aggregator': self.aggregator_type,
        }
        
        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)
    
    def _extra_ckpt_meta(self) -> Dict[str, Any]:
        """체크포인트 메타데이터"""
        meta = super()._extra_ckpt_meta() if hasattr(super(), '_extra_ckpt_meta') else {}
        meta.update({
            'fl_aggregator': self.aggregator_type,
            'fl_total_communication_bytes': self.total_communication_bytes,
        })
        
        if self._client_optimizers:
            meta['fl_client_optim_states'] = [opt.state_dict() for opt in self._client_optimizers]
        if self._client_schedulers and self._client_schedulers[0] is not None:
            meta['fl_client_scheduler_states'] = [
                s.state_dict() if s else None for s in self._client_schedulers
            ]
        if self._global_params is not None:
            meta['fl_global_params'] = self._global_params
        
        return meta
    
    def load_model_resume(self, strict: bool = True):
        """Resume"""
        from easytorch.core.checkpoint import load_ckpt
        
        try:
            ckpt = load_ckpt(self.ckpt_save_dir, logger=self.logger)
            
            self._validate_partition_meta(ckpt)
            self._validate_extra_ckpt_meta(ckpt)
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.load_state_dict(ckpt["model_state_dict"], strict=strict)
            
            self.start_epoch = ckpt["epoch"]
            if ckpt.get("best_metrics") is not None:
                self.best_metrics = ckpt["best_metrics"]
            
            if ckpt.get("fl_total_communication_bytes") is not None:
                self.total_communication_bytes = ckpt["fl_total_communication_bytes"]
            if ckpt.get("fl_global_params") is not None:
                self._global_params = ckpt["fl_global_params"]
            
            self._pending_optim_states = ckpt.get("fl_client_optim_states", None)
            self._pending_scheduler_states = ckpt.get("fl_client_scheduler_states", None)
            
            if ckpt.get("early_stopping_completed", False):
                self.early_stopping_completed = True
                self.logger.info("Early stopping completed.")
            else:
                self.logger.info(f"Resume from epoch {self.start_epoch}")
                self.logger.info(f"  Restored communication: {self.total_communication_bytes/(1024*1024):.2f}MB")
        
        except (FileNotFoundError, IndexError):
            self.logger.info("No checkpoint found.")
            self._pending_optim_states = None
            self._pending_scheduler_states = None
