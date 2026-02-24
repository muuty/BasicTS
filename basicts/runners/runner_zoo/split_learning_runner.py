#!/usr/bin/env python3
"""
Split Learning Runner

Split Learning의 특성을 반영한 Runner입니다.
- 클라이언트 모델들과 서버 모델이 분리
- 각각 독립적인 optimizer 관리
- Forward/Backward 통신량 측정

핵심 구조:
- Forward: 클라이언트 → 서버 (activation 전송)
- Backward: 서버 → 클라이언트 (gradient 전송)
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class SplitLearningRunner(SimpleTimeSeriesForecastingRunner):
    """
    Split Learning Runner
    
    클라이언트 모델들과 서버 모델에 대해 개별 optimizer를 관리하고,
    통신량을 측정합니다.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        # 클라이언트/서버별 optimizer
        self._client_optimizers: Optional[List[torch.optim.Optimizer]] = None
        self._server_optimizer: Optional[torch.optim.Optimizer] = None
        
        # 클라이언트/서버별 scheduler
        self._client_schedulers: Optional[List] = None
        self._server_scheduler = None
    
    def _setup_split_optimizers(self, cfg: Dict):
        """클라이언트별 + 서버 optimizer 설정"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        optim_cfg = cfg['TRAIN']['OPTIM']
        optim_class = getattr(torch.optim, optim_cfg['TYPE'])
        optim_params = optim_cfg['PARAM']
        
        # 클라이언트 optimizers
        self._client_optimizers = []
        for client_model in model.get_client_models():
            optimizer = optim_class(client_model.parameters(), **optim_params)
            self._client_optimizers.append(optimizer)
        
        # 서버 optimizer (서버가 Identity 등으로 파라미터가 0개일 수 있음)
        server_params = [p for p in model.get_server_model().parameters() if p.requires_grad]
        if len(server_params) == 0:
            self._server_optimizer = None
            self.logger.warning(
                "Server model has no trainable parameters. "
                "Skip server optimizer/scheduler. (This is expected for Identity server modules like STGCNServerSpatial.)"
            )
        else:
            self._server_optimizer = optim_class(server_params, **optim_params)
        
        # Schedulers
        self._client_schedulers = []
        self._server_scheduler = None
        
        sched_cfg = cfg['TRAIN']['LR_SCHEDULER']
        sched_class = getattr(torch.optim.lr_scheduler, sched_cfg['TYPE'])
        sched_params = sched_cfg['PARAM']
        
        for optimizer in self._client_optimizers:
            scheduler = sched_class(optimizer, **sched_params)
            self._client_schedulers.append(scheduler)
        
        if self._server_optimizer is not None:
            self._server_scheduler = sched_class(self._server_optimizer, **sched_params)
        
        num_clients = len(self._client_optimizers)
        self.logger.info(f'Setup Split Learning optimizers: {num_clients} clients + 1 server')
    
    def init_training(self, cfg: Dict):
        """Initialize training for Split Learning"""
        super().init_training(cfg)
        
        # Split Learning용 optimizer 설정
        self._setup_split_optimizers(cfg)
    
    def train_iters(
        self,
        epoch: int,
        iter_index: int,
        data: Union[torch.Tensor, Tuple]
    ) -> torch.Tensor:
        """
        Split Learning training iteration
        
        1. 모든 optimizer zero_grad
        2. Forward (통신량 포함)
        3. Loss 계산 & backward
        4. Backward 통신량 계산
        5. 모든 optimizer step
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 1. Zero grad (클라이언트들 + 서버)
        for optimizer in self._client_optimizers:
            optimizer.zero_grad(set_to_none=True)
        if self._server_optimizer is not None:
            self._server_optimizer.zero_grad(set_to_none=True)
        
        # 2. Forward pass
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        
        # 3. Loss & backward
        loss = self.metric_forward(self.loss, forward_return)
        loss.backward()
        
        # 4. Backward 통신량 계산 (model 내부에서 누적)
        model.compute_backward_comm(forward_return)
        
        
        # 6. Optimizer step (클라이언트들 + 서버)
        for optimizer in self._client_optimizers:
            optimizer.step()
        if self._server_optimizer is not None:
            self._server_optimizer.step()
        
        # Metrics 업데이트
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', loss.item(), weight)
        
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)
        
        # None 반환하여 BaseEpochRunner.backward() 호출 방지
        return None
    
    def on_epoch_end(self, epoch: int) -> None:
        """Epoch 종료 시 scheduler step"""
        # 클라이언트 schedulers
        for scheduler in self._client_schedulers:
            if scheduler is not None:
                scheduler.step()
        
        # 서버 scheduler
        if self._server_scheduler is not None:
            self._server_scheduler.step()
        
        # LR 로깅 (서버 우선, 없으면 첫 클라이언트 기준)
        lr_val = None
        if self._server_scheduler is not None:
            lr_val = self._server_scheduler.get_last_lr()[0]
        elif self._client_schedulers is not None and len(self._client_schedulers) > 0 and self._client_schedulers[0] is not None:
            lr_val = self._client_schedulers[0].get_last_lr()[0]
        if lr_val is not None:
            self.update_epoch_meter('train/lr', lr_val)
        
        # 부모의 scheduler step 방지
        original_scheduler = self.scheduler
        self.scheduler = None
        super().on_epoch_end(epoch)
        self.scheduler = original_scheduler
    
    def _save_test_metrics(self):
        """Save test metrics with communication stats"""
        metrics_results = {}
        metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}
        
        # 통신량 추가
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'get_communication_stats'):
            metrics_results['communication_cost'] = model.get_communication_stats()

        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)
