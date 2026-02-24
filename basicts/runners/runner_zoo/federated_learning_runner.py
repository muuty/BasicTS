#!/usr/bin/env python3
"""
Federated Learning Runner

Federated Learning의 round 기반 학습을 구현합니다.
각 round에서:
1. 각 클라이언트가 local_epochs만큼 로컬 학습
2. Aggregator로 파라미터 집계
3. 글로벌 파라미터 배포

Aggregator는 모델에서 DI로 주입됩니다.
"""

import json
import os
import time
from typing import Dict, Tuple

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class FederatedLearningRunner(SimpleTimeSeriesForecastingRunner):
    """
    Federated Learning Runner
    
    BaseEpochRunner의 train() 메서드를 오버라이드하여
    round 기반 학습을 구현합니다.
    """
    
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        # Federated Learning 설정
        self.num_rounds = cfg['TRAIN'].get('NUM_ROUNDS', cfg['TRAIN']['NUM_EPOCHS'])
        self.local_epochs = cfg['TRAIN'].get('LOCAL_EPOCHS', 1)
        
        # 클라이언트별 optimizer 및 scheduler
        self._client_optimizers = None
        self._client_schedulers = None
        self._client_data_loaders = None
        
        # 통신 비용 추적
        self.total_communication_bytes = 0
    
    def _setup_client_optimizers(self, cfg: Dict):
        """클라이언트별 optimizer 및 scheduler 설정"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        num_clients = model.num_clients
        
        self._client_optimizers = []
        self._client_schedulers = []
        
        for client_idx in range(num_clients):
            client_model = model.client_models[client_idx]
            
            # Optimizer
            optim_cfg = cfg['TRAIN']['OPTIM']
            optimizer = getattr(torch.optim, optim_cfg['TYPE'])(
                client_model.parameters(),
                **optim_cfg['PARAM']
            )
            self._client_optimizers.append(optimizer)
            
            # Scheduler
            if cfg.has('TRAIN.LR_SCHEDULER'):
                sched_cfg = cfg['TRAIN']['LR_SCHEDULER']
                scheduler = getattr(torch.optim.lr_scheduler, sched_cfg['TYPE'])(
                    optimizer,
                    **sched_cfg['PARAM']
                )
                self._client_schedulers.append(scheduler)
            else:
                self._client_schedulers.append(None)
    
    def _setup_client_data_loaders(self, cfg: Dict):
        """클라이언트별 데이터 로더 설정"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 전체 데이터셋 사용 (각 클라이언트가 자신의 노드만 사용)
        # Federated Learning에서는 보통 각 클라이언트가 전체 시계열 데이터의 자신 노드 부분만 사용
        self._client_data_loaders = []
        
        for client_idx in range(model.num_clients):
            # 같은 데이터로더를 사용하되, forward에서 노드 선택
            self._client_data_loaders.append(self.train_data_loader)
    
    def init_training(self, cfg: Dict):
        """Initialize training for Federated Learning"""
        super().init_training(cfg)
        
        # 클라이언트별 optimizer/scheduler 설정
        self._setup_client_optimizers(cfg)
        self._setup_client_data_loaders(cfg)
    
    def train(self, cfg: Dict) -> None:
        """
        Federated Learning Train Loop
        
        epoch 대신 round 기반으로 학습합니다.
        """
        self.init_training(cfg)
        
        self.logger.info(f'Starting Federated Learning: {self.num_rounds} rounds, {self.local_epochs} local epochs')
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        for round_num in range(1, self.num_rounds + 1):
            # Early stopping
            if self.check_early_stopping():
                break
            
            self.on_epoch_start(round_num)  # 기존 hook 재사용
            round_start_time = time.time()
            
            # 1. Local Training
            round_loss = 0.0
            round_metrics = {name: 0.0 for name in self.metrics.keys()}
            
            for client_idx in range(model.num_clients):
                client_loss, client_metrics = self._local_train(
                    client_idx=client_idx,
                    local_epochs=self.local_epochs,
                    round_num=round_num,
                )
                round_loss += client_loss
                for name in round_metrics:
                    round_metrics[name] += client_metrics.get(name, 0.0)
            
            # 평균
            round_loss /= model.num_clients
            for name in round_metrics:
                round_metrics[name] /= model.num_clients
            
            # 2. Aggregation & Distribution
            comm_info = model.aggregate_and_distribute()
            self.total_communication_bytes += comm_info['total_communication_bytes']
            
            # Metrics 업데이트
            self.update_epoch_meter('train/loss', round_loss)
            for name, value in round_metrics.items():
                self.update_epoch_meter(f'train/{name}', value)
            
            # LR scheduler step
            for scheduler in self._client_schedulers:
                if scheduler is not None:
                    scheduler.step()
            
            if self._client_schedulers[0] is not None:
                self.update_epoch_meter('train/lr', self._client_schedulers[0].get_last_lr()[0])
            
            round_end_time = time.time()
            self.update_epoch_meter('train/time', round_end_time - round_start_time)
            
            self.on_epoch_end(round_num)  # validation, test, save
        
        self.on_training_end(cfg=cfg, train_epoch=round_num)
    
    def _local_train(
        self,
        client_idx: int,
        local_epochs: int,
        round_num: int,
    ) -> Tuple[float, Dict[str, float]]:
        """
        클라이언트 로컬 학습
        
        Args:
            client_idx: 클라이언트 인덱스
            local_epochs: 로컬 에포크 수
            round_num: 현재 라운드
            
        Returns:
            avg_loss: 평균 loss
            avg_metrics: 평균 metrics
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        client_model = model.client_models[client_idx]
        client_nodes = model.get_client_nodes(client_idx)
        optimizer = self._client_optimizers[client_idx]
        
        client_model.train()
        
        total_loss = 0.0
        total_metrics = {name: 0.0 for name in self.metrics.keys()}
        total_batches = 0
        
        for local_epoch in range(local_epochs):
            for batch_idx, data in enumerate(self.train_data_loader):
                optimizer.zero_grad()
                
                # Forward
                forward_return = self._client_forward(
                    client_idx=client_idx,
                    data=data,
                    train=True,
                )
                
                # Loss
                loss = self.metric_forward(self.loss, forward_return)
                
                # Backward
                loss.backward()
                
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
                
                for name, metric_func in self.metrics.items():
                    metric_val = self.metric_forward(metric_func, forward_return)
                    total_metrics[name] += metric_val.item()
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        avg_metrics = {
            name: total_metrics[name] / total_batches if total_batches > 0 else 0.0
            for name in total_metrics
        }
        
        return avg_loss, avg_metrics
    
    def _client_forward(
        self,
        client_idx: int,
        data: Dict,
        train: bool,
    ) -> Dict:
        """
        단일 클라이언트 forward pass
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        client_model = model.client_models[client_idx]
        client_nodes = model.get_client_nodes(client_idx)
        
        # Preprocessing
        data = self.preprocessing(data)
        
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        
        # 해당 클라이언트 노드만 선택
        history_client = history_data[:, :, client_nodes, :]
        future_client = future_data[:, :, client_nodes, :]
        
        # Feature selection
        history_client = self.select_input_features(history_client)
        
        # Client model forward
        pred_client = client_model(
            history_data=history_client,
            future_data=future_client,
            batch_seen=0,
            epoch=0,
            train=train,
        )
        
        if isinstance(pred_client, dict):
            pred_client = pred_client.get('prediction', pred_client)
        
        # Target
        target_client = self.select_target_features(future_client)
        
        # Postprocessing (rescale)
        forward_return = {
            'prediction': pred_client,
            'target': target_client,
            'inputs': self.select_target_features(history_client),
        }
        forward_return = self.postprocessing(forward_return)
        
        return forward_return
    
    def _save_test_metrics(self):
        """Save test metrics with communication stats"""
        metrics_results = {}
        metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}
        
        # 통신량 추가
        metrics_results['communication_cost'] = {
            'total_communication_bytes': self.total_communication_bytes,
            'total_communication_mb': self.total_communication_bytes / (1024 * 1024),
        }

        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)
