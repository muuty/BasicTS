#!/usr/bin/env python3
"""
Independent Learning Runner

각 클라이언트가 완전히 독립적인 모델을 가지고,
자신의 노드에 대한 데이터만으로 학습합니다.

핵심:
- forward에서 전체 prediction 생성 (각 클라이언트 모델이 해당 노드만 처리)
- 각 클라이언트별 노드에 대해 loss 계산
- torch.autograd.backward(tensors=losses)로 각 loss를 독립적으로 backward
- 노드 집합이 disjoint하므로, 각 loss의 gradient는 해당 클라이언트 모델에만 전파

수학적 동치:
- 단일 optimizer 사용 == 개별 optimizer 사용
- 파라미터가 disjoint하고 gradient가 각 모델에만 흐르므로 결과 동일
- 단일 optimizer가 함수 호출 오버헤드 면에서 더 효율적

논문 정의:
"Independent learning (IL), where each client trains their model locally 
without interaction between clients."
"""

from typing import Dict, Optional, Tuple, Union

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class IndependentLearningRunner(SimpleTimeSeriesForecastingRunner):
    """
    Independent Learning을 위한 Runner
    
    SimpleTimeSeriesForecastingRunner를 상속하고,
    train_iters에서 각 클라이언트별 loss backward를 수행합니다.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        # 모델에서 client_nodes_list 가져오기
        self._client_nodes_list = None
    
    def _get_client_nodes_list(self):
        """모델에서 client_nodes_list 가져오기"""
        if self._client_nodes_list is None:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            self._client_nodes_list = model.client_nodes_list
        return self._client_nodes_list
    
    def train_iters(
        self,
        epoch: int,
        iter_index: int,
        data: Union[torch.Tensor, Tuple]
    ) -> torch.Tensor:
        """
        Training iteration with independent client-wise backward
        
        각 클라이언트의 loss가 해당 클라이언트 모델의 파라미터에만 
        gradient를 생성하므로, 단일 optimizer로도 IL이 구현됩니다.
        
        Args:
            epoch: 현재 에포크
            iter_index: 현재 iteration 인덱스
            data: DataLoader에서 제공된 데이터
            
        Returns:
            None (backward를 내부에서 직접 처리)
        """
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index
        
        # Forward pass
        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)
        
        # 클라이언트별 노드 리스트 가져오기
        client_nodes_list = self._get_client_nodes_list()
        
        # optimizer zero_grad
        self.optim.zero_grad()
        
        # 각 클라이언트별 loss 계산
        losses = []
        device = forward_return['prediction'].device
        
        for client_nodes in client_nodes_list:
            client_nodes_tensor = torch.tensor(client_nodes, device=device, dtype=torch.long)
            
            client_pred = forward_return['prediction'][:, :, client_nodes_tensor, :]
            client_target = forward_return['target'][:, :, client_nodes_tensor, :]
            
            # loss 계산
            client_loss = self.metric_forward(
                self.loss,
                {'prediction': client_pred, 'target': client_target}
            )
            losses.append(client_loss)
        
        # 각 loss에 대해 독립적으로 backward
        # 노드 집합이 disjoint하므로, 각 loss의 gradient는 해당 클라이언트 모델에만 전파
        torch.autograd.backward(tensors=losses)
        

        # optimizer step
        self.optim.step()
        
        # 로깅용 전체 loss 계산
        total_loss = self.metric_forward(self.loss, forward_return)
        weight = self._get_metric_weight(forward_return['target'])
        self.update_epoch_meter('train/loss', total_loss.item(), weight)
        
        # 메트릭 업데이트
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f'train/{metric_name}', metric_item.item(), weight)
        
        # None 반환하여 BaseEpochRunner.backward() 호출 방지
        return None
