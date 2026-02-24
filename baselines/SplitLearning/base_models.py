#!/usr/bin/env python3
"""추상 베이스 모델 클래스들"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseClientModel(nn.Module, ABC):
    """클라이언트 모델 베이스 클래스"""
    
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        인코더 forward pass
        Args:
            x: 입력 데이터
        Returns:
            encoded: 인코딩된 특징
        """
        pass
    
    @abstractmethod
    def forward_decoder(self, *args, **kwargs) -> torch.Tensor:
        """
        디코더 forward pass
        Returns:
            prediction: 예측 결과
        """
        pass


class BaseServerModel(nn.Module, ABC):
    """서버 모델 베이스 클래스"""
    
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        서버 forward pass
        Args:
            x: 클라이언트들로부터 받은 특징
        Returns:
            processed: 처리된 특징
        """
        pass


class PoolingClientMixin:
    """Pooling 기능을 가진 클라이언트를 위한 Mixin"""
    
    @abstractmethod
    def make_tokens(self, h: torch.Tensor) -> torch.Tensor:
        """
        노드 특징을 토큰으로 압축
        Args:
            h: (B, N, D) 노드 특징
        Returns:
            tokens: (B, K, D) 압축된 토큰
        """
        pass
    
    @abstractmethod
    def tokens_to_nodes(self, tokens: torch.Tensor, client_h: torch.Tensor) -> torch.Tensor:
        """
        토큰을 다시 노드 특징으로 확장
        Args:
            tokens: (B, K, D) 토큰
            client_h: (B, N, D) 원래 노드 특징
        Returns:
            expanded: (B, N, D) 확장된 특징
        """
        pass

