from typing import Optional, Dict, Any, List
import torch
from .replay_memory import UniformIndexBuffer, ReplayBuffer, WeightedIndexBuffer
from abc import ABC, abstractmethod
import numpy as np


class ExperienceReplay:
    def __init__(
        self,
        *,
        capacity: int,
        # topq_nodes, node_thr 제거
        seed: Optional[int] = None,
        batch_size: int,
    ):
        assert capacity > 0
        self.memory = ReplayBuffer(capacity=capacity, seed=seed)
        self.x_key, self.y_key, self.pred_key = "inputs", "target", "prediction"
        self.batch_size = batch_size

    @torch.no_grad()
    def push_batch(self, *, data, forward_return) -> int:
        pred = forward_return[self.pred_key]
        target = data[self.y_key].to(pred.device)
        mae_b = (pred - target).abs().mean(dim=(1, 2, 3))
        
        data_index = data["index"]
        for b in range(mae_b.numel()):
            score = float(mae_b[b].item())
            index = int(data_index[b].item())
            self.memory.enqueue(score=score, index=index)
        
        return mae_b.numel()

    def get_all_indices(self) -> Optional[List[int]]:
        """
        버퍼에 저장된 모든 인덱스를 반환합니다.
        """
        return self.memory.get_all_indices()

    def sample(self, batch_size: int) -> List[int]:
        return self.memory.sample(batch_size)

    def build_loss_mask(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """모든 노드에 대해 True 반환 (전체 노드 학습)"""
        # [B, T_out, N, C_out] → [B, N] 크기의 all-True mask
        B, T, N, C = prediction.shape
        return torch.ones(B, N, dtype=torch.bool, device=prediction.device)

    def size(self) -> int:
        return self.memory.size

    def clear(self) -> None:
        self.memory.clear()

    def get_first_n_items(self, n: int):
        return self.memory.get_first_n_items(n)



class BaseReplay(ABC):
    @abstractmethod
    def push_batch(self, *, data, forward_return) -> int:
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> List[int]:
        pass
    
    @abstractmethod
    def size(self) -> int:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    def initialize(self, **kwargs) -> None:
        pass


class DifficultyReplay(BaseReplay):
    def __init__(self, *, capacity: int, seed: Optional[int] = None, batch_size: int):
        self.memory = WeightedIndexBuffer(capacity=capacity, seed=seed)
        self.batch_size = batch_size
        self.pred_key, self.y_key = "prediction", "target"
    
    @torch.no_grad()
    def push_batch(self, *, data, forward_return) -> int:
        pred = forward_return[self.pred_key]
        target = data[self.y_key].to(pred.device)
        mae_b = (pred - target).abs().mean(dim=(1, 2, 3))
        
        data_index = data["index"]
        for b in range(mae_b.numel()):
            score = float(mae_b[b].item())
            index = int(data_index[b].item())
            self.memory.enqueue(score=score, index=index)
        
        return mae_b.numel()
    
    def sample(self, batch_size: int = None) -> List[int]:
        if batch_size is None:
            batch_size = self.batch_size
        return self.memory.sample(batch_size)
    
    def size(self) -> int:
        return self.memory.size
    
    def clear(self) -> None:
        self.memory.clear()
    
    def get_all_indices(self) -> List[int]:
        return self.memory.get_all_indices()


class RepresentativeReplay(BaseReplay):
    def __init__(self, *, capacity: int, seed: Optional[int] = None, batch_size: int):
        self.memory = UniformIndexBuffer(capacity=capacity, seed=seed)
        self.batch_size = batch_size
    
    def initialize(self, *, indices: List[int]) -> None:
        self.memory.enqueue_batch(indices)
    
    def push_batch(self, *, data, forward_return) -> int:
        return 0
    
    def sample(self, batch_size: int = None) -> List[int]:
        if batch_size is None:
            batch_size = self.batch_size
        return self.memory.sample(batch_size)
    
    def size(self) -> int:
        return self.memory.size
    
    def clear(self) -> None:
        pass 
    
    def get_all_indices(self) -> List[int]:
        return self.memory.get_all_indices()



class RandomReplay(BaseReplay):
    def __init__(self, *, capacity: int, seed: Optional[int] = None, batch_size: int, dataset_size: int):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        if seed is not None:
            np.random.seed(seed)
    
    def push_batch(self, *, data, forward_return) -> int:
        return 0  # push 안 함
    
    def sample(self, batch_size: int = None) -> List[int]:
        if batch_size is None:
            batch_size = self.batch_size
        return np.random.choice(self.dataset_size, size=batch_size, replace=False).tolist()
    
    def size(self) -> int:
        return self.dataset_size
    
    def clear(self) -> None:
        pass
    
    def get_all_indices(self) -> List[int]:
        return list(range(self.dataset_size))

class AdaptiveReplay(BaseReplay):
    def __init__(
        self, 
        *, 
        capacity: int, 
        seed: Optional[int] = None, 
        batch_size: int,
        difficulty_ratio: float = 0.5
    ):
        self.batch_size = batch_size
        self.difficulty_ratio = difficulty_ratio
        
        diff_capacity = int(capacity * difficulty_ratio)
        repr_capacity = capacity - diff_capacity
        
        self.difficulty_buffer = WeightedIndexBuffer(capacity=max(diff_capacity, 1), seed=seed)
        self.repr_buffer = UniformIndexBuffer(capacity=max(repr_capacity, 1), seed=seed)
        
        self.pred_key, self.y_key = "prediction", "target"
    
    def initialize(self, *, indices: List[int]) -> None:
        self.repr_buffer.set_indices(indices)
    
    @torch.no_grad()
    def push_batch(self, *, data, forward_return) -> int:
        pred = forward_return[self.pred_key]
        target = data[self.y_key].to(pred.device)
        mae_b = (pred - target).abs().mean(dim=(1, 2, 3))
        
        data_index = data["index"]
        for b in range(mae_b.numel()):
            score = float(mae_b[b].item())
            index = int(data_index[b].item())
            self.difficulty_buffer.enqueue(score=score, index=index)
        
        return mae_b.numel()
    
    def sample(self, batch_size: int = None) -> List[int]:
        if batch_size is None:
            batch_size = self.batch_size
        
        diff_size = int(batch_size * self.difficulty_ratio)
        repr_size = batch_size - diff_size
        
        diff_indices = self.difficulty_buffer.sample(diff_size) if diff_size > 0 else []
        repr_indices = self.repr_buffer.sample(repr_size) if repr_size > 0 else []
        
        return diff_indices + repr_indices
    
    def size(self) -> int:
        return self.difficulty_buffer.size + self.repr_buffer.size
    
    def clear(self) -> None:
        self.difficulty_buffer.clear()
        self.repr_buffer.clear()
    
    def get_all_indices(self) -> List[int]:
        return self.difficulty_buffer.get_all_indices() + self.repr_buffer.get_all_indices()
