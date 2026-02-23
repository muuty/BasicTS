# replay_memory.py
from typing import List, Optional, NamedTuple
import random
import heapq
import numpy as np


class ReplayItem(NamedTuple):
    """
    리플레이 버퍼에 저장되는 최소 정보 아이템.
    score가 첫 번째 필드이므로 heapq가 자동으로 score를 기준으로 비교합니다.
    """
    score: float
    index: int



from collections import deque
from typing import List, Optional
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def size(self) -> int:
        return len(self._buffer)
    
    def enqueue(self, score: float, index: int) -> None:
        """FIFO로 추가. O(1)."""
        self._buffer.append(ReplayItem(score=score, index=index))
    
    def sample(self, batch_size: int) -> List[int]:
        """Score 기반 weighted sampling."""
        if len(self._buffer) == 0:
            return []
        
        num_samples = min(batch_size, len(self._buffer))
        items = list(self._buffer)
        scores = np.array([item.score for item in items])
        
        # score가 0 이하인 경우 처리
        scores = np.maximum(scores, 1e-8)
        probs = scores / scores.sum()
        
        indices = np.random.choice(len(items), size=num_samples, replace=False, p=probs)
        return [items[i].index for i in indices]
    
    def clear(self) -> None:
        self._buffer.clear()


class WeightedIndexBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def size(self) -> int:
        return len(self._buffer)
    
    def enqueue(self, score: float, index: int) -> None:
        """FIFO로 추가. O(1)."""
        self._buffer.append(ReplayItem(score=score, index=index))
    
    def sample(self, batch_size: int) -> List[int]:
        """Score 기반 weighted sampling."""
        if len(self._buffer) == 0:
            return []
        
        num_samples = min(batch_size, len(self._buffer))
        items = list(self._buffer)
        scores = np.array([item.score for item in items])
        
        # score가 0 이하인 경우 처리
        scores = np.maximum(scores, 1e-8)
        probs = scores / scores.sum()
        
        indices = np.random.choice(len(items), size=num_samples, replace=False, p=probs)
        return [items[i].index for i in indices]
    
    def clear(self) -> None:
        self._buffer.clear()


class UniformIndexBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        if seed is not None:
            np.random.seed(seed)
    
    @property
    def size(self) -> int:
        return len(self._buffer)
    
    def enqueue(self, index: int) -> None:
        self._buffer.append(index)
    
    def enqueue_batch(self, indices: List[int]) -> None:
        for idx in indices:
            self._buffer.append(idx)
    
    def sample(self, batch_size: int) -> List[int]:
        if len(self._buffer) == 0:
            return []
        
        num_samples = min(batch_size, len(self._buffer))
        chosen = np.random.choice(len(self._buffer), size=num_samples, replace=False)
        return [self._buffer[i] for i in chosen]
    
    def clear(self) -> None:
        self._buffer.clear()
    
    def get_all_indices(self) -> List[int]:
        return list(self._buffer)