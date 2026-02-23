from typing import Optional, List, Any
import torch
from .replay_memory_new import WeightedIndexBuffer, UniformIndexBuffer

class ExperienceReplayNew:
    def __init__(
        self,
        *,
        capacity: int,
        seed: Optional[int] = None,
        batch_size: int,
        difficulty_ratio: float = 1.0,
    ):
        assert capacity > 0
        assert 0.0 <= difficulty_ratio <= 1.0
        
        self.capacity = capacity
        self.batch_size = batch_size
        self.difficulty_ratio = difficulty_ratio
        
        difficulty_capacity = int(capacity * difficulty_ratio)
        repr_capacity = capacity - difficulty_capacity
        
        self.difficulty_buffer = WeightedIndexBuffer(capacity=max(difficulty_capacity, 1), seed=seed)
        self.repr_buffer = UniformIndexBuffer(capacity=max(repr_capacity, 1), seed=seed)
        
        self.x_key, self.y_key, self.pred_key = "inputs", "target", "prediction"

    @torch.no_grad()
    def push_batch(self, *, data, forward_return) -> int:
        if self.difficulty_ratio == 0:
            return 0
        
        pred = forward_return[self.pred_key]
        target = data[self.y_key].to(pred.device)
        mae_b = (pred - target).abs().mean(dim=(1, 2, 3))
        
        data_index = data["index"]
        for b in range(mae_b.numel()):
            score = float(mae_b[b].item())
            index = int(data_index[b].item())
            self.difficulty_buffer.enqueue(score=score, index=index)
        
        return mae_b.numel()

    def set_repr_indices(self, indices: List[int]) -> None:
        self.repr_buffer.set_indices(indices)

    def sample(self, batch_size: int = None) -> List[int]:
        if batch_size is None:
            batch_size = self.batch_size
        
        diff_size = int(batch_size * self.difficulty_ratio)
        repr_size = batch_size - diff_size
        
        diff_indices = self.difficulty_buffer.sample(diff_size) if diff_size > 0 else []
        repr_indices = self.repr_buffer.sample(repr_size) if repr_size > 0 else []
        
        return diff_indices + repr_indices

    def get_all_indices(self) -> List[int]:
        diff_indices = []
        if hasattr(self.difficulty_buffer, '_buffer'):
            diff_indices = [item.index for item in self.difficulty_buffer._buffer]
        repr_indices = self.repr_buffer.get_all_indices()
        return diff_indices + repr_indices

    def size(self) -> int:
        return self.difficulty_buffer.size + self.repr_buffer.size

    def clear(self) -> None:
        self.difficulty_buffer.clear()
        self.repr_buffer.clear()

    def build_loss_mask(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, T, N, C = prediction.shape
        return torch.ones(B, N, dtype=torch.bool, device=prediction.device)