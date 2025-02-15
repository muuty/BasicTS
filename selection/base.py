from typing import List
from typing import Protocol


class BaseSelection(Protocol):
    """Abstract base class for custom samplers."""

    def select_indices(self) -> List[int]:
        ...
