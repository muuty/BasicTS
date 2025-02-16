import time
from typing import List, Protocol
from functools import wraps


def measure_time(func):
    """
    함수 실행 시간을 측정하는 데코레이터.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱ {func.__name__} 실행 시간: {end_time - start_time:.4f}초")
        return result
    return wrapper


class BaseSelection(Protocol):
    """Abstract base class for custom samplers."""

    @measure_time
    def select_indices(self) -> List[int]:
        ...
