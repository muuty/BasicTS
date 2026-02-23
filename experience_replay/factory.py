from typing import Optional
from torch.utils.data import Dataset
from experience_replay.experience_replay import (
    BaseReplay, DifficultyReplay, RepresentativeReplay, RandomReplay, AdaptiveReplay
)
from coreset.k_medoids import KMedoidsSelection


def create_replay(cfg, dataset: Dataset) -> BaseReplay:
    method = cfg.EXPERIENCE_REPLAY.METHOD.lower()
    dataset_size = len(dataset)
    capacity = int(cfg.EXPERIENCE_REPLAY.CAPACITY_RATIO * dataset_size)
    capacity_ratio = cfg.EXPERIENCE_REPLAY.CAPACITY_RATIO
    seed = cfg.EXPERIENCE_REPLAY.get('SEED', None)
    batch_size = cfg.EXPERIENCE_REPLAY.BATCH_SIZE
    
    if method == "difficult":
        return DifficultyReplay(capacity=capacity, seed=seed, batch_size=batch_size)
    
    elif method == "representative":
        indices = _compute_kmedoids_indices(cfg, dataset, capacity_ratio, seed)
        print(f"Representative indices: {indices}")
        replay = RepresentativeReplay(capacity=capacity, seed=seed, batch_size=batch_size)
        replay.initialize(indices=indices)
        return replay
    
    elif method == "random":
        return RandomReplay(seed=seed, batch_size=batch_size, dataset_size=dataset_size)
    
    elif method == "adaptive":
        indices = _compute_kmedoids_indices(cfg, dataset, capacity_ratio, seed)
        difficulty_ratio = cfg.EXPERIENCE_REPLAY.get('DIFFICULTY_RATIO', 0.5)
        replay = AdaptiveReplay(
            capacity=capacity, 
            seed=seed, 
            batch_size=batch_size,
            difficulty_ratio=difficulty_ratio
        )
        replay.initialize(indices=indices)
        return replay
    
    else:
        raise ValueError(f"Unknown replay method: {method}")


def _compute_kmedoids_indices(cfg, dataset: Dataset, capacity_ratio: float, seed: Optional[int] = None) -> list:
    selection = KMedoidsSelection(
        dataset=dataset,
        ratio=capacity_ratio,
        embedding_model=None,
        model_config=cfg,
        seed=seed
    )
    return selection.select_indices()