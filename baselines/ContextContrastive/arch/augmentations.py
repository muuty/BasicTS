"""
Augmentation strategies for Contrastive Learning.

Supports individual and combined augmentation strategies for ablation studies.
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import scipy.sparse as sp


class TemporalMasking(nn.Module):
    """
    Temporal Masking: Randomly mask entire time steps.

    This tests the model's ability to learn temporal patterns
    even when some time steps are missing.
    """

    def __init__(self, mask_ratio: float = 0.15, return_mask: bool = False):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.return_mask = return_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            x_masked: [B, T, N, C] - masked time series (or tuple with mask if return_mask=True)
        """
        B, T, N, C = x.shape
        # Create mask: True = MASKED (to be reconstructed), False = visible
        mask = torch.rand(B, T, 1, 1, device=x.device) < self.mask_ratio
        mask = mask.expand(B, T, N, C)

        if self.return_mask:
            return x, mask
        # Legacy behavior: zero out masked positions
        return x * (~mask).float()


class FeatureMasking(nn.Module):
    """
    Feature Masking: Randomly mask entire feature channels.

    This tests the model's ability to learn from partial feature information.
    For traffic data: flow, time_of_day, day_of_week
    """

    def __init__(self, mask_ratio: float = 0.3, return_mask: bool = False):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.return_mask = return_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            x_masked: [B, T, N, C] - masked time series (or tuple with mask if return_mask=True)
        """
        B, T, N, C = x.shape
        # Create mask: True = MASKED (to be reconstructed), False = visible
        mask = torch.rand(B, 1, 1, C, device=x.device) < self.mask_ratio
        mask = mask.expand(B, T, N, C)

        if self.return_mask:
            return x, mask
        # Legacy behavior: zero out masked positions
        return x * (~mask).float()


class NodeMasking(nn.Module):
    """
    Node Masking: Randomly mask entire nodes (sensors).

    This tests the model's ability to infer from spatial neighbors.
    """

    def __init__(self, mask_ratio: float = 0.15, return_mask: bool = False):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.return_mask = return_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            x_masked: [B, T, N, C] - masked time series (or tuple with mask if return_mask=True)
        """
        B, T, N, C = x.shape
        # Create mask: True = MASKED (to be reconstructed), False = visible
        mask = torch.rand(B, 1, N, 1, device=x.device) < self.mask_ratio
        mask = mask.expand(B, T, N, C)

        if self.return_mask:
            return x, mask
        # Legacy behavior: zero out masked positions
        return x * (~mask).float()


class NeighborhoodMasking(nn.Module):
    """
    Neighborhood Masking: Mask spatially contiguous regions based on graph structure.

    Instead of random masking, this selects seed nodes and masks their top-k
    nearest neighbors together. This forces the model to learn long-range spatial
    dependencies by reconstructing entire masked regions.

    Key insight: Traffic patterns propagate spatially (e.g., congestion spreads).
    By masking connected regions, the model must learn to infer local patterns
    from distant observations.

    Design for dense graphs: Uses top-k neighbors per node instead of k-hop,
    which works better for distance-decay weighted adjacency matrices.
    """

    def __init__(
        self,
        adj_mx_path: str = None,
        mask_ratio: float = 0.3,
        neighbors_per_seed: int = 10,  # Top-k neighbors to mask with each seed
        seed_strategy: str = 'random',  # 'random' or 'hub'
    ):
        """
        Args:
            adj_mx_path: Path to adjacency matrix pickle file
            mask_ratio: Target ratio of nodes to mask (approximate)
            neighbors_per_seed: Number of nearest neighbors to mask with each seed
            seed_strategy: 'random' for random seeds, 'hub' for high-degree nodes
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.neighbors_per_seed = neighbors_per_seed
        self.seed_strategy = seed_strategy

        # Load and process adjacency matrix
        if adj_mx_path and os.path.exists(adj_mx_path):
            self._load_adj_matrix(adj_mx_path)
        else:
            self.top_k_neighbors = None
            self.num_nodes = None

    def _load_adj_matrix(self, adj_mx_path: str):
        """Load adjacency matrix and precompute top-k neighborhoods."""
        with open(adj_mx_path, 'rb') as f:
            data = pickle.load(f)

        # Standard format: (sensor_ids, sensor_id_to_ind, adj_mx) or just array
        adj_mx = data[2] if hasattr(data, '__getitem__') and not hasattr(data, 'shape') else data

        # Convert to numpy array (handles sparse, tensor, or ndarray)
        if sp.issparse(adj_mx):
            adj_mx = adj_mx.toarray()
        adj_mx = np.asarray(adj_mx)

        self.num_nodes = adj_mx.shape[0]

        # Keep weighted matrix for top-k selection
        adj_weighted = adj_mx.astype(np.float32)
        np.fill_diagonal(adj_weighted, 0)  # Exclude self for neighbor selection

        # Compute node degrees (using threshold for "strong" connections)
        adj_strong = (adj_mx > 0.5).astype(np.float32)
        degrees = adj_strong.sum(axis=1)
        self.register_buffer('node_degrees', torch.from_numpy(degrees))

        # Precompute top-k neighbors for each node (by weight = proximity)
        self._precompute_top_k_neighbors(adj_weighted)

    def _precompute_top_k_neighbors(self, adj_weighted: np.ndarray):
        """Precompute top-k nearest neighbors for each node."""
        N = adj_weighted.shape[0]
        k = self.neighbors_per_seed

        self.top_k_neighbors = []
        for i in range(N):
            # Get top-k neighbors by weight (highest = closest)
            weights = adj_weighted[i]
            # argsort descending, take top k
            top_k_idx = np.argsort(weights)[-k:][::-1]
            # Filter out zero-weight neighbors
            top_k_idx = top_k_idx[weights[top_k_idx] > 0]
            self.top_k_neighbors.append(top_k_idx)

        # Neighborhood size = seed + neighbors
        self.avg_neighborhood_size = 1 + np.mean([len(n) for n in self.top_k_neighbors])

    def _select_seed_nodes(self, B: int, N: int, device: torch.device) -> list:
        """Select seed nodes for each batch item."""
        # Estimate number of seeds needed to achieve mask_ratio
        target_masked = int(N * self.mask_ratio)
        num_seeds = max(1, int(target_masked / max(1, self.avg_neighborhood_size)))

        seeds_per_batch = []
        for _ in range(B):
            if self.seed_strategy == 'hub' and self.node_degrees is not None:
                # Sample proportionally to degree (hub nodes more likely)
                probs = self.node_degrees.cpu().numpy() + 1  # Add 1 to avoid zero
                probs = probs / probs.sum()
                seeds = np.random.choice(N, size=min(num_seeds, N), replace=False, p=probs)
            else:
                # Random selection
                seeds = np.random.choice(N, size=min(num_seeds, N), replace=False)
            seeds_per_batch.append(seeds)

        return seeds_per_batch

    def _get_mask_from_seeds(self, seeds: np.ndarray, N: int) -> np.ndarray:
        """Expand seeds to their top-k neighborhoods and create mask."""
        masked_nodes = set()

        for seed in seeds:
            # Add seed itself
            masked_nodes.add(seed)
            # Add top-k neighbors
            if self.top_k_neighbors is not None:
                neighbors = self.top_k_neighbors[seed]
                masked_nodes.update(neighbors)

        # Create binary mask (1 = keep, 0 = mask)
        mask = np.ones(N, dtype=np.float32)
        for node in masked_nodes:
            mask[node] = 0

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, C] - input time series
        Returns:
            x_masked: [B, T, N, C] - masked time series
        """
        B, T, N, C = x.shape

        if self.top_k_neighbors is None:
            # Fallback to random node masking if no adj matrix
            mask = torch.rand(B, 1, N, 1, device=x.device) > self.mask_ratio
            return x * mask.float()

        if not self.training:
            # No masking during eval
            return x

        # Select seeds and create masks for each batch item
        seeds_per_batch = self._select_seed_nodes(B, N, x.device)

        masks = []
        for seeds in seeds_per_batch:
            mask = self._get_mask_from_seeds(seeds, N)
            masks.append(mask)

        # Stack masks: [B, N] -> [B, 1, N, 1]
        mask_tensor = torch.tensor(np.stack(masks), device=x.device, dtype=x.dtype)
        mask_tensor = mask_tensor.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]

        return x * mask_tensor


class GaussianNoise(nn.Module):
    """Add Gaussian noise to input."""

    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class Scaling(nn.Module):
    """Random scaling augmentation."""

    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        super().__init__()
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            scale = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(
                self.scale_min, self.scale_max
            )
            return x * scale
        return x


class ConfigurableAugmentation(nn.Module):
    """
    Configurable augmentation module for ablation studies.

    Allows specifying which augmentations to apply and their parameters.
    """

    AUGMENTATION_REGISTRY = {
        'temporal_masking': TemporalMasking,
        'feature_masking': FeatureMasking,
        'node_masking': NodeMasking,
        'neighborhood_masking': NeighborhoodMasking,
        'gaussian_noise': GaussianNoise,
        'scaling': Scaling,
    }

    def __init__(self, augmentation_config: dict):
        """
        Args:
            augmentation_config: dict mapping augmentation name to its params
                Example: {
                    'temporal_masking': {'mask_ratio': 0.15},
                    'gaussian_noise': {'noise_std': 0.05},
                }
        """
        super().__init__()
        self.augmentations = nn.ModuleList()

        for aug_name, aug_params in augmentation_config.items():
            if aug_name in self.AUGMENTATION_REGISTRY:
                aug_class = self.AUGMENTATION_REGISTRY[aug_name]
                self.augmentations.append(aug_class(**aug_params))
            else:
                raise ValueError(f"Unknown augmentation: {aug_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all configured augmentations sequentially."""
        for aug in self.augmentations:
            x = aug(x)
        return x


# Convenience functions for common configurations
def get_temporal_masking_augmentation(mask_ratio: float = 0.15) -> ConfigurableAugmentation:
    """Temporal masking only."""
    return ConfigurableAugmentation({
        'temporal_masking': {'mask_ratio': mask_ratio},
    })


def get_feature_masking_augmentation(mask_ratio: float = 0.3) -> ConfigurableAugmentation:
    """Feature masking only."""
    return ConfigurableAugmentation({
        'feature_masking': {'mask_ratio': mask_ratio},
    })


def get_node_masking_augmentation(mask_ratio: float = 0.15) -> ConfigurableAugmentation:
    """Node masking only."""
    return ConfigurableAugmentation({
        'node_masking': {'mask_ratio': mask_ratio},
    })


def get_combined_augmentation(
    temporal_mask_ratio: float = 0.15,
    noise_std: float = 0.05,
) -> ConfigurableAugmentation:
    """Combined augmentation (temporal masking + noise)."""
    return ConfigurableAugmentation({
        'temporal_masking': {'mask_ratio': temporal_mask_ratio},
        'gaussian_noise': {'noise_std': noise_std},
    })
