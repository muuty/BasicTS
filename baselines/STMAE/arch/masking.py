"""
Masking strategies for STMAE.

Feature Masking: Masks patches of time steps uniformly across the temporal dimension.
Structure Masking: Masks edges in the graph using random-walk based path masking.
"""

import torch
import torch.nn as nn
from einops import repeat


def mask_path(
    binaried_support: torch.Tensor,
    mask_ratio: float,
    walks_per_node: int = 1,
    walk_length: int = 3,
    start: str = 'node',
    p: float = 1.0,
    q: float = 1.0,
) -> tuple:
    """
    Random-walk based path masking on adjacency matrix.

    Args:
        binaried_support: Binary adjacency matrix [N, N]
        mask_ratio: Ratio of edges to mask
        walks_per_node: Number of random walks per starting node
        walk_length: Length of each random walk
        start: Starting point strategy ('node' or 'edge')
        p: Return parameter for biased random walk
        q: In-out parameter for biased random walk

    Returns:
        masked_edge_index: [2, num_masked] indices of masked edges
        num_masked: Number of masked edges
    """
    try:
        import torch_cluster
        random_walk = torch.ops.torch_cluster.random_walk
        from torch_geometric.utils import degree
        HAS_TORCH_CLUSTER = True
    except ImportError:
        HAS_TORCH_CLUSTER = False

    if not HAS_TORCH_CLUSTER:
        # Fallback to uniform random masking if torch_cluster not available
        return _uniform_edge_masking(binaried_support, mask_ratio)

    assert start in ['node', 'edge']

    edge_index = binaried_support.nonzero().t().contiguous()
    num_edges = edge_index.size(1)

    if mask_ratio == 0.0:
        return None, 0

    num_nodes = binaried_support.size(0)
    row, col = edge_index

    if start == 'edge':
        sample_mask = torch.rand(row.size(0), device=edge_index.device) <= mask_ratio
        start_nodes = row[sample_mask].repeat(walks_per_node)
    else:
        start_nodes = torch.randperm(num_nodes, device=edge_index.device)[:round(num_nodes * mask_ratio)].repeat(walks_per_node)

    if len(start_nodes) == 0:
        return None, 0

    deg = degree(row, num_nodes=num_nodes)

    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = random_walk(rowptr, col, start_nodes, walk_length, p, q)

    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    edge_mask[e_id] = False

    masked_edge_index = edge_index[:, ~edge_mask]
    num_masked = masked_edge_index.size(1)

    return masked_edge_index, num_masked


def _uniform_edge_masking(binaried_support: torch.Tensor, mask_ratio: float) -> tuple:
    """
    Fallback uniform random edge masking when torch_cluster is not available.

    Args:
        binaried_support: Binary adjacency matrix [N, N]
        mask_ratio: Ratio of edges to mask

    Returns:
        masked_edge_index: [2, num_masked] indices of masked edges
        num_masked: Number of masked edges
    """
    edge_index = binaried_support.nonzero().t().contiguous()
    num_edges = edge_index.size(1)

    if mask_ratio == 0.0 or num_edges == 0:
        return None, 0

    num_to_mask = round(num_edges * mask_ratio)
    perm = torch.randperm(num_edges, device=edge_index.device)[:num_to_mask]
    masked_edge_index = edge_index[:, perm]

    return masked_edge_index, num_to_mask


class FeatureMasking(nn.Module):
    """
    Feature masking module using patch_uniform strategy.

    Masks patches of time steps uniformly across the temporal dimension.
    mask: 1 = keep, 0 = masked
    """

    def __init__(self, patch_length: int = 1):
        """
        Args:
            patch_length: Length of each patch for masking. Default 1 collapses to uniform masking.
        """
        super().__init__()
        self.patch_length = patch_length

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask: torch.Tensor = None,
        patch_length: int = None,
    ) -> tuple:
        """
        Apply feature masking to input tensor.

        Args:
            x: Input tensor [B, T, N, D]
            mask_ratio: Ratio of patches to mask (0-1)
            mask: Pre-computed mask [B, T, N], if provided
            patch_length: Override default patch_length

        Returns:
            x_masked: Masked input tensor [B, T, N, D]
            mask: Binary mask [B, T, N] where 1=keep, 0=masked
        """
        B, T, N, D = x.shape
        patch_length = patch_length if patch_length is not None else self.patch_length

        # If mask is provided, use it directly
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1)
            return x_masked, mask

        # No masking case
        if mask_ratio == 0:
            mask = torch.ones([B, T, N], device=x.device)
            return x, mask

        assert patch_length <= T / 2 and T % patch_length == 0, \
            f'patch_length ({patch_length}) must be smaller than T/2 ({T/2}) and divide T ({T}) evenly.'

        num_patches = T // patch_length
        num_masked_patches = round(num_patches * mask_ratio)

        # Initialize mask with all ones (keep all)
        mask = torch.ones([B, T, N], device=x.device)

        # Randomly select patches to mask (same for all samples in batch)
        masked_indices = torch.randperm(num_patches, device=x.device)[:num_masked_patches]

        # Generate indices to mask
        start_indices = (masked_indices * patch_length).to(dtype=torch.long)
        end_indices = (start_indices + patch_length).to(dtype=torch.long)

        # Create ranges for all patches to mask
        ranges = torch.stack([torch.arange(start, end, device=x.device) for start, end in zip(start_indices, end_indices)])
        all_indices = torch.flatten(ranges)

        # Apply mask using scatter
        mask.scatter_(1, repeat(all_indices, 't -> b t n', b=B, n=N), 0)

        # Mask by zero
        x_masked = x * mask.unsqueeze(-1)

        return x_masked, mask


class StructureMasking(nn.Module):
    """
    Structure masking module using rw_fill strategy.

    Uses random-walk based path masking, then fills remaining quota with uniform masking.
    mask: 1 = keep, 0 = masked
    """

    def __init__(
        self,
        walks_per_node: int = 10,
        walk_length: int = 20,
        start: str = 'node',
        p: float = 1.0,
        q: float = 1.0,
    ):
        """
        Args:
            walks_per_node: Number of random walks per starting node
            walk_length: Length of each random walk
            start: Starting point strategy ('node' or 'edge')
            p: Return parameter for biased random walk
            q: In-out parameter for biased random walk
        """
        super().__init__()
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.start = start
        self.p = p
        self.q = q

    def forward(
        self,
        support: torch.Tensor,
        mask_ratio: float,
        mask: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply structure masking to adjacency matrix.

        Args:
            support: Adjacency/support matrix [N, N]
            mask_ratio: Ratio of edges to mask (0-1)
            mask: Pre-computed mask [N, N], if provided

        Returns:
            mask: Binary mask [N, N] where 1=keep, 0=masked
        """
        N = support.size(0)

        # If mask is provided, use it directly
        if mask is not None:
            return mask

        # No masking case
        if mask_ratio == 0:
            return torch.ones(N, N, device=support.device)

        goal_discard = round(N * N * mask_ratio)

        # STEP 1: Random-walk based path masking on fully connected graph
        binaried_support = torch.ones_like(support)

        walks_per_node = kwargs.get('walks_per_node', self.walks_per_node)
        walk_length = kwargs.get('walk_length', self.walk_length)
        start = kwargs.get('start', self.start)
        p = kwargs.get('p', self.p)
        q = kwargs.get('q', self.q)

        masked_edge_index, num_discard = mask_path(
            binaried_support,
            mask_ratio=mask_ratio,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            start=start,
            p=p,
            q=q
        )

        # STEP 2: If more needed, discard from path; else, uniform fill
        mask = torch.ones_like(binaried_support)

        if masked_edge_index is not None:
            if goal_discard > num_discard:
                # Use all random walk masks and add uniform masking
                mask[masked_edge_index[0, :], masked_edge_index[1, :]] = 0

                # Uniform masking for remaining quota
                remain_discard = goal_discard - num_discard
                remain_idx = torch.nonzero(binaried_support * mask)

                if remain_idx.size(0) > 0:
                    shuffled_idx = torch.randperm(remain_idx.size(0), device=mask.device)
                    mask_indices = shuffled_idx[:min(remain_discard, remain_idx.size(0))]
                    remain_actual_mask_idx = remain_idx[mask_indices]
                    mask[remain_actual_mask_idx[:, 0], remain_actual_mask_idx[:, 1]] = 0
            else:
                # Truncate random walk masks to goal
                masked_edge_index = masked_edge_index[:, :goal_discard]
                mask[masked_edge_index[0, :], masked_edge_index[1, :]] = 0
        else:
            # Fallback to pure uniform masking
            all_idx = torch.nonzero(binaried_support)
            if all_idx.size(0) > 0:
                shuffled_idx = torch.randperm(all_idx.size(0), device=mask.device)
                mask_indices = shuffled_idx[:min(goal_discard, all_idx.size(0))]
                actual_mask_idx = all_idx[mask_indices]
                mask[actual_mask_idx[:, 0], actual_mask_idx[:, 1]] = 0

        return mask
