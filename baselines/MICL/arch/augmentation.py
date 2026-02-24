"""Missing pattern augmentation for MICL pretraining.

Applies synthetic missing patterns to traffic data to create
asymmetric teacher-student views for contrastive learning.
"""

import random
import torch


class MissingAugmentation:
    """Apply missing pattern augmentation to traffic data.

    Operates on preprocessed data [B, T, N, C] where C=8:
    [flow, occ, speed, mask_flow, mask_occ, mask_speed, tod, dow]

    Augmentation types:
    - node_death: zero all physical channels for entire window
    - intermittent: randomly drop individual timesteps
    - block: zero a contiguous block of timesteps
    """

    def __init__(self, aug_types=('node_death', 'intermittent', 'block'),
                 rate_range=(0.1, 0.5),
                 physical_channels=(0, 1, 2),
                 mask_channels=(3, 4, 5)):
        self.aug_types = aug_types
        self.rate_range = rate_range
        self.physical_channels = list(physical_channels)
        self.mask_channels = list(mask_channels)
        self.all_channels = self.physical_channels + self.mask_channels

    def __call__(self, x):
        """
        Args:
            x: [B, T, N, C] preprocessed data

        Returns:
            augmented: [B, T, N, C] with synthetic missing patterns
            obs_weight: [B, N] per-node observation weight after augmentation
        """
        B, T, N, C = x.shape
        augmented = x.clone()

        rate = random.uniform(*self.rate_range)
        n_corrupt = max(1, int(N * rate))

        # Select nodes to corrupt (same for all batch items)
        corrupt_indices = torch.randperm(N, device=x.device)[:n_corrupt]

        # Select augmentation type
        aug_type = random.choice(self.aug_types)

        # Build temporal mask: 1=keep, 0=corrupt. Shape [B, T, n_corrupt]
        if aug_type == 'node_death':
            tmask = torch.zeros(B, T, n_corrupt, device=x.device)

        elif aug_type == 'intermittent':
            drop_rate = random.uniform(0.3, 0.8)
            tmask = (torch.rand(B, T, n_corrupt, device=x.device) > drop_rate).float()

        elif aug_type == 'block':
            block_len = random.randint(T // 4, T)
            start = random.randint(0, T - block_len)
            tmask = torch.ones(B, T, n_corrupt, device=x.device)
            tmask[:, start:start + block_len, :] = 0

        # Apply: zero out physical + mask channels where tmask=0
        tmask_expand = tmask.unsqueeze(-1)  # [B, T, n_corrupt, 1]
        for ch in self.all_channels:
            vals = augmented[:, :, corrupt_indices, ch]  # [B, T, n_corrupt]
            augmented[:, :, corrupt_indices, ch] = vals * tmask.squeeze()

        # Observation weight: mean of mask_flow (ch3) after augmentation
        obs_weight = augmented[:, :, :, self.mask_channels[0]].mean(dim=1)  # [B, N]
        obs_weight = obs_weight.clamp(min=0.01)

        return augmented, obs_weight
