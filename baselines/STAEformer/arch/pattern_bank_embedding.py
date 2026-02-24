"""Pattern Bank Embedding: factorized adaptive embedding.

Replaces the standard adaptive_embedding (T, N, D) with:
  pattern_bank: (K, T, D) - shared prototypes
  node_weights: (N, K)    - per-node mixing logits

Output: softmax(node_weights) @ pattern_bank -> (T, N, D)

Advantages for expanding sensor networks:
  - Per-node params: K (e.g. 8) vs D (e.g. 24)
  - New nodes share learned prototypes, only need to learn K mixing weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternBankEmbedding(nn.Module):

    def __init__(self, num_nodes: int, in_steps: int, adaptive_embedding_dim: int,
                 num_patterns: int = 8):
        super().__init__()
        self.num_patterns = num_patterns
        self.in_steps = in_steps
        self.adaptive_embedding_dim = adaptive_embedding_dim

        self.pattern_bank = nn.Parameter(
            torch.empty(num_patterns, in_steps, adaptive_embedding_dim))
        self.node_weights = nn.Parameter(
            torch.zeros(num_nodes, num_patterns))

        nn.init.xavier_uniform_(self.pattern_bank.view(num_patterns, -1).unsqueeze(0))

    @property
    def shape(self):
        """Compatible with adaptive_embedding.shape -> (T, N, D)."""
        N = self.node_weights.shape[0]
        return (self.in_steps, N, self.adaptive_embedding_dim)

    @property
    def device(self):
        return self.pattern_bank.device

    def forward(self) -> torch.Tensor:
        """Compute adaptive embedding from pattern bank.

        Returns:
            (T, N, D) tensor, same shape as standard adaptive_embedding.
        """
        weights = F.softmax(self.node_weights, dim=-1)  # (N, K)
        K, T, D = self.pattern_bank.shape
        pb_flat = self.pattern_bank.reshape(K, T * D)  # (K, T*D)
        emb_flat = weights @ pb_flat  # (N, T*D)
        emb = emb_flat.reshape(-1, T, D).permute(1, 0, 2)  # (T, N, D)
        return emb

    def expand_nodes(self, new_total: int, existing_idx: list):
        """Expand node_weights to accommodate new nodes.

        New nodes get zero weights (uniform mixing over all prototypes).
        Pattern bank stays unchanged (shared).

        Args:
            new_total: total number of nodes after expansion
            existing_idx: indices where existing nodes map in the new layout
        """
        old_weights = self.node_weights.data  # (N_old, K)
        K = old_weights.shape[1]
        new_weights = torch.zeros(new_total, K, device=old_weights.device)
        new_weights[existing_idx] = old_weights
        self.node_weights = nn.Parameter(new_weights)
