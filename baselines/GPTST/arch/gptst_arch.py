"""
GPT-ST: Generalist Pre-training for Spatio-Temporal Forecasting.

Main architecture including:
- GPTST_Model: Full pre-training model with encoder and decoder
- Hypergraph_encoder: Masked encoding with adaptive masking
- Hypergraph_decoder: Reconstruction decoder
"""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sthcn import STHCN
from .modules import MLP_RL, TimeFeature


class HypergraphEncoder(nn.Module):
    """Hypergraph encoder with masking for pre-training.

    Supports both random masking (initial training) and adaptive masking
    (curriculum learning after change_epoch).
    """

    def __init__(
        self,
        num_nodes: int,
        input_base_dim: int,
        input_extra_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int,
        embed_dim: int,
        embed_dim_spa: int,
        HS: int,
        HT: int,
        HT_Tem: int,
        num_route: int,
        mode: str = 'pretrain',
        scaler_zeros: float = 0.0,
        mask_ratio: float = 0.3,
        ada_mask_ratio: float = 1.0,
        ada_type: str = 'all',
        change_epoch: int = 10,
        epochs: int = 100,
        device: str = 'cuda:0',
    ):
        super(HypergraphEncoder, self).__init__()

        self.device = device
        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.input_extra_dim = input_extra_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.embed_dim_spa = embed_dim_spa
        self.HS = HS
        self.HT = HT
        self.HT_Tem = HT_Tem
        self.num_route = num_route
        self.mode = mode
        self.scaler_zeros = scaler_zeros
        self.mask_ratio = mask_ratio
        self.ada_mask_ratio = ada_mask_ratio
        self.ada_type = ada_type
        self.change_epoch = change_epoch
        self.epochs = epochs

        # Input projection
        self.dim_in_flow = nn.Linear(input_base_dim, hidden_dim, bias=True)

        # STHCN encoder
        self.STHCN_encode = STHCN(
            num_nodes=num_nodes,
            input_base_dim=input_base_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            embed_dim=embed_dim,
            embed_dim_spa=embed_dim_spa,
            HS=HS,
            HT=HT,
            HT_Tem=HT_Tem,
            num_route=num_route,
        )

        # MLP for adaptive masking classification
        self.MLP_RL = MLP_RL(input_base_dim, HS, hidden_dim, embed_dim, device)
        self.teb4mask = TimeFeature(embed_dim)
        self.neb4mask = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)

        self.act = nn.LeakyReLU()

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.neb4mask)

    def _random_mask(self, source: torch.Tensor) -> torch.Tensor:
        """Generate random mask for initial pre-training phase."""
        device = source.device
        mask_random_init = torch.rand_like(source[..., 0:self.input_base_dim].reshape(-1)).to(device)
        _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)
        mask_num = int(mask_random_init.shape[0] * self.mask_ratio)
        max_idx = max_idx_random[:mask_num]
        mask_random = torch.ones_like(max_idx_random)
        mask_random = mask_random.scatter_(0, max_idx, 0)
        mask_random = mask_random.reshape(-1, self.horizon, self.num_node, self.input_base_dim)
        return mask_random

    def _adaptive_mask(self, source: torch.Tensor, label_c: torch.Tensor,
                       epoch: int) -> torch.Tensor:
        """Generate adaptive mask based on classification labels."""
        device = source.device

        # Calculate number of random mask and adaptive mask
        train_process = ((epoch - self.change_epoch) / (self.epochs - self.change_epoch)) * self.ada_mask_ratio
        train_process = min(train_process, 1.0)

        mask_num_sum = int(source[:, :, :, 0].reshape(-1).shape[0] * self.mask_ratio)
        adaptive_mask_num = int(mask_num_sum * train_process)
        random_mask_num = mask_num_sum - adaptive_mask_num

        # Adaptive mask: randomly choose mask class until the adaptive_mask_num <= select_num
        list_c = list(range(0, self.HS))
        random.shuffle(list_c)
        select_c = torch.zeros_like(label_c).to(device)
        select_d = torch.zeros_like(label_c).to(device)
        select_f = torch.zeros_like(label_c).to(device)
        select_num = 0
        i = 0

        if self.ada_type == 'all':
            while select_num < adaptive_mask_num:
                select_c[label_c == list_c[i]] = 1
                select_num = torch.sum(select_c)
                i = i + 1
            if i >= 2:
                for k in range(i - 1):
                    select_d[label_c == list_c[k]] = 1
                adaptive_dnum = torch.sum(select_d)
                select_f[label_c == list_c[i - 1]] = 1
            else:
                adaptive_dnum = 0
                select_f = select_c.clone()
        else:
            while select_num < adaptive_mask_num:
                select_c[label_c == list_c[i]] = 1
                select_num = torch.sum(select_c)
                i = i + 1
            adaptive_dnum = 0
            select_f = select_c.clone()

        # Randomly choose top adaptive_mask_num to mask
        select_f = select_f.reshape(-1, self.horizon * self.num_node).reshape(-1)
        select_d = select_d.reshape(-1, self.horizon * self.num_node).reshape(-1)
        mask_adaptive_init = torch.rand_like(source[..., 0:1].reshape(-1)).to(device)
        mask_adaptive_init = select_f * mask_adaptive_init
        _, max_idx_adaptive = torch.sort(mask_adaptive_init, dim=0, descending=True)

        select_idx_adaptive = max_idx_adaptive[:(adaptive_mask_num - int(adaptive_dnum))]

        mask_adaptive = torch.ones_like(max_idx_adaptive)
        mask_adaptive = mask_adaptive.scatter_(0, select_idx_adaptive, 0)
        mask_adaptive = mask_adaptive * (1 - select_d)

        # Random mask
        mask_random_init = torch.rand_like(source[..., 0:1].reshape(-1)).to(device)
        mask_random_init = mask_adaptive * mask_random_init
        _, max_idx_random = torch.sort(mask_random_init, dim=0, descending=True)

        select_idx_random = max_idx_random[:random_mask_num]
        mask_random = torch.ones_like(max_idx_random)
        mask_random = mask_random.scatter_(0, select_idx_random, 0)
        mask_random = mask_random.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon, self.num_node)

        # Final mask
        mask_adaptive = mask_adaptive.reshape(-1, self.horizon * self.num_node).reshape(-1, self.horizon, self.num_node)
        final_mask = (mask_adaptive * mask_random).unsqueeze(-1)
        if self.input_base_dim != 1:
            final_mask = final_mask.repeat(1, 1, 1, self.input_base_dim)

        return final_mask

    def forward(self, source: torch.Tensor, label: torch.Tensor,
                epoch: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            source: Input data [B, T, N, D] where D = input_base_dim + temporal features
            label: Target labels (not used in current implementation)
            epoch: Current epoch number (for adaptive masking)

        Returns:
            If mode == 'pretrain':
                x_flow_encode: Encoded features [B, T, N, hidden_dim]
                final_mask: Binary mask [B, T, N, input_base_dim]
                softmax_guide_weight: Classification probabilities
                HS_cat: Spatial hyperedge weights
            If mode != 'pretrain':
                x_flow_encode: Encoded features [B, T, N, hidden_dim]
        """
        device = source.device

        if self.mode == 'pretrain':
            # Get time embeddings for classification
            day_index_ori = source[:, :, 0, self.input_base_dim:self.input_base_dim + 1]
            week_index_ori = source[:, :, 0, self.input_base_dim + 1:self.input_base_dim + 2]
            time_eb_logits = self.teb4mask(torch.cat([day_index_ori, week_index_ori], dim=-1))
            guide_weight = self.MLP_RL(source[..., 0:self.input_base_dim], time_eb_logits, self.neb4mask)
            softmax_guide_weight = F.softmax(guide_weight, dim=-1)

            if epoch is None or epoch <= self.change_epoch:
                # Random mask for initial training
                final_mask = self._random_mask(source)
            else:
                # Adaptive mask after change_epoch
                max_value, max_idx_all = torch.sort(softmax_guide_weight, dim=-1, descending=True)
                label_c = max_idx_all[..., 0]  # [batch_size, time_steps, num_node]
                final_mask = self._adaptive_mask(source, label_c, epoch)

            final_mask = final_mask.detach()
            mask_source = final_mask * source[..., 0:self.input_base_dim]
            mask_source[final_mask == 0] = self.scaler_zeros
            x_flow_eb = self.dim_in_flow(mask_source)
        else:
            x_flow_eb = self.dim_in_flow(source[..., 0:self.input_base_dim])

        x_flow_encode, HS1, HS2 = self.STHCN_encode(source, x_flow_eb)

        if self.mode == 'pretrain':
            HS_cat = HS1.squeeze(-1).transpose(-1, -2)
            return x_flow_encode, final_mask[..., :self.input_base_dim], softmax_guide_weight, HS_cat
        else:
            return x_flow_encode


class HypergraphDecoder(nn.Module):
    """Hypergraph decoder for reconstruction."""

    def __init__(
        self,
        num_nodes: int,
        input_base_dim: int,
        input_extra_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int,
        embed_dim: int,
        embed_dim_spa: int,
        HS: int,
        HT: int,
        HT_Tem: int,
        num_route: int,
        mode: str = 'pretrain',
    ):
        super(HypergraphDecoder, self).__init__()

        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.input_extra_dim = input_extra_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.embed_dim_spa = embed_dim_spa
        self.HS = HS
        self.HT = HT
        self.HT_Tem = HT_Tem
        self.num_route = num_route
        self.mode = mode

        # STHCN decoder
        self.STHCN_decode = STHCN(
            num_nodes=num_nodes,
            input_base_dim=input_base_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            embed_dim=embed_dim,
            embed_dim_spa=embed_dim_spa,
            HS=HS,
            HT=HT,
            HT_Tem=HT_Tem,
            num_route=num_route,
        )

        # Output projection
        self.dim_flow_out = nn.Linear(hidden_dim, input_base_dim, bias=True)
        self.act = nn.LeakyReLU()

    def forward(self, source: torch.Tensor, flow_encode_eb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source: Original input with temporal features [B, T, N, D]
            flow_encode_eb: Encoded features from encoder [B, T, N, hidden_dim]

        Returns:
            flow_out: Reconstructed output [B, T, N, input_base_dim]
            flow_decode: Decoded hidden features [B, T, N, hidden_dim]
        """
        flow_decode, HS1, HS2 = self.STHCN_decode(source, flow_encode_eb)
        flow_out = self.dim_flow_out(flow_decode)
        return flow_out, flow_decode


class GPTSTModel(nn.Module):
    """GPT-ST: Generalist Pre-training for Spatio-Temporal Forecasting.

    Full pre-training model with hypergraph encoder and decoder.
    Supports both pre-training (with masking and reconstruction) and
    fine-tuning modes.
    """

    def __init__(
        self,
        num_nodes: int,
        input_base_dim: int = 1,
        input_extra_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 1,
        horizon: int = 12,
        embed_dim: int = 16,
        embed_dim_spa: int = 8,
        HS: int = 4,
        HT: int = 4,
        HT_Tem: int = 4,
        num_route: int = 3,
        mode: str = 'pretrain',
        scaler_zeros: float = 0.0,
        mask_ratio: float = 0.3,
        ada_mask_ratio: float = 1.0,
        ada_type: str = 'all',
        change_epoch: int = 10,
        epochs: int = 100,
        device: str = 'cuda:0',
    ):
        """
        Args:
            num_nodes: Number of nodes in the graph
            input_base_dim: Number of input features (e.g., 1 for flow)
            input_extra_dim: Number of extra features (e.g., 2 for tod, dow)
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            horizon: Sequence length (number of timesteps)
            embed_dim: Embedding dimension for temporal features
            embed_dim_spa: Embedding dimension for spatial features
            HS: Number of spatial hyperedge heads
            HT: Number of temporal hyperedge heads
            HT_Tem: Number of temporal hypergraph heads
            num_route: Number of routing iterations
            mode: 'pretrain' or 'eval'
            scaler_zeros: Value to use for masked positions
            mask_ratio: Ratio of masked positions
            ada_mask_ratio: Maximum ratio for adaptive masking
            ada_type: Type of adaptive masking ('all' or other)
            change_epoch: Epoch to switch from random to adaptive masking
            epochs: Total number of epochs
            device: Device to use
        """
        super(GPTSTModel, self).__init__()

        self.num_node = num_nodes
        self.input_base_dim = input_base_dim
        self.input_extra_dim = input_extra_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.embed_dim_spa = embed_dim_spa
        self.HS = HS
        self.HT = HT
        self.HT_Tem = HT_Tem
        self.num_route = num_route
        self.mode = mode

        self.encoder = HypergraphEncoder(
            num_nodes=num_nodes,
            input_base_dim=input_base_dim,
            input_extra_dim=input_extra_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            embed_dim=embed_dim,
            embed_dim_spa=embed_dim_spa,
            HS=HS,
            HT=HT,
            HT_Tem=HT_Tem,
            num_route=num_route,
            mode=mode,
            scaler_zeros=scaler_zeros,
            mask_ratio=mask_ratio,
            ada_mask_ratio=ada_mask_ratio,
            ada_type=ada_type,
            change_epoch=change_epoch,
            epochs=epochs,
            device=device,
        )

        self.decoder = HypergraphDecoder(
            num_nodes=num_nodes,
            input_base_dim=input_base_dim,
            input_extra_dim=input_extra_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            horizon=horizon,
            embed_dim=embed_dim,
            embed_dim_spa=embed_dim_spa,
            HS=HS,
            HT=HT,
            HT_Tem=HT_Tem,
            num_route=num_route,
            mode=mode,
        )

    def forward_pretrain(self, source: torch.Tensor, label: torch.Tensor,
                         batch_seen: Optional[int] = None,
                         epoch: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """Forward pass for pre-training with masking and reconstruction."""
        flow_encode_eb, mask, probability, HS1 = self.encoder(source, label, epoch)
        flow_out, flow_decode = self.decoder(source, flow_encode_eb)
        # Return: reconstruction, decoded features, mask (1 = masked), probability, spatial heads
        return flow_out, flow_decode, 1 - mask, probability, HS1

    def forward_finetune(self, source: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass for fine-tuning (no masking)."""
        flow_encode_eb = self.encoder(source, label)
        # Return same format as pretrain for compatibility
        return flow_encode_eb, flow_encode_eb, flow_encode_eb, flow_encode_eb, flow_encode_eb

    def forward(self, source: torch.Tensor, label: torch.Tensor,
                batch_seen: Optional[int] = None,
                epoch: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            source: Input data [B, T, N, D]
            label: Target labels
            batch_seen: Number of batches seen (optional)
            epoch: Current epoch number (optional)

        Returns:
            Tuple of tensors depending on mode
        """
        if self.mode == 'pretrain':
            return self.forward_pretrain(source, label, batch_seen, epoch)
        else:
            return self.forward_finetune(source, label)

    def get_encoder_output(self, source: torch.Tensor) -> torch.Tensor:
        """Get encoder output without masking (for downstream tasks).

        Args:
            source: Input data [B, T, N, D]

        Returns:
            Encoded features [B, T, N, hidden_dim]
        """
        # Temporarily set mode to eval
        original_mode = self.encoder.mode
        self.encoder.mode = 'eval'
        flow_encode_eb = self.encoder(source, None)
        self.encoder.mode = original_mode
        return flow_encode_eb
