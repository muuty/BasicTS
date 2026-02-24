"""MultiTask STAEformer: Forecasting + Cross-Variable Reconstruction.

No encoder bottleneck. STAEformer processes raw 5 features directly.
Reconstruction head is attached to intermediate representation as auxiliary task.
During training, 1 of 3 physical variables is randomly masked per sample.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.STAEformer.arch.staeformer_arch import STAEformer


class MultiTaskSTAEformer(STAEformer):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=5,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        # Multi-task specific
        tod_feat_idx=3,
        dow_feat_idx=4,
        physical_feat_indices=(0, 1, 2),
        recon_hidden_dim=None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            in_steps=in_steps,
            out_steps=out_steps,
            steps_per_day=steps_per_day,
            input_dim=input_dim,
            output_dim=output_dim,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=spatial_embedding_dim,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=use_mixed_proj,
        )
        self.tod_feat_idx = tod_feat_idx
        self.dow_feat_idx = dow_feat_idx
        self.physical_feat_indices = list(physical_feat_indices)
        self.num_physical_vars = len(self.physical_feat_indices)

        # Reconstruction head from model_dim -> num_physical_vars
        hidden = recon_hidden_dim or self.model_dim
        self.recon_head = nn.Sequential(
            nn.Linear(self.model_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.num_physical_vars),
        )

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x = history_data  # [B, T, N, C]
        batch_size = x.shape[0]

        # Save reconstruction target (original physical variables)
        recon_target = torch.stack(
            [x[..., i] for i in self.physical_feat_indices], dim=-1
        )  # [B, T, N, num_physical_vars]

        # Random masking during training
        if train:
            mask_idx = torch.randint(0, self.num_physical_vars, (batch_size,), device=x.device)
            recon_mask = F.one_hot(mask_idx, self.num_physical_vars).float()
            recon_mask = recon_mask.view(batch_size, 1, 1, self.num_physical_vars)
            x = x.clone()
            for i, feat_idx in enumerate(self.physical_feat_indices):
                x[..., feat_idx] = x[..., feat_idx] * (1 - recon_mask[..., i])
        else:
            recon_mask = torch.ones(
                batch_size, 1, 1, self.num_physical_vars, device=x.device
            )

        # === STAEformer forward (with configurable tod/dow indices) ===
        if self.tod_embedding_dim > 0:
            tod = x[..., self.tod_feat_idx] * self.steps_per_day
        if self.dow_embedding_dim > 0:
            dow = x[..., self.dow_feat_idx] * 7
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # [B, T, N, input_embedding_dim]
        features = [x]
        if self.tod_embedding_dim > 0:
            features.append(self.tod_embedding(tod.long()))
        if self.dow_embedding_dim > 0:
            features.append(self.dow_embedding(dow.long()))
        if self.spatial_embedding_dim > 0:
            features.append(
                self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
            )
        if self.adaptive_embedding_dim > 0:
            features.append(
                self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape)
            )
        x = torch.cat(features, dim=-1)  # [B, T, N, model_dim]

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # x: [B, T, N, model_dim]

        # Reconstruction head
        recon_pred = self.recon_head(x)  # [B, T, N, num_physical_vars]

        # Forecasting head (same as parent STAEformer)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # [B, N, T, model_dim]
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # [B, out_steps, N, output_dim]
        else:
            out = x.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))

        return {
            "prediction": out,
            "recon_pred": recon_pred,
            "recon_target": recon_target,
            "recon_mask": recon_mask,
        }
