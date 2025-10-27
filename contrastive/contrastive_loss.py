import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalAntiSmoothingLoss(nn.Module):
    def __init__(self, adj: torch.Tensor, margin: float = 0.5, 
                 alpha: float = 2.0, beta: float = 0.5,
                 eps: float = 1e-6, input_channel: int = 0):
        super().__init__()
        # row-normalize if not normalized
        A = adj.to(device="cuda")
        rs = A.sum(dim=1, keepdim=True).clamp_min(eps).to(device="cuda")
        self.register_buffer("A", A / rs)
        self.margin = margin
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.input_channel = input_channel

    def _to_B_L_N_D(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,C,L,N) -> (B,L,N,D)
        return z.permute(0, 2, 3, 1)

    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]            # (B,C,L,N)
        x = forward_return["inputs"]          # (B,L,N,Cx)

        h_t = h.permute(0, 2, 3, 1)[:, -1]                    # (B,N,D)
        x_t = x[:, -1, :, self.input_channel]  # (B,N)

        # neighbor mean in representation & input spaces
        # (B,N,D): bmm with A^T to apply row-normalized neighbor mix
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t)     # (B,N,D)
        Ax_t = torch.einsum("ij,bj->bi",     self.A, x_t)   # (B,N)

        # 2. d를 cosine distance로 계산 (폭주 방지)
        h_n  = F.normalize(h_t,  dim=-1)
        Ah_n = F.normalize(Ah_t, dim=-1)
        d = 1 - (h_n * Ah_n).sum(dim=-1)

        # 3. margin 제거, 단순 가중
        w = (x_t - Ax_t).abs()
        w = w / (w.mean(dim=(0,1), keepdim=True) + 1e-6)
        # loss = (w * d).mean()


        # smoothing
        gate = torch.sigmoid(self.alpha * (w - self.beta))
        loss = (gate * w * d).mean()

        return loss


class CosineDistanceLoss(nn.Module):
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        # row-normalize if not normalized
        A = adj.to(device="cuda")
        rs = A.sum(dim=1, keepdim=True).clamp_min(eps).to(device="cuda")
        self.register_buffer("A", A / rs)
        self.eps = eps


    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]            # (B,C,L,N)
        x = forward_return["inputs"]          # (B,L,N,Cx)

        h_t = h.permute(0, 2, 3, 1)[:, -1]                    # (B,N,D)
        x_t = x[:, -1, :, :]  # (B,N,Cx)

        # neighbor mean in representation & input spaces
        # (B,N,D): bmm with A^T to apply row-normalized neighbor mix
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t)     # (B,N,D)
        Ax_t = torch.einsum("ij,bjk->bik", self.A, x_t)   # (B,N,Cx)

        # 2. d를 cosine distance로 계산 (폭주 방지)
        h_n  = F.normalize(h_t,  dim=-1)
        Ah_n = F.normalize(Ah_t, dim=-1)
        d = 0.5 * (1.0 - (h_n * Ah_n).sum(dim=-1))

        w = (x_t - Ax_t).pow(2).sum(dim=-1).sqrt()                     # (B,N)  L2
        w = w / (w.mean(dim=(0,1), keepdim=True) + 1e-6)  
        # loss = (w * d).mean()

        return (w * d).mean()


class EuclideanDistanceLoss(nn.Module):
    """
    Anti-smoothing with Euclidean (normalized L2) distance, squared to emphasize outliers.

    Loss = - mean( w * d^2 )
    where
        w  = mean-normalized input deviation (L2 over channels)
        d  = normalized L2 distance between h_t and A h_t ∈ [0,1]
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        # row-normalize once (keep same device)
        A = adj.float()
        rs = A.sum(dim=1, keepdim=True).clamp_min(eps)
        self.register_buffer("A", (A / rs).to(device="cuda"))
        self.eps = eps

    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]            # (B,C,L,N)
        x = forward_return["inputs"]          # (B,L,N,Cx)

        # --- last-step slices ---
        h_t = h.permute(0, 2, 3, 1)[:, -1]    # (B,N,D)
        x_t = x[:, -1, :, :]                  # (B,N,Cx)

        # --- neighbor means ---
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t)   # (B,N,D)
        Ax_t = torch.einsum("ij,bjk->bik", self.A, x_t)   # (B,N,Cx)

        # --- representation distance (normalized L2, squared) ---
        diff  = (h_t - Ah_t).pow(2).sum(dim=-1).sqrt()                    # (B,N)
        denom = (h_t.pow(2).sum(dim=-1).sqrt()
                 + Ah_t.pow(2).sum(dim=-1).sqrt() + self.eps)             # (B,N)
        d = ((diff / denom).clamp(0.0, 1.0)) ** 2                         # (B,N), squared

        # --- input deviation (L2 over channel dim) + mean-normalize ---
        w = (x_t - Ax_t).pow(2).sum(dim=-1).sqrt()                        # (B,N)
        w = w / (w.mean(dim=(0,1), keepdim=True) + self.eps)              # mean ≈ 1

        # --- anti-smoothing: emphasize outlier nodes ---
        loss = (w * d).mean()                                            # weight(λ)는 외부에서 곱하기
        return loss