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
        x_t = x[:, -1, :, 0]  # (B,N, flow)

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

        return - (w * d).mean()


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
        x_t = x[:, -1, :, 0]                  # (B,N, flow)

        # --- neighbor means ---
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t)   # (B,N,D)
        Ax_t = torch.einsum("ij,bj->bi", self.A, x_t)        # (B,N,Cx)

        # --- representation distance (normalized L2, squared) ---
        diff  = (h_t - Ah_t).pow(2).sum(dim=-1).sqrt()                    # (B,N)
        denom = (h_t.pow(2).sum(dim=-1).sqrt()
                 + Ah_t.pow(2).sum(dim=-1).sqrt() + self.eps)             # (B,N)
        d = ((diff / denom).clamp(0.0, 1.0)) ** 2                         # (B,N), squared

        # --- input deviation (L2 over channel dim) + mean-normalize ---
        w = (x_t - Ax_t).pow(2).sum(dim=-1).sqrt()                        # (B,N)

        # --- anti-smoothing: emphasize outlier nodes ---
        loss = - (w * d).mean()                                            # weight(λ)는 외부에서 곱하기
        return loss



class EuclideanDistanceLossV2(nn.Module):
    """
    Anti-smoothing loss with Chebyshev k-hop neighbors
    - Excludes T₀ (self) to focus on neighbor difference
    - Uses flow channel only for input deviation
    - Detaches neighbor aggregation to prevent circular gradients
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1, "Ks must be >= 1"
        
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        
        # Remove self-loops from original adjacency
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        
        # Row-normalized GSO (STGCN style)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D  # D^{-1}A
        
        # Chebyshev polynomials (excluding T₀)
        cheb_list = [G]  # T₁
        
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)
            
            for k in range(3, Ks + 1):  # T₃, T₄, ..., T_Ks
                Tk = 2 * (G @ cheb_list[-1]) - cheb_list[-2]
                cheb_list.append(Tk)
        
        # Aggregate and normalize
        A_cheb = sum(cheb_list) / len(cheb_list)
        
        # Ensure no self-loops (safety check)
        A_cheb = A_cheb.clone()
        A_cheb.fill_diagonal_(0.0)
        
        # Row-normalize
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs
        
        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks
    
    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]       # (B,C,L,N)
        x = forward_return["inputs"]     # (B,L,N,Cx)
        
        # Last timestep
        h_t = h.permute(0, 2, 3, 1)[:, -1]  # (B,N,D)
        x_t = x[:, -1, :, 0]                # (B,N) flow only
        
        # Neighbor aggregation (detached)
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t.detach())
        Ax_t = torch.einsum("ij,bj->bi", self.A, x_t.detach())
        
        # Normalized L2 distance (squared)
        diff = (h_t - Ah_t).norm(dim=-1)
        denom = (h_t.norm(dim=-1) + Ah_t.norm(dim=-1) + self.eps)
        d = ((diff / denom).clamp(0.0, 1.0)) ** 2
        
        # Input deviation (flow only, no normalization)
        w = (x_t - Ax_t).abs()
        
        return -(w * d).mean()

class EuclideanDistanceLossReverse(nn.Module):
    """
    Anti-smoothing loss with Chebyshev k-hop neighbors
    - Excludes T₀ (self) to focus on neighbor difference
    - Uses flow channel only for input deviation
    - Detaches neighbor aggregation to prevent circular gradients
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1, "Ks must be >= 1"
        
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        
        # Remove self-loops from original adjacency
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        
        # Row-normalized GSO (STGCN style)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D  # D^{-1}A
        
        # Chebyshev polynomials (excluding T₀)
        cheb_list = [G]  # T₁
        
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)

        
        # Aggregate and normalize
        A_cheb = sum(cheb_list) / len(cheb_list)
        
        # Ensure no self-loops (safety check)
        A_cheb = A_cheb.clone()
        A_cheb.fill_diagonal_(0.0)
        
        # Row-normalize
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs
        
        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks
    
    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]       # (B,C,L,N)
        x = forward_return["inputs"]     # (B,L,N,Cx)
        
        # Last timestep
        h_t = h.permute(0, 2, 3, 1)[:, -1]  # (B,N,D)
        x_t = x[:, -1, :, 0]                # (B,N) flow only
        
        # Neighbor aggregation (detached)
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t.detach())
        Ax_t = torch.einsum("ij,bj->bi", self.A, x_t.detach())
        
        # Normalized L2 distance (squared)
        diff = (h_t - Ah_t).norm(dim=-1)
        denom = (h_t.norm(dim=-1) + Ah_t.norm(dim=-1) + self.eps)
        d = ((diff / denom).clamp(0.0, 1.0)) ** 2
        
        # Input deviation (flow only, no normalization)
        w = (x_t - Ax_t).abs()
        
        return (w * d).mean()


class CosineDistanceLossReverse(nn.Module):
    """
    Anti-smoothing loss with Chebyshev k-hop neighbors (cosine version)
    - Excludes T₀ (self) to focus on neighbor difference
    - Uses flow channel only for input deviation
    - Detaches neighbor aggregation to prevent circular gradients
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1, "Ks must be >= 1"
        
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        
        # Remove self-loops
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        
        # Row-normalized GSO (STGCN style)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D
        
        # Chebyshev polynomials (excluding T₀)
        cheb_list = [G]  # T₁
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)

        # Aggregate and normalize
        A_cheb = sum(cheb_list) / len(cheb_list)
        A_cheb = A_cheb.clone()
        A_cheb.fill_diagonal_(0.0)
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs

        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks

    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]       # (B,C,L,N)
        x = forward_return["inputs"]     # (B,L,N,Cx)
        
        # Last timestep
        h_t = h.permute(0, 2, 3, 1)[:, -1]  # (B,N,D)
        x_t = x[:, -1, :, 0]                # (B,N)  flow only
        
        # Neighbor aggregation (detached)
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t.detach())
        Ax_t = torch.einsum("ij,bj->bi",   self.A, x_t.detach())
        
        # === Cosine distance ===
        h_n  = F.normalize(h_t,  dim=-1)
        Ah_n = F.normalize(Ah_t, dim=-1)
        d = 0.5 * (1.0 - (h_n * Ah_n).sum(dim=-1))  # [0,1]

        # Input deviation (flow only, no normalization)
        w = (x_t - Ax_t).abs()

        return (w * d).mean()


class EuclideanDistanceLossV2(nn.Module):
    """
    Anti-smoothing loss with Chebyshev k-hop neighbors
    - Excludes T₀ (self) to focus on neighbor difference
    - Uses flow channel only for input deviation
    - Detaches neighbor aggregation to prevent circular gradients
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1, "Ks must be >= 1"
        
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        
        # Remove self-loops from original adjacency
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        
        # Row-normalized GSO (STGCN style)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D  # D^{-1}A
        
        # Chebyshev polynomials (excluding T₀)
        cheb_list = [G]  # T₁
        
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)

        
        # Aggregate and normalize
        A_cheb = sum(cheb_list) / len(cheb_list)
        
        # Ensure no self-loops (safety check)
        A_cheb = A_cheb.clone()
        A_cheb.fill_diagonal_(0.0)
        
        # Row-normalize
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs
        
        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks
    
    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]       # (B,C,L,N)
        x = forward_return["inputs"]     # (B,L,N,Cx)
        
        # Last timestep
        h_t = h.permute(0, 2, 3, 1)[:, -1]  # (B,N,D)
        x_t = x[:, -1, :, 0]                # (B,N) flow only
        
        # Neighbor aggregation (detached)
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t.detach())
        Ax_t = torch.einsum("ij,bj->bi", self.A, x_t.detach())
        
        # Normalized L2 distance (squared)
        diff = (h_t - Ah_t).norm(dim=-1)
        denom = (h_t.norm(dim=-1) + Ah_t.norm(dim=-1) + self.eps)
        d = ((diff / denom).clamp(0.0, 1.0)) ** 2
        
        # Input deviation (flow only, no normalization)
        w = (x_t - Ax_t).abs()
        
        return -(w * d).mean()


class CosineDistanceLossV2(nn.Module):
    """
    Anti-smoothing loss with Chebyshev k-hop neighbors (cosine version)
    - Excludes T₀ (self) to focus on neighbor difference
    - Uses flow channel only for input deviation
    - Detaches neighbor aggregation to prevent circular gradients
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1, "Ks must be >= 1"
        
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        
        # Remove self-loops
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        
        # Row-normalized GSO (STGCN style)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D
        
        # Chebyshev polynomials (excluding T₀)
        cheb_list = [G]  # T₁
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)

        # Aggregate and normalize
        A_cheb = sum(cheb_list) / len(cheb_list)
        A_cheb = A_cheb.clone()
        A_cheb.fill_diagonal_(0.0)
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs

        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks

    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]       # (B,C,L,N)
        x = forward_return["inputs"]     # (B,L,N,Cx)
        
        # Last timestep
        h_t = h.permute(0, 2, 3, 1)[:, -1]  # (B,N,D)
        x_t = x[:, -1, :, 0]                # (B,N)  flow only
        
        # Neighbor aggregation (detached)
        Ah_t = torch.einsum("ij,bjd->bid", self.A, h_t.detach())
        Ax_t = torch.einsum("ij,bj->bi",   self.A, x_t.detach())
        
        # === Cosine distance ===
        h_n  = F.normalize(h_t,  dim=-1)
        Ah_n = F.normalize(Ah_t, dim=-1)
        d = 0.5 * (1.0 - (h_n * Ah_n).sum(dim=-1))  # [0,1]

        # Input deviation (flow only, no normalization)
        w = (x_t - Ax_t).abs()

        return -(w * d).mean()

class PredictionGuidedBidirectionalLoss(nn.Module):
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)

        # remove self-loops then row-normalize (D^{-1}A)
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D

        # Chebyshev basis (exclude T0): T1=G, T2=2GT1-I
        cheb_list = [G]
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)
            for k in range(3, Ks):  # <- stop before Ks (Ks=3 => T1,T2 only)
                Tk = 2 * (G @ cheb_list[-1]) - cheb_list[-2]
                cheb_list.append(Tk)

        A_cheb = (sum(cheb_list) / len(cheb_list)).clone()
        A_cheb.fill_diagonal_(0.0)
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs

        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks

    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]          # (B,C,L,N)
        x = forward_return["inputs"]        # (B,L,N,Cx)
        y_true = forward_return["target"].squeeze(-1)   # (B,L,N)
        y_pred = forward_return["prediction"].squeeze(-1)  # (B,L,N)

        B, L, N, = y_true.shape

        # --- feature & input at each step ---
        h_t = h.permute(0, 2, 3, 1)         # (B,L,N,D)
        x_t = x[..., 0]                     # (B,L,N) flow only

        # --- neighbor aggregation for all timesteps (detach to avoid circular grads) ---
        Ah_t   = torch.einsum("ij,bljd->blid", self.A, h_t.detach())    # (B,L,N,D)
        Ax_t   = torch.einsum("ij,blj->bli",   self.A, x_t.detach())    # (B,L,N)
        Ay_true= torch.einsum("ij,blj->bli",   self.A, y_true.detach()) # (B,L,N)

        # --- representation distance (normalized L2, squared), per timestep ---
        diff  = (h_t - Ah_t).norm(dim=-1)                                # (B,L,N)
        denom = (h_t.norm(dim=-1) + Ah_t.norm(dim=-1) + self.eps)
        d = ((diff / denom).clamp(0.0, 1.0) ** 2)                        # (B,L,N)

        # --- input deviation (flow) ---
        w = (x_t - Ax_t).abs().clamp_min(1e-6)                           # (B,L,N)

        # --- prediction-guided continuous sign over all steps ---
        err_pred = (y_pred.detach() - y_true).abs()                      # (B,L,N)
        err_nbr  = (Ay_true - y_true).abs()                              # (B,L,N)
        denom_err = (err_nbr + err_pred).clamp_min(self.eps)
        ratio = (err_nbr - err_pred) / denom_err                         # (B,L,N)

        # --- aggregate across timesteps (mean or sum) ---
        # horizon 평균으로 안정화
        ratio_mean = ratio.mean(dim=1)   # (B,N)
        w_mean     = w.mean(dim=1)       # (B,N)
        d_mean     = d.mean(dim=1)       # (B,N)

        loss = (ratio_mean * w_mean * d_mean).mean()
        return loss

class PredictionGuidedRepelLoss(nn.Module):
    """
    Simplified version of Prediction-Guided loss:
    - Uses mean of product (ratio * w * d)
    - Applies normalization only over incident-like nodes (ratio > 0)
    """
    def __init__(self, adj: torch.Tensor, eps: float = 1e-6, Ks: int = 3):
        super().__init__()
        assert Ks >= 1
        A = adj.float()
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)

        # remove self-loops then row-normalize (D^{-1}A)
        A_no_self = A.clone()
        A_no_self.fill_diagonal_(0.0)
        D = A_no_self.sum(dim=1, keepdim=True).clamp_min(eps)
        G = A_no_self / D

        # Chebyshev basis (exclude T0): T1=G, T2=2GT1-I
        cheb_list = [G]
        if Ks >= 2:
            T2 = 2 * (G @ cheb_list[0]) - I
            cheb_list.append(T2)
            for k in range(3, Ks):  # <- stop before Ks (Ks=3 => T1,T2 only)
                Tk = 2 * (G @ cheb_list[-1]) - cheb_list[-2]
                cheb_list.append(Tk)

        A_cheb = (sum(cheb_list) / len(cheb_list)).clone()
        A_cheb.fill_diagonal_(0.0)
        rs = A_cheb.sum(dim=1, keepdim=True).clamp_min(eps)
        A_cheb = A_cheb / rs

        self.register_buffer("A", A_cheb.to("cuda"))
        self.eps = eps
        self.Ks = Ks

    def forward(self, forward_return: dict) -> torch.Tensor:
        h = forward_return["repr"]          # (B,C,L,N)
        x = forward_return["inputs"]        # (B,L,N,Cx)
        y_true = forward_return["target"].squeeze(-1)   # (B,L,N)
        y_pred = forward_return["prediction"].squeeze(-1)  # (B,L,N)

        Ls = 3
        B, L, N, = y_true.shape
        # 뒤에서 L*개만 사용 (마지막 구간 정렬)
        h_t   = h.permute(0, 2, 3, 1)[:, -Ls:, :, :]    # (B,L*,N,D)
        x_t   = x[..., 0][:, -Ls:, :]                   # (B,L*,N)
        y_t   = y_true[:, -Ls:, :]                      # (B,L*,N)
        yhat_t= y_pred[:, -Ls:, :]                      # (B,L*,N)

        # ----------------------------
        # 1) 이웃 집계 (기존 그대로)
        # ----------------------------
        Ah_t   = torch.einsum("ij,bljd->blid", self.A, h_t.detach())  # (B,L*,N,D)
        Ax_t   = torch.einsum("ij,blj->bli",   self.A, x_t.detach())  # (B,L*,N)
        Ay_true= torch.einsum("ij,blj->bli",   self.A, y_t.detach())  # (B,L*,N)

        # ----------------------------
        # 2) 코사인 거리 d (기존 그대로)
        # ----------------------------
        h_n  = F.normalize(h_t,  dim=-1)
        Ah_n = F.normalize(Ah_t, dim=-1)
        d = (0.5 * (1.0 - (h_n * Ah_n).sum(dim=-1))).clamp_(0, 1)     # (B,L*,N)

        # ----------------------------
        # 3) 중요도 w (기존 그대로)
        # ----------------------------
        w = (x_t - Ax_t).abs().clamp_min(1e-6)                         # (B,L*,N)

        # ----------------------------
        # 4) prediction-guided ratio (기존 그대로)
        # ----------------------------
        err_pred = (yhat_t.detach() - y_t).abs()                       # (B,L*,N)
        err_nbr  = (Ay_true - y_t).abs()                               # (B,L*,N)
        denom_err = (err_nbr + err_pred).clamp_min(self.eps)
        ratio = (err_nbr - err_pred) / denom_err                       # (B,L*,N)

        # ----------------------------
        # 5) 시간 평균 (기존 그대로)
        # ----------------------------
        term = (ratio * w * d).mean(dim=1)                             # (B,N)
        loss = term.mean()                                             # 스칼라

        return loss


class InfoNCELoss(nn.Module):
    """
    Simple InfoNCE contrastive loss for representation learning.

    Treats (z_weak[i], z_strong[i]) as positive pairs,
    and all other combinations as negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_weak: torch.Tensor,
        z_strong: torch.Tensor,
        severity_weak: torch.Tensor = None,
        severity_strong: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            z_weak: (B, T, N, D) representations from weak augmentation
            z_strong: (B, T, N, D) representations from strong augmentation
            severity_weak: unused (for API compatibility)
            severity_strong: unused (for API compatibility)

        Returns:
            loss: scalar
        """
        B, T, N, D = z_weak.shape

        # Flatten: (B, T, N, D) -> (B*T*N, D)
        z_w = z_weak.reshape(-1, D)
        z_s = z_strong.reshape(-1, D)

        # L2 normalize
        z_w = F.normalize(z_w, dim=1)
        z_s = F.normalize(z_s, dim=1)

        # Positive pairs: z_w[i] and z_s[i]
        # Similarity matrix: (BTN, BTN)
        sim_matrix = torch.mm(z_w, z_s.t()) / self.temperature

        # Positive scores on diagonal
        pos_sim = torch.diag(sim_matrix)

        # InfoNCE loss: -log(exp(pos) / sum(exp(all)))
        loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)

        return loss.mean()


class SeverityAwareContrastiveLoss(nn.Module):
    """
    Contrastive loss that considers severity levels.

    Samples with similar severity levels are treated as positives,
    and the negative weight is proportional to severity difference.
    """

    def __init__(self, temperature: float = 0.1, severity_weight_scale: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.severity_weight_scale = severity_weight_scale

    def forward(
        self,
        z_weak: torch.Tensor,
        z_strong: torch.Tensor,
        severity_weak: torch.Tensor,
        severity_strong: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_weak: (B, T, N, D) representations from weak augmentation
            z_strong: (B, T, N, D) representations from strong augmentation
            severity_weak: (B, T, N) severity labels for weak aug (mostly 0)
            severity_strong: (B, T, N) severity labels for strong aug

        Returns:
            loss: scalar
        """
        B, T, N, D = z_weak.shape

        # Flatten
        z_w = z_weak.reshape(-1, D)
        z_s = z_strong.reshape(-1, D)
        sev_w = severity_weak.reshape(-1)
        sev_s = severity_strong.reshape(-1)

        # Normalize
        z_w = F.normalize(z_w, dim=1)
        z_s = F.normalize(z_s, dim=1)

        # Concatenate for computing all similarities
        z_all = torch.cat([z_w, z_s], dim=0)  # (2BTN, D)
        sev_all = torch.cat([sev_w, sev_s], dim=0)  # (2BTN,)

        n_samples = z_all.shape[0]

        # Similarity matrix
        sim = torch.mm(z_all, z_all.t()) / self.temperature  # (2BTN, 2BTN)

        # Mask out self-similarity
        mask_self = torch.eye(n_samples, device=z_all.device, dtype=torch.bool)
        sim = sim.masked_fill(mask_self, -float('inf'))

        # Severity difference matrix
        sev_diff = torch.abs(sev_all.unsqueeze(0) - sev_all.unsqueeze(1))  # (2BTN, 2BTN)

        # Positive mask: same severity level (and not self)
        pos_mask = (sev_diff == 0) & ~mask_self

        # For samples with no positives, use the corresponding aug pair
        # (i.e., z_w[i] <-> z_s[i])
        n_half = n_samples // 2
        for i in range(n_half):
            if not pos_mask[i].any():
                pos_mask[i, i + n_half] = True
            if not pos_mask[i + n_half].any():
                pos_mask[i + n_half, i] = True

        # Compute loss
        # For each sample, compute InfoNCE with severity-weighted negatives
        loss = 0.0
        valid_count = 0

        for i in range(n_samples):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0:
                continue

            # Positive similarity (average if multiple positives)
            pos_sim = sim[i, pos_indices].mean()

            # All similarities (for denominator)
            all_sim = sim[i]
            all_sim = all_sim.masked_fill(mask_self[i], -float('inf'))

            # Standard InfoNCE
            loss += -pos_sim + torch.logsumexp(all_sim, dim=0)
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=z_all.device, requires_grad=True)

        return loss / valid_count