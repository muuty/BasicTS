import torch
import torch.nn as nn


class NodePairGateMVP(nn.Module):
    # [B,T,N,C] -> [B,N,N]  (비대칭 허용, Top-K 없음, 진짜 게이팅+self-fill)
    def __init__(self, T:int, C:int, emb_dim:int=32, hidden:int=64):
        super().__init__()
        self.T, self.C = T, C
        self.embed = nn.Sequential(
            nn.Linear(T*C, hidden), nn.GELU(),
            nn.Linear(hidden, emb_dim)
        )
        self.W = nn.Parameter(torch.empty(emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):   # x: [B,T,N,C]
        B,T,N,C = x.shape
        z = self.embed(x.permute(0,2,1,3).reshape(B, N, T*C))             # [B,N,E]
        G = torch.sigmoid(torch.einsum('bnd,de,bme->bnm', z, self.W, z))   # [B,N,N]
        Gsum_wo_diag = G.sum(-1) - torch.diagonal(G, dim1=-2, dim2=-1)    # [B,N]
        A = G.clone()
        A_diag = (1.0 - Gsum_wo_diag).clamp_min(0.0)                      # [B,N]
        eye = torch.eye(N, device=x.device).unsqueeze(0)
        A = A * (1 - eye) + eye * A_diag.unsqueeze(-1)                     # [B,N,N]
        return A