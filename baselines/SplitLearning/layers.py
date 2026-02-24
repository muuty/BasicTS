import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        self.last_attn = None

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose(-1, -2)
        attn_score = (query @ key) / (self.head_dim ** 0.5)
        if self.mask:
            mask = torch.ones(tgt_length, src_length, device=query.device).tril()
            attn_score.masked_fill_(~mask, -torch.inf)
        attn_score = torch.softmax(attn_score, dim=-1)

        self.last_attn = attn_score 
        
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out.transpose(dim, -2)


class CrossAttentionLayer(nn.Module):
    """Cross attention layer for Level 4: query는 spatial attention output, key/value는 client attention output"""
    
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Args:
            query: [B, T, num_nodes_in_group, model_dim] - spatial attention output
            key: [B, T, K, model_dim] - client attention output
            value: [B, T, K, model_dim] - client attention output
            
        Returns:
            out: [B, T, num_nodes_in_group, model_dim] - cross attention output
        """
        # Cross attention 수행
        residual = query
        out = self.attn(query, key, value)
        out = self.dropout1(out)
        out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        
        return out


class NodeSummaryLayer(nn.Module):
    """클라이언트별 temporal output을 서브그래프를 사용해서 요약하는 레이어"""
    
    def __init__(self, model_dim, subgraph_adj, num_tokens=1):
        super().__init__()
        self.model_dim = model_dim
        self.subgraph_adj = subgraph_adj  # (N, N) adjacency matrix for the subgraph
        self.num_tokens = num_tokens
        self.node_reduction = nn.Linear(subgraph_adj.shape[0], num_tokens)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, N, model_dim] - temporal attention output
            
        Returns:
            summary: [B, T, num_tokens, model_dim] - node dimension reduced to num_tokens
        """
        subgraph_adj = self.subgraph_adj.to(x.device)
        x = subgraph_adj @ x
        x = x.transpose(-2, -1)  # [B, T, D, N] or [B, D, N]
        x = self.node_reduction(x)  # [B, T, D, num_tokens] or [B, D, num_tokens]
        x = x.transpose(-2, -1)  # [B, T, num_tokens, D] or [B, num_tokens, D]
        return x


class AttentionPoolFromNodes(nn.Module):
    def __init__(self, model_dim=64, num_tokens=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, model_dim))
        self.attn = AttentionLayer(model_dim, num_heads, mask=False)
        self.proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_bnD):
        B, N, D = x_bnD.shape
        q = self.tokens.expand(B, -1, -1)
        out = self.attn(q, x_bnD, x_bnD)
        out = self.dropout(out)
        return self.proj(out)


class AttentionPoolLayer(nn.Module):
    """
    Learnable summary tokens로 노드·시간 정보를 요약.
    시간별로 K개 토큰을 생성.
    """
    def __init__(self, model_dim, num_tokens=4, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens

        # 학습 가능한 summary token
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, model_dim))

        # Multi-head attention using custom AttentionLayer
        self.attn = AttentionLayer(model_dim, num_heads, mask=False)

        self.proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, N, D]
        return: [B, T, num_tokens, D]
        """
        B, T, N, D = x.shape

        # 시간 단위로 풀어 처리 → (B*T, N, D)
        x_flat = x.reshape(B*T, N, D)
        q = self.tokens.expand(B*T, -1, -1)    # (B*T, num_tokens, D)
        out = self.attn(q, x_flat, x_flat)
        out = self.dropout(out)                # dropout 적용
        out = self.proj(out)                   # (B*T, num_tokens, D)
        return out.reshape(B, T, self.num_tokens, D)


class TokenAttentionFusion(nn.Module):
    """
    안전한 토큰-축 cross-attention 융합
    클라이언트 모델에 넣어 두면 최적화가 깔끔하고 grad 끊김 위험이 없습니다.
    """
    def __init__(self, c_out: int, attn_dim: int = None, dropout: float = 0.0, gate: bool = True):
        super().__init__()
        d = attn_dim or c_out
        self.q = nn.Linear(c_out, d)
        self.k = nn.Linear(c_out, d)
        self.v = nn.Linear(c_out, d)
        self.out = nn.Linear(d, c_out)
        self.drop = nn.Dropout(dropout)
        self.scale = d ** -0.5
        self.gate = nn.Parameter(torch.zeros(1)) if gate else None

    def forward(self, client_proj, server_tok):
        # client_proj: [B,L,N,C], server_tok: [B,L,T,C]
        Q = self.q(client_proj)                 # [B,L,N,d]
        K = self.k(server_tok)                  # [B,L,T,d]
        V = self.v(server_tok)                  # [B,L,T,d]
        attn = torch.softmax(torch.einsum('blnd,bltd->blnt', Q, K) * self.scale, dim=-1)
        attn = self.drop(attn)
        fused = torch.einsum('blnt,bltd->blnd', attn, V)          # [B,L,N,d]
        fused = self.out(fused)                                   # [B,L,N,C]
        if self.gate is None:
            return fused
        return torch.sigmoid(self.gate) * fused                   # 게이트드 델타


# def mlp(in_dim: int, hidden: list, out_dim: int, act=nn.ReLU, bias: bool = True) -> nn.Sequential:
#     """Simple MLP builder"""
#     layers = []
#     d = in_dim
#     for h in hidden:
#         layers += [nn.Linear(d, h, bias=bias), act()]
#         d = h
#     layers += [nn.Linear(d, out_dim, bias=bias)]
#     return nn.Sequential(*layers)


# class GraphNetBlock(nn.Module):
#     """
#     정식 GN 블록:
#       e'_k = phi_e([e_k, v_{r_k}, v_{s_k}, u])
#       v'_i = phi_v([v_i, sum_{k: r_k=i} e'_k, u])
#       u'   = phi_u([u, mean_k e'_k, mean_i v'_i])
#     """
#     def __init__(self, node_dim=64, edge_dim=64, global_dim=64):
#         super().__init__()
#         h = [256, 256, 128]
#         self.edge_mlp   = mlp(edge_dim + 2*node_dim + global_dim, h, edge_dim)
#         self.node_mlp   = mlp(node_dim + edge_dim + global_dim,    h, node_dim)
#         self.global_mlp = mlp(global_dim + edge_dim + node_dim,    h, global_dim)

#     def forward(self, v, e, u, edge_index, edge_weight=None):
#         # v: (B,N,Dn), e: (B,E,De), u: (B,Dg)
#         B, N, Dn = v.shape
#         E       = edge_index.size(1)
#         De      = e.size(-1)
#         Dg      = u.size(-1)

#         dst = edge_index[0]  # (E,)
#         src = edge_index[1]  # (E,)

#         # ----- Edge update -----
#         u_e   = u[:, None, :].expand(B, E, Dg)     # (B,E,Dg)
#         v_src = v[:, src, :]                       # (B,E,Dn)
#         v_dst = v[:, dst, :]                       # (B,E,Dn)

#         e_in  = torch.cat([e, v_dst, v_src, u_e], dim=-1)  # (B,E,De+2*Dn+Dg)
#         e_upd = self.edge_mlp(e_in)                         # (B,E,De)
#         if edge_weight is not None:
#             e_upd = e_upd * edge_weight.view(1, E, 1)

#         # ----- Node update (aggregate incoming edges) -----
#         agg = torch.zeros(B, N, De, device=v.device, dtype=v.dtype)
#         for b in range(B):
#             agg[b].index_add_(0, dst, e_upd[b])  # sum over incoming edges

#         u_v   = u[:, None, :].expand(B, N, Dg)    # (B,N,Dg)
#         v_in  = torch.cat([v, agg, u_v], dim=-1)  # (B,N,Dn+De+Dg)
#         v_upd = self.node_mlp(v_in)               # (B,N,Dn)

#         # ----- Global update (pool edges/nodes) -----
#         e_pool = e_upd.mean(dim=1) if E > 0 else torch.zeros(B, De, device=v.device, dtype=v.dtype)  # (B,De)
#         v_pool = v_upd.mean(dim=1)                                                                      # (B,Dn)
#         u_in   = torch.cat([u, e_pool, v_pool], dim=-1)                                                 # (B,Dg+De+Dn)
#         u_upd  = self.global_mlp(u_in)                                                                  # (B,Dg)

#         return e_upd, v_upd, u_upd


# class GraphNet(nn.Module):
#     """
#     GraphNet: edge 초기화 + 여러 GN 블록 적용 (residual connection)
    
#     GRU Encoder 후 서브그래프에서 메시지 패싱을 수행하는 전체 파이프라인.
#     """
#     def __init__(
#         self, 
#         hidden_dim: int = 64, 
#         num_blocks: int = 2,
#         edge_index: torch.Tensor = None,
#         edge_weight: torch.Tensor = None,
#     ):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_blocks = num_blocks
        
#         # Edge/node/global dimensions
#         self.node_dim = self.edge_dim = self.global_dim = hidden_dim
#         # Edge init MLP hidden sizes (GraphNetBlock과 동일하게)
#         h_dims = [256, 256, 128]
        
#         # Edge 초기화 MLP: [v_dst, v_src, edge_weight] -> edge_dim
#         self.edge_init = mlp(2 * self.node_dim + 1, h_dims, self.edge_dim)
        
#         # GN 블록들
#         self.gn_blocks = nn.ModuleList([
#             GraphNetBlock(self.node_dim, self.edge_dim, self.global_dim)
#             for _ in range(num_blocks)
#         ])
        
#         # Edge index/weight 등록 (None이면 나중에 forward에서 제공)
#         if edge_index is not None:
#             self.register_buffer("edge_index", edge_index.long(), persistent=False)
#             self.register_buffer("edge_weight", 
#                                  edge_weight.float() if edge_weight is not None else None,
#                                  persistent=False)
#         else:
#             self.edge_index = None
#             self.edge_weight = None
    
#     def forward(
#         self, 
#         h: torch.Tensor, 
#         edge_index: torch.Tensor = None, 
#         edge_weight: torch.Tensor = None
#     ) -> torch.Tensor:
#         """
#         Args:
#             h: (B, N, hidden_dim) 노드 특성
#             edge_index: (2, E) 선택적, 없으면 등록된 버퍼 사용
#             edge_weight: (E,) 선택적
#         Returns:
#             v: (B, N, hidden_dim) 업데이트된 노드 특성
#         """
#         # 등록된 버퍼 또는 입력 사용
#         if edge_index is None:
#             edge_index = self.edge_index
#         if edge_weight is None:
#             edge_weight = self.edge_weight
        
#         if edge_index is None:
#             raise ValueError("edge_index must be provided either in __init__ or forward()")
        
#         B, N, H = h.shape
#         E = edge_index.size(1)
#         dst, src = edge_index[0], edge_index[1]
        
#         # Global feature 초기화
#         u = h.mean(dim=1)  # (B, hidden_dim)
        
#         # Edge weight 기본값
#         w = edge_weight if edge_weight is not None else torch.ones(E, device=h.device, dtype=h.dtype)
        
#         # Edge 초기화: [v_dst, v_src, w] -> edge feature
#         v_src = h[:, src, :]  # (B, E, H)
#         v_dst = h[:, dst, :]  # (B, E, H)
#         e_in = torch.cat([v_dst, v_src, w.view(1, E, 1).expand(B, E, 1)], dim=-1)
#         e = self.edge_init(e_in)  # (B, E, H)
        
#         v = h
#         g = u
        
#         # GN 블록들 적용 (residual connection)
#         for block in self.gn_blocks:
#             e_upd, v_upd, g_upd = block(v, e, g, edge_index, edge_weight)
#             e = e + e_upd
#             v = v + v_upd
#             g = g + g_upd
        
#         return v

