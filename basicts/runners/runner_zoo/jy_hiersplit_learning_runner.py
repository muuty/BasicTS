#!/usr/bin/env python3
"""
Hierarchical Split Learning (HierSplit) Runner

- Inherits IndependentLearningRunner style:
  each client computes (encoder -> spatial) locally, then decoder locally.
- Difference:
  after spatial, each client pools node-dimension into tokens and sends to server.
  server performs self-attention over all clients' tokens, sends back.
  client expands tokens back to node-dimension and fuses with spatial output, then decodes.

Pooling/Expansion:
- pooling:   'simple' (MLP along node-axis) or 'attention' (learned queries cross-attend to nodes)
- expansion: 'simple' (MLP along token-axis) or 'attention' (nodes cross-attend to returned tokens)

Communication cost (bytes):
Forward:
  C->S : pooled tokens (+ optional extra tensors; currently tokens only)
  S->C : processed tokens
Backward:
  C->S : grad(processed tokens)
  S->C : grad(pooled tokens)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, Type

import numpy as np
import torch
import torch.nn as nn

from ..node_partition import setup_split_learning_nodes
from .jy_independent_learning_runner import IndependentLearningRunner
from easytorch.utils import master_only


# =========================
# Meter registration helper
# =========================
def _register_meter_like_train_loss(meter_pool, name: str, meter_type: str = "train"):
    """
    basicts meter_pool expects entries with keys like {'meter','type','index',...}.
    We clone the template entry from 'train/loss' to avoid KeyError in print_meters.
    """
    if not hasattr(meter_pool, "_pool"):
        return
    pool = meter_pool._pool
    tmpl_key = "train/loss"
    if tmpl_key not in pool:
        return
    if name in pool:
        return

    tmpl = pool[tmpl_key]
    max_idx = 0
    for v in pool.values():
        if isinstance(v, dict) and "index" in v:
            max_idx = max(max_idx, int(v["index"]))

    entry = dict(tmpl)
    entry["meter"] = type(tmpl["meter"])()  # fresh meter instance
    entry["type"] = meter_type
    entry["index"] = max_idx + 1
    pool[name] = entry


def _il_partition_kwargs(cfg: Dict) -> Dict[str, object]:
    il = cfg.get("IL", {})
    return dict(
        imbalance=il.get("IMBALANCE", None),
        imbalance_alpha=il.get("IMBALANCE_ALPHA", None),
        min_nodes_per_client=int(il.get("MIN_NODES_PER_CLIENT", 1)),
        random_seed=il.get("PARTITION_SEED", None),
    )


def _same_nodes(a, b) -> bool:
    return list(map(int, a)) == list(map(int, b))  # 순서까지 동일해야 안전


# =========================
# Attention Layer (from S2Former style)
# =========================
class AttentionLayer(nn.Module):
    """Custom multi-head attention layer (S2Former style)."""

    def __init__(self, model_dim: int, num_heads: int = 4, mask: bool = False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
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
        attn_score = (query @ key) / (self.head_dim**0.5)
        if self.mask:
            mask = torch.ones(tgt_length, src_length, device=query.device).tril()
            attn_score.masked_fill_(~mask.bool(), float("-inf"))
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    """Self-attention layer with feed-forward network (S2Former style)."""

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        mask: bool = False,
    ):
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

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
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


# =========================
# Pooling / Expansion blocks (S2Former-equivalent for simple mode)
# =========================
class SimpleNodePooling(nn.Module):
    """
    S2Former NodeSummaryLayer equivalent:
    Uses subgraph adjacency: A @ x, then Linear(N -> K) over node axis.
    Input:  [B,T,N,D]
    Output: [B,T,K,D]
    """

    def __init__(
        self,
        num_nodes: int,
        num_tokens: int,
        model_dim: int,
        subgraph_adj: torch.Tensor,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        # keep adj as buffer (moves with .to(device))
        if isinstance(subgraph_adj, np.ndarray):
            subgraph_adj = torch.from_numpy(subgraph_adj)
        self.register_buffer("subgraph_adj", subgraph_adj.float(), persistent=False)
        self.node_reduction = nn.Linear(num_nodes, num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,N,D]
        A = self.subgraph_adj.to(device=x.device, dtype=x.dtype)  # [N,N]
        x = A @ x                          # [B,T,N,D]
        x = x.transpose(2, 3)              # [B,T,D,N]
        x = self.node_reduction(x)         # [B,T,D,K]
        x = x.transpose(2, 3)              # [B,T,K,D]
        return x


class SimpleNodeExpansion(nn.Module):
    """
    S2Former concat fusion equivalent:
    - If num_tokens > 1: token_fusion (Linear K->1) to reduce tokens to 1
    - Broadcast the single token to all nodes
    - Concat with spatial output: [B,T,N,2D]
    - Apply fusion_layer (Linear 2D->D)

    Input:  node_x [B,T,N,D], tokens [B,T,K,D]
    Output: fused  [B,T,N,D]
    """

    def __init__(
        self,
        num_tokens: int,
        num_nodes: int,
        model_dim: int,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_nodes = num_nodes
        self.model_dim = model_dim

        # fusion_layer: concat [node_x, token_broadcast] -> Linear(2D -> D)
        self.fusion_layer = nn.Linear(2 * model_dim, model_dim)

        # token_fusion: reduce K tokens to 1 (only if K > 1)
        self.token_fusion = None
        if num_tokens > 1:
            self.token_fusion = nn.Linear(num_tokens, 1)

    def forward(self, node_x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # node_x: [B,T,N,D], tokens: [B,T,K,D]
        B, T, N, D = node_x.shape

        if self.num_tokens == 1:
            # tokens: [B,T,1,D] -> broadcast to [B,T,N,D]
            token_broadcast = tokens.expand(B, T, N, D)
        else:
            # [B,T,K,D] -> [B,T,D,K] -> Linear(K->1) -> [B,T,D,1] -> [B,T,1,D] -> broadcast
            tmp = tokens.transpose(-2, -1)          # [B,T,D,K]
            tmp = self.token_fusion(tmp)            # [B,T,D,1]
            tmp = tmp.transpose(-2, -1)             # [B,T,1,D]
            token_broadcast = tmp.expand(B, T, N, D)

        cat = torch.cat([node_x, token_broadcast], dim=-1)  # [B,T,N,2D]
        fused = self.fusion_layer(cat)                       # [B,T,N,D]
        return fused


class AttentionNodePooling(nn.Module):
    def __init__(self, model_dim, num_tokens, ff_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(1, num_tokens, model_dim))
        self.attn = AttentionLayer(model_dim, num_heads, mask=False)

        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        B, T, N, D = x.shape
        q = self.tokens.expand(B, T, -1, -1)

        a = self.attn(q, x, x)
        h = self.ln1(q + self.drop1(a))  # ✅ residual on q

        f = self.ff(h)
        out = self.ln2(h + self.drop2(f))  # ✅ FFN residual
        return out


class AttentionNodeExpansion(nn.Module):
    """
    Cross-attention expansion (Transformer-block style):
      node queries (N,D) cross-attend to tokens (K,D) => message (N,D)

    Input:  node_x [B,T,N,D], tokens [B,T,K,D]
    Output: msg    [B,T,N,D]
    """

    def __init__(self, model_dim: int, ff_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.model_dim = model_dim
        self.attn = AttentionLayer(model_dim, num_heads, mask=False)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, model_dim),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, node_x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(node_x, tokens, tokens)  # [B,T,N,D]
        h = self.ln1(node_x + self.drop1(attn_out))  # [B,T,N,D]
        f = self.ff(h)
        out = self.ln2(h + self.drop2(f)) # [B,T,N,D]
        return out


# =========================
# Server token mixer
# =========================
class ServerTokenMixer(nn.Module):
    """
    tokens_all: [B,T,M,D] -> self-attn over M (token axis), per time step
    Uses SelfAttentionLayer (S2Former style).
    """

    def __init__(self, model_dim: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SelfAttentionLayer(model_dim, feed_forward_dim=ff_dim, num_heads=num_heads, dropout=dropout, mask=False)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for layer in self.layers:
            x = layer(x, dim=-2)
        return x


# =========================
# HierSplit model wrapper
# =========================
def _bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def _infer_kind(model_cls: Type[nn.Module]) -> str:
    n = model_cls.__name__.lower()
    if "staeformer" in n:
        return "staeformer"
    if "stgformer" in n:
        return "stgformer"
    if "stgcn" in n:
        return "stgcn"
    if "gruseq2seq" in n or "graphnet" in n:
        return "gruseq2seq_graphnet"
    raise ValueError(f"Unsupported model kind for HierSplit: {model_cls.__name__}")


def _adj_to_edge_index(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj[adj > 0].contiguous()
    return edge_index, edge_attr


def _partition_supports_for_client(supports: List[torch.Tensor], nodes: List[int]) -> List[torch.Tensor]:
    out = []
    for s in supports:
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        out.append(s[nodes][:, nodes].contiguous())
    return out


class HierSplitModel(nn.Module):
    """
    Returns dict:
      - prediction: [B, out_steps, N, out_dim]
      - comm: forward/backward bytes (tokens only)
    """

    def __init__(
        self,
        base_model_cls: Type[nn.Module],
        base_model_params: Dict[str, Any],
        client_nodes_list: List[List[int]],
        subgraph_adj_list: Optional[List[torch.Tensor]],
        total_nodes: int,
        pooling: str,
        expansion: str,
        num_tokens: int,
        token_heads: int,
        server_heads: int,
        server_layers: int,
        server_ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.base_model_cls = base_model_cls
        self.base_model_params = dict(base_model_params)
        self.client_nodes_list = client_nodes_list
        self.total_nodes = total_nodes
        self.kind = _infer_kind(base_model_cls)

        self.num_clients = len(client_nodes_list)
        self.num_tokens = int(num_tokens)

        # buffers for nodes
        for cid, nodes in enumerate(client_nodes_list):
            self.register_buffer(f"client_nodes_{cid}", torch.tensor(nodes, dtype=torch.long), persistent=False)

        # build clients (and per-client supports/edges if needed)
        self.clients = nn.ModuleList()
        uses_supports = "supports" in self.base_model_params
        uses_edges = ("edge_index" in self.base_model_params) or ("edge_attr" in self.base_model_params)
        global_supports = self.base_model_params.get("supports", None) if uses_supports else None

        for cid, nodes in enumerate(client_nodes_list):
            p = dict(self.base_model_params)
            p["num_nodes"] = len(nodes)

            # supports (STGformer)
            if uses_supports:
                if subgraph_adj_list is not None and cid < len(subgraph_adj_list):
                    if p.get("supports", None) is None:
                        p["supports"] = [subgraph_adj_list[cid]]
                    else:
                        if isinstance(p["supports"], list) and len(p["supports"]) > 0:
                            p["supports"] = _partition_supports_for_client(p["supports"], nodes)
                else:
                    if isinstance(global_supports, list) and len(global_supports) > 0:
                        p["supports"] = _partition_supports_for_client(global_supports, nodes)

            # edges (GRU)
            if uses_edges and (subgraph_adj_list is not None) and cid < len(subgraph_adj_list):
                adj = subgraph_adj_list[cid]
                edge_index, edge_attr = _adj_to_edge_index(adj)
                p["edge_index"] = edge_index
                p["edge_attr"] = edge_attr

            # adj_matrix (STGCN) - subgraph_adj가 있으면 그걸 사용
            if "adj_matrix" in p and p["adj_matrix"] is not None:
                if subgraph_adj_list is not None and cid < len(subgraph_adj_list):
                    p["adj_matrix"] = subgraph_adj_list[cid]
                else:
                    # fallback: global adj에서 분할
                    global_adj = p["adj_matrix"]
                    if isinstance(global_adj, np.ndarray):
                        global_adj = torch.from_numpy(global_adj).float()
                    p["adj_matrix"] = global_adj[nodes][:, nodes].contiguous()

            self.clients.append(base_model_cls(**p))

        # representation dim (for tokens)
        if self.kind in ("staeformer", "stgformer", "stgcn"):
            model_dim = getattr(self.clients[0], "model_dim")
        else:
            model_dim = getattr(self.clients[0], "hidden_size")
        self.model_dim = int(model_dim)

        self.no_token_mode = (int(num_tokens) == 0)

        if self.no_token_mode:
            self.pooling_type = str(pooling).lower()
            self.expansion_type = str(expansion).lower()
            self.poolers = nn.ModuleList()
            self.expanders = nn.ModuleList()
            self.server = nn.Identity()  # params 없음
            self.total_forward_bytes = 0
            self.total_backward_bytes = 0
            return

        # pooling/expansion modules per client
        pooling = pooling.lower()
        expansion = expansion.lower()
        self.pooling_type = pooling
        self.expansion_type = expansion

        self.poolers = nn.ModuleList()
        self.expanders = nn.ModuleList()

        for cid, nodes in enumerate(client_nodes_list):
            n_i = len(nodes)

            # pooler
            if pooling == "simple":
                # S2Former NodeSummaryLayer equivalent: requires subgraph_adj
                if subgraph_adj_list is None or cid >= len(subgraph_adj_list):
                    raise ValueError("POOLING='simple' requires subgraph_adj_list for each client.")
                self.poolers.append(
                    SimpleNodePooling(
                        num_nodes=n_i,
                        num_tokens=self.num_tokens,
                        model_dim=self.model_dim,
                        subgraph_adj=subgraph_adj_list[cid],
                    )
                )
            elif pooling == "attention":
                self.poolers.append(
                    AttentionNodePooling(
                        self.model_dim,
                        self.num_tokens,
                        ff_dim=server_ff_dim,
                        num_heads=token_heads,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            # expander
            if expansion == "simple":
                # S2Former concat fusion equivalent
                self.expanders.append(
                    SimpleNodeExpansion(
                        num_tokens=self.num_tokens,
                        num_nodes=n_i,
                        model_dim=self.model_dim,
                    )
                )
            elif expansion == "attention":
                self.expanders.append(
                    AttentionNodeExpansion(
                        self.model_dim,
                        ff_dim=server_ff_dim,
                        num_heads=token_heads,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unknown expansion: {expansion}")

        # server token mixer (shared)
        self.server = ServerTokenMixer(
            model_dim=self.model_dim,
            num_heads=server_heads,
            num_layers=server_layers,
            ff_dim=server_ff_dim,
            dropout=dropout,
        )

        # comm accumulators (MB stats can be exposed)
        self.total_forward_bytes = 0
        self.total_backward_bytes = 0

    def get_client_nodes(self, cid: int) -> torch.Tensor:
        return getattr(self, f"client_nodes_{cid}")

    def get_client_models(self) -> List[nn.Module]:
        return list(self.clients)

    def get_server_model(self) -> nn.Module:
        return self.server

    def get_communication_stats(self) -> Dict[str, float]:
        return {
            "total_forward_mb": self.total_forward_bytes / (1024 * 1024),
            "total_backward_mb": self.total_backward_bytes / (1024 * 1024),
            "total_mb": (self.total_forward_bytes + self.total_backward_bytes) / (1024 * 1024),
        }

    def forward(self, history_data, future_data=None, batch_seen=0, epoch=0, train=True, **kwargs) -> Dict:
        device = history_data.device
        dtype = history_data.dtype

        comm = {
            "forward": {"client_to_server_bytes": 0, "server_to_client_bytes": 0, "total_bytes": 0},
            "backward": {"client_to_server_bytes": 0, "server_to_client_bytes": 0, "total_bytes": 0},
        }

        if getattr(self, "no_token_mode", False):
            # ✅ Independent Learning과 동일: 각 client가 자기 모델을 그대로 forward하고 합친다.
            if self.kind in ("staeformer", "stgformer", "stgcn"):
                B = history_data.shape[0]
                out_steps = self.clients[0].out_steps
                out_dim = self.clients[0].output_dim
                pred_full = torch.zeros((B, out_steps, self.total_nodes, out_dim), device=device, dtype=dtype)

                for cid, client in enumerate(self.clients):
                    nodes = self.get_client_nodes(cid)
                    x_i = history_data.index_select(2, nodes)
                    y_i = future_data.index_select(2, nodes) if future_data is not None else None

                    pred_i = client(
                        history_data=x_i,
                        future_data=y_i,
                        batch_seen=batch_seen,
                        epoch=epoch,
                        train=train,
                        **kwargs,
                    )
                    if isinstance(pred_i, dict):
                        pred_i = pred_i["prediction"]
                    pred_full = pred_full.index_copy(2, nodes, pred_i)
            else:
                B = history_data.shape[0]
                horizon = self.clients[0].horizon
                out_dim = self.clients[0].output_dim
                pred_full = torch.zeros((B, horizon, self.total_nodes, out_dim), device=device, dtype=dtype)

                for cid, client in enumerate(self.clients):
                    nodes = self.get_client_nodes(cid)
                    x_i = history_data.index_select(2, nodes)
                    y_i = future_data.index_select(2, nodes) if future_data is not None else None

                    pred_i = client(
                        history_data=x_i,
                        future_data=y_i,
                        batch_seen=batch_seen,
                        epoch=epoch,
                        train=train,
                        **kwargs,
                    )
                    if isinstance(pred_i, dict):
                        pred_i = pred_i["prediction"]
                    pred_full = pred_full.index_copy(2, nodes, pred_i)

            return {"prediction": pred_full, "comm": comm}

        # -------------------------------------------------------
        # 1) Local: encoder -> spatial (per client)
        # -------------------------------------------------------
        spatial_list = []
        aux_list = []  # GRU: (h_encode, last_input, graph_encoding_4d)

        for cid, client in enumerate(self.clients):
            nodes = self.get_client_nodes(cid)
            x_i = history_data.index_select(2, nodes)

            if self.kind in ("staeformer", "stgformer", "stgcn"):
                enc_i = client.forward_encoder(x_i, **kwargs)
                spat_i = client.forward_spatial(enc_i, **kwargs)  # local spatial 그대로
                spatial_list.append(spat_i)
                aux_list.append(None)
            else:
                B, L, n_i, C = x_i.shape
                h_encode, last_in = client.forward_encoder(x_i, **kwargs)
                graph_enc = client.forward_spatial(h_encode, batch_size=B, num_nodes=n_i, **kwargs)
                Lr, BN, H = graph_enc.shape
                graph4d = graph_enc.view(Lr, B, n_i, H).permute(1, 0, 2, 3).contiguous()
                spatial_list.append(graph4d)
                aux_list.append((h_encode, last_in, graph4d))

        # -------------------------------------------------------
        # 2) Pooling (node -> tokens) and send to server
        # -------------------------------------------------------
        token_list = []
        for cid in range(self.num_clients):
            s_i = spatial_list[cid]  # [B,T,N,D]
            tokens_i = self.poolers[cid](s_i)  # [B,T,K,D]
            token_list.append(tokens_i)
            comm["forward"]["client_to_server_bytes"] += _bytes(tokens_i)

        tokens_all = torch.cat(token_list, dim=2)
        comm["forward"]["server_to_client_bytes"] = _bytes(tokens_all)
        comm["forward"]["total_bytes"] = (
            comm["forward"]["client_to_server_bytes"] + comm["forward"]["server_to_client_bytes"]
        )

        if train and torch.is_grad_enabled():
            self.total_forward_bytes += comm["forward"]["total_bytes"]

        # backward hook: server -> clients (grad of pooled tokens)
        if torch.is_grad_enabled():

            def _hook_tokens_in(g):
                b = _bytes(g)
                comm["backward"]["server_to_client_bytes"] += b
                self.total_backward_bytes += b

            tokens_all.register_hook(_hook_tokens_in)

        # -------------------------------------------------------
        # 3) Server self-attention over all tokens
        # -------------------------------------------------------
        tokens_all_out = self.server(tokens_all)

        # backward hook: clients -> server (grad of returned tokens)
        if torch.is_grad_enabled():

            def _hook_tokens_out(g):
                b = _bytes(g)
                comm["backward"]["client_to_server_bytes"] += b
                self.total_backward_bytes += b

            tokens_all_out.register_hook(_hook_tokens_out)

        tokens_out_list = torch.split(tokens_all_out, self.num_tokens, dim=2)

        # -------------------------------------------------------
        # 4) Expansion (tokens -> nodes), fusion, local decode
        # -------------------------------------------------------
        if self.kind in ("staeformer", "stgformer", "stgcn"):
            B = spatial_list[0].shape[0]
            out_steps = self.clients[0].out_steps
            out_dim = self.clients[0].output_dim
            pred_full = torch.zeros((B, out_steps, self.total_nodes, out_dim), device=device, dtype=dtype)

            for cid, client in enumerate(self.clients):
                nodes = self.get_client_nodes(cid)
                s_i = spatial_list[cid]
                t_i = tokens_out_list[cid]
                msg = self.expanders[cid](s_i, t_i)
                p_i = client.forward_decoder(msg, **kwargs)
                pred_full = pred_full.index_copy(2, nodes, p_i)
        else:
            B = history_data.shape[0]
            horizon = self.clients[0].horizon
            out_dim = self.clients[0].output_dim
            pred_full = torch.zeros((B, horizon, self.total_nodes, out_dim), device=device, dtype=dtype)

            for cid, client in enumerate(self.clients):
                nodes = self.get_client_nodes(cid)
                h_encode, last_in, g4d = aux_list[cid]
                t_i = tokens_out_list[cid]
                msg = self.expanders[cid](g4d, t_i)

                Lr = msg.shape[1]
                n_i = msg.shape[2]
                fused_graph = msg.permute(1, 0, 2, 3).contiguous().view(Lr, B * n_i, self.model_dim)

                y_i = future_data.index_select(2, nodes) if future_data is not None else None
                p_i = client.forward_decoder(
                    h_encode=h_encode,
                    graph_encoding=fused_graph,
                    last_input=last_in,
                    future_data=y_i,
                    batch_seen=batch_seen,
                    batch_size=B,
                    num_nodes=n_i,
                )
                pred_full = pred_full.index_copy(2, nodes, p_i)

        comm["backward"]["total_bytes"] = (
            comm["backward"]["client_to_server_bytes"] + comm["backward"]["server_to_client_bytes"]
        )
        return {"prediction": pred_full, "comm": comm}


# =========================
# FedAvg helper (for HierSplit + FedAvg, STAEformer only)
# =========================

# "세그먼트" 기준으로 안전하게 매칭 (substring보다 덜 위험)
FEDAVG_SHAREABLE_MODULES = {
    "input_proj",
    "tod_embedding",
    "dow_embedding",
    "attn_layers_t",
}

def _is_fedavg_shareable_param(key: str) -> bool:
    # 예: "encoder.input_proj.weight" 처럼 중간에 있어도 잡히게
    parts = key.split(".")
    return any(p in FEDAVG_SHAREABLE_MODULES for p in parts)

def _select_shareable_keys(state_dicts: List[Dict[str, torch.Tensor]]) -> List[str]:
    """
    - 화이트리스트에 해당
    - 모든 클라이언트에서 존재
    - shape 동일
    - floating tensor만
    인 키만 공유 대상으로 선택
    """
    if len(state_dicts) == 0:
        return []
    sd0 = state_dicts[0]
    keys: List[str] = []
    for k, v0 in sd0.items():
        if not _is_fedavg_shareable_param(k):
            continue
        if (not torch.is_tensor(v0)) or (not v0.is_floating_point()):
            continue
        ok = True
        for sd in state_dicts[1:]:
            v = sd.get(k, None)
            if v is None or (not torch.is_tensor(v)) or (v.shape != v0.shape):
                ok = False
                break
        if ok:
            keys.append(k)
    return keys

def _compute_fedavg_comm_bytes_from_keys(sd0: Dict[str, torch.Tensor], keys: List[str], num_clients: int) -> Tuple[int, int]:
    """
    FedAvg 통신량(바이트):
      - upload: 각 클라이언트가 서버로 업로드
      - download: 서버가 각 클라이언트로 다운로드
    """
    param_bytes = 0
    for k in keys:
        t = sd0[k]
        param_bytes += t.numel() * t.element_size()
    return param_bytes * num_clients, param_bytes * num_clients



# =========================
# Runner (inherits IL runner)
# =========================
class HierarchicalSplitLearningRunner(IndependentLearningRunner):
    def __init__(self, cfg: Dict):
        hs = cfg.get("HIERSPLIT", {})
        self._hs_pooling = str(hs.get("POOLING", "simple")).lower()
        self._hs_expansion = str(hs.get("EXPANSION", self._hs_pooling)).lower()
        self._hs_num_tokens = int(hs.get("NUM_TOKENS", 8))
        self._hs_use_fedavg = bool(hs.get("USE_FEDAVG", False))

        super().__init__(cfg)

        # ✅ FedAvg 실행 상태 추적
        self._fedavg_init_done = False          # epoch1 시작 초기 동기화 완료 여부
        self._last_fedavg_epoch = None          # epoch 시작 FedAvg(라운드) 수행 여부

        if self._hs_use_fedavg:
            model = self.model.module if hasattr(self.model, "module") else self.model
            model_kind = getattr(model, "kind", None)
            if model_kind == "staeformer":
                self.logger.info("HierSplit + FedAvg enabled (staeformer)")
            else:
                self.logger.warning(
                    f"HierSplit + FedAvg is only supported for staeformer, but got '{model_kind}'. FedAvg will be skipped."
                )

    def _extra_ckpt_meta(self) -> Dict[str, Any]:
        # 모델에서 통신량 가져오기
        model = self.model.module if hasattr(self.model, "module") else self.model
        total_forward_bytes = getattr(model, "total_forward_bytes", 0)
        total_backward_bytes = getattr(model, "total_backward_bytes", 0)
        
        return {
            "hiersplit_pooling": self._hs_pooling,
            "hiersplit_expansion": self._hs_expansion,
            "hiersplit_num_tokens": self._hs_num_tokens,
            "hiersplit_total_forward_bytes": total_forward_bytes,
            "hiersplit_total_backward_bytes": total_backward_bytes,
            "hiersplit_use_fedavg": self._hs_use_fedavg,
        }

    def _validate_extra_ckpt_meta(self, ckpt: Dict[str, Any]):
        if "hiersplit_num_tokens" not in ckpt:
            return
        if (
            ckpt.get("hiersplit_pooling") != self._hs_pooling
            or ckpt.get("hiersplit_expansion") != self._hs_expansion
            or int(ckpt.get("hiersplit_num_tokens")) != int(self._hs_num_tokens)
        ):
            raise RuntimeError(
                f"HierSplit config mismatch:\n"
                f"saved pooling/expansion/tokens = {ckpt.get('hiersplit_pooling')}/{ckpt.get('hiersplit_expansion')}/{ckpt.get('hiersplit_num_tokens')}\n"
                f"current = {self._hs_pooling}/{self._hs_expansion}/{self._hs_num_tokens}"
            )

    def load_model_resume(self, strict: bool = True):
        """Resume 시 통신량도 복원"""
        from easytorch.core.checkpoint import load_ckpt

        try:
            ckpt = load_ckpt(self.ckpt_save_dir, logger=self.logger)

            # validate meta
            self._validate_partition_meta(ckpt)
            self._validate_extra_ckpt_meta(ckpt)

            # load model
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.load_state_dict(ckpt["model_state_dict"], strict=strict)

            # 통신량 복원 (FedAvg 통신량도 forward/backward에 이미 포함됨)
            if "hiersplit_total_forward_bytes" in ckpt:
                model.total_forward_bytes = ckpt["hiersplit_total_forward_bytes"]
            if "hiersplit_total_backward_bytes" in ckpt:
                model.total_backward_bytes = ckpt["hiersplit_total_backward_bytes"]

            # load optim
            if getattr(self, "optim", None) is not None and ckpt.get("optim_state_dict", None) is not None:
                self.optim.load_state_dict(ckpt["optim_state_dict"])

            self.start_epoch = ckpt["epoch"]
            if ckpt.get("best_metrics") is not None:
                self.best_metrics = ckpt["best_metrics"]

            if self.scheduler is not None:
                self.scheduler.last_epoch = ckpt["epoch"]

            if ckpt.get("early_stopping_completed", False):
                self.early_stopping_completed = True
                self.logger.info("Early stopping was already completed. Skipping training.")
            else:
                self.logger.info(f"Resume training from epoch {self.start_epoch}")
                self.logger.info(f"  Restored communication: forward={model.total_forward_bytes/(1024*1024):.2f}MB, backward={model.total_backward_bytes/(1024*1024):.2f}MB")

        except (FileNotFoundError, IndexError):
            self.logger.info("No checkpoint found, start training from scratch.")

    def define_model(self, cfg: Dict) -> nn.Module:
        base_cls = cfg["MODEL"].get("CLASS", None) or cfg["MODEL"].get("ARCH", None)
        if base_cls is None:
            raise ValueError("cfg['MODEL']['CLASS'] or cfg['MODEL']['ARCH'] must be provided.")
        base_params = dict(cfg["MODEL"]["PARAM"])

        hs = cfg.get("HIERSPLIT", {})
        pooling = str(hs.get("POOLING", "simple")).lower()
        expansion = str(hs.get("EXPANSION", pooling)).lower()
        num_tokens = int(hs.get("NUM_TOKENS", 8))

        token_heads = int(hs.get("TOKEN_HEADS", 4))
        server_heads = int(hs.get("SERVER_HEADS", 4))
        server_layers = int(hs.get("SERVER_LAYERS", 1))
        server_ff_dim = int(hs.get("SERVER_FF_DIM", 256))
        dropout = float(hs.get("DROPOUT", 0.1))

        return HierSplitModel(
            base_model_cls=base_cls,
            base_model_params=base_params,
            client_nodes_list=self._il_client_nodes_list,
            subgraph_adj_list=self._il_subgraph_adj_list,
            total_nodes=self._il_total_nodes,
            pooling=pooling,
            expansion=expansion,
            num_tokens=num_tokens,
            token_heads=token_heads,
            server_heads=server_heads,
            server_layers=server_layers,
            server_ff_dim=server_ff_dim,
            dropout=dropout,
        )

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        _register_meter_like_train_loss(self.meter_pool, "train/comm_forward_mb", "train")
        _register_meter_like_train_loss(self.meter_pool, "train/comm_backward_mb", "train")

    def _save_test_metrics(self):
        """Save test metrics with communication stats to JSON file."""
        import json
        import os

        metrics_results = {}
        metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}

        # 통신량 추가 (FedAvg 통신량도 forward/backward에 이미 포함됨)
        model = self.model.module if hasattr(self.model, "module") else self.model
        total_forward_bytes = getattr(model, "total_forward_bytes", 0)
        total_backward_bytes = getattr(model, "total_backward_bytes", 0)
        total_bytes = total_forward_bytes + total_backward_bytes

        metrics_results['communication_cost'] = {
            'total_forward_bytes': total_forward_bytes,
            'total_backward_bytes': total_backward_bytes,
            'total_bytes': total_bytes,
            'total_forward_mb': total_forward_bytes / (1024 * 1024),
            'total_backward_mb': total_backward_bytes / (1024 * 1024),
            'total_mb': total_bytes / (1024 * 1024),
            'use_fedavg': self._hs_use_fedavg,
            'pooling': self._hs_pooling,
            'expansion': self._hs_expansion,
            'num_tokens': self._hs_num_tokens,
        }

        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]):
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index

        # ✅ FedAvg는 "epoch 시작"에 수행 (1번 S2Former와 타이밍 정렬)
        if self._hs_use_fedavg:
            model = self.model.module if hasattr(self.model, "module") else self.model
            if getattr(model, "kind", None) == "staeformer":
                # epoch 시작 판정: iter_index==0 기준 (필요시 환경에 맞게 조정)
                if iter_index == 0:
                    # (1) epoch1 시작: 초기 동기화(공유 파트 동일 초기화)
                    if (epoch == 1) and (not self._fedavg_init_done):
                        c2s, s2c = self._fedavg_aggregate_clients(epoch=epoch, init_sync=True)
                        model.total_forward_bytes += c2s
                        model.total_backward_bytes += s2c
                        self._fedavg_init_done = True
                        self._last_fedavg_epoch = 1  # epoch1은 이미 동기화 완료

                    # (2) epoch>=2 시작: 이전 epoch 로컬 학습 결과를 평균
                    elif self._last_fedavg_epoch != epoch:
                        c2s, s2c = self._fedavg_aggregate_clients(epoch=epoch, init_sync=False)
                        model.total_forward_bytes += c2s
                        model.total_backward_bytes += s2c
                        self._last_fedavg_epoch = epoch

        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        self.optim.zero_grad()

        pred = forward_return["prediction"]
        tgt = forward_return["target"]
        loss = self.metric_forward(self.loss, {"prediction": pred, "target": tgt})

        loss.backward()
        self.optim.step()

        total_loss = loss

        weight = self._get_metric_weight(forward_return["target"])
        self.update_epoch_meter("train/loss", total_loss.item(), weight)

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item(), weight)

        comm = forward_return.get("comm", None)
        if comm is not None:
            comm["backward"]["total_bytes"] = (
                comm["backward"]["client_to_server_bytes"] + comm["backward"]["server_to_client_bytes"]
            )
            self.update_epoch_meter("train/comm_forward_mb", comm["forward"]["total_bytes"] / (1024 * 1024), 1.0)
            self.update_epoch_meter("train/comm_backward_mb", comm["backward"]["total_bytes"] / (1024 * 1024), 1.0)

        return None

    @torch.no_grad()
    def _fedavg_aggregate_clients(self, epoch: int, init_sync: bool = False) -> Tuple[int, int]:
        """
        init_sync=True:
          - epoch1 시작에 "초기 공유 파트 동기화" (표준 FedAvg처럼 동일 초기 모델 배포)
          - 여기서는 client0의 공유 파트를 그대로 broadcast (mean-of-random으로 스케일 줄이는 문제 회피)

        init_sync=False:
          - epoch 시작(=이전 epoch 로컬 학습 이후)에 uniform mean으로 FedAvg
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        clients = model.clients
        num_clients = len(clients)
        if num_clients <= 1:
            return 0, 0

        # state_dict 수집 (deepcopy 불필요)
        sds = [c.state_dict() for c in clients]

        share_keys = _select_shareable_keys(sds)
        if epoch == 1 and init_sync:
            self.logger.info(f"[FedAvg:init] shareable keys = {len(share_keys)}")
            if len(share_keys) > 0:
                self.logger.info(f"[FedAvg:init] example keys: {share_keys[:12]}{'...' if len(share_keys) > 12 else ''}")

        if not share_keys:
            # 공유할 게 없으면 아무것도 하지 않음 (키 네이밍 mismatch 가능)
            if epoch == 1:
                self.logger.warning("[FedAvg] No shareable keys found. Check FEDAVG_SHAREABLE_MODULES vs model state_dict keys.")
            return 0, 0

        # ✅ update dict 생성
        update_sd: Dict[str, torch.Tensor] = {}

        if init_sync:
            # client0 기준으로 그대로 배포 (초기 표현공간 정렬)
            for k in share_keys:
                update_sd[k] = sds[0][k].detach().clone()
        else:
            # uniform mean (1번 코드와 동일)
            for k in share_keys:
                v0 = sds[0][k]
                avg = torch.zeros_like(v0, dtype=torch.float32)
                for sd in sds:
                    avg += sd[k].float()
                avg /= float(num_clients)
                update_sd[k] = avg.to(dtype=v0.dtype)

        # ✅ 배포 (부분 dict로 strict=False)
        for c in clients:
            c.load_state_dict(update_sd, strict=False)

        # 통신량 계산
        c2s_bytes, s2c_bytes = _compute_fedavg_comm_bytes_from_keys(sds[0], share_keys, num_clients)

        # init_sync는 "서버가 이미 글로벌 모델을 갖고 있다"는 전형적 상황을 가정하면 upload는 0이 더 자연스럽지만,
        # 기존 통계 스타일 유지하려면 그대로 두어도 됨.
        if init_sync:
            c2s_bytes = 0  # ✅ 초기 배포는 다운로드만 카운트 (원하면 주석처리해서 양방향 카운트 가능)

        return c2s_bytes, s2c_bytes


    def on_epoch_end(self, epoch: int):
       
        super().on_epoch_end(epoch)
