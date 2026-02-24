#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn


def _adj_to_edge_index(adj: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if adj is None:
        return None, None
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj).float()
    edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()
    edge_attr = adj[adj > 0].contiguous()
    return edge_index, edge_attr


def _partition_supports_for_client(supports: List[torch.Tensor], client_nodes: List[int]) -> List[torch.Tensor]:
    out = []
    for s in supports:
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        out.append(s[client_nodes][:, client_nodes].contiguous())
    return out


def _ensure_tensor(adj: torch.Tensor) -> torch.Tensor:
    if isinstance(adj, np.ndarray):
        return torch.from_numpy(adj).float()
    return adj


class IndependentClientEnsemble(nn.Module):
    """
    여러 client model을 들고 full-node prediction으로 합친다.
    - differentiable index_copy_ 사용(grad 정상)
    """
    def __init__(
        self,
        base_model_class: Type[nn.Module],
        base_model_params: Dict[str, Any],
        client_nodes_list: List[List[int]],
        subgraph_adj_list: Optional[List[torch.Tensor]],
        full_adj: Optional[torch.Tensor],
        output_dim: int,
    ):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = dict(base_model_params)
        self.client_nodes_list = client_nodes_list
        self.output_dim = int(output_dim)

        uses_supports = ("supports" in self.base_model_params)
        uses_edges = ("edge_index" in self.base_model_params) or ("edge_attr" in self.base_model_params)
        uses_adj_matrix = ("adj_matrix" in self.base_model_params)
        global_supports = self.base_model_params.get("supports", None) if uses_supports else None

        self.client_models = nn.ModuleList()

        for ci, nodes in enumerate(self.client_nodes_list):
            self.register_buffer(f"client_nodes_{ci}", torch.tensor(nodes, dtype=torch.long), persistent=False)

        for ci, nodes in enumerate(self.client_nodes_list):
            params = dict(self.base_model_params)
            params["num_nodes"] = len(nodes)

            # supports (STGformer)
            if uses_supports:
                if subgraph_adj_list is not None and ci < len(subgraph_adj_list):
                    if params.get("supports", None) is None:
                        params["supports"] = [subgraph_adj_list[ci]]
                    elif isinstance(global_supports, list) and len(global_supports) > 0:
                        params["supports"] = _partition_supports_for_client(global_supports, nodes)
                else:
                    if isinstance(global_supports, list) and len(global_supports) > 0:
                        params["supports"] = _partition_supports_for_client(global_supports, nodes)
                    else:
                        params.pop("supports", None)
            else:
                params.pop("supports", None)

            # edges (GRU)
            if uses_edges:
                if subgraph_adj_list is not None and ci < len(subgraph_adj_list):
                    adj = subgraph_adj_list[ci]
                    edge_index, edge_attr = _adj_to_edge_index(adj)
                    params["edge_index"] = edge_index
                    params["edge_attr"] = edge_attr
                elif full_adj is not None:
                    adj = full_adj[nodes][:, nodes]
                    edge_index, edge_attr = _adj_to_edge_index(adj)
                    params["edge_index"] = edge_index
                    params["edge_attr"] = edge_attr
                else:
                    params.setdefault("edge_index", None)
                    params.setdefault("edge_attr", None)

            # adj_matrix (STGCN 등)
            if uses_adj_matrix:
                # NOTE:
                # subgraph_adj_list가 제공되더라도, 일부 설정/경로에서 잘못된 값이 섞이는 케이스가 있어
                # 가장 신뢰 가능한 full_adj로 항상 슬라이싱해서 client용 adj를 만든다.
                if full_adj is not None:
                    full = _ensure_tensor(full_adj)
                    params["adj_matrix"] = full[nodes][:, nodes].contiguous()
                elif subgraph_adj_list is not None and ci < len(subgraph_adj_list):
                    params["adj_matrix"] = _ensure_tensor(subgraph_adj_list[ci]).contiguous()
                else:
                    raise ValueError("base_model_params requires 'adj_matrix' but no adjacency is available for partitioning.")

            self.client_models.append(base_model_class(**params))

    def get_client_nodes(self, client_idx: int) -> torch.Tensor:
        return getattr(self, f"client_nodes_{client_idx}")

    def forward(self, history_data, future_data=None, batch_seen=0, epoch=0, train=True, **kwargs) -> Dict[str, torch.Tensor]:
        B, _, N, _ = history_data.shape
        device = history_data.device
        dtype = history_data.dtype

        L_out = future_data.shape[1] if future_data is not None else self.base_model_params.get("out_steps", history_data.shape[1])
        pred_all = torch.zeros((B, L_out, N, self.output_dim), device=device, dtype=dtype)

        for ci, client_model in enumerate(self.client_models):
            nodes = self.get_client_nodes(ci)
            x_client = history_data.index_select(2, nodes)
            y_client = future_data.index_select(2, nodes) if future_data is not None else None

            out = client_model(history_data=x_client, future_data=y_client, batch_seen=batch_seen, epoch=epoch, train=train, **kwargs)
            if isinstance(out, dict):
                out = out["prediction"]

            # index_copy_는 autograd 지원됨(grad 살아있음)
            pred_all.index_copy_(2, nodes, out)

        return {"prediction": pred_all}
