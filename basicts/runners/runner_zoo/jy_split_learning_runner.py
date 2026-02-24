#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import os

from easytorch.utils import master_only
from .jy_partitioned_runner_base import PartitionedRunnerBase

def _bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()

def _register_meter_like_train_loss(meter_pool, name: str, meter_type: str = "train"):
    """
    basicts meter_pool entry 형식({'meter','type','index',...})을 맞추기 위해
    'train/loss'를 템플릿으로 복사해서 등록한다.
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
    entry["meter"] = type(tmpl["meter"])()
    entry["type"] = meter_type
    entry["index"] = max_idx + 1
    pool[name] = entry


def _ensure_tensor_adj(adj: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if adj is None:
        return None
    if isinstance(adj, torch.Tensor):
        return adj
    return torch.tensor(adj, dtype=torch.float32)


class _STGCNServerSpatial(nn.Module):
    """
    STGCN Split Learning용 Server Spatial.
    
    전체 그래프를 사용하여 Graph Convolution 수행.
    클라이언트 간 정보 교환이 이 레이어에서 발생.
    
    입력/출력:
      - in : (B, T, N, D)
      - out: (B, T, N, D)
    """

    def __init__(
        self,
        num_nodes: int,
        feature_dim: int,
        Ks: int,
        adj_matrix: torch.Tensor,
        graph_conv_type: str = "cheb_graph_conv",
        bias: bool = True,
        droprate: float = 0.1,
    ) -> None:
        super().__init__()
        from baselines.STGCN.arch.layers import GraphConvLayer
        
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        
        self._gcn = GraphConvLayer(
            graph_conv_type=graph_conv_type,
            c_in=feature_dim,
            c_out=feature_dim,
            Ks=Ks,
            gso=adj_matrix,
            bias=bias,
        )
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout(p=droprate)

    def forward_spatial(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # (B, T, N, D) -> (B, D, T, N)
        x_ = x.permute(0, 3, 1, 2)
        x_ = self._gcn(x_)
        x_ = self._relu(x_)
        x_ = self._dropout(x_)
        # (B, D, T, N) -> (B, T, N, D)
        return x_.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_spatial(x)


class SplitLearningModel(nn.Module):
    """
    Clients: forward_encoder + forward_decoder
    Server : forward_spatial

    Comm (per iter):
      Forward:
        c2s = encoder activations (+ STGformer adaptive_embedding slices)
        s2c = spatial activations
      Backward:
        c2s = grad(spatial activations)
        s2c = grad(encoder activations) (+ grad(global adaptive_embedding) for STGformer)
    """

    def __init__(
        self,
        base_model_cls: type[nn.Module],
        base_model_params: Dict,
        client_nodes_list: List[List[int]],
        total_nodes: int,
        kind: str,
        full_adj: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.kind = kind
        self.client_nodes_list = client_nodes_list
        self.total_nodes = total_nodes
        self.full_adj = _ensure_tensor_adj(full_adj)

        for cid, nodes in enumerate(client_nodes_list):
            self.register_buffer(f"client_nodes_{cid}", torch.tensor(nodes, dtype=torch.long), persistent=False)

        # clients: separate model per client (num_nodes = local)
        self.clients = nn.ModuleList()
        for nodes in client_nodes_list:
            p = dict(base_model_params)
            p["num_nodes"] = len(nodes)
            # GRU client doesn't need full edges
            if self.kind == "gruseq2seq_graphnet":
                p["edge_index"] = None
                p["edge_attr"] = None
            # STGCN needs per-client subgraph adjacency
            if self.kind == "stgcn":
                if self.full_adj is None and p.get("adj_matrix", None) is None:
                    raise ValueError("STGCN split needs full adjacency (DATASET adj) or MODEL.PARAM.adj_matrix.")
                full = self.full_adj if self.full_adj is not None else _ensure_tensor_adj(p["adj_matrix"])
                p["adj_matrix"] = full[nodes][:, nodes].contiguous()
            self.clients.append(base_model_cls(**p))

        # server: STGCN은 전체 그래프를 사용하는 Graph Conv 서버
        if self.kind == "stgcn":
            if self.full_adj is None and base_model_params.get("adj_matrix", None) is None:
                raise ValueError("STGCN split needs full adjacency for server.")
            full_adj_for_server = self.full_adj if self.full_adj is not None else _ensure_tensor_adj(base_model_params["adj_matrix"])
            
            # encoder 출력 feature dim (blocks[2][-1])
            feature_dim = base_model_params.get("blocks", [[1], [64, 16, 64], [64, 16, 64]])[2][-1]
            Ks = base_model_params.get("Ks", 3)
            graph_conv_type = base_model_params.get("graph_conv_type", "cheb_graph_conv")
            bias = base_model_params.get("bias", True)
            droprate = base_model_params.get("droprate", 0.1)
            
            self.server = _STGCNServerSpatial(
                num_nodes=total_nodes,
                feature_dim=feature_dim,
                Ks=Ks,
                adj_matrix=full_adj_for_server,
                graph_conv_type=graph_conv_type,
                bias=bias,
                droprate=droprate,
            )
        else:
            # server: one model (num_nodes = total)
            pS = dict(base_model_params)
            pS["num_nodes"] = total_nodes

            # GRU server needs full edges
            if self.kind == "gruseq2seq_graphnet":
                if pS.get("edge_index", None) is None:
                    if self.full_adj is None:
                        raise ValueError("GRU split needs full adjacency (DATASET adj) or edge_index in MODEL.PARAM.")
                    edge_index = (self.full_adj > 0).nonzero(as_tuple=False).t().contiguous()
                    edge_attr = self.full_adj[self.full_adj > 0].contiguous()
                    pS["edge_index"] = edge_index
                    pS["edge_attr"] = edge_attr

            self.server = base_model_cls(**pS)

        # cumulative stats
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

        # ---------------------------
        # STAEformer / STGformer / STGCN
        # ---------------------------
        if self.kind in ("staeformer", "stgformer", "stgcn"):
            B = history_data.shape[0]

            # 1) client enc
            enc = {}
            for cid, client in enumerate(self.clients):
                nodes = self.get_client_nodes(cid)
                x_i = history_data.index_select(2, nodes)
                enc_i = client.forward_encoder(x_i, **kwargs)
                enc[cid] = enc_i
                comm["forward"]["client_to_server_bytes"] += _bytes(enc_i)

            # 2) assemble global enc in original node order (differentiable)
            T_feat = next(iter(enc.values())).shape[1]
            D_feat = next(iter(enc.values())).shape[-1]
            global_enc = torch.zeros((B, T_feat, self.total_nodes, D_feat), device=device, dtype=dtype)
            for cid in range(len(self.clients)):
                nodes = self.get_client_nodes(cid)
                global_enc = global_enc.index_copy(2, nodes, enc[cid])

            # hook: server->clients (grad of global_enc)
            if torch.is_grad_enabled():
                def _hook_genc(g):
                    b = _bytes(g)
                    comm["backward"]["server_to_client_bytes"] += b
                    self.total_backward_bytes += b
                global_enc.register_hook(_hook_genc)

            # 2.5) STGformer: send adaptive_embedding slices to server for GLOBAL graph (as you requested)
            global_adp = None
            if self.kind == "stgformer":
                Tin, _, Dadp = self.clients[0].adaptive_embedding.shape
                global_adp = torch.zeros((Tin, self.total_nodes, Dadp), device=device, dtype=self.clients[0].adaptive_embedding.dtype)
                for cid, client in enumerate(self.clients):
                    nodes = self.get_client_nodes(cid)
                    adp_i = client.adaptive_embedding
                    global_adp = global_adp.index_copy(1, nodes, adp_i)
                    comm["forward"]["client_to_server_bytes"] += _bytes(adp_i)

                if torch.is_grad_enabled():
                    def _hook_gadp(g):
                        b = _bytes(g)
                        comm["backward"]["server_to_client_bytes"] += b
                        self.total_backward_bytes += b
                    global_adp.register_hook(_hook_gadp)

            # 3) server spatial (uses model's own forward_spatial)
            if self.kind == "stgformer":
                spatial_global = self.server.forward_spatial(global_enc, adaptive_embedding=global_adp, **kwargs)
            else:
                spatial_global = self.server.forward_spatial(global_enc, **kwargs)

            comm["forward"]["server_to_client_bytes"] = _bytes(spatial_global)
            comm["forward"]["total_bytes"] = comm["forward"]["client_to_server_bytes"] + comm["forward"]["server_to_client_bytes"]
            if train and torch.is_grad_enabled():
                self.total_forward_bytes += comm["forward"]["total_bytes"]

            # hook: clients->server (grad of spatial_global)
            if torch.is_grad_enabled():
                def _hook_spatial(g):
                    b = _bytes(g)
                    comm["backward"]["client_to_server_bytes"] += b
                    self.total_backward_bytes += b
                spatial_global.register_hook(_hook_spatial)

            # 4) client decode + assemble prediction
            out_steps = self.clients[0].out_steps
            out_dim = self.clients[0].output_dim
            pred_full = torch.zeros((B, out_steps, self.total_nodes, out_dim), device=device, dtype=dtype)

            for cid, client in enumerate(self.clients):
                nodes = self.get_client_nodes(cid)
                s_i = spatial_global.index_select(2, nodes)
                p_i = client.forward_decoder(s_i, **kwargs)
                pred_full = pred_full.index_copy(2, nodes, p_i)

            return {"prediction": pred_full, "comm": comm}

        # ---------------------------
        # GRUSeq2SeqWithGraphNet
        # ---------------------------
        if self.kind == "gruseq2seq_graphnet":
            B, L, N, C = history_data.shape

            # 1) client enc
            h_list = {}
            last_list = {}
            for cid, client in enumerate(self.clients):
                nodes = self.get_client_nodes(cid)
                x_i = history_data.index_select(2, nodes)

                h_i, last_i = client.forward_encoder(x_i, **kwargs)  # h_i: [Lr, B*n, H]
                h_list[cid] = h_i
                last_list[cid] = last_i
                comm["forward"]["client_to_server_bytes"] += _bytes(h_i)

            Lr, BN0, H = next(iter(h_list.values())).shape
            # assemble global h4d: [Lr, B, N_total, H]
            global_h4d = torch.zeros((Lr, B, self.total_nodes, H), device=device, dtype=dtype)
            for cid, h_i in h_list.items():
                nodes = self.get_client_nodes(cid)
                n_i = len(nodes)
                h4d_i = h_i.view(Lr, B, n_i, H)
                global_h4d = global_h4d.index_copy(2, nodes, h4d_i)

            global_h = global_h4d.reshape(Lr, B * self.total_nodes, H)

            # hook: server->clients (grad of global_h)
            if torch.is_grad_enabled():
                def _hook_gh(g):
                    b = _bytes(g)
                    comm["backward"]["server_to_client_bytes"] += b
                    self.total_backward_bytes += b
                global_h.register_hook(_hook_gh)

            # 2) server spatial (graphnet)
            # server.forward_spatial uses its own edge_index/edge_attr by default
            graph_global = self.server.forward_spatial(global_h, batch_size=B, num_nodes=self.total_nodes, **kwargs)
            comm["forward"]["server_to_client_bytes"] = _bytes(graph_global)
            comm["forward"]["total_bytes"] = comm["forward"]["client_to_server_bytes"] + comm["forward"]["server_to_client_bytes"]
            if train and torch.is_grad_enabled():
                self.total_forward_bytes += comm["forward"]["total_bytes"]

            # hook: clients->server (grad of graph_global)
            if torch.is_grad_enabled():
                def _hook_gg(g):
                    b = _bytes(g)
                    comm["backward"]["client_to_server_bytes"] += b
                    self.total_backward_bytes += b
                graph_global.register_hook(_hook_gg)

            # 3) client decode
            horizon = self.clients[0].horizon
            out_dim = self.clients[0].output_dim
            pred_full = torch.zeros((B, horizon, self.total_nodes, out_dim), device=device, dtype=dtype)

            graph4d = graph_global.view(Lr, B, self.total_nodes, H)
            for cid, client in enumerate(self.clients):
                nodes = self.get_client_nodes(cid)
                n_i = len(nodes)

                h_i = h_list[cid]  # [Lr, B*n_i, H]
                g_i = graph4d.index_select(2, nodes).reshape(Lr, B * n_i, H)
                last_i = last_list[cid]
                y_i = future_data.index_select(2, nodes) if future_data is not None else None

                p_i = client.forward_decoder(
                    h_encode=h_i,
                    graph_encoding=g_i,
                    last_input=last_i,
                    future_data=y_i,
                    batch_seen=batch_seen,
                    batch_size=B,
                    num_nodes=n_i,
                )
                pred_full = pred_full.index_copy(2, nodes, p_i)

            return {"prediction": pred_full, "comm": comm}

        raise ValueError(f"Unknown split kind: {self.kind}")

class SplitLearningRunner(PartitionedRunnerBase):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self._client_optimizers = None
        self._server_optimizer = None
        self._client_schedulers = None
        self._server_scheduler = None

        self._pending_split_opt = None
        self._pending_split_sched = None

    def define_model(self, cfg: Dict) -> nn.Module:
        base_cls = cfg["MODEL"].get("CLASS", None) or cfg["MODEL"].get("ARCH", None)
        base_params = dict(cfg["MODEL"]["PARAM"])

        kind = cfg.get("SPLIT", {}).get("KIND", None)
        if kind is None:
            name = base_cls.__name__.lower()
            if "staeformer" in name:
                kind = "staeformer"
            elif "stgformer" in name:
                kind = "stgformer"
            elif "stgcn" in name:
                kind = "stgcn"
            else:
                kind = "gruseq2seq_graphnet"

        return SplitLearningModel(
            base_model_cls=base_cls,
            base_model_params=base_params,
            client_nodes_list=self._il_client_nodes_list,
            total_nodes=self._il_total_nodes,
            kind=kind,
            full_adj=self._il_full_adj,
        )

    def _setup_split_optimizers(self, cfg: Dict):
        model = self.model.module if hasattr(self.model, "module") else self.model

        optim_cfg = cfg["TRAIN"]["OPTIM"]
        optim_class = getattr(torch.optim, optim_cfg["TYPE"])
        optim_params = optim_cfg["PARAM"]

        self._client_optimizers = [optim_class(m.parameters(), **optim_params) for m in model.get_client_models()]
        server_params = [p for p in model.get_server_model().parameters() if p.requires_grad]
        if len(server_params) == 0:
            self._server_optimizer = None
        else:
            self._server_optimizer = optim_class(server_params, **optim_params)

        sched_cfg = cfg["TRAIN"]["LR_SCHEDULER"]
        sched_class = getattr(torch.optim.lr_scheduler, sched_cfg["TYPE"])
        sched_params = sched_cfg["PARAM"]

        self._client_schedulers = [sched_class(o, **sched_params) for o in self._client_optimizers]
        if self._server_optimizer is not None:
            self._server_scheduler = sched_class(self._server_optimizer, **sched_params)
        else:
            self._server_scheduler = None

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        self._setup_split_optimizers(cfg)

        # ✅ comm meters 등록 (KeyError 방지)
        _register_meter_like_train_loss(self.meter_pool, "train/comm_forward_mb", "train")
        _register_meter_like_train_loss(self.meter_pool, "train/comm_backward_mb", "train")

        # pending state 적용
        if self._pending_split_opt is not None:
            c_states, s_state = self._pending_split_opt
            if c_states:
                for o, st in zip(self._client_optimizers, c_states):
                    o.load_state_dict(st)
            if s_state and self._server_optimizer is not None:
                self._server_optimizer.load_state_dict(s_state)

        if self._pending_split_sched is not None:
            c_s, s_s = self._pending_split_sched
            if c_s:
                for sch, st in zip(self._client_schedulers, c_s):
                    sch.load_state_dict(st)
            if s_s and self._server_scheduler is not None:
                self._server_scheduler.load_state_dict(s_s)

        # base scheduler는 split에서는 안 씀
        self.scheduler = None

    @master_only
    def save_model(self, epoch: int):
        from easytorch.core.checkpoint import save_ckpt, backup_last_ckpt, clear_ckpt

        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        # 통신량 가져오기
        total_forward_bytes = getattr(model, "total_forward_bytes", 0)
        total_backward_bytes = getattr(model, "total_backward_bytes", 0)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_metrics": self.best_metrics,
            **self._partition_meta_for_ckpt(),

            # ✅ split optimizer/scheduler states
            "client_optim_state_dicts": [o.state_dict() for o in self._client_optimizers],
            "server_optim_state_dict": self._server_optimizer.state_dict() if self._server_optimizer is not None else None,
            "client_sched_state_dicts": [s.state_dict() for s in self._client_schedulers],
            "server_sched_state_dict": self._server_scheduler.state_dict() if self._server_scheduler else None,

            # ✅ 통신량 저장
            "split_total_forward_bytes": total_forward_bytes,
            "split_total_backward_bytes": total_backward_bytes,
        }

        if epoch > 1:
            last_ckpt_path = self.get_ckpt_path(epoch - 1)
            if os.path.exists(last_ckpt_path):
                backup_last_ckpt(last_ckpt_path, epoch, self.ckpt_save_strategy)

        ckpt_path = self.get_ckpt_path(epoch)
        save_ckpt(ckpt, ckpt_path, self.logger)

        if epoch % 10 == 0 or epoch == self.num_epochs:
            clear_ckpt(self.ckpt_save_dir)

    def load_model_resume(self, strict: bool = True):
        from easytorch.core.checkpoint import load_ckpt
        try:
            ckpt = load_ckpt(self.ckpt_save_dir, logger=self.logger)

            self._validate_partition_meta(ckpt)

            # model weights
            model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            model.load_state_dict(ckpt["model_state_dict"], strict=strict)

            # ✅ 통신량 복원
            if "split_total_forward_bytes" in ckpt:
                model.total_forward_bytes = ckpt["split_total_forward_bytes"]
            if "split_total_backward_bytes" in ckpt:
                model.total_backward_bytes = ckpt["split_total_backward_bytes"]

            self.start_epoch = ckpt["epoch"]
            if ckpt.get("best_metrics") is not None:
                self.best_metrics = ckpt["best_metrics"]

            # ✅ optim/sched state는 init_training 후에 적용
            self._pending_split_opt = (
                ckpt.get("client_optim_state_dicts", None),
                ckpt.get("server_optim_state_dict", None),
            )
            self._pending_split_sched = (
                ckpt.get("client_sched_state_dicts", None),
                ckpt.get("server_sched_state_dict", None),
            )

            self.logger.info(f"Resume training from epoch {self.start_epoch}")
            self.logger.info(f"  Restored communication: forward={model.total_forward_bytes/(1024*1024):.2f}MB, backward={model.total_backward_bytes/(1024*1024):.2f}MB")

        except (FileNotFoundError, IndexError):
            self.logger.info("No checkpoint found, start training from scratch.")

    def _register_comm_meters(self):
        """
        basicts meter_pool은 각 항목이 {'meter', 'type', 'index', ...} 형태여야 print_meters가 안 터짐.
        그래서 기존 train/loss 엔트리를 템플릿으로 복사해서 comm meter를 등록한다.
        """
        if not hasattr(self, "meter_pool") or not hasattr(self.meter_pool, "_pool"):
            return

        pool = self.meter_pool._pool
        template_key = "train/loss"
        if template_key not in pool:
            return

        template = pool[template_key]
        # 다음 index는 전체 pool 기준 max+1로
        max_idx = 0
        for v in pool.values():
            if isinstance(v, dict) and "index" in v:
                max_idx = max(max_idx, v["index"])

        def _add(name: str, meter_type: str = "train"):
            nonlocal max_idx
            if name in pool:
                return
            new_entry = dict(template)  # index/type/format 등 템플릿 그대로
            # meter는 템플릿 meter와 같은 클래스의 새 인스턴스 사용
            new_meter = type(template["meter"])()
            if hasattr(new_meter, "reset"):
                new_meter.reset()
            new_entry["meter"] = new_meter
            max_idx += 1
            new_entry["index"] = max_idx
            new_entry["type"] = meter_type
            pool[name] = new_entry

        _add("train/comm_forward_mb", "train")
        _add("train/comm_backward_mb", "train")


    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]):
        iter_num = (epoch - 1) * self.iter_per_epoch + iter_index

        for opt in self._client_optimizers:
            opt.zero_grad(set_to_none=True)
        if self._server_optimizer is not None:
            self._server_optimizer.zero_grad(set_to_none=True)

        forward_return = self.forward(data=data, epoch=epoch, iter_num=iter_num, train=True)

        loss = self.metric_forward(self.loss, forward_return)
        loss.backward()

        # finalize backward total
        comm = forward_return.get("comm", None)
        if comm is not None:
            comm["backward"]["total_bytes"] = comm["backward"]["client_to_server_bytes"] + comm["backward"]["server_to_client_bytes"]

        for opt in self._client_optimizers:
            opt.step()
        if self._server_optimizer is not None:
            self._server_optimizer.step()

        weight = self._get_metric_weight(forward_return["target"])
        self.update_epoch_meter("train/loss", loss.item(), weight)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_epoch_meter(f"train/{metric_name}", metric_item.item(), weight)

        if comm is not None:
            self.update_epoch_meter("train/comm_forward_mb", comm["forward"]["total_bytes"] / (1024 * 1024), 1.0)
            self.update_epoch_meter("train/comm_backward_mb", comm["backward"]["total_bytes"] / (1024 * 1024), 1.0)

        return None

    def on_epoch_end(self, epoch: int) -> None:
        for s in self._client_schedulers:
            s.step()
        if self._server_scheduler is not None:
            self._server_scheduler.step()
            self.update_epoch_meter("train/lr", self._server_scheduler.get_last_lr()[0])
        elif self._client_schedulers is not None and len(self._client_schedulers) > 0:
            self.update_epoch_meter("train/lr", self._client_schedulers[0].get_last_lr()[0])

        original = self.scheduler
        self.scheduler = None
        super().on_epoch_end(epoch)
        self.scheduler = original

    def _save_test_metrics(self):
        """Save test metrics with communication stats to JSON file."""
        import json

        metrics_results = {}
        metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
        for i in self.evaluation_horizons:
            metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}

        # 통신량 추가
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
        }

        with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics_results, f, indent=4)
