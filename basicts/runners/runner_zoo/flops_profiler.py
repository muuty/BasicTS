#!/usr/bin/env python3
"""
FLOPs Profiler for Time Series Forecasting Models

별도의 스크립트로 실행하여 모델의 FLOPs를 측정합니다.
기존 runner들을 수정하지 않고, config 파일만 읽어서 모델을 생성하고 측정합니다.

Usage:
    python experiments/measure_flops.py -c baselines/STAEformer/METR-LA.py
    python experiments/measure_flops.py -c new_baselines_jy/split/staeformer/METR_LA.py

Supported methods:
    - central: 전체 모델 FLOPs
    - independent: 클라이언트별 모델 FLOPs 합산
    - split: client + server 분리 측정
    - hiersplit: client + pooler/expander + server 분리 측정
    - federated: 클라이언트별 모델 FLOPs 합산
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Manual FLOPs calculation functions
# =============================================================================

def _manual_linear_flops(in_features: int, out_features: int, batch_elements: int) -> int:
    """FLOPs for nn.Linear: 2 * in * out * batch (multiply-add)."""
    return 2 * in_features * out_features * batch_elements


def _manual_attention_flops(batch_size: int, num_heads: int, seq_len: int, head_dim: int) -> int:
    """
    FLOPs for multi-head attention:
    - Q, K, V projections: 3 * 2 * seq * dim * dim
    - QK^T: 2 * seq * seq * dim
    - softmax: ~5 * seq * seq (approx)
    - attn @ V: 2 * seq * seq * dim
    - output projection: 2 * seq * dim * dim
    """
    model_dim = num_heads * head_dim
    # Q, K, V projections
    qkv_flops = 3 * _manual_linear_flops(model_dim, model_dim, batch_size * seq_len)
    # QK^T matmul
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    # softmax (approx 5 ops per element)
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    # attn @ V
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    # output projection
    out_flops = _manual_linear_flops(model_dim, model_dim, batch_size * seq_len)

    return qkv_flops + qk_flops + softmax_flops + av_flops + out_flops


def _manual_ffn_flops(batch_size: int, seq_len: int, model_dim: int, ff_dim: int) -> int:
    """FLOPs for feed-forward network (2 linear layers)."""
    ff1 = _manual_linear_flops(model_dim, ff_dim, batch_size * seq_len)
    ff2 = _manual_linear_flops(ff_dim, model_dim, batch_size * seq_len)
    return ff1 + ff2


def _manual_gru_flops(
    seq_len: int,
    batch_size: int,
    input_size: int,
    hidden_size: int,
    num_layers: int,
) -> int:
    """
    Manual FLOPs for GRU layer.
    GRU cell: r_t = sigmoid(W_r @ [x_t, h_{t-1}])
              z_t = sigmoid(W_z @ [x_t, h_{t-1}])
              n_t = tanh(W_n @ [x_t, r_t * h_{t-1}])
              h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    
    Per timestep:
    - 3 gates: 3 * (input_size + hidden_size) * hidden_size * 2 MACs
    - Element-wise ops: ~5 * hidden_size (sigmoid, tanh, multiply, add)
    """
    # Per timestep FLOPs
    gate_flops = 3 * 2 * (input_size + hidden_size) * hidden_size  # 3 gates
    elem_flops = 5 * hidden_size  # sigmoid, tanh, multiply, add (approx)
    per_timestep = gate_flops + elem_flops
    
    # For each layer
    total_flops = 0
    for layer in range(num_layers):
        if layer == 0:
            input_dim = input_size
        else:
            input_dim = hidden_size
        layer_flops = seq_len * batch_size * (
            3 * 2 * (input_dim + hidden_size) * hidden_size + 5 * hidden_size
        )
        total_flops += layer_flops
    
    return total_flops


def _manual_gru_encoder_flops(
    batch_size: int,
    seq_len: int,
    num_nodes: int,
    input_dim: int,
    hidden_size: int,
    num_layers: int,
) -> int:
    """Manual FLOPs for GRU encoder (forward_encoder)."""
    # Input reshape: (B, L, N, C) -> (L, B*N, C)
    # GRU forward: (L, B*N, C) -> hidden states
    effective_batch = batch_size * num_nodes
    return _manual_gru_flops(seq_len, effective_batch, input_dim, hidden_size, num_layers)


def _manual_gru_decoder_flops(
    batch_size: int,
    horizon: int,
    num_nodes: int,
    input_dim: int,
    hidden_size: int,
    num_layers: int,
    output_dim: int,
) -> int:
    """
    Manual FLOPs for GRU decoder (Seq2SeqDecoder).
    - GRU: autoregressive, horizon steps
    - Linear projection: horizon * (2*hidden_size -> output_dim)
    """
    effective_batch = batch_size * num_nodes
    
    # GRU decoder: horizon steps, input is 2*hidden_size (h_encode + graph_encoding)
    gru_input_size = 2 * hidden_size
    gru_flops = _manual_gru_flops(horizon, effective_batch, gru_input_size, 2 * hidden_size, num_layers)
    
    # Linear projection: (2*hidden_size -> output_dim) for each timestep
    proj_flops = horizon * _manual_linear_flops(2 * hidden_size, output_dim, effective_batch)
    
    return gru_flops + proj_flops


def _manual_cheb_graph_conv_flops(
    batch_size: int,
    time_steps: int,
    num_nodes: int,
    c_in: int,
    c_out: int,
    Ks: int,
) -> int:
    """
    Manual FLOPs for Chebyshev Graph Convolution.
    - Chebyshev polynomial computation: K iterations of (2*A*x - x_prev)
    - Final einsum: weight multiplication
    """
    # Chebyshev polynomial iterations
    # Each iteration: A @ x (N x N @ N x D = N^2 * D * 2 MACs per batch*time)
    cheb_flops = 0
    for k in range(Ks):
        if k == 0:
            continue  # x_0 = x, no computation
        elif k == 1:
            # x_1 = A @ x
            cheb_flops += 2 * batch_size * time_steps * num_nodes * num_nodes * c_in
        else:
            # x_k = 2 * A @ x_{k-1} - x_{k-2}
            cheb_flops += 2 * batch_size * time_steps * num_nodes * num_nodes * c_in  # A @ x
            cheb_flops += batch_size * time_steps * num_nodes * c_in  # 2 * ...
            cheb_flops += batch_size * time_steps * num_nodes * c_in  # - x_{k-2}

    # Final einsum: (B, T, K, N, c_in) @ (K, c_in, c_out) -> (B, T, N, c_out)
    # This is K * c_in * c_out MACs per (B, T, N) element
    weight_flops = 2 * batch_size * time_steps * num_nodes * Ks * c_in * c_out

    return cheb_flops + weight_flops


def _manual_stgcn_server_flops(
    batch_size: int,
    time_steps: int,
    num_nodes: int,
    feature_dim: int,
    Ks: int,
) -> int:
    """Manual FLOPs for STGCN server spatial (single GraphConvLayer)."""
    # Align layer (if needed, assume c_in == c_out, skip)
    # ChebGraphConv
    gcn_flops = _manual_cheb_graph_conv_flops(
        batch_size, time_steps, num_nodes, feature_dim, feature_dim, Ks
    )
    # Residual add
    gcn_flops += batch_size * time_steps * num_nodes * feature_dim
    # ReLU (negligible but count)
    gcn_flops += batch_size * time_steps * num_nodes * feature_dim

    return gcn_flops


def _manual_pooler_flops(
    batch_size: int,
    time_steps: int,
    num_nodes: int,
    num_tokens: int,
    model_dim: int,
    pooling_type: str,
    num_heads: int = 4,
    ff_dim: int = 256,
) -> int:
    """Manual FLOPs for pooling module."""
    if pooling_type == "simple":
        # A @ x: (N, N) @ (B, T, N, D) -> N^2 * D * B * T * 2
        adj_flops = 2 * batch_size * time_steps * num_nodes * num_nodes * model_dim
        # Linear(N -> K): (B, T, D, N) @ (N, K) -> B * T * D * N * K * 2
        linear_flops = _manual_linear_flops(num_nodes, num_tokens, batch_size * time_steps * model_dim)
        return adj_flops + linear_flops
    elif pooling_type == "attention":
        head_dim = model_dim // num_heads
        # Cross-attention: query=tokens(K), key/value=nodes(N)
        # Q projection: K tokens
        q_flops = _manual_linear_flops(model_dim, model_dim, batch_size * time_steps * num_tokens)
        # K, V projections: N nodes
        kv_flops = 2 * _manual_linear_flops(model_dim, model_dim, batch_size * time_steps * num_nodes)
        # QK^T: (B*T*H, K, d) @ (B*T*H, d, N) -> K * N * d * 2
        qk_flops = 2 * batch_size * time_steps * num_heads * num_tokens * num_nodes * head_dim
        # softmax
        softmax_flops = 5 * batch_size * time_steps * num_heads * num_tokens * num_nodes
        # attn @ V: (B*T*H, K, N) @ (B*T*H, N, d) -> K * N * d * 2
        av_flops = 2 * batch_size * time_steps * num_heads * num_tokens * num_nodes * head_dim
        # output projection
        out_flops = _manual_linear_flops(model_dim, model_dim, batch_size * time_steps * num_tokens)
        # FFN
        ffn_flops = _manual_ffn_flops(batch_size * time_steps, num_tokens, model_dim, ff_dim)

        return q_flops + kv_flops + qk_flops + softmax_flops + av_flops + out_flops + ffn_flops
    else:
        return 0


def _manual_expander_flops(
    batch_size: int,
    time_steps: int,
    num_nodes: int,
    num_tokens: int,
    model_dim: int,
    expansion_type: str,
    num_heads: int = 4,
    ff_dim: int = 256,
) -> int:
    """Manual FLOPs for expansion module."""
    if expansion_type == "simple":
        # token_fusion if K > 1: Linear(K -> 1)
        fusion_flops = 0
        if num_tokens > 1:
            fusion_flops = _manual_linear_flops(num_tokens, 1, batch_size * time_steps * model_dim)
        # concat + fusion_layer: Linear(2D -> D)
        cat_flops = _manual_linear_flops(2 * model_dim, model_dim, batch_size * time_steps * num_nodes)
        return fusion_flops + cat_flops
    elif expansion_type == "attention":
        head_dim = model_dim // num_heads
        # Cross-attention: query=nodes(N), key/value=tokens(K)
        q_flops = _manual_linear_flops(model_dim, model_dim, batch_size * time_steps * num_nodes)
        kv_flops = 2 * _manual_linear_flops(model_dim, model_dim, batch_size * time_steps * num_tokens)
        qk_flops = 2 * batch_size * time_steps * num_heads * num_nodes * num_tokens * head_dim
        softmax_flops = 5 * batch_size * time_steps * num_heads * num_nodes * num_tokens
        av_flops = 2 * batch_size * time_steps * num_heads * num_nodes * num_tokens * head_dim
        out_flops = _manual_linear_flops(model_dim, model_dim, batch_size * time_steps * num_nodes)
        ffn_flops = _manual_ffn_flops(batch_size * time_steps, num_nodes, model_dim, ff_dim)

        return q_flops + kv_flops + qk_flops + softmax_flops + av_flops + out_flops + ffn_flops
    else:
        return 0


def _gru_flop_jit(inputs: list, outputs: list) -> int:
    """
    fvcore JIT handle for aten::gru.
    inputs[0]: (seq_len, batch, input_size)
    inputs[1]: (num_layers, batch, hidden_size) = h_0
    outputs[0]: (seq_len, batch, hidden_size)
    """
    try:
        from fvcore.nn.jit_handles import get_shape
    except Exception:
        return 0

    inp = get_shape(inputs[0]) if inputs else None
    h0 = get_shape(inputs[1]) if len(inputs) > 1 else None
    out = get_shape(outputs[0]) if outputs else None

    if not inp or not out:
        return 0
    seq_len, batch, input_size = inp[0], inp[1], inp[2]
    hidden_size = out[2]
    num_layers = h0[0] if (h0 and len(h0) >= 1) else 1

    return _manual_gru_flops(seq_len, batch, input_size, hidden_size, num_layers)


def measure_flops(model: nn.Module, sample_input: Tuple[torch.Tensor, ...]) -> int:
    """
    Measure FLOPs using fvcore (PyTorch/Meta official library).
    Registers aten::gru handler so GRU/RNN FLOPs are counted.

    Args:
        model: PyTorch model (must accept tuple input)
        sample_input: Tuple of input tensors

    Returns:
        Total MACs (int), or -1 if measurement failed
    """
    import warnings

    try:
        from fvcore.nn import FlopCountAnalysis

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flop_counter = FlopCountAnalysis(model, sample_input)
            flop_counter.unsupported_ops_warnings(False)
            flop_counter.set_op_handle("aten::gru", _gru_flop_jit)
            return int(flop_counter.total())
    except Exception as e:
        print(f"[WARNING] fvcore FLOPs measurement failed: {e}")
        return -1


class FLOPsProfiler:
    """
    FLOPs Profiler for various learning methods.
    
    Automatically detects model type and measures FLOPs accordingly:
    - Central/Independent/Federated: single model or ensemble of clients
    - Split/HierSplit: client + server separation
    """
    
    def __init__(
        self,
        model: nn.Module,
        sample_history: torch.Tensor,
        sample_future: Optional[torch.Tensor] = None,
        method: str = "central",
        cfg: Optional[Dict] = None,
    ):
        """
        Args:
            model: The model to profile
            sample_history: Sample history data [B, T, N, C]
            sample_future: Sample future data [B, T, N, C] (optional)
            method: One of 'central', 'independent', 'split', 'hiersplit', 'federated'
            cfg: Config dictionary for additional info
        """
        self.model = model
        self.sample_history = sample_history
        self.sample_future = sample_future if sample_future is not None else torch.zeros_like(sample_history)
        self.method = method.lower()
        self.cfg = cfg or {}
        
        self.device = sample_history.device
        self.dtype = sample_history.dtype
        
    def profile(self) -> Dict[str, Any]:
        """
        Run FLOPs profiling based on method type.
        
        Returns:
            Dict with profiling results
        """
        if self.method in ("central",):
            return self._profile_central()
        elif self.method in ("independent", "federated", "independent_clientwise"):
            return self._profile_independent()
        elif self.method == "split":
            return self._profile_split()
        elif self.method == "hiersplit":
            return self._profile_hiersplit()
        else:
            print(f"[WARNING] Unknown method '{self.method}', using central profiling")
            return self._profile_central()
    
    def _profile_central(self) -> Dict[str, Any]:
        """Profile a single central model using fvcore."""
        model = self._unwrap_model(self.model)
        model.eval()

        total_params = count_parameters(model)

        # Create wrapper for fvcore (needs tuple input)
        wrapper = self._create_model_wrapper(model)
        total_macs = measure_flops(wrapper, (self.sample_history, self.sample_future))

        return {
            "method": self.method,
            "total_macs": total_macs,
            "total_flops": total_macs * 2 if total_macs > 0 else -1,
            "total_params": total_params,
            "client_macs": total_macs,
            "client_flops": total_macs * 2 if total_macs > 0 else -1,
            "server_macs": 0,
            "server_flops": 0,
            "client_params": total_params,
            "server_params": 0,
            **self._get_input_info(),
        }

    def _profile_independent(self) -> Dict[str, Any]:
        """Profile independent learning (ensemble of clients)."""
        model = self._unwrap_model(self.model)
        model.eval()
        
        # Check if it's an ensemble model
        if hasattr(model, "client_models"):
            clients = model.client_models
            total_client_flops = 0
            total_client_params = 0
            
            for cid, client in enumerate(clients):
                nodes = self._get_client_nodes(model, cid)
                x_i = self.sample_history.index_select(2, nodes)
                f_i = self.sample_future.index_select(2, nodes)
                
                wrapper = self._create_model_wrapper(client)
                flops = measure_flops(wrapper, (x_i, f_i))
                if flops > 0:
                    total_client_flops += flops
                total_client_params += count_parameters(client)
            
            return {
                "method": self.method,
                "total_macs": total_client_flops,
                "total_flops": total_client_flops * 2,  # MACs * 2 = FLOPs
                "total_params": total_client_params,
                "client_macs": total_client_flops,
                "client_flops": total_client_flops * 2,
                "server_macs": 0,
                "server_flops": 0,
                "client_params": total_client_params,
                "server_params": 0,
                "num_clients": len(clients),
                **self._get_input_info(),
            }
        else:
            # Fallback to central profiling
            return self._profile_central()
    
    def _profile_split(self) -> Dict[str, Any]:
        """Profile split learning (client encoder/decoder + server spatial)."""
        model = self._unwrap_model(self.model)
        model.eval()
        
        if not hasattr(model, "clients") or not hasattr(model, "server"):
            print("[WARNING] Model doesn't have clients/server, using central profiling")
            return self._profile_central()
        
        clients = model.clients
        server = model.server
        
        # Detect model kind
        model_kind = getattr(model, "kind", "unknown")
        is_gru = model_kind == "gruseq2seq_graphnet" or "GRUSeq2Seq" in type(clients[0]).__name__
        
        # ─────────────────────────────────────────────────────────────
        # Client FLOPs: encoder + decoder
        # ─────────────────────────────────────────────────────────────
        total_client_flops = 0
        total_client_params = 0
        
        for cid, client in enumerate(clients):
            nodes = self._get_client_nodes(model, cid)
            x_i = self.sample_history.index_select(2, nodes)
            f_i = self.sample_future.index_select(2, nodes) if self.sample_future is not None else None
            
            # GRU: encoder via fvcore (aten::gru handler); decoder via manual (complex signature)
            if is_gru:
                enc_flops = self._measure_encoder(client, x_i)
                if enc_flops <= 0:
                    B_i, T_i, N_i, C_i = x_i.shape
                    enc_flops = _manual_gru_encoder_flops(
                        batch_size=B_i,
                        seq_len=T_i,
                        num_nodes=N_i,
                        input_dim=C_i,
                        hidden_size=getattr(client, 'hidden_size', 128),
                        num_layers=getattr(client, 'gru_num_layers', 2),
                    )
                    print(f"[DEBUG] GRU client {cid} encoder fallback manual FLOPs: {enc_flops:,}")

                B_i, T_i, N_i, C_i = x_i.shape
                dec_flops = _manual_gru_decoder_flops(
                    batch_size=B_i,
                    horizon=getattr(client, 'horizon', T_i),
                    num_nodes=N_i,
                    input_dim=getattr(client, 'input_dim', C_i),
                    hidden_size=getattr(client, 'hidden_size', 128),
                    num_layers=getattr(client, 'gru_num_layers', 2),
                    output_dim=getattr(client, 'output_dim', 1),
                )
                client_flops = enc_flops + dec_flops
            else:
                # Encoder
                enc_flops = self._measure_encoder(client, x_i)
                
                # Decoder (need encoder output shape)
                with torch.no_grad():
                    enc_out_raw = client.forward_encoder(x_i)
                    # Handle tuple return (GRUSeq2SeqWithGraphNet returns (h_encode, last_input))
                    if isinstance(enc_out_raw, tuple):
                        enc_out = enc_out_raw[0]  # h_encode
                    else:
                        enc_out = enc_out_raw
                dec_flops = self._measure_decoder(client, enc_out)
                
                client_flops = (enc_flops if enc_flops > 0 else 0) + (dec_flops if dec_flops > 0 else 0)
            
            total_client_flops += client_flops
            total_client_params += count_parameters(client)
        
        # ─────────────────────────────────────────────────────────────
        # Server FLOPs: spatial
        # ─────────────────────────────────────────────────────────────
        server_params = count_parameters(server)
        
        # Create dummy global encoding
        with torch.no_grad():
            test_client = clients[0]
            test_nodes = self._get_client_nodes(model, 0)
            test_x = self.sample_history.index_select(2, test_nodes)
            test_enc_raw = test_client.forward_encoder(test_x)
            # Handle tuple return
            if isinstance(test_enc_raw, tuple):
                test_enc = test_enc_raw[0]
            else:
                test_enc = test_enc_raw
        
        B = self.sample_history.shape[0]
        total_nodes = model.total_nodes
        
        # GRUSeq2SeqWithGraphNet encoder returns (num_layers, B*N, hidden_dim)
        # Other models return (B, T, N, D)
        if is_gru and len(test_enc.shape) == 3:
            # (num_layers, B*N, hidden_dim)
            num_layers = test_enc.shape[0]
            D = test_enc.shape[-1]
            # Create dummy for server spatial: (num_layers, B*total_nodes, D)
            dummy_global = torch.zeros((num_layers, B * total_nodes, D), device=self.device, dtype=self.dtype)
            server_flops = self._measure_gru_spatial(server, dummy_global, B, total_nodes)
        else:
            T_feat = test_enc.shape[1]
            D = test_enc.shape[-1]
            dummy_global = torch.zeros((B, T_feat, total_nodes, D), device=self.device, dtype=self.dtype)
            server_flops = self._measure_spatial(server, dummy_global)

        # Manual calculation fallback if server_flops is 0 or failed
        if server_flops <= 0 and server_params > 0:
            print(f"[DEBUG] Server FLOPs measurement failed, using manual calculation")
            if model_kind == "stgcn":
                # STGCN server is _STGCNServerSpatial with GraphConvLayer
                # GraphConvLayer contains cheb_graph_conv or graph_conv
                Ks_val = 3  # default
                gcn_layer = getattr(server, '_gcn', None)
                if gcn_layer is not None:
                    # Check if it's ChebGraphConv (has Ks) or GraphConv (Ks=1)
                    if hasattr(gcn_layer, 'cheb_graph_conv'):
                        cheb_conv = gcn_layer.cheb_graph_conv
                        if hasattr(cheb_conv, 'Ks'):
                            Ks_val = cheb_conv.Ks
                    elif hasattr(gcn_layer, 'graph_conv'):
                        # GraphConv is equivalent to Ks=1
                        Ks_val = 1
                    elif hasattr(gcn_layer, 'Ks'):
                        # Direct Ks attribute
                        Ks_val = gcn_layer.Ks

                server_flops = _manual_stgcn_server_flops(
                    batch_size=B,
                    time_steps=T_feat,
                    num_nodes=total_nodes,
                    feature_dim=D,
                    Ks=Ks_val,
                )
                print(f"[DEBUG] STGCN server manual FLOPs: {server_flops:,} (Ks={Ks_val})")
            elif model_kind in ("staeformer", "stgformer"):
                # For STAEformer/STGformer split, server is full spatial model
                # Estimate based on spatial attention
                num_heads = getattr(server, 'num_heads', 4)
                num_layers_s = len(getattr(server, 'attn_layers_s', [])) if hasattr(server, 'attn_layers_s') else 3
                ff_dim = 256

                server_flops = 0
                for _ in range(num_layers_s):
                    server_flops += _manual_attention_flops(B * T_feat, num_heads, total_nodes, D // num_heads)
                    server_flops += _manual_ffn_flops(B * T_feat, total_nodes, D, ff_dim)
                print(f"[DEBUG] STAEformer/STGformer server manual FLOPs: {server_flops:,}")

        total_macs = total_client_flops + server_flops
        total_params = total_client_params + server_params

        return {
            "method": self.method,
            "total_macs": total_macs,
            "total_flops": total_macs * 2,  # MACs * 2 = FLOPs
            "total_params": total_params,
            "client_macs": total_client_flops,
            "client_flops": total_client_flops * 2,
            "server_macs": server_flops,
            "server_flops": server_flops * 2,
            "client_params": total_client_params,
            "server_params": server_params,
            "num_clients": len(clients),
            **self._get_input_info(),
        }
    
    def _profile_hiersplit(self) -> Dict[str, Any]:
        """Profile hierarchical split learning (client + pooler/expander + server)."""
        model = self._unwrap_model(self.model)
        model.eval()
        
        # Debug model structure
        print(f"[DEBUG] Model type: {type(model)}")
        print(f"[DEBUG] Model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
        
        # HierSplitModel uses client_models and server_model (not clients and server)
        if hasattr(model, "client_models"):
            clients = model.client_models
            print(f"[DEBUG] Using model.client_models")
        elif hasattr(model, "clients"):
            clients = model.clients
            print(f"[DEBUG] Using model.clients")
        else:
            print("[WARNING] Model doesn't have client_models/clients, using central profiling")
            return self._profile_central()
        
        if hasattr(model, "server_model"):
            server = model.server_model
            print(f"[DEBUG] Using model.server_model, type={type(server)}")
        elif hasattr(model, "server"):
            server = model.server
            print(f"[DEBUG] Using model.server, type={type(server)}")
        else:
            print("[WARNING] Model doesn't have server_model/server, using central profiling")
            return self._profile_central()
        
        # ─────────────────────────────────────────────────────────────
        # Client FLOPs: encoder + spatial + decoder (or whole client)
        # ─────────────────────────────────────────────────────────────
        total_client_flops = 0
        total_client_params = 0
        
        # Check if clients have sub-methods (baselines/HierSplit) or are full models (jy_hiersplit)
        first_client = clients[0]
        has_sub_methods = hasattr(first_client, 'forward_encoder') and hasattr(first_client, 'forward_spatial')
        
        # Detect model kind
        model_kind = getattr(model, "kind", "unknown")
        is_gru = model_kind == "gruseq2seq_graphnet" or "GRUSeq2Seq" in type(first_client).__name__
        
        for cid, client in enumerate(clients):
            nodes = self._get_client_nodes(model, cid)
            x_i = self.sample_history.index_select(2, nodes)
            f_i = self.sample_future.index_select(2, nodes) if self.sample_future is not None else None
            B = x_i.shape[0]
            N_i = len(nodes)
            
            # GRU models have special forward_decoder signature, use full client measurement
            if is_gru:
                client_flops = self._measure_full_client(client, x_i, f_i)
                if client_flops <= 0:
                    print(f"[WARNING] HierSplit GRU client {cid} FLOPs measurement failed, using fallback")
                    # Fallback: measure encoder only and estimate
                    enc_flops = self._measure_encoder(client, x_i)
                    client_flops = enc_flops * 3 if enc_flops > 0 else 0  # rough estimate: enc + spatial + dec
            elif has_sub_methods:
                # baselines/HierSplit model structure (non-GRU)
                enc_flops = self._measure_encoder(client, x_i)
                
                with torch.no_grad():
                    enc_out_raw = client.forward_encoder(x_i)
                    if isinstance(enc_out_raw, tuple):
                        enc_out = enc_out_raw[0]
                    else:
                        enc_out = enc_out_raw
                
                spatial_flops = self._measure_spatial(client, enc_out)
                
                with torch.no_grad():
                    try:
                        spatial_out = client.forward_spatial(enc_out)
                    except Exception as e:
                        print(f"[DEBUG] forward_spatial failed: {e}")
                        spatial_out = enc_out
                dec_flops = self._measure_decoder(client, spatial_out)
                
                client_flops = sum(f for f in [enc_flops, spatial_flops, dec_flops] if f > 0)
            else:
                # jy_hiersplit model structure - measure whole client
                client_flops = self._measure_full_client(client, x_i, f_i)
            
            total_client_flops += client_flops
            total_client_params += count_parameters(client)
        
        # ─────────────────────────────────────────────────────────────
        # Pooler/Expander FLOPs
        # ─────────────────────────────────────────────────────────────
        pooler_flops = 0
        expander_flops = 0
        
        # Get pooling/expansion types for manual calculation
        pooling_type = getattr(model, "pooling_type", "attention")
        expansion_type = getattr(model, "expansion_type", "attention")
        num_tokens = getattr(model, "num_tokens", 1)
        model_dim = getattr(model, "model_dim", 64)

        if hasattr(model, "poolers") and len(model.poolers) > 0:
            for cid, pooler in enumerate(model.poolers):
                nodes = self._get_client_nodes(model, cid)
                x_i = self.sample_history.index_select(2, nodes)
                B = x_i.shape[0]
                N_i = len(nodes)

                with torch.no_grad():
                    enc_out_raw = clients[cid].forward_encoder(x_i)
                    if isinstance(enc_out_raw, tuple):
                        enc_out = enc_out_raw[0]
                    else:
                        enc_out = enc_out_raw
                    try:
                        if is_gru:
                            has_edge_index = getattr(clients[cid], 'edge_index', None) is not None
                            if has_edge_index:
                                spatial_out = clients[cid].forward_spatial(enc_out, batch_size=B, num_nodes=N_i)
                            else:
                                spatial_out = enc_out  # Skip spatial
                        else:
                            spatial_out = clients[cid].forward_spatial(enc_out)
                    except Exception as e:
                        print(f"[DEBUG] pooler forward_spatial failed for client {cid}: {type(e).__name__}: {e}")
                        spatial_out = enc_out

                p_flops = self._measure_module(pooler, (spatial_out,))

                # Manual calculation fallback for pooler
                if p_flops <= 0:
                    if len(spatial_out.shape) == 4:
                        B_s, T_s, N_s, D_s = spatial_out.shape
                        p_flops = _manual_pooler_flops(
                            batch_size=B_s, time_steps=T_s, num_nodes=N_s,
                            num_tokens=num_tokens, model_dim=D_s,
                            pooling_type=pooling_type,
                        )
                        print(f"[DEBUG] Pooler {cid} manual FLOPs: {p_flops:,}")

                if p_flops > 0:
                    pooler_flops += p_flops
                total_client_params += count_parameters(pooler)
        
        if hasattr(model, "expanders") and len(model.expanders) > 0:
            for cid, expander in enumerate(model.expanders):
                nodes = self._get_client_nodes(model, cid)
                x_i = self.sample_history.index_select(2, nodes)
                B = x_i.shape[0]
                N_i = len(nodes)

                with torch.no_grad():
                    enc_out_raw = clients[cid].forward_encoder(x_i)
                    if isinstance(enc_out_raw, tuple):
                        enc_out = enc_out_raw[0]
                    else:
                        enc_out = enc_out_raw
                    try:
                        if is_gru:
                            has_edge_index = getattr(clients[cid], 'edge_index', None) is not None
                            if has_edge_index:
                                spatial_out = clients[cid].forward_spatial(enc_out, batch_size=B, num_nodes=N_i)
                            else:
                                spatial_out = enc_out  # Skip spatial
                        else:
                            spatial_out = clients[cid].forward_spatial(enc_out)
                    except Exception as e:
                        print(f"[DEBUG] expander forward_spatial failed for client {cid}: {type(e).__name__}: {e}")
                        spatial_out = enc_out

                # Handle different spatial_out shapes
                if len(spatial_out.shape) == 3:
                    # GRU: (L, B*N, H) -> create dummy tokens
                    L, _, D = spatial_out.shape
                    dummy_tokens = torch.zeros((L, B * num_tokens, D), device=self.device, dtype=self.dtype)
                else:
                    B_s, T_s, N_i_s, D = spatial_out.shape
                    dummy_tokens = torch.zeros((B_s, T_s, num_tokens, D), device=self.device, dtype=self.dtype)

                e_flops = self._measure_module(expander, (spatial_out, dummy_tokens))

                # Manual calculation fallback for expander
                if e_flops <= 0:
                    if len(spatial_out.shape) == 4:
                        B_s, T_s, N_i_s, D_s = spatial_out.shape
                        e_flops = _manual_expander_flops(
                            batch_size=B_s, time_steps=T_s, num_nodes=N_i_s,
                            num_tokens=num_tokens, model_dim=D_s,
                            expansion_type=expansion_type,
                        )
                        print(f"[DEBUG] Expander {cid} manual FLOPs: {e_flops:,}")

                if e_flops > 0:
                    expander_flops += e_flops
                total_client_params += count_parameters(expander)
        
        total_client_flops += pooler_flops + expander_flops
        
        # ─────────────────────────────────────────────────────────────
        # Server FLOPs: token mixer
        # ─────────────────────────────────────────────────────────────
        print(f"[DEBUG] Server type: {type(server)}")
        print(f"[DEBUG] Server is Identity: {isinstance(server, nn.Identity)}")
        server_params = count_parameters(server)
        print(f"[DEBUG] Server params: {server_params}")
        
        num_clients = len(clients)
        # Get num_tokens from model or first client
        num_tokens = getattr(model, "num_tokens", None)
        if num_tokens is None and hasattr(clients[0], "num_tokens"):
            num_tokens = clients[0].num_tokens
        if num_tokens is None:
            num_tokens = 1
            
        # Get model_dim from model, server, or first client
        model_dim = getattr(model, "model_dim", None)
        if model_dim is None and hasattr(server, "attention"):
            model_dim = server.attention.embed_dim
        if model_dim is None and hasattr(clients[0], "model_dim"):
            model_dim = clients[0].model_dim
        if model_dim is None:
            model_dim = 64
        
        # Get T from encoder output
        with torch.no_grad():
            test_nodes = self._get_client_nodes(model, 0)
            test_x = self.sample_history.index_select(2, test_nodes)
            test_enc_raw = clients[0].forward_encoder(test_x)
            if isinstance(test_enc_raw, tuple):
                test_enc = test_enc_raw[0]
            else:
                test_enc = test_enc_raw
        
        B = self.sample_history.shape[0]
        
        # Handle different encoder output shapes
        if is_gru and len(test_enc.shape) == 3:
            # GRU: (num_layers, B*N, hidden_dim)
            num_layers = test_enc.shape[0]
            D = test_enc.shape[-1]
            model_dim = D
            # Create dummy for server spatial
            dummy_tokens_all = torch.randn(
                (num_layers, B * num_clients * num_tokens, model_dim),
                device=self.device, dtype=self.dtype
            )
            # Measure server FLOPs with GRU-style call
            server_flops = self._measure_gru_spatial(server, dummy_tokens_all, B, num_clients * num_tokens)
        else:
            T_feat = test_enc.shape[1]
            # Create proper input for server (not zeros - use randn for proper FLOPs)
            dummy_tokens_all = torch.randn(
                (B, T_feat, num_clients * num_tokens, model_dim),
                device=self.device, dtype=self.dtype
            )
            # Measure server FLOPs
            server_flops = self._measure_module(server, (dummy_tokens_all,))
        
        total_macs = total_client_flops + server_flops
        total_params = total_client_params + server_params
        
        return {
            "method": self.method,
            "total_macs": total_macs,
            "total_flops": total_macs * 2,  # MACs * 2 = FLOPs
            "total_params": total_params,
            "client_macs": total_client_flops,
            "client_flops": total_client_flops * 2,
            "server_macs": server_flops,
            "server_flops": server_flops * 2,
            "client_params": total_client_params,
            "server_params": server_params,
            "pooler_macs": pooler_flops,
            "pooler_flops": pooler_flops * 2,
            "expander_macs": expander_flops,
            "expander_flops": expander_flops * 2,
            "num_clients": num_clients,
            "num_tokens": num_tokens,
            **self._get_input_info(),
        }
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap DDP model if needed."""
        if hasattr(model, "module"):
            return model.module
        return model
    
    def _get_client_nodes(self, model: nn.Module, cid: int) -> torch.Tensor:
        """Get client node indices."""
        if hasattr(model, "get_client_nodes"):
            return model.get_client_nodes(cid)
        elif hasattr(model, f"client_nodes_{cid}"):
            return getattr(model, f"client_nodes_{cid}")
        else:
            # Fallback: assume equal split
            total = self.sample_history.shape[2]
            num_clients = len(model.clients) if hasattr(model, "clients") else 1
            nodes_per = total // num_clients
            start = cid * nodes_per
            end = start + nodes_per if cid < num_clients - 1 else total
            return torch.arange(start, end, device=self.device)
    
    def _create_model_wrapper(self, model: nn.Module) -> nn.Module:
        """Create a wrapper for FlopCountAnalysis that returns only Tensor."""
        class ModelWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            
            def forward(self, history, future):
                out = self.m(history_data=history, future_data=future, 
                            batch_seen=0, epoch=1, train=False)
                # FlopCountAnalysis needs Tensor output, not dict
                if isinstance(out, dict):
                    pred = out.get("prediction", None)
                    if pred is not None:
                        return pred
                    # Return first tensor value found
                    for v in out.values():
                        if isinstance(v, torch.Tensor):
                            return v
                    # Fallback: return zeros
                    return torch.zeros(1, device=history.device)
                return out
        
        wrapper = ModelWrapper(model)
        wrapper.eval()
        return wrapper
    
    def _measure_encoder(self, client: nn.Module, x: torch.Tensor) -> int:
        """Measure encoder FLOPs."""
        if not hasattr(client, "forward_encoder"):
            return 0
        
        class EncoderWrapper(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c
            def forward(self, x):
                return self.c.forward_encoder(x)
        
        wrapper = EncoderWrapper(client)
        wrapper.eval()
        return measure_flops(wrapper, (x,))
    
    def _measure_spatial(self, module: nn.Module, x: torch.Tensor) -> int:
        """Measure spatial FLOPs."""
        if hasattr(module, "forward_spatial"):
            class SpatialWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m.forward_spatial(x)
            wrapper = SpatialWrapper(module)
        else:
            class DirectWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x)
            wrapper = DirectWrapper(module)
        
        wrapper.eval()
        flops = measure_flops(wrapper, (x,))
        return flops if flops > 0 else 0
    
    def _measure_gru_spatial(self, module: nn.Module, h_encode: torch.Tensor, batch_size: int, num_nodes: int) -> int:
        """Measure GRUSeq2SeqWithGraphNet spatial FLOPs (needs batch_size, num_nodes args)."""
        if hasattr(module, "forward_spatial"):
            class GRUSpatialWrapper(nn.Module):
                def __init__(self, m, bs, nn):
                    super().__init__()
                    self.m = m
                    self.bs = bs
                    self.nn = nn
                def forward(self, x):
                    return self.m.forward_spatial(x, batch_size=self.bs, num_nodes=self.nn)
            wrapper = GRUSpatialWrapper(module, batch_size, num_nodes)
        elif hasattr(module, "spatial"):
            # Server has a spatial attribute (GraphNetSpatialEncoder)
            class ServerSpatialWrapper(nn.Module):
                def __init__(self, m, bs, nn):
                    super().__init__()
                    self.m = m
                    self.bs = bs
                    self.nn = nn
                def forward(self, x):
                    return self.m.spatial(x, batch_size=self.bs, num_nodes=self.nn, edge_index=self.m.edge_index, edge_attr=self.m.edge_attr)
            wrapper = ServerSpatialWrapper(module, batch_size, num_nodes)
        else:
            # Fallback to direct call
            class DirectWrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, x):
                    return self.m(x)
            wrapper = DirectWrapper(module)
        
        wrapper.eval()
        flops = measure_flops(wrapper, (h_encode,))
        return flops if flops > 0 else 0
    
    def _measure_decoder(self, client: nn.Module, x: torch.Tensor) -> int:
        """Measure decoder FLOPs."""
        if not hasattr(client, "forward_decoder"):
            return 0
        
        class DecoderWrapper(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.c = c
            def forward(self, x):
                return self.c.forward_decoder(x)
        
        wrapper = DecoderWrapper(client)
        wrapper.eval()
        return measure_flops(wrapper, (x,))
    
    def _measure_full_client(self, client: nn.Module, history: torch.Tensor, future: torch.Tensor = None) -> int:
        """Measure full client model FLOPs using fvcore."""
        class ClientWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, history_data, future_data):
                out = self.m(history_data=history_data, future_data=future_data,
                            batch_seen=0, epoch=1, train=False)
                if isinstance(out, dict):
                    return out.get('prediction', list(out.values())[0])
                return out

        wrapper = ClientWrapper(client)
        wrapper.eval()

        if future is None:
            future = torch.zeros_like(history)

        return measure_flops(wrapper, (history, future))
    
    def _measure_module(self, module: nn.Module, inputs: tuple) -> int:
        """Measure generic module FLOPs using fvcore."""
        class ModuleWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, *args):
                out = self.m(*args)
                if isinstance(out, dict):
                    return out.get('prediction', list(out.values())[0])
                return out

        wrapper = ModuleWrapper(module)
        wrapper.eval()
        return measure_flops(wrapper, inputs)
    
    def _get_input_info(self) -> Dict[str, Any]:
        """Get input tensor info."""
        shape = self.sample_history.shape
        return {
            "batch_size": shape[0],
            "input_len": shape[1],
            "num_nodes": shape[2],
            "num_features": shape[3] if len(shape) > 3 else 1,
        }


def save_flops_csv(
    results: Dict[str, Any],
    output_dir: str = "flops_results",
    model_name: str = "unknown",
    dataset: str = "unknown",
) -> str:
    """
    Save FLOPs profiling results to CSV.
    
    Args:
        results: Dict from FLOPsProfiler.profile()
        output_dir: Output directory
        model_name: Model name for filename
        dataset: Dataset name for filename
        
    Returns:
        Path to saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Add metadata
    results["model_name"] = model_name
    results["dataset"] = dataset
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate filename
    method = results.get("method", "unknown")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"flops_{method}_{model_name}_{dataset}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Define field order
    fieldnames = [
        "method", "model_name", "dataset", "timestamp",
        "total_flops", "total_params",
        "client_flops", "server_flops",
        "client_params", "server_params",
        "batch_size", "input_len", "num_nodes", "num_features",
    ]
    
    # Add extra keys
    for key in results.keys():
        if key not in fieldnames:
            fieldnames.append(key)
    
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results)
    
    return filepath


def _format_number(n: int, unit: str = "") -> str:
    """Format large numbers with K/M/G suffix."""
    if n < 0:
        return "N/A"
    if n >= 1e12:
        return f"{n / 1e12:.2f}T{unit}"
    elif n >= 1e9:
        return f"{n / 1e9:.2f}G{unit}"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M{unit}"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K{unit}"
    else:
        return f"{n:,}{unit}"


def print_flops_summary(results: Dict[str, Any]) -> None:
    """Print FLOPs profiling summary."""
    print("\n" + "=" * 60)
    print("FLOPs Profiling Results")
    print("=" * 60)
    
    print(f"Method:        {results.get('method', 'N/A')}")
    print(f"Model:         {results.get('model_name', 'N/A')}")
    print(f"Dataset:       {results.get('dataset', 'N/A')}")
    print("-" * 60)
    
    total_macs = results.get("total_macs", results.get("total_flops", 0))
    total_params = results.get("total_params", 0)
    
    print(f"Total MACs:    {_format_number(total_macs)}")
    print(f"Total FLOPs:   {_format_number(total_macs * 2 if total_macs > 0 else -1)} (MACs * 2)")
    print(f"Total Params:  {_format_number(total_params)}")
    print("-" * 60)
    
    client_macs = results.get("client_macs", results.get("client_flops", 0))
    server_macs = results.get("server_macs", results.get("server_flops", 0))
    client_params = results.get("client_params", 0)
    server_params = results.get("server_params", 0)
    
    print(f"Client MACs:   {_format_number(client_macs)}")
    print(f"Server MACs:   {_format_number(server_macs)}")
    print(f"Client Params: {_format_number(client_params)}")
    print(f"Server Params: {_format_number(server_params)}")
    
    if "num_clients" in results:
        print(f"Num Clients:   {results['num_clients']}")
    if "num_tokens" in results:
        print(f"Num Tokens:    {results['num_tokens']}")
    if "pooler_macs" in results:
        print(f"Pooler MACs:   {_format_number(results['pooler_macs'])}")
    if "expander_macs" in results:
        print(f"Expander MACs: {_format_number(results['expander_macs'])}")
    
    print("-" * 60)
    print(f"Input Shape:   [{results.get('batch_size', '?')}, {results.get('input_len', '?')}, "
          f"{results.get('num_nodes', '?')}, {results.get('num_features', '?')}]")
    print("=" * 60 + "\n")
