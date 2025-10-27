import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

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
        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()
            attn_score.masked_fill_(~mask, float('-inf'))

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out


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

        out = out.transpose(dim, -2)
        return out


class SpatialGATLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0):
        super().__init__()
        self.model_dim = model_dim
        self.gat = GATv2Conv(
            in_channels=model_dim,
            out_channels=model_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, n_node_edge_index): # 인자 이름을 n_node_edge_index로 변경하여 명확화
        B, T, N, D = x.shape
        # n_node_edge_index는 N개 노드로 구성된 단일 그래프의 edge_index (예: GATSTAEformer의 self.edge_index)
        # shape: (2, num_edges_for_N_nodes)

        # 1. B*T 개의 그래프 인스턴스를 위한 edge_index 확장
        num_instances = B * T
        
        # 각 인스턴스별 노드 인덱스 오프셋 생성: [0, N, 2N, ..., (num_instances-1)*N]
        # 이 오프셋은 각 복제된 n_node_edge_index의 노드 인덱스에 더해집니다.
        instance_offsets = torch.arange(num_instances, device=x.device) * N
        # instance_offsets shape: (num_instances,) -> (B*T,)

        # n_node_edge_index (2, E_N)를 (1, 2, E_N)로 만들어 브로드캐스팅 준비
        # instance_offsets (B*T)를 (B*T, 1, 1)로 만들어 브로드캐스팅 준비
        # 덧셈 결과의 shape: (B*T, 2, E_N)
        # 각 슬라이스 expanded_edge_index[i, :, :]는 i번째 인스턴스의 조정된 edge_index가 됩니다.
        expanded_edge_index = n_node_edge_index.unsqueeze(0) + instance_offsets.view(-1, 1, 1)
        
        # 최종적으로 (2, E_N * B*T) 형태로 변환
        # (B*T, 2, E_N) -> transpose (2, B*T, E_N) -> reshape (2, B*T * E_N)
        expanded_edge_index = expanded_edge_index.transpose(0, 1).reshape(2, -1)

        # 2. GATv2Conv 연산
        x_flat = x.reshape(B * T * N, D)
        
        # 확장된 edge_index 사용 (필요시 디바이스 일치 확인, 버퍼는 보통 자동 이동됨)
        x_gat = self.gat(x_flat, expanded_edge_index) 
        x_gat = x_gat.reshape(B, T, N, D) # 원래 형태로 복원

        # 3. 이후 연산 (FeedForward, LayerNorm 등)
        out = self.dropout1(x_gat)
        out = self.ln1(x + out) # x는 reshape 이전의 원본 x
        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        return out


class GATSTAEformer(nn.Module):
    def __init__(self,
                 in_steps,
                 out_steps,
                 adj_mx,
                 steps_per_day=288,
                 input_dim=3,
                 output_dim=1,
                 input_embedding_dim=48,
                 tod_embedding_dim=24,
                 dow_embedding_dim=24,
                 feed_forward_dim=256,
                 model_dim=96,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day

        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        self.temporal_attns = nn.ModuleList([
            SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout, mask=True)
            for _ in range(num_layers)
        ])
        self.spatial_attns = nn.ModuleList([
            SpatialGATLayer(model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(in_steps * model_dim, out_steps * output_dim),
            nn.Unflatten(-1, (out_steps, output_dim))
        )

        self.register_buffer("edge_index", dense_to_sparse(adj_mx)[0])

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs):
        x = history_data
        batch_size = x.shape[0]

        tod = x[..., 1] * self.steps_per_day
        dow = x[..., 2] * 7
        x = x[..., :self.input_dim]

        x = self.input_proj(x)
        features = [x]

        tod_emb = self.tod_embedding(tod.long())
        features.append(tod_emb)
        dow_emb = self.dow_embedding(dow.long())
        features.append(dow_emb)

        x = torch.cat(features, dim=-1)

        for t_attn, s_attn in zip(self.temporal_attns, self.spatial_attns):
            x = t_attn(x, dim=1)
            x = s_attn(x, self.edge_index)

        x = x.transpose(1, 2).reshape(batch_size, x.shape[2], -1)
        x = self.output_proj(x)
        return x.transpose(1, 2)
