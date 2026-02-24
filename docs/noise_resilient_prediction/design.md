# Context-Contrastive Representation Learning for Robust Spatiotemporal Traffic Prediction

**Technical Design Document**  
Version 1.0 | January 2025

---

## 1. Executive Summary

본 문서는 교통 예측 태스크에서 representation learning의 **performance variance를 줄이고**, 사고 등 이상 상황에서도 **robust한 예측**을 가능하게 하는 **Context-Contrastive Learning** 방법론을 제안합니다.

**핵심 아이디어**: 각 노드의 representation을 학습할 때, 해당 노드의 'context' (temporal window 내 다른 시점들)를 함께 모델링하여, context 대비 현재 상태가 얼마나 다른지를 representation에 자연스럽게 인코딩합니다.

### Key Benefits

| 항목 | 설명 |
|------|------|
| **Variance 감소** | Context-aware representation으로 학습 안정성 향상 |
| **Anomaly Sensitivity** | Context와 다른 상태는 자연스럽게 다른 representation |
| **Decoder 불필요** | Pure contrastive learning으로 decoder 의존성 제거 |
| **Spatial 정보 Optional** | 위치 정보 없이도 동작, 있으면 추가 활용 가능 |

---

## 2. Problem Statement

### 2.1 현재 문제점

- 기존 representation learning 방법들의 performance variance가 큼
- 사고 등 이상 상황에서 예측 성능이 급격히 저하
- Reconstruction-based 방법은 decoder 품질에 크게 의존
- 일반적인 contrastive learning은 context 정보를 명시적으로 활용하지 않음

### 2.2 목표

Context 대비 노드의 상태를 sensitive하게 표현하는 representation 학습. 구체적으로:

- **평상시**: 노드가 context와 유사하면 → context와 유사한 representation
- **이상시**: 노드가 context와 다르면 → context와 다른 representation

---

## 3. Related Work & Inspiration

### 3.1 DCdetector (KDD 2023)

Dual attention으로 patch-wise view와 in-patch view를 생성하고, 두 view 간의 discrepancy로 anomaly를 탐지. Reconstruction loss 없이 pure contrastive learning만으로 SOTA 달성.

- **핵심 인사이트**: 정상 데이터는 두 view가 일관되고, 이상 데이터는 불일치
- **적용점**: Node view와 Context view의 discrepancy 학습

### 3.2 ST-SSL (AAAI 2022)

Spatiotemporal traffic prediction을 위한 self-supervised learning. Similarity-guided augmentation으로 의미적 일관성 유지.

- **핵심 인사이트**: Random augmentation보다 similarity 기반 augmentation이 효과적
- **적용점**: Temporal similarity를 context 모델링에 활용

### 3.3 CARLA (Pattern Recognition 2024)

Anomaly injection을 통해 명시적인 negative sample 생성. 다양한 anomaly type을 주입하여 robust한 representation 학습.

- **핵심 인사이트**: 명시적 anomaly 정보가 representation 품질 향상
- **적용점**: Context-inconsistent sample을 negative로 활용 가능

---

## 4. Proposed Method

### 4.1 Overall Architecture

**입력 데이터 형태**: `[N, T, V, C]` where N=batch, T=12 timestamps, V=nodes, C=3~5 channels

**출력 representation**: `[N, V, D]` node-wise representation (D=hidden dimension)

```
                    Input [N, T=12, V, C]
                            │
                            ▼
               ┌─────────────────────────────────┐
               │     Temporal Encoder (shared)   │
               └─────────────────────────────────┘
                            │
                            ▼
                    z_full [N, T, V, D]
                      │           │
                      ▼           ▼
              ┌──────────────┐  ┌──────────────┐
              │  Node Branch │  │Context Branch│
              │   (Current)  │  │  (Temporal)  │
              └──────────────┘  └──────────────┘
                      │           │
                      ▼           ▼
              z_node [N,V,D]   z_context [N,V,D]
                      └─────┬─────┘
                            ▼
                 Contrastive Loss (Discrepancy)
```

### 4.2 Component Details

#### 4.2.1 Temporal Encoder

각 노드의 T=12 timestamps에 대한 temporal dependency를 학습합니다. Transformer 기반으로 구현합니다.

```python
class TemporalEncoder(nn.Module):
    def __init__(self, c_in, d_model, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(c_in, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=4, 
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x):
        # x: [N, T, V, C]
        N, T, V, C = x.shape
        
        # Reshape: treat each node independently
        x = x.permute(0, 2, 1, 3)  # [N, V, T, C]
        x = x.reshape(N * V, T, C)  # [N*V, T, C]
        
        # Project and encode
        x = self.input_proj(x)  # [N*V, T, D]
        x = self.transformer(x)  # [N*V, T, D]
        
        # Reshape back
        x = x.reshape(N, V, T, -1)  # [N, V, T, D]
        x = x.permute(0, 2, 1, 3)   # [N, T, V, D]
        
        return x
```

#### 4.2.2 Context Attention Module

각 노드의 'context'를 temporal window 내 다른 시점들의 attention-weighted aggregate로 정의합니다.

```python
class ContextAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Learnable query tokens
        self.node_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.context_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.recency_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, z_full):
        """
        Args:
            z_full: [N, T, V, D] - full temporal representation
        Returns:
            z_node: [N, V, D] - current state representation
            z_context: [N, V, D] - context representation
        """
        N, T, V, D = z_full.shape
        z = z_full.permute(0, 2, 1, 3).reshape(N * V, T, D)
        
        K = self.k_proj(z)
        V_val = self.v_proj(z)
        
        # === Node representation (focus on recent) ===
        Q_node = self.node_query.expand(N * V, -1, -1)
        attn_node = torch.matmul(Q_node, K.transpose(-2, -1)) / (D ** 0.5)
        
        # Recency bias
        recency_bias = torch.linspace(-1, 0, T, device=z.device).view(1, 1, T)
        attn_node = attn_node + self.recency_scale * recency_bias
        attn_node = F.softmax(attn_node, dim=-1)
        z_node = torch.matmul(attn_node, V_val).squeeze(1)
        
        # === Context representation (past only) ===
        Q_ctx = self.context_query.expand(N * V, -1, -1)
        attn_ctx = torch.matmul(Q_ctx, K.transpose(-2, -1)) / (D ** 0.5)
        
        # Mask last timestamp
        mask = torch.zeros(1, 1, T, device=z.device)
        mask[:, :, -1] = float('-inf')
        attn_ctx = attn_ctx + mask
        attn_ctx = F.softmax(attn_ctx, dim=-1)
        z_context = torch.matmul(attn_ctx, V_val).squeeze(1)
        
        return z_node.reshape(N, V, D), z_context.reshape(N, V, D)
```

#### 4.2.3 Spatial Context (Optional)

Graph 정보가 있을 경우, 공간적 context도 함께 모델링할 수 있습니다.

```python
class SpatialContextAttention(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
    def forward(self, z_node, adj_matrix=None):
        """
        Args:
            z_node: [N, V, D] - node representations
            adj_matrix: [V, V] - optional adjacency matrix
        Returns:
            z_spatial_ctx: [N, V, D] - spatial context
        """
        z_spatial_ctx, attn_weights = self.cross_attn(
            query=z_node, key=z_node, value=z_node
        )
        return z_spatial_ctx, attn_weights
```

### 4.3 Pretext Task: Context-Contrastive Learning

#### 4.3.1 핵심 아이디어

DCdetector의 dual-view discrepancy 아이디어를 차용하되, Node view와 Context view로 재정의합니다.

| 상태 | z_node vs z_context | 학습 효과 |
|------|---------------------|-----------|
| **평상시** | 유사 (low discrepancy) | Context와 aligned representation |
| **이상시** | 상이 (high discrepancy) | Context와 다른 representation |

#### 4.3.2 Loss Function

DCdetector 스타일의 KL divergence 기반 contrastive loss + InfoNCE를 사용합니다.

```python
class ContextContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_node, z_context):
        N, V, D = z_node.shape
        
        # Normalize
        z_node = F.normalize(z_node, dim=-1)
        z_context = F.normalize(z_context, dim=-1)
        
        # === KL Divergence (DCdetector style) ===
        p_node = F.softmax(z_node / self.temperature, dim=-1)
        p_context = F.softmax(z_context / self.temperature, dim=-1)
        
        kl_1 = F.kl_div(p_node.log(), p_context, reduction='batchmean')
        kl_2 = F.kl_div(p_context.log(), p_node, reduction='batchmean')
        loss_kl = (kl_1 + kl_2) / 2
        
        # === InfoNCE ===
        z_node_flat = z_node.reshape(N * V, D)
        z_context_flat = z_context.reshape(N * V, D)
        
        sim_matrix = torch.mm(z_node_flat, z_context_flat.t()) / self.temperature
        labels = torch.arange(N * V, device=z_node.device)
        loss_nce = F.cross_entropy(sim_matrix, labels)
        
        return loss_kl + 0.1 * loss_nce
```

### 4.4 Full Model Integration

```python
class ContextContrastiveModel(nn.Module):
    def __init__(self, c_in, d_model, num_nodes, output_len=12, output_dim=2):
        super().__init__()
        
        self.temporal_encoder = TemporalEncoder(c_in, d_model)
        self.context_attention = ContextAttention(d_model)
        self.spatial_context = SpatialContextAttention(d_model)
        self.use_spatial = False  # Ablation flag
        
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_len * output_dim)
        )
        self.contrastive_loss = ContextContrastiveLoss()
        
    def forward(self, x, adj_matrix=None):
        z_full = self.temporal_encoder(x)
        z_node, z_context = self.context_attention(z_full)
        
        if self.use_spatial and adj_matrix is not None:
            z_spatial_ctx, _ = self.spatial_context(z_node, adj_matrix)
            z_context = z_context + z_spatial_ctx
            
        return z_node, z_context
    
    def compute_loss(self, x, y_true, adj_matrix=None, 
                     pred_weight=1.0, contrast_weight=0.5):
        z_node, z_context = self.forward(x, adj_matrix)
        
        y_pred = self.predictor(z_node)
        loss_pred = F.mse_loss(y_pred, y_true)
        loss_contrast = self.contrastive_loss(z_node, z_context)
        
        loss = pred_weight * loss_pred + contrast_weight * loss_contrast
        
        return loss, {'pred': loss_pred.item(), 'contrast': loss_contrast.item()}
```

---

## 5. Training Strategy

### 5.1 Two-Stage Training (권장)

- **Stage 1 - Pre-training**: Contrastive loss만으로 representation 학습 (epoch 50~100)
- **Stage 2 - Fine-tuning**: Prediction loss 추가하여 downstream task 최적화 (epoch 100~200)

### 5.2 End-to-End Training (대안)

Prediction loss와 Contrastive loss를 동시에 최적화. Dynamic weight averaging (DWA)로 loss balance 조절.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

for epoch in range(200):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss, loss_dict = model.compute_loss(
            batch_x, batch_y,
            pred_weight=1.0,
            contrast_weight=0.5
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    scheduler.step()
```

---

## 6. Experiment Plan

### 6.1 Datasets

| Dataset | Nodes | Time Span | 특징 |
|---------|-------|-----------|------|
| METR-LA | 207 | 4 months | Speed, Graph ○ |
| PEMS-BAY | 325 | 6 months | Speed, Graph ○ |
| NYC-Taxi | 263 | 6 months | Flow, Graph × |

### 6.2 Ablation Study

- **Baseline**: Temporal encoder + prediction head only
- **+ Context Attention**: temporal context 추가
- **+ Spatial Context**: spatial context 추가 (graph 있는 경우)
- **+ Contrastive Loss**: context-contrastive learning 추가

### 6.3 Evaluation Metrics

- **Performance**: MAE, RMSE, MAPE
- **Variance**: 5회 random seed 실험의 std
- **Robustness**: 사고 구간 / 평상시 구간 성능 비교

### 6.4 Baselines

- STGCN, DCRNN, Graph WaveNet (GNN-based)
- ST-SSL (Self-supervised ST)
- DCdetector 변형 (Anomaly detection baseline)

---

## 7. Implementation Notes

### 7.1 Hyperparameters

| Parameter | Default | Note |
|-----------|---------|------|
| d_model | 64 | Hidden dimension |
| num_heads | 4 | Attention heads |
| temperature | 0.1 | Contrastive loss temperature |
| contrast_weight | 0.5 | Loss weight (tune: 0.1~1.0) |
| learning_rate | 1e-3 | With cosine annealing |
| batch_size | 32 | Adjust for GPU memory |

### 7.2 Key Implementation Details

- **Gradient clipping**: max_norm=5.0 (학습 안정성)
- **Representation normalization**: L2 normalize before contrastive loss
- **Context masking**: 마지막 timestamp를 context에서 제외하여 정보 누출 방지
- **Early stopping**: validation loss 기준 patience=20

---

## 8. Expected Results & Discussion

### 8.1 예상 효과

- **Performance variance 감소**: Context-aware learning으로 representation 일관성 향상
- **Anomaly sensitivity**: Context 대비 이상 상태가 자연스럽게 다른 representation으로 인코딩
- **Decoder 의존성 제거**: Pure contrastive learning으로 encoder 품질에 집중

### 8.2 잠재적 한계 및 대응

| 잠재적 한계 | 대응 방안 |
|-------------|-----------|
| Representation collapse | Temperature tuning, stop-gradient 적용 |
| Context 정의의 민감도 | Ablation으로 최적 context window 탐색 |
| Contrastive weight 민감도 | DWA 또는 grid search |

---

## 9. Next Steps

1. 코드 구현 및 단위 테스트 (1주)
2. Small-scale 실험 (METR-LA subset)으로 feasibility 검증 (1주)
3. Full-scale 실험 및 ablation study (2주)
4. 결과 분석 및 논문 작성 (2주)

---

## Appendix: Code Files

전체 구현 코드는 `context_contrastive_model.py` 파일에 포함되어 있습니다.

주요 클래스:
- `TemporalEncoder`: 시간적 의존성 인코딩
- `ContextAttention`: Node/Context representation 분리
- `SpatialContextAttention`: 공간적 context (optional)
- `ContextContrastiveLoss`: Contrastive loss 계산
- `ContextContrastiveModel`: 전체 모델 통합