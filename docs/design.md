# Context-aware Severity Contrastive Learning for Robust Traffic Prediction

## 1. 연구 배경 및 문제 정의

### 1.1 문제 상황
- 교통 예측 모델에서 전체 평균 MAE는 좋지만, node/timestamp 간 성능 차이가 큼
- 특히 **사고 상황에서 예측 성능이 크게 저하**됨
- 사고 발생 시 특정 노드의 flow가 0으로 급감하지만, 모델은 평소와 유사한 값을 예측

### 1.2 데이터 특성
- **Dataset:** PEMS (PEMS04, PEMS08 등)
- **Backbone 모델:** AGCRN, STAEformer, STGformer, STGCN
- **사고 timestamp 비율:** 전체의 약 10%
- **사고 timestamp 내 실제 사고 노드:** 수백 개 중 1개
- **전체 데이터에서 사고 노드×사고 시점 비율:** 0.01~0.1%
- PEMS에서 사고 report 제공 (type 포함)

### 1.3 기존 시도 및 실패
| 시도 | 결과 | 문제점 |
|------|------|--------|
| Message passing 후 similarity-based loss | 사고 노드 개선, overall MAE 하락 | 정상 케이스 성능 저하 |
| Gating module 도입 | 동일 | 동일 |
| Experience replay (difficulty/representative based) | MAE 악화 | 학습 불안정 |

**공통 패턴:** 사고 케이스 개선 시도 → overall MAE 1~10% 악화

### 1.4 핵심 발견
- Input에 특정 노드 flow가 0으로 들어와도 모델은 평소처럼 output
- Message passing에서 주변 정상 노드 정보가 사고 노드의 0 signal을 덮어씀
- 모델이 input의 급격한 변화에 둔감함

---

## 2. 제안 Framework 개요

### 2.1 핵심 아이디어
**Self-supervised pre-training으로 anomaly-aware representation 학습**

- Anomaly detection 논문들(DCDetector, CARLA)에서 영감
- Traffic-specific anomaly injection으로 synthetic negative sample 생성
- Severity-aware contrastive learning으로 정상/비정상 representation 분리

### 2.2 기대 효과
- Representation space에서 정상과 anomaly 분리
- Severity에 따라 정상에서 점점 멀어지는 구조
- 사고 input → representation이 anomaly 영역 → prediction head가 적절한 낮은 output 생성
- **목표:** 1~5% overall MAE 손해로 worst-case MAE 30% 개선

### 2.3 Framework 구조

```
┌──────────────────────────────────────────────────────────────────┐
│         Two-Stage Training Framework                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Stage 1: Context-aware Contrastive Pre-training]               │
│                                                                   │
│   Raw Data ──┬── Weak Aug (Jittering) ──→ Encoder ──→ z_weak     │
│              │                              ↓                     │
│              │                    Severity-aware                  │
│              │                    Contrastive Loss                │
│              │                              ↑                     │
│              └── Strong Aug (Anomaly Inj) ─→ Encoder ──→ z_strong │
│                                                                   │
│   Output: Pre-trained Context-aware ST Encoder                    │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Stage 2: Prediction Fine-tuning]                               │
│                                                                   │
│   Raw X ──→ [Frozen Encoder] ──→ Z ──→ [Backbone] ──→ Prediction │
│                                                                   │
│   - Encoder: frozen (from Stage 1)                                │
│   - Z replaces backbone's original input                          │
│   - Loss: MAE/MSE                                                 │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 상세 구현 명세

### 3.1 Context-aware ST Encoder

#### 구조
```python
class ContextAwareSTEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 1,           # flow feature dimension
        d_model: int = 256,           # hidden dimension
        n_nodes: int = 307,           # number of nodes (dataset-specific)
        n_layers: int = 3,            # transformer layers
        n_heads: int = 8,             # attention heads
        T: int = 12,                  # input sequence length
    ):
        super().__init__()
        
        # Context embeddings
        self.time_of_day_embed = nn.Embedding(288, d_model)   # 5분 단위, 하루=288
        self.day_of_week_embed = nn.Embedding(7, d_model)      # 요일
        self.node_embed = nn.Embedding(n_nodes, d_model)       # 노드별 특성
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Learnable temporal positional encoding (within window)
        self.temporal_pos = nn.Parameter(torch.randn(1, T, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm (CLEAR 논문 참고)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection (optional)
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, time_idx, dow_idx, node_idx=None):
        """
        Args:
            x: (B, T, N, F) - traffic flow values
            time_idx: (B, T) - time of day index (0-287)
            dow_idx: (B,) - day of week (0-6)
            node_idx: (N,) - node indices (optional, default: 0 to N-1)
        
        Returns:
            z: (B, T, N, D) - encoded representations
        """
        B, T, N, F = x.shape
        
        if node_idx is None:
            node_idx = torch.arange(N, device=x.device)
        
        # 1. Project input
        h = self.input_proj(x)  # (B, T, N, D)
        
        # 2. Add context embeddings
        tod_emb = self.time_of_day_embed(time_idx)  # (B, T, D)
        dow_emb = self.day_of_week_embed(dow_idx)   # (B, D)
        node_emb = self.node_embed(node_idx)        # (N, D)
        
        # Broadcast and add
        h = h + tod_emb.unsqueeze(2)                # (B, T, N, D) + temporal context
        h = h + dow_emb.unsqueeze(1).unsqueeze(2)   # + day context  
        h = h + node_emb.unsqueeze(0).unsqueeze(0)  # + node identity
        h = h + self.temporal_pos[:, :T, :, :]      # + position in window
        
        # 3. Reshape for transformer: (B, T*N, D)
        h = h.view(B, T * N, -1)
        
        # 4. Transformer encoding
        z = self.transformer(h)  # (B, T*N, D)
        
        # 5. Reshape back and project
        z = z.view(B, T, N, -1)  # (B, T, N, D)
        z = self.output_proj(z)
        
        return z
```

#### 설계 근거
- **Learnable positional encoding:** CLEAR 논문에서 sinusoidal보다 성능 좋음
- **Pre-norm residual:** 학습 안정성 향상 (CLEAR ablation 결과)
- **Context embeddings:** time_of_day, day_of_week, node_id를 모두 포함하여 같은 flow 값도 context에 따라 다른 representation 생성

---

### 3.2 Data Augmentation

#### 3.2.1 Weak Augmentation: Jittering

```python
def weak_augmentation(x, noise_std=0.1):
    """
    Simple Gaussian jittering
    
    Args:
        x: (B, T, N, F) - normalized input
        noise_std: standard deviation of Gaussian noise
    
    Returns:
        x_aug: (B, T, N, F) - augmented input
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise
```

#### 3.2.2 Strong Augmentation: Traffic-aware Anomaly Injection

```python
class TrafficAnomalyInjector:
    """
    Traffic-specific anomaly injection for creating negative samples
    """
    
    def __init__(
        self,
        severity_levels: dict = {
            1: (0.3, 0.5),   # 경미: 30-50% 감소
            2: (0.5, 0.8),   # 중간: 50-80% 감소
            3: (0.8, 1.0),   # 심각: 80-100% 감소
        },
        propagation_hops: int = 2,
        propagation_decay: float = 0.5,
        recovery_steps: int = 6,
    ):
        self.severity_levels = severity_levels
        self.propagation_hops = propagation_hops
        self.propagation_decay = propagation_decay
        self.recovery_steps = recovery_steps
    
    def inject(self, x, adj_matrix, injection_config=None):
        """
        Inject traffic anomaly into the data
        
        Args:
            x: (B, T, N, F) - original traffic data
            adj_matrix: (N, N) - adjacency matrix
            injection_config: dict with keys:
                - target_nodes: list of node indices to inject anomaly
                - start_time: injection start timestep
                - severity: severity level (1, 2, or 3)
                - anomaly_type: 'incident', 'congestion', 'lane_closure'
        
        Returns:
            x_aug: (B, T, N, F) - augmented data
            severity_labels: (B, T, N) - severity labels for each position
        """
        B, T, N, F = x.shape
        x_aug = x.clone()
        severity_labels = torch.zeros(B, T, N, device=x.device)
        
        if injection_config is None:
            injection_config = self._random_config(B, T, N)
        
        for b in range(B):
            config = injection_config[b] if isinstance(injection_config, list) else injection_config
            x_aug[b], severity_labels[b] = self._inject_single(
                x[b], adj_matrix, config
            )
        
        return x_aug, severity_labels
    
    def _inject_single(self, x, adj_matrix, config):
        """
        Inject anomaly into single sample
        
        Args:
            x: (T, N, F)
            adj_matrix: (N, N)
            config: injection configuration
        
        Returns:
            x_aug: (T, N, F)
            severity_labels: (T, N)
        """
        T, N, F = x.shape
        x_aug = x.clone()
        severity_labels = torch.zeros(T, N, device=x.device)
        
        target_nodes = config['target_nodes']
        start_time = config['start_time']
        severity = config['severity']
        anomaly_type = config.get('anomaly_type', 'incident')
        
        # Get reduction ratio for this severity
        min_red, max_red = self.severity_levels[severity]
        reduction = torch.rand(1).item() * (max_red - min_red) + min_red
        
        # 1. Inject at target nodes
        for node in target_nodes:
            for t in range(start_time, min(start_time + self.recovery_steps, T)):
                # Gradual onset (first 2 steps)
                if t < start_time + 2:
                    factor = (t - start_time + 1) / 2 * reduction
                # Recovery phase
                elif t >= start_time + self.recovery_steps - 2:
                    remaining = start_time + self.recovery_steps - t
                    factor = remaining / 2 * reduction
                else:
                    factor = reduction
                
                x_aug[t, node] = x[t, node] * (1 - factor)
                severity_labels[t, node] = severity
        
        # 2. Propagate to neighbors (with decay)
        if self.propagation_hops > 0:
            x_aug, severity_labels = self._propagate_anomaly(
                x_aug, severity_labels, adj_matrix, 
                target_nodes, start_time, severity, reduction
            )
        
        return x_aug, severity_labels
    
    def _propagate_anomaly(self, x, severity_labels, adj, target_nodes, start_time, severity, reduction):
        """
        Propagate anomaly effect to neighboring nodes
        """
        T, N, F = x.shape
        
        for hop in range(1, self.propagation_hops + 1):
            decay = self.propagation_decay ** hop
            delay = hop  # Time delay for propagation
            
            # Find k-hop neighbors
            neighbors = self._get_k_hop_neighbors(adj, target_nodes, hop)
            neighbors = neighbors - set(target_nodes)  # Exclude already affected
            
            for node in neighbors:
                for t in range(start_time + delay, min(start_time + self.recovery_steps + delay, T)):
                    current_factor = reduction * decay
                    x[t, node] = x[t, node] * (1 - current_factor)
                    severity_labels[t, node] = max(severity_labels[t, node], severity - hop)
        
        return x, severity_labels
    
    def _get_k_hop_neighbors(self, adj, source_nodes, k):
        """Get k-hop neighbors from source nodes"""
        adj_np = adj.cpu().numpy() if torch.is_tensor(adj) else adj
        current = set(source_nodes)
        
        for _ in range(k):
            new_neighbors = set()
            for node in current:
                neighbors = np.where(adj_np[node] > 0)[0]
                new_neighbors.update(neighbors)
            current = current.union(new_neighbors)
        
        return current
    
    def _random_config(self, B, T, N):
        """Generate random injection configuration"""
        configs = []
        for _ in range(B):
            n_target_nodes = np.random.randint(1, 4)  # 1-3 nodes
            target_nodes = np.random.choice(N, n_target_nodes, replace=False).tolist()
            start_time = np.random.randint(0, T - self.recovery_steps)
            severity = np.random.randint(1, 4)  # 1, 2, or 3
            
            configs.append({
                'target_nodes': target_nodes,
                'start_time': start_time,
                'severity': severity,
                'anomaly_type': np.random.choice(['incident', 'congestion', 'lane_closure'])
            })
        
        return configs
```

---

### 3.3 Pseudo-severity Label 계산

```python
class PseudoSeverityComputer:
    """
    Compute context-aware pseudo-severity labels from historical statistics
    """
    
    def __init__(self, historical_data, threshold_sigmas=[1.5, 2.5, 3.5]):
        """
        Args:
            historical_data: (total_time, N, F) - historical traffic data
            threshold_sigmas: thresholds for severity levels (in standard deviations)
        """
        self.threshold_sigmas = threshold_sigmas
        self._compute_statistics(historical_data)
    
    def _compute_statistics(self, data):
        """
        Compute per-(node, time_of_day, day_of_week) statistics
        """
        total_time, N, F = data.shape
        
        # Assuming 5-min intervals, 288 per day
        self.stats = {
            'mean': torch.zeros(N, 288, 7, F),
            'std': torch.zeros(N, 288, 7, F),
        }
        
        # Group by (time_of_day, day_of_week) and compute stats
        for tod in range(288):
            for dow in range(7):
                # Find all matching timestamps
                mask = [(t % 288 == tod) and ((t // 288) % 7 == dow) 
                        for t in range(total_time)]
                matching_data = data[mask]  # (num_matches, N, F)
                
                if len(matching_data) > 0:
                    self.stats['mean'][:, tod, dow, :] = matching_data.mean(dim=0)
                    self.stats['std'][:, tod, dow, :] = matching_data.std(dim=0) + 1e-6
    
    def compute_severity(self, x, time_idx, dow_idx):
        """
        Compute pseudo-severity labels
        
        Args:
            x: (B, T, N, F) - current traffic data
            time_idx: (B, T) - time of day indices
            dow_idx: (B,) - day of week indices
        
        Returns:
            severity: (B, T, N) - severity labels (0=normal, 1,2,3=anomaly levels)
        """
        B, T, N, F = x.shape
        severity = torch.zeros(B, T, N, device=x.device)
        
        for b in range(B):
            for t in range(T):
                tod = time_idx[b, t].item()
                dow = dow_idx[b].item()
                
                mean = self.stats['mean'][:, tod, dow, :]  # (N, F)
                std = self.stats['std'][:, tod, dow, :]    # (N, F)
                
                # Z-score
                z_score = torch.abs(x[b, t] - mean) / std  # (N, F)
                z_score = z_score.mean(dim=-1)  # (N,) - average over features
                
                # Assign severity levels
                for level, thresh in enumerate(self.threshold_sigmas, 1):
                    mask = z_score > thresh
                    severity[b, t, mask] = level
        
        return severity
```

---

### 3.4 Severity-aware Contrastive Loss

```python
class SeverityAwareContrastiveLoss(nn.Module):
    """
    Contrastive loss that considers severity levels
    """
    
    def __init__(self, temperature=0.1, severity_weight_scale=1.0):
        super().__init__()
        self.temperature = temperature
        self.severity_weight_scale = severity_weight_scale
    
    def forward(self, z_weak, z_strong, severity_weak, severity_strong):
        """
        Args:
            z_weak: (B, T, N, D) - representations from weak augmentation
            z_strong: (B, T, N, D) - representations from strong augmentation
            severity_weak: (B, T, N) - severity labels for weak aug (mostly 0)
            severity_strong: (B, T, N) - severity labels for strong aug (injected)
        
        Returns:
            loss: scalar
        """
        B, T, N, D = z_weak.shape
        
        # Flatten
        z_w = z_weak.view(B * T * N, D)      # (BTN, D)
        z_s = z_strong.view(B * T * N, D)    # (BTN, D)
        sev_w = severity_weak.view(-1)        # (BTN,)
        sev_s = severity_strong.view(-1)      # (BTN,)
        
        # Normalize
        z_w = F.normalize(z_w, dim=1)
        z_s = F.normalize(z_s, dim=1)
        
        # Concatenate for computing all similarities
        z_all = torch.cat([z_w, z_s], dim=0)  # (2BTN, D)
        sev_all = torch.cat([sev_w, sev_s], dim=0)  # (2BTN,)
        
        # Similarity matrix
        sim = torch.mm(z_all, z_all.t()) / self.temperature  # (2BTN, 2BTN)
        
        # Mask out self-similarity
        mask_self = torch.eye(len(z_all), device=z_all.device).bool()
        sim.masked_fill_(mask_self, -float('inf'))
        
        # Severity difference matrix
        sev_diff = torch.abs(sev_all.unsqueeze(0) - sev_all.unsqueeze(1))  # (2BTN, 2BTN)
        
        # Positive mask: same severity level
        pos_mask = (sev_diff == 0) & ~mask_self
        
        # Negative weights: proportional to severity difference
        neg_weights = sev_diff / (sev_diff.max() + 1e-6) * self.severity_weight_scale
        neg_weights = torch.clamp(neg_weights, min=0.1)  # Minimum weight
        
        # Compute loss
        loss = 0
        valid_count = 0
        
        for i in range(len(z_all)):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = (~pos_mask[i] & ~mask_self[i]).nonzero(as_tuple=True)[0]
            
            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
            
            pos_sim = sim[i, pos_indices]
            neg_sim = sim[i, neg_indices]
            neg_w = neg_weights[i, neg_indices]
            
            # Weighted InfoNCE
            pos_exp = torch.exp(pos_sim).sum()
            neg_exp = (neg_w * torch.exp(neg_sim)).sum()
            
            loss += -torch.log(pos_exp / (pos_exp + neg_exp + 1e-6))
            valid_count += 1
        
        return loss / (valid_count + 1e-6)
```

---

### 3.5 Training Pipeline

```python
class ContrastivePretrainer:
    """
    Stage 1: Contrastive pre-training
    """
    
    def __init__(
        self,
        encoder: ContextAwareSTEncoder,
        anomaly_injector: TrafficAnomalyInjector,
        severity_computer: PseudoSeverityComputer,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.encoder = encoder
        self.anomaly_injector = anomaly_injector
        self.severity_computer = severity_computer
        
        self.optimizer = torch.optim.AdamW(
            encoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.loss_fn = SeverityAwareContrastiveLoss(temperature=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
    
    def train_epoch(self, dataloader, adj_matrix):
        """
        Train one epoch
        
        Args:
            dataloader: yields (x, time_idx, dow_idx)
            adj_matrix: (N, N) adjacency matrix
        """
        self.encoder.train()
        total_loss = 0
        
        for batch_idx, (x, time_idx, dow_idx) in enumerate(dataloader):
            # x: (B, T, N, F), time_idx: (B, T), dow_idx: (B,)
            
            # 1. Weak augmentation: jittering
            x_weak = weak_augmentation(x, noise_std=0.1)
            severity_weak = self.severity_computer.compute_severity(x_weak, time_idx, dow_idx)
            
            # 2. Strong augmentation: anomaly injection
            x_strong, severity_strong = self.anomaly_injector.inject(x, adj_matrix)
            
            # 3. Encode both
            z_weak = self.encoder(x_weak, time_idx, dow_idx)
            z_strong = self.encoder(x_strong, time_idx, dow_idx)
            
            # 4. Compute loss
            loss = self.loss_fn(z_weak, z_strong, severity_weak, severity_strong)
            
            # 5. Backprop
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)


class PredictionFinetuner:
    """
    Stage 2: Prediction fine-tuning with frozen encoder
    """
    
    def __init__(
        self,
        encoder: ContextAwareSTEncoder,
        backbone: nn.Module,  # AGCRN, STAEformer, etc.
        prediction_horizon: int = 12,
        learning_rate: float = 1e-3,
    ):
        self.encoder = encoder
        self.backbone = backbone
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            self.backbone.parameters(),
            lr=learning_rate
        )
        
        self.loss_fn = nn.L1Loss()  # MAE
    
    def train_epoch(self, dataloader):
        """
        Train one epoch
        """
        self.encoder.eval()
        self.backbone.train()
        total_loss = 0
        
        for batch_idx, (x, time_idx, dow_idx, y) in enumerate(dataloader):
            # x: (B, T, N, F), y: (B, L, N, F) - ground truth
            
            # 1. Get representations (frozen)
            with torch.no_grad():
                z = self.encoder(x, time_idx, dow_idx)  # (B, T, N, D)
            
            # 2. Backbone prediction
            # Note: backbone input interface may need adaptation
            y_pred = self.backbone(z)  # (B, L, N, F)
            
            # 3. Loss
            loss = self.loss_fn(y_pred, y)
            
            # 4. Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

---

## 4. 실험 계획

### 4.1 Datasets
- **PEMS04:** 307 nodes, 59 days, flow data
- **PEMS08:** 170 nodes, 62 days, flow data  
- **METR-LA:** 207 nodes, 4 months, speed data
- **PEMS-BAY:** 325 nodes, 6 months, speed data

### 4.2 Backbone Models
- AGCRN
- STAEformer
- STGCN
- Graph WaveNet

### 4.3 Evaluation Metrics
- **Overall:** MAE, RMSE, MAPE
- **Anomaly-specific:** 
  - MAE on accident timestamps
  - MAE on accident nodes at accident timestamps
  - Variance across nodes/timestamps

### 4.4 Ablation Studies
1. Context embedding 효과 (w/ vs w/o)
2. Severity-aware loss vs vanilla contrastive loss
3. Anomaly injection types
4. Encoder freeze vs fine-tune in Stage 2

### 4.5 Baseline Comparisons
- No pre-training (original backbone)
- CLEAR (TKDE 2025)
- STD-MAE (IJCAI 2024)
- SCPT (DMKD 2023)

---

## 5. 구현 우선순위

### Phase 1: Core Components
1. [ ] ContextAwareSTEncoder
2. [ ] TrafficAnomalyInjector
3. [ ] PseudoSeverityComputer
4. [ ] SeverityAwareContrastiveLoss

### Phase 2: Training Pipeline
1. [ ] ContrastivePretrainer
2. [ ] PredictionFinetuner
3. [ ] DataLoader with context indices

### Phase 3: Experiments
1. [ ] PEMS04/08 데이터셋 준비
2. [ ] 사고 timestamp/node 라벨링
3. [ ] Backbone 연동 (input interface 수정)
4. [ ] Evaluation pipeline

---

## 6. 참고 논문

1. **CLEAR** (TKDE 2025): Contrastive Learning of spatial-tEmporal trAffic data Representations
   - Dual-branch (time-series + graph) representation learning
   - Weak/strong augmentation strategy
   - 4 types of bootstrapping methods

2. **STD-MAE** (IJCAI 2024): Spatio-Temporal-Decoupled Masked Pre-training
   - Masked autoencoder for traffic
   - Representation을 backbone hidden에 add하는 방식

3. **SCPT** (DMKD 2023): Spatial Contrastive Pre-Training
   - Spatial encoder pre-training
   - Spatially Gated Addition (SGA)

4. **DCDetector** (KDD 2023): Dual Attention Contrastive Representation Learning
   - Anomaly detection with contrastive learning
   - Dual-view representation discrepancy

5. **CARLA** (Pattern Recognition 2024): Contrastive Learning with Anomaly Injection
   - Synthetic anomaly injection for negative samples

---

## 7. 핵심 Design Decisions 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| Encoder | Unified Transformer | 구현 단순화, 검증 용이 |
| Stage 2 Encoder | Freeze | Clean separation, overfitting 방지 |
| Weak Aug | Jittering | 무난하고 검증된 방법 |
| Strong Aug | Anomaly Injection | Traffic-specific novelty |
| Context | Time-of-day + Day-of-week + Node embedding | Context-dependent anomaly 해결 |
| Severity | Pseudo-labeling (historical stats) + Injection labels | End-to-end와 절충 |
| Backbone 연결 | Input replacement | Model-agnostic, clean interface |

---

## 8. 예상 Contribution

1. **Traffic-aware Anomaly Injection Framework**
   - 교통 도메인 특화된 anomaly injection
   - Spatial propagation, recovery pattern 반영

2. **Context-aware Severity Contrastive Learning**
   - Context embedding으로 same-value-different-meaning 문제 해결
   - Severity-aware loss로 정상/비정상 representation 분리

3. **Model-agnostic Pre-training**
   - 다양한 backbone에 plug-and-play로 적용 가능
   - Overall MAE 유지하면서 worst-case 성능 개선