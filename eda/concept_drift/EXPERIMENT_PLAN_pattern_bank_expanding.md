# Experiment Plan: Pattern Bank for Expanding Sensor Networks (v2)

## 1. 실험 목적

### 핵심 질문
> "Node-specific embedding을 Pattern Bank (shared prototypes + per-node mixing weights)로
> factorize하면, unseen 노드에서 더 적은 데이터로 더 빠르게 적응할 수 있는가?"

### 배경
- Init strategy 실험 (v1)에서 확인: **초기화 방법은 중요하지 않다** (random ≈ zero ≈ global_avg)
- 핵심은 **구조적 차이**: per-node D-dim embedding 직접 학습 vs K개 prototype의 mixing weights 학습
- Per-node 파라미터: embedding=D=24 vs PB weights=K=8 → PB가 3배 적음

### 논문에서의 위치
"Universal core + Node-specific embedding" 주장 강화:
1. Scale drift → Instance Norm (확인)
2. Embedding drift → Few-shot embedding FT (확인)
3. **[이 실험] Expanding network → PB factorization이 더 data-efficient한 adaptation을 가능하게 하는가?**

---

## 2. 실험 설계

### 2.1 Clean Cold-Start Setup (Unseen Leakage 방지)

**이전 실험의 문제**: 893개 노드로 학습한 모델에서 200개를 "new"로 지정 →
core weights가 이미 해당 노드 패턴을 본 상태 → 진짜 cold-start가 아님.

**해결**: Source year 학습 단계부터 700개 노드만 사용.

```
전체 893 nodes (functional에서 선택)
├── 693 existing nodes  ← source year 학습에 참여
└── 200 new nodes       ← source year 학습에서 완전 제외, target year에서만 등장
```

- Node 선택: functional 노드(dead/major_fail 제외)에서 random split (seed=42)
- Dataset의 `node_indices` 파라미터로 학습 시 700개만 사용
- **두 모델 모두** (baseline STAEformer, PB-STAEformer) 동일한 700 노드로 학습

### 2.2 Two Model Architectures

#### (A) Baseline STAEformer
기존 그대로. `adaptive_embedding = nn.Parameter(T=12, N=700, D=24)`

#### (B) PB-STAEformer
`adaptive_embedding`을 PB 모듈로 **교체**:

```python
class PatternBankEmbedding(nn.Module):
    def __init__(self, num_patterns, in_steps, adaptive_embedding_dim):
        self.pattern_bank = nn.Parameter(K, T, D)   # shared prototypes
        self.node_weights = nn.Parameter(N, K)       # per-node mixing logits

    def forward(self):
        weights = softmax(self.node_weights, dim=-1)  # (N, K)
        # (N, K) @ (K, T*D) → (N, T*D) → reshape to (T, N, D)
        return emb  # (T, N, D)
```

이 모듈이 STAEformer의 encoder 내부에서 `adaptive_embedding` 자리에 들어감.
**매 forward마다** `softmax(weights) @ prototypes`를 계산하여 adaptive embedding 생성.

→ `.data.copy_()` 같은 hack 없이 autograd graph가 자연스럽게 연결됨.

### 2.3 Source Year 학습 (접근 2: 처음부터)

두 모델 모두 source year에서 처음부터 학습:
- Dataset: SAN_BERNARDINO_{year}_Q1, node_indices=existing_693
- Runner: InstanceNormRunner (scale drift 대응 내장)
- Epochs: 30 (기존과 동일)
- 기타 하이퍼파라미터: 기존 STAEformer config과 동일

**K-means 같은 변환 과정 없음**. PB는 학습 과정에서 자연스럽게 최적화됨.

### 2.4 Expanding to 893 Nodes

학습 완료 후, 200개 new 노드 추가:

#### Baseline STAEformer
```python
# adaptive_embedding: (T, 700, D) → (T, 893, D)
old_emb = model.encoder.adaptive_embedding.data  # (T, 700, D)
new_emb = torch.zeros(T, 893, D)
new_emb[:, existing_idx, :] = old_emb
# new_emb[:, new_idx, :] = 0  (zero init, init 방법은 중요하지 않음을 이미 확인)
model.encoder.adaptive_embedding = nn.Parameter(new_emb)
```

#### PB-STAEformer
```python
# pattern_bank: (K, T, D) → 그대로 유지 (shared)
# node_weights: (700, K) → (893, K)
old_weights = model.encoder.pb.node_weights.data  # (700, K)
new_weights = torch.zeros(893, K)
new_weights[existing_idx] = old_weights
# new_weights[new_idx] = 0  → softmax(0) = uniform → 모든 prototype 균등 혼합
model.encoder.pb.node_weights = nn.Parameter(new_weights)
```

**핵심 차이**: Baseline은 new 노드에 24-dim 벡터를 0에서 학습해야 하지만,
PB는 8-dim mixing weights만 학습하면 됨 + prototype은 이미 학습된 것 공유.

### 2.5 Fine-tuning Configurations

| ID | Method | FT 대상 | Per-node params | 설명 |
|----|--------|---------|-----------------|------|
| A | `emb_all` | adaptive_embedding 전체 (893) | 24 × 893 = 21,432 | Baseline: embedding 전체 FT |
| B | `emb_new_only` | adaptive_embedding (200 new만) | 24 × 200 = 4,800 | Baseline: new만 FT |
| C | `pb_weights_all` | node_weights 전체 (proto frozen) | 8 × 893 = 7,144 | PB: weights 전체 FT |
| D | `pb_weights_new_only` | node_weights (200 new만, proto frozen) | 8 × 200 = 1,600 | PB: new weights만 FT |
| E | `pb_both` | pattern_bank + node_weights 전체 | 8×893 + 8×12×24 = 9,448 | PB: 전부 FT |

**공정성 보고**: 각 method의 trainable param 수를 명시.

### 2.6 Fine-tuning 구현 주의사항

**pb_weights_new_only (Method D)에서 optimizer state 누수 방지:**

Gradient masking 대신 파라미터 분리:
```python
# BAD: gradient masking (momentum/weight decay 누수 가능)
optimizer = Adam([model.node_weights])
loss.backward()
model.node_weights.grad[existing_idx] = 0  # momentum still leaks

# GOOD: 파라미터 분리
new_weights = nn.Parameter(model.node_weights.data[new_idx].clone())
optimizer = Adam([new_weights])
# forward에서 new_weights를 node_weights[new_idx]에 scatter
```

### 2.7 Few-shot Budget
3h, 12h, 1d, 7d (이전 실험과 동일)

### 2.8 Year Pairs
- 2022→2023 (1년 후)
- 2023→2024 (1년 후)

시간 순서대로 1년씩 이동: "매년 네트워크가 확장될 때 어떻게 적응하나"라는
현실적 시나리오에 대응. 2022→2024처럼 2년 점프보다 자연스러운 narrative.

---

## 3. 평가 지표

### 기본 지표
- **MAE (all)**: 전체 893 노드
- **MAE (existing)**: 693 기존 노드
- **MAE (new)**: 200 새 노드
- Instance Norm 적용 (input window에서 mean/std 추출 → denorm)

### 추가 지표
- **Cold-start MAE@0**: FT 전 (expanding 직후, adaptation 없이) 성능
  - PB의 uniform mixing이 zero embedding보다 나은가?
- **Adaptation speed curve**: budget별 MAE 곡선 (3h→12h→1d→7d)
  - AUC로 요약 가능: 낮을수록 빠른 적응
- **Forgetting**: FT 후 existing 노드 성능 하락량
  - forgetting = MAE_existing(after FT) - MAE_existing(before FT)
  - 양수 = 성능 하락 (나쁨)

---

## 4. 기대 결과 및 가설

### 가설 1: Few-shot에서 PB가 더 data-efficient
- 3h, 12h에서 D(pb_weights_new_only)의 new MAE < B(emb_new_only)의 new MAE
- 이유: per-node 8 params vs 24 params, 적은 데이터에서 overfitting 적음

### 가설 2: 충분한 데이터에서는 수렴
- 7d에서 D ≈ B
- 이유: 충분한 데이터면 24-dim도 잘 학습됨

### 가설 3: Cold-start에서 PB 우위
- PB의 MAE@0 < Baseline의 MAE@0
- 이유: PB는 uniform mixing으로 "평균적인 센서" 역할 가능, zero embedding은 정보 없음

### 가설 4: Forgetting이 PB에서 적음
- C(pb_weights_all)의 forgetting < A(emb_all)의 forgetting
- 이유: PB는 prototype이 frozen이면 기존 노드의 representational basis가 보존됨

---

## 5. 실험 실행 순서

### Phase 1: Model Training (~1시간)
1. **STAEformer-Baseline**: 700 nodes, source year, InstanceNorm, 30 epochs
2. **STAEformer-PB**: 700 nodes, source year, InstanceNorm, 30 epochs
   - K=8 (default), 추후 K ablation 가능
3. 각 source year (2022, 2023)에 대해 → 총 4개 모델 학습

### Phase 2: Expanding + Evaluation (~1시간)
4. 모델 확장 (700→893 nodes)
5. Cold-start MAE@0 측정
6. Method A~E × Budget 4개 × Year pair 2개 fine-tune + evaluate
7. 결과 저장

### Phase 3: Analysis
8. Method별 비교 테이블
9. Adaptation speed curve
10. Forgetting 분석

---

## 6. 구현해야 할 것

### 새로 만들 파일
1. `baselines/STAEformer/arch/pattern_bank_embedding.py`
   - `PatternBankEmbedding` 모듈 (nn.Module)
   - STAEformer encoder에 통합 가능한 형태

2. `baselines/STAEformer/SAN_BERNARDINO_PB_700nodes.py`
   - PB-STAEformer config (700 nodes, K=8)

3. `baselines/STAEformer/SAN_BERNARDINO_baseline_700nodes.py`
   - Baseline STAEformer config (700 nodes)

4. `eda/concept_drift/pattern_bank_expanding.py`
   - 메인 실험 스크립트: expanding + FT + evaluation

### 수정할 파일
- `baselines/STAEformer/arch/encoder.py`: PatternBankEmbedding 통합 옵션 추가

---

## 7. 이전 실험과의 차이 (혼동 방지)

| | v1 (expanding_sensor_experiment.py) | v2 (pattern_bank_expanding.py) |
|---|---|---|
| 핵심 질문 | "어떤 init이 좋은가?" | "PB factorization이 더 data-efficient한가?" |
| 학습 노드 수 | 893 (전체) | **700** (200 제외 → 진짜 cold-start) |
| 모델 학습 | 기존 checkpoint 사용 | **처음부터 학습** (PB 구조 내장) |
| adaptive_embedding | 값만 reset | PB 모듈로 **구조적 교체** |
| 학습 대상 | embedding 자체 | node_weights (±pattern_bank) |
| Per-node params | D=24 (고정) | K=8 (PB) vs D=24 (baseline) |
| Gradient 처리 | masking (누수 위험) | 파라미터 분리 |
| 추가 지표 | MAE만 | + cold-start MAE@0, adaptation AUC, forgetting |
| Unseen leakage | 있음 (893으로 학습) | **없음** (700으로 학습) |
