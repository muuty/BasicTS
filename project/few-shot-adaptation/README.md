# Few-Shot Node Onboarding in Expanding Traffic Networks

## Paper Story

```
1. Problem: Few-shot node onboarding in expanding STGNNs
   → 새 센서 설치 후 소량 데이터(3h~7d)만으로 기존 모델을 adapt
   → 기존 연구는 전부 full data 가정 (TrafficStream, PECPM, EAC 등)

2. Analysis: N-Dependency Analysis
   → STGNN의 파라미터 중 Node-Specific Parameters(NSP)만 N에 의존
   → NSP 형태는 모델마다 다름: (N,D), (T,N,D), 추가 nn.Embedding 등
   → Network expansion 시 NSP만 adaptation하면 됨
   → 기존 노드의 NSP freeze → zero forgetting 보장

3. Method: Embedding Predictor Network (EPN)
   → 기존 노드의 (traffic pattern, graph context) → learned NSP 관계를
     synthetic episode로 학습
   → 새 노드 도착 시: few-shot data + graph context → NSP 예측
   → NSP 형태에 따라 EPN architecture를 adapt

4. Experiments:
   → 3+ models (AGCRN, STAEformer, STGCN+Emb)
   → 5 budgets (3h, 12h, 1d, 7d, full)
   → Ablation: mean init, graph-only, pattern-only, EPN, EPN+FT
   → Visualization: embedding space, pattern matching, geographic case study
```

### Core Contributions

1. **Problem Formulation**: First study of few-shot node onboarding in expanding STGNNs
2. **N-Dependency Analysis**: Node-Specific Parameters (NSP) 정의 및 체계적 분석
3. **Method**: EPN — synthetic episode 기반 NSP predictor, 모델별 architecture 적응

---

## N-Dependency Analysis: Node-Specific Parameters (NSP)

**NSP 정의**: 모델 파라미터 중 노드 수 N에 scale하는 learnable parameters. Network expansion 시 이 부분만 adaptation 필요.

| Model | NSP | Shape | 특성 |
|---|---|---|---|
| **AGCRN** | node_embeddings | **(N, D=10)** | Per-node identity vector. Adaptive adj + node-specific GRU에 사용 |
| **STAEformer** | adaptive_embedding | **(T=12, N, D=80)** | Per-node spatio-temporal representation. Timestep별 다른 값 |
| **STGCN+Emb** | nn.Embedding (추가) | **(N, D)** | 앞단에 붙이는 node identity. Representation learning 논문 관례 |
| **GWNet** | nodevec1, nodevec2 | **(N, D), (D, N)** | Adaptive adj matrix 생성용 |
| **DCRNN+Emb** | nn.Embedding (추가) | **(N, D)** | STGCN과 동일하게 추가 |

**N-Independent parameters** (변경 불필요): temporal conv, graph conv, attention layers, FFN, output projection 등

**Key insight**: NSP 형태는 모델마다 다르지만, EPN의 입력(traffic pattern + graph context)은 공통. EPN의 출력 head만 NSP shape에 맞게 adapt.

---

## Proposed Method: Embedding Predictor Network (EPN)

### Motivation
Mean/random initialization 대신, **기존 노드의 학습된 NSP를 supervision으로 활용**하여 새 노드의 NSP를 예측하는 네트워크를 학습.

### Synthetic Episode Training (Offline)

```
Pre-trained model의 기존 노드 N_old개에서 반복:
  1. 노드 i를 randomly mask (NSP 제거)
  2. Input 구성:
     - 노드 i의 traffic data (budget만큼, e.g., 3h=36 steps × C channels)
     - 노드 i의 이웃 노드들의 NSP (graph context)
  3. Target: 노드 i의 원래 학습된 NSP
  4. Loss: MSE(predicted_NSP, true_NSP)

Budget별로 별도 EPN 학습 (3h용, 12h용, 1d용, 7d용)
또는 variable-length input을 처리하는 single EPN
```

### EPN Architecture

```
Traffic Pattern Encoder:
  Input: (budget_steps, num_channels)  e.g., (36, 5) for 3h
  → 1D-Conv layers or MLP
  → pattern_feature (d_hidden)

Graph Context Encoder:
  Input: neighbor NSPs + adjacency weights
  → Weighted mean or attention aggregation
  → context_feature (d_hidden)

Fusion + NSP Head (model-specific):
  concat(pattern_feature, context_feature)
  → MLP
  → predicted NSP

  NSP head adapts to target model:
  - AGCRN: output (D=10)
  - STAEformer: output (T×D=12×80=960)
  - STGCN+Emb: output (D)
```

### Deployment (New nodes)

```
1. 새 노드의 few-shot traffic data 수집 (3h~7d)
2. Graph에서 이웃 노드의 NSP 가져옴
3. EPN → NSP 예측 (initialization)
4. (Optional) Few-shot data로 추가 fine-tuning
   - Gradient mask: 기존 노드 NSP freeze → zero forgetting
```

---

## Experiment Design

### Dataset
- **SAN_BERNARDINO**: 893 nodes, 5 features (flow, occ, speed, tod, dow)
- 5-min intervals, 288 steps/day
- Location: `datasets/xtraffic/SAN_BERNARDINO/`

### Expansion Scenario
- Quarterly: Q1(558) → Q2(669) → Q3(781) → Q4(893)
- Source model trained on Q1 (558 nodes)
- Each quarter: new nodes onboarded

### Few-Shot Budgets
| Budget | Steps | Windows (~24-step) | Practical Meaning |
|---|---|---|---|
| 3h | 36 | ~13 | Immediate deployment |
| 12h | 144 | ~121 | Half-day collection |
| 1d | 288 | ~265 | One full daily cycle |
| 7d | 2016 | ~1993 | One week collection |
| full | All quarter data | ~15000+ | Upper bound |

### Ablation Design

| Method | Traffic Pattern | Graph Context | FT | Forgetting |
|---|---|---|---|---|
| Cold-start (no adapt) | - | - | - | 0 |
| Mean init | - | - | - | 0 |
| Mean init + FT | - | - | O | 0 (grad mask) |
| Graph-only init | - | O | - | 0 |
| Pattern-only init | O | - | - | 0 |
| **EPN (full)** | **O** | **O** | **-** | **0** |
| **EPN + FT** | **O** | **O** | **O** | **0** |
| full_model (ref) | - | - | O (all params) | >0 |
| Oracle | Full retrain | | | - |

### Metrics
- **MAE_all**: Overall MAE (all nodes, test set)
- **MAE_existing**: MAE on existing nodes
- **MAE_new**: MAE on new nodes
- **Forgetting**: MAE_existing_after - MAE_existing_coldstart (0 = perfect)
- **NSP MSE**: Embedding prediction quality (EPN evaluation)

### Training Details
- EPN training: offline on source model's existing nodes
- Fine-tuning: max 200 epochs, early stopping patience 10, lr=1e-4, Adam
- GPU: cuda:1 (GPU 0 reserved)

---

## Baseline Experiment Results

### v1 Results (Fixed 10 epochs)
- Location: `project/concept_drift/results/results.json`
- Key findings:
  - pb_adapt failed (K-means factorization too lossy)
  - full_model best MAE_all but highest forgetting
  - emb_new_only best stability (near-zero forgetting)

### v2 Results (Early stopping, max 200 epochs)
- Location: `project/concept_drift/results/results_v2_earlystop.json`
- Status: **Running** (nohup PID 364817)
- Log: `project/concept_drift/results/experiment_v2_log.txt`

---

## File Structure

```
project/few-shot-adaptation/
├── README.md                  # This file
├── related_work.md            # Survey of 7 related papers
├── epn_model.py               # EPN model definition
├── train_epn.py               # Synthetic episode training
├── evaluate_epn.py            # Evaluation on real expansion
└── results/                   # EPN experiment results

project/concept_drift/
├── expanding_adaptation.py    # Baseline adaptation experiments (v1/v2)
├── baselines/
│   └── expanding_q1_source.py # Source model config (Q1, 558 nodes)
└── results/
    ├── results.json           # v1 results
    ├── results_v2_earlystop.json  # v2 results (early stopping)
    └── experiment_v2_log.txt  # v2 log
```

## Related Work
- See [related_work.md](related_work.md) for detailed survey of 7 papers
- Most relevant: EAC (ICLR'25), PECPM (KDD'23), TrafficStream (IJCAI'21)
- Our differentiators: **few-shot budget**, **zero forgetting guarantee**, **N-dependency (NSP) analysis**

## Environment
- Conda: `basicts`
- GPU: Always use GPU 1 (`gpus='1'`)
- Python: Use `python` (not `python3`) after conda activate
