# Adjacency Matrix Experiment: Complete Context for Server Transfer

> **Date**: 2026-02-15
> **Author**: Auto-generated for experiment transfer to slurm research server
> **Status**: Ready to run (48 experiments)

---

## 1. Research Background

### 1.1 Research Question

GNN 기반 교통 예측에서 adjacency matrix 구성 방법론에 대한 best practice가 확립되어 있지 않다.
기존 연구들은 각자 자기 모델 내에서만 ablation을 수행하며, **동일한 adjacency method set을 여러 GNN architecture에 걸쳐 체계적으로 비교한 cross-model controlled experiment는 부재**하다.

이 실험은 distance-based adjacency의 sparsity를 체계적으로 변화시키면서, 2개 GNN 모델 x 2개 dataset에서의 영향을 비교한다.

### 1.2 Research Gap

| 기존 연구 유형 | 수행 내용 | 한계 |
|---------------|----------|------|
| Survey (STG4Traffic 등) | Graph construction 방법론 분류 | 실험적 비교 없음 |
| 개별 논문 (GWNet, MTGNN 등) | 자기 모델 내 ablation | Model-specific, 체계적이지 않음 |
| CLEAR (TKDE 2025) | Representation bootstrapping | 자체 방법론만 테스트 |

### 1.3 Advisor's Proposal

교수님이 제안한 연구 방향:
- Distance-based adjacency에서 **threshold (sparsity)** 를 체계적으로 변화
- 동일 adjacency를 여러 GNN에 적용하여 **cross-model** 비교
- 문헌에 근거한 방법론만 사용 (knn, correlation 등 traffic 분야 레퍼런스 없는 방법 제외)

### 1.4 Expected Contributions

1. **Cross-model controlled experiment** - 동일 프로토콜 하에서 같은 adjacency를 여러 모델에 적용
2. **Sparsity sensitivity analysis** - threshold 변화에 따른 성능 변화 체계적 분석
3. **Practical guidelines** - 모델/데이터 특성에 따른 adjacency 선택 가이드
4. **Reproducibility** - 공정 비교 프로토콜, 코드, 실험 로그 공개

---

## 2. Experiment Design

### 2.1 Adjacency Methods (Independent Variable)

4가지 adjacency matrix 구성 방법을 비교한다. 모두 **sensor의 위도/경도 좌표**에서 직접 생성한다.

| Method | Description | Sigma | Threshold | Expected Avg Degree |
|--------|-------------|-------|-----------|-------------------|
| `distance` | Standard Gaussian kernel | 2km | 0.1 | ~50-60 |
| `distance_sparse` | Sparse (nearby only) | 2km | 0.5 | ~25-30 |
| `distance_dense` | Dense (farther connections) | 2km | 0.01 | ~85-95 |
| `identity` | No spatial info baseline | N/A | N/A | 0 (self-loop only) |

**Adjacency 생성 과정:**
```
Sensor Lat/Lng coordinates
    ↓ Haversine distance (km)
Pairwise distance matrix D (N x N)
    ↓ Gaussian kernel: exp(-d² / (2σ²)), σ=2km
Weight matrix W (N x N), values in [0, 1]
    ↓ Threshold: W[W < threshold] = 0
Raw adjacency matrix A (N x N)
    ↓ Model-specific normalization
Model-ready format (list of tensors or single tensor)
```

**왜 이 방법들인가:**
- Gaussian kernel adjacency는 DCRNN (ICLR 2018), GWNet (IJCAI 2019) 등에서 표준적으로 사용
- Sigma를 고정하고 threshold만 변화시켜 **sparsity만 독립적으로 제어**
- Identity는 "graph가 아예 없는" baseline으로, spatial information의 가치를 검증

**실측 Average Degree (sigma=2km 기준):**

| Dataset | distance (t=0.1) | sparse (t=0.5) | dense (t=0.01) |
|---------|----------------:|----------------:|---------------:|
| SAN_BERNARDINO (893 nodes) | 50.8 | 26.3 | 84.5 |
| CONTRA_COSTA (773 nodes) | 62.0 | 30.6 | 92.3 |

### 2.2 Models (2개)

| Model | Category | Graph Conv | Adj Handling | Reference |
|-------|----------|-----------|-------------|-----------|
| **GWNet** | C2 (Adaptive+Predefined) | Diffusion + adaptive E₁E₂ᵀ | `supports` (list of 2 tensors) + adaptive | Wu et al., IJCAI 2019 |
| **MTGNN** | C2 (Mixprop+Adaptive) | Mixprop diffusion | `predefined_A` (single tensor) | Wu et al., KDD 2020 |

**GWNet 설정:**
- `addaptadj=True`: adaptive adjacency도 함께 학습 (predefined + adaptive)
- `gcn_bool=True`: GCN convolution 사용
- Predefined adjacency를 교체하면서 adaptive는 자유롭게 학습

**MTGNN 설정:**
- `buildA_true=False`: adaptive graph 비활성화, predefined만 사용
- 이유: predefined adjacency의 효과를 isolated하게 측정하기 위함
- `gcn_true=True`: GCN convolution 사용

### 2.3 Datasets (2개)

xtraffic 데이터셋 사용. 센서별 Lat/Lng 좌표가 있어 adjacency를 직접 제어할 수 있음.

| Dataset | Nodes | Shape | Features | Interval | Year |
|---------|-------|-------|----------|----------|------|
| **SAN_BERNARDINO** | 893 | (105120, 893, 5) | flow, occ, speed, tod, dow | 5min | 2023 |
| **CONTRA_COSTA** | 773 | (105120, 773, 5) | flow, occ, speed, tod, dow | 5min | 2023 |

**Channel order (중요!):**
- ch0: flow (예측 대상)
- ch1: occupancy
- ch2: speed
- ch3: time of day (0~1 normalized)
- ch4: day of week (0~1 normalized)

**FORWARD_FEATURES = [0, 3]** (flow + time_of_day) — 모델 입력
**TARGET_FEATURES = [0]** (flow) — 예측 대상

> **주의:** METR-LA 등 기존 데이터셋은 [0, 1] (speed + tod)을 사용하지만, xtraffic은 channel order가 다르므로 반드시 [0, 3]을 사용해야 함.

**데이터 위치:**
```
datasets/SAN_BERNARDINO/     (또는 datasets/xtraffic/SAN_BERNARDINO/)
├── data.dat                 # 메인 데이터 (memmap)
├── desc.json                # 데이터셋 메타정보
├── metadata.csv             # 센서 좌표 (idx, station_id, Lat, Lng, ...)
└── adj_mx.pkl               # 기존 adjacency (사용하지 않음)

datasets/CONTRA_COSTA/       (또는 datasets/xtraffic/CONTRA_COSTA/)
├── data.dat
├── desc.json
├── metadata.csv
└── adj_mx.pkl
```

**metadata.csv 예시:**
```csv
idx,station_id,Abs_PM,Lat,Lng,Length,Type,Lanes,Name,City,County,Fwy Name,District
0,1205476,0.000,34.01764,-117.35268,0.39,ML,4,,San Bernardino,San Bernardino,I-10,8
1,1205479,0.390,34.01709,-117.34699,0.67,ML,4,,San Bernardino,San Bernardino,I-10,8
...
```

### 2.4 Experiment Matrix

```
4 adj methods × 2 models × 2 datasets × 3 seeds = 48 runs
```

| Group | Configs |
|-------|---------|
| GWNet × SAN_BERNARDINO × 4 adj × 3 seeds | 12 |
| GWNet × CONTRA_COSTA × 4 adj × 3 seeds | 12 |
| MTGNN × SAN_BERNARDINO × 4 adj × 3 seeds | 12 |
| MTGNN × CONTRA_COSTA × 4 adj × 3 seeds | 12 |
| **Total** | **48** |

Seeds: 42, 123, 456

### 2.5 Fair Comparison Protocol

| 항목 | 설정 |
|------|------|
| Data split | 6:2:2 (train/val/test), chronological |
| Lookback | 12 steps (1시간) |
| Horizon | 12 steps (1시간) |
| Max epochs | 100 |
| Early stopping | Patience 15 (val MAE 기준) |
| Normalization | Z-score, `norm_each_channel=False` |
| NULL_VAL | 0.0 |
| Seeds | 3개 (42, 123, 456), mean±std 보고 |
| Gradient clipping | max_norm=5.0 |
| Metrics | MAE (primary), RMSE, MAPE |
| Evaluation horizons | h3 (15min), h6 (30min), h12 (60min) |

**Model-specific hyperparameters:**

| Param | GWNet | MTGNN |
|-------|-------|-------|
| Optimizer | Adam, lr=0.002 | Adam, lr=0.001 |
| LR scheduler | MultiStepLR [1, 50], γ=0.5 | None |
| Batch size (train) | 8 | 32 |
| Batch size (val/test) | 32 | 64 |
| Weight decay | 0.0001 | 0.0001 |
| Curriculum learning | No | Yes (warm=0, cl_epochs=3) |

---

## 3. Code Structure

### 3.1 Files to Transfer

```
project/adjacency_matrix/
├── __init__.py                        # Package init (exports get_adjacency, normalize_for_model)
├── adjacency_methods.py               # Registry + 4 adjacency generation functions
├── normalize.py                       # Model-specific normalization (GWNet, MTGNN)
├── generate_phase1a_configs.py        # Config file generator (generates 48 .py files)
├── run_phase1a.sh                     # Sequential run script (GPU 1)
├── configs/                           # Generated config files (48개)
│   ├── GWNet_SAN_BERNARDINO_distance_s42.py
│   ├── GWNet_SAN_BERNARDINO_distance_sparse_s42.py
│   ├── GWNet_SAN_BERNARDINO_distance_dense_s42.py
│   ├── GWNet_SAN_BERNARDINO_identity_s42.py
│   ├── ... (44 more)
├── experiment_plan.md                 # Full experiment plan (broader scope)
├── literature_review.md               # Literature review
├── model_analysis.md                  # Model architecture analysis
└── EXPERIMENT_CONTEXT.md              # This document
```

**Dependencies (BasicTS 내):**

```
other_baselines/GWNet/
├── arch/
│   ├── __init__.py
│   └── gwnet_arch.py                  # GraphWaveNet model
├── METR-LA.py                         # Reference config

other_baselines/MTGNN/
├── arch/
│   ├── __init__.py
│   └── mtgnn_arch.py                  # MTGNN model
│   └── mtgnn_layers.py                # Mixprop layers
├── runner/
│   ├── __init__.py
│   └── mtgnn_runner.py                # MTGNN custom runner (curriculum learning)
├── METR-LA.py                         # Reference config

basicts/utils/
├── adjacent_matrix_norm.py            # calculate_transition_matrix() etc.
├── serialization.py                   # load_pkl(), load_dataset_desc()
```

### 3.2 Core Module: adjacency_methods.py

```python
# Registry pattern
ADJ_METHOD_REGISTRY = {}

def get_adjacency(method, dataset_name, **kwargs):
    """Main entry point. Returns raw adj matrix (np.ndarray, [N,N])."""
    return ADJ_METHOD_REGISTRY[method](dataset_name, **kwargs)

# 4 registered methods:
# - "distance":        sigma=2km, threshold=0.1  → avg_deg ~50-60
# - "distance_sparse": sigma=2km, threshold=0.5  → avg_deg ~25-30
# - "distance_dense":  sigma=2km, threshold=0.01 → avg_deg ~85-95
# - "identity":        np.eye(N)

# 내부 함수:
# - _load_sensor_coords(dataset_name) → metadata.csv에서 Lat/Lng 읽기
# - _haversine_distance_matrix(lats, lngs) → pairwise km distances
# - _gaussian_kernel(dist, sigma) → exp(-d²/(2σ²))
# - _build_distance_adj(dataset_name, sigma, threshold) → thresholded adj
```

### 3.3 Core Module: normalize.py

```python
def normalize_for_model(raw_adj, model_name):
    """raw adj → model-specific format"""

# GWNet: doubletransition → [D_out⁻¹A, D_in⁻¹Aᵀ] (list of 2 torch.Tensors)
# MTGNN: raw - identity → single torch.Tensor
#   (MTGNN's mixprop internally adds self-loops and row-normalizes)
```

### 3.4 Config Structure

각 config 파일은 self-contained Python 파일로, BasicTS의 표준 config 형식을 따른다.

핵심 구조:
```python
# 1. Adjacency injection
raw_adj = get_adjacency(ADJ_METHOD, DATA_NAME)
supports = normalize_for_model(raw_adj, 'GWNet')  # or 'MTGNN'

# 2. Model parameter에 주입
MODEL_PARAM = {
    "supports": supports,  # GWNet
    # 또는
    "predefined_A": predefined_A,  # MTGNN
    ...
}

# 3. Checkpoint 경로
CFG.TRAIN.CKPT_SAVE_DIR = f'checkpoints/adj_experiment/{MODEL}/{DATA}_{ADJ}_s{SEED}'
```

---

## 4. Setup Instructions (New Server)

### 4.1 Prerequisites

- Python 3.8+ with conda
- PyTorch 1.12+ with CUDA
- BasicTS framework installed and working
- xtraffic datasets (SAN_BERNARDINO, CONTRA_COSTA) with metadata.csv

### 4.2 Step-by-Step Setup

**Step 1: Dataset 준비**

xtraffic 데이터가 없다면, 기존 서버에서 복사:
```bash
# 필요한 파일들:
datasets/SAN_BERNARDINO/data.dat       # (105120, 893, 5) float32
datasets/SAN_BERNARDINO/desc.json      # 메타정보
datasets/SAN_BERNARDINO/metadata.csv   # 센서 좌표 (Lat, Lng)

datasets/CONTRA_COSTA/data.dat         # (105120, 773, 5) float32
datasets/CONTRA_COSTA/desc.json
datasets/CONTRA_COSTA/metadata.csv
```

> **중요:** `metadata.csv`가 반드시 있어야 한다. 이 파일에서 Lat/Lng 좌표를 읽어 adjacency를 생성한다. 원본 위치: `/data/XTraffic/process/data/counties/{COUNTY}/metadata.csv`

**desc.json 내용 확인:**
```json
{
    "num_nodes": 893,
    "regular_settings": {
        "INPUT_LEN": 12,
        "OUTPUT_LEN": 12,
        "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
        "NORM_EACH_CHANNEL": false,
        "RESCALE": true,
        "NULL_VAL": 0.0
    }
}
```

**Step 2: 코드 복사**

```bash
# project/adjacency_matrix/ 전체 복사
scp -r project/adjacency_matrix/ user@slurm-server:/path/to/basicts/project/

# model 코드 확인 (이미 있어야 함)
ls other_baselines/GWNet/arch/gwnet_arch.py
ls other_baselines/MTGNN/arch/mtgnn_arch.py
ls other_baselines/MTGNN/runner/mtgnn_runner.py
```

**Step 3: Config 생성 (또는 기존 configs/ 사용)**

```bash
cd /path/to/basicts
python project/adjacency_matrix/generate_phase1a_configs.py
# → project/adjacency_matrix/configs/ 에 48개 .py 파일 생성
```

**Step 4: 단일 실험 테스트**

```bash
python -c "
from basicts import launch_training
launch_training('project/adjacency_matrix/configs/GWNet_SAN_BERNARDINO_distance_s42.py', gpus='0')
"
```

### 4.3 Slurm Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=adj_GWNet_SB_dist_s42
#SBATCH --output=logs/adj/%x_%j.out
#SBATCH --error=logs/adj/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00

source ~/.conda/etc/profile.d/conda.sh
conda activate basicts
cd /path/to/basicts

CONFIG=$1   # e.g., project/adjacency_matrix/configs/GWNet_SAN_BERNARDINO_distance_s42.py
GPU_ID=0    # slurm이 할당한 GPU는 항상 0번으로 보임

python -c "
from basicts import launch_training
launch_training('${CONFIG}', gpus='${GPU_ID}')
"
```

### 4.4 Batch Submit Script (48 experiments)

```bash
#!/bin/bash
# submit_all_adj_experiments.sh

CONFIG_DIR="project/adjacency_matrix/configs"
mkdir -p logs/adj

for config in "${CONFIG_DIR}"/*.py; do
    name=$(basename "$config" .py)

    # Skip if already completed
    found=$(find checkpoints/adj_experiment -path "*/${name}/*test_metrics.json" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
        echo "[SKIP] $name (already completed)"
        continue
    fi

    echo "[SUBMIT] $name"
    sbatch --job-name="adj_${name}" run_adj_single.sh "$config"
done

echo "All jobs submitted. Check with: squeue -u $USER"
```

---

## 5. Critical Gotchas

### 5.1 NORM_EACH_CHANNEL = False (CRITICAL)

```python
# MUST use False for xtraffic datasets
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']  # Returns False

# NEVER hardcode True:
'norm_each_channel': True   # CATASTROPHIC: dead sensors (std=0) → scaler (0,1) → wrong predictions
```

`True`로 설정하면 per-node normalization을 수행하는데, dead sensor (값이 항상 0인 센서)의 std=0이므로 scaler가 불일치하여 예측이 완전히 망가진다.

### 5.2 FORWARD_FEATURES = [0, 3] (xtraffic)

xtraffic의 channel order: flow(0), occupancy(1), speed(2), tod(3), dow(4)

```python
CFG.MODEL.FORWARD_FEATURES = [0, 3]   # flow + time_of_day
CFG.MODEL.TARGET_FEATURES = [0]        # flow only
```

METR-LA 등 기존 데이터셋은 `[0, 1]`이지만, xtraffic은 tod가 channel 3에 있으므로 `[0, 3]`을 사용해야 한다.

### 5.3 MTGNN buildA_true = False

이 실험에서는 MTGNN의 adaptive graph learning을 끈다:
```python
"buildA_true": False   # predefined adjacency만 사용
```
이유: predefined adjacency의 효과를 isolated하게 측정하기 위함. adaptive가 켜져 있으면 predefined의 영향이 희석된다.

### 5.4 GWNet addaptadj = True

GWNet은 adaptive adjacency를 활성화한 상태:
```python
"addaptadj": True   # adaptive + predefined 모두 사용
```
이유: GWNet의 원래 설계를 유지하면서, predefined만 교체. Adaptive는 자유롭게 학습되므로, predefined의 "추가적 가치"를 측정할 수 있다.

### 5.5 GPU Selection

BasicTS에서 GPU는 `gpus` 파라미터로 지정:
```python
launch_training('config.py', gpus='0')   # GPU 0 사용
launch_training('config.py', gpus='1')   # GPU 1 사용
```
`CUDA_VISIBLE_DEVICES`는 사용하지 말 것 - `gpus` 파라미터가 내부적으로 처리함.

Slurm에서는 할당된 GPU가 항상 device 0으로 매핑되므로 `gpus='0'` 사용.

### 5.6 metadata.csv 경로

adjacency_methods.py는 다음 경로에서 센서 좌표를 읽는다:
```python
meta = pd.read_csv(f"datasets/{dataset_name}/metadata.csv")
```
따라서 `datasets/SAN_BERNARDINO/metadata.csv`와 `datasets/CONTRA_COSTA/metadata.csv`가 반드시 존재해야 한다.

### 5.7 Checkpoint 경로

결과는 다음 경로에 저장된다:
```
checkpoints/adj_experiment/{MODEL}/{DATASET}_{ADJ_METHOD}_s{SEED}/
├── {hash}/
│   ├── best_model.pth
│   ├── test_metrics.json    ← 최종 성능 지표
│   └── ...
```

---

## 6. Results Collection

### 6.1 Test Metrics

각 실험 완료 후 `test_metrics.json`에서 결과를 읽는다:

```bash
find checkpoints/adj_experiment -name "test_metrics.json" -exec echo "=== {} ===" \; -exec cat {} \;
```

**주요 지표:**
- `overall.MAE` — 전체 MAE (primary metric)
- `overall.RMSE` — 전체 RMSE
- `overall.MAPE` — 전체 MAPE
- `3.MAE`, `6.MAE`, `12.MAE` — horizon별 MAE

### 6.2 Results Aggregation Script (추후 작성)

```python
# project/adjacency_matrix/aggregate_results.py
# checkpoints/adj_experiment/**/test_metrics.json → CSV/DataFrame

import json, glob, pandas as pd

results = []
for path in glob.glob("checkpoints/adj_experiment/**/test_metrics.json", recursive=True):
    parts = path.split("/")
    # Parse: model, dataset_adj_seed from path
    metrics = json.load(open(path))
    results.append({
        "model": ...,
        "dataset": ...,
        "adj_method": ...,
        "seed": ...,
        "MAE": metrics["overall"]["MAE"],
        "RMSE": metrics["overall"]["RMSE"],
        "MAPE": metrics["overall"]["MAPE"],
    })

df = pd.DataFrame(results)
# Mean ± Std per (model, dataset, adj_method)
summary = df.groupby(["model", "dataset", "adj_method"]).agg(
    MAE_mean=("MAE", "mean"),
    MAE_std=("MAE", "std"),
).reset_index()
```

### 6.3 Expected Analysis

1. **Main Table**: Model × Adj Method → MAE (mean±std), 각 dataset별
2. **Sparsity Effect**: sparse → standard → dense에 따른 MAE 변화 curve
3. **Identity Baseline**: graph가 없을 때 vs 있을 때 성능 차이
4. **Cross-model Consistency**: 두 모델에서 동일한 adjacency 순위가 나오는지
5. **Statistical Tests**: Paired t-test (동일 seed 매칭), Friedman + Nemenyi

---

## 7. Experiment Config Naming Convention

```
{Model}_{Dataset}_{AdjMethod}_s{Seed}.py
```

Examples:
```
GWNet_SAN_BERNARDINO_distance_s42.py
GWNet_SAN_BERNARDINO_distance_sparse_s123.py
GWNet_CONTRA_COSTA_distance_dense_s456.py
MTGNN_SAN_BERNARDINO_identity_s42.py
```

**Full list (48 configs):**

| # | Config |
|---|--------|
| 1-12 | GWNet × SAN_BERNARDINO × {distance, distance_sparse, distance_dense, identity} × {42, 123, 456} |
| 13-24 | GWNet × CONTRA_COSTA × {distance, distance_sparse, distance_dense, identity} × {42, 123, 456} |
| 25-36 | MTGNN × SAN_BERNARDINO × {distance, distance_sparse, distance_dense, identity} × {42, 123, 456} |
| 37-48 | MTGNN × CONTRA_COSTA × {distance, distance_sparse, distance_dense, identity} × {42, 123, 456} |

---

## 8. Estimated Resources

| Model | Dataset | ~Epoch Time | ~Total (100 epochs, early stop) | GPU Memory |
|-------|---------|-------------|--------------------------------|------------|
| GWNet | SAN_BERNARDINO (893 nodes) | ~4 min | ~2-4 hours | ~8-10 GB |
| GWNet | CONTRA_COSTA (773 nodes) | ~3.5 min | ~2-3.5 hours | ~7-9 GB |
| MTGNN | SAN_BERNARDINO | ~3 min | ~1.5-3 hours | ~6-8 GB |
| MTGNN | CONTRA_COSTA | ~2.5 min | ~1.5-3 hours | ~5-7 GB |

> **Note:** Early stopping (patience=15) 때문에 100 epoch 전에 종료될 가능성 높음. 실제로 30-60 epoch에서 수렴하는 경우가 많음.

**총 소요 시간 (sequential):** ~48 × 3시간 = ~144시간 (6일)
**Slurm parallel (4 GPU):** ~36시간 (1.5일)
**Slurm parallel (8 GPU):** ~18시간

---

## 9. Future Extensions (Phase 1b+)

현재 Phase 1a (pilot)에서 유의미한 결과가 나오면 다음으로 확장:

1. **추가 모델**: DGCRN (C4), STGCN (C1), DCRNN (C1)
2. **추가 dataset**: RIVERSIDE, SACRAMENTO, ALAMEDA (xtraffic 내 다른 county)
3. **Parameter sensitivity**: sigma 변화 (1km, 2km, 5km, 10km)
4. **Data-driven adjacency**: correlation-based (train-only), DTW similarity
5. **Dynamic adjacency**: attention bias/mask for STAEFormer

### normalize.py 확장 시 참고

새 모델을 추가할 때는 normalize.py에 normalizer를 등록:
```python
@register_normalizer("STGCN")
def _stgcn_normalize(raw_adj):
    """STGCN: normalized Laplacian"""
    from basicts.utils.adjacent_matrix_norm import calculate_scaled_laplacian
    lap = calculate_scaled_laplacian(raw_adj)
    return torch.tensor(np.array(lap), dtype=torch.float32)
```

---

## 10. Quick Start Checklist

- [ ] xtraffic 데이터셋 2개 준비 (data.dat, desc.json, metadata.csv)
- [ ] `project/adjacency_matrix/` 전체 복사
- [ ] `other_baselines/GWNet/`, `other_baselines/MTGNN/` 코드 확인
- [ ] `basicts/utils/adjacent_matrix_norm.py` 존재 확인
- [ ] Config 생성: `python project/adjacency_matrix/generate_phase1a_configs.py`
- [ ] 단일 테스트: `launch_training('configs/GWNet_SAN_BERNARDINO_distance_s42.py', gpus='0')`
- [ ] Slurm batch submit
- [ ] 완료 후 `test_metrics.json` 수집 및 분석
