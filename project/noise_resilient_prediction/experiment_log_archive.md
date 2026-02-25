# Experiment Log Archive (2026-01 ~ 2026-02-15)

> **Archived experiments**: Contrastive pre-training (failed), GPTST/STEP/STMAE baselines, MICL v1-v3 (failed).
> These approaches did not improve over baseline. Kept for reference only.
> For active experiments, see [experiment_log.md](experiment_log.md).

---

# Contrastive Pre-training for Traffic Prediction: Experiment Report

> **Dataset**: SAN_BERNARDINO (xtraffic)
> **Period**: 3 months (26,280 samples)
> **Task**: Traffic flow prediction (12 steps → 12 steps)
> **Date**: January 2026

---

## Table of Contents
1. [Research Overview](#1-research-overview)
2. [Experimental Setup](#2-experimental-setup)
3. [Baseline Experiments](#3-baseline-experiments)
4. [Contrastive Pre-training Experiments](#4-contrastive-pre-training-experiments)
5. [Ablation Study: Masking Dimension Comparison](#5-ablation-study-masking-dimension-comparison)
6. [Comprehensive Results Summary](#6-comprehensive-results-summary)
7. [Key Insights & Lessons Learned](#7-key-insights--lessons-learned)
8. [Future Directions](#8-future-directions)
9. [Appendix: Configuration Details](#9-appendix-configuration-details)

---

## 1. Research Overview

### 1.1 Motivation
Traffic prediction 모델의 성능 향상을 위해 **self-supervised contrastive pre-training**을 적용. Labeled data가 풍부한 traffic domain에서도 representation learning이 downstream task 성능을 개선할 수 있는지 검증.

### 1.2 Research Questions
1. Contrastive pre-training이 traffic prediction 성능을 개선하는가?
2. Encoder를 freeze vs fine-tune 중 어느 것이 효과적인가?
3. 어떤 augmentation 전략이 가장 효과적인가?
4. Masking 차원(temporal vs feature vs spatial)에 따른 성능 차이는?
5. 서로 다른 masking 전략의 조합이 시너지 효과를 내는가?

### 1.3 Approach
- **SimCLR-style contrastive learning**: 같은 샘플의 두 augmented view를 positive pair로 학습
- **Transformer encoder**: 시공간 패턴을 학습하는 encoder
- **Two-stage training**: Pre-training (contrastive) → Fine-tuning (downstream STAEformer)

---

## 2. Experimental Setup

### 2.1 Dataset Configuration
```python
DATA_NAME = 'xtraffic/SAN_BERNARDINO'
NUM_NODES = 893  # 센서 수
INPUT_LEN = 12   # 1시간 (5분 간격)
OUTPUT_LEN = 12  # 1시간 예측
TRAIN_VAL_TEST_RATIO = [0.6, 0.2, 0.2]
data_range = (0, 26280)  # 3개월
```

### 2.2 Encoder Architecture
```python
ENCODER_CONFIG = {
    'input_dim': 3,      # flow, time_of_day, day_of_week
    'd_model': 64,       # embedding dimension
    'num_layers': 2,     # transformer layers
    'nhead': 4,          # attention heads
    'dropout': 0.1,
}
```

### 2.3 Downstream Model (STAEformer)
```python
STAEFORMER_CONFIG = {
    'input_dim': 66,     # encoded (64) + tod (1) + dow (1)
    'num_layers': 1,     # lighter than encoder
    'num_heads': 4,
    'feed_forward_dim': 256,
}
```

### 2.4 Training Configuration
| Phase | Epochs | Optimizer | Learning Rate | Scheduler |
|-------|--------|-----------|---------------|-----------|
| Pre-training | 30 | Adam | 1e-3 | CosineAnnealing |
| Fine-tuning | 30 | Adam | Encoder: 1e-5, Downstream: 1e-3 | MultiStepLR [20,25] |

---

## 3. Baseline Experiments

### 3.1 Vanilla STAEformer (Baseline)
> **목적**: Pre-training 없이 STAEformer만 학습했을 때의 성능 기준선

**Configuration:**
- 표준 STAEformer 구조
- input_dim: 3 (flow, tod, dow)
- 30 epochs 학습

**Results:**
| Metric | Value |
|--------|-------|
| Test MAE | **12.10** |
| MAPE | 18.5% |

**Insight:**
- 3개월 데이터로도 합리적인 성능 달성
- 이 값이 모든 pre-training 실험의 기준선

---

## 4. Contrastive Pre-training Experiments

### 4.1 Experiment A: Frozen Encoder
> **목적**: Pre-trained encoder를 완전히 고정하고 downstream만 학습

**Configuration:**
```python
PRETRAINED_ENCODER = {
    'freeze': True,
    'encoder_lr': 0,  # 학습 안함
}
```

**Results:**
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Test MAE | **11.94** | -1.3% |

**Insight:**
- Pre-trained representation이 유의미한 정보 포함
- 그러나 downstream task에 최적화되지 않아 개선폭 제한적
- Encoder가 downstream task와 완전히 align되지 않음

---

### 4.2 Experiment B: Scratch with Temporal Contrastive Loss
> **목적**: Encoder만 contrastive loss로 학습 (downstream 없이)

**Configuration:**
```python
# Pre-training only, no downstream fine-tuning
LOSS = ContrastiveLoss(temperature=0.1)
AUGMENTATION = {
    'jitter': {'sigma': 0.1},
    'scaling': {'sigma': 0.1},
}
```

**Results:**
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Test MAE | **11.89** | -1.7% |

**Insight:**
- Contrastive learning만으로도 baseline 초과 성능
- Augmentation을 통한 invariance 학습이 효과적
- 그러나 downstream task supervision 없이는 한계

---

### 4.3 Experiment C: Fine-tuned Encoder (Best Practice)
> **목적**: Pre-trained encoder를 discriminative LR로 fine-tune

**Configuration:**
```python
PRETRAINED_ENCODER = {
    'freeze': False,
    'encoder_lr': 1e-5,      # 낮은 LR로 조심스럽게
    'unfreeze_after': 0,     # 처음부터 fine-tune
}
DOWNSTREAM_LR = 1e-3         # 100배 높은 LR
```

**Results:**
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Test MAE | **11.86** | -2.0% |

**Insight:**
- **Discriminative learning rate가 핵심**: Encoder는 천천히, downstream은 빠르게
- Pre-trained weights를 보존하면서 task-specific adaptation
- Frozen보다 fine-tune이 더 효과적 (+0.7%)

---

### 4.4 Failed Experiment: GAT-based Spatial Encoder
> **목적**: Graph Attention으로 spatial relationship 학습

**Configuration:**
```python
SPATIAL_ENCODER = {
    'type': 'GAT',
    'heads': 4,
    'hidden_dim': 64,
}
```

**Results:**
| Epoch | Train MAE | Test MAE | Status |
|-------|-----------|----------|--------|
| 1 | 14.07 | - | OK |
| 5 | 17.35 | - | Degrading |
| - | - | - | **Stopped** |

**Insight:**
- Training loss가 증가하는 불안정한 학습
- **원인 분석**:
  1. GAT + Contrastive loss 조합의 gradient 충돌
  2. Spatial attention의 initial randomness
  3. Learning rate가 너무 높음
- **권장 해결책**:
  - Lower LR (1e-4 → 1e-5)
  - Gradient clipping (max_norm=1.0)
  - Warmup scheduler
  - GAT를 pre-train 후 freeze하고 contrastive 적용

---

## 5. Ablation Study: Masking Dimension Comparison

### 5.1 Overview
> **목적**: Augmentation에서 어떤 차원을 마스킹하는 것이 효과적인지 비교

| Dimension | Description | Hypothesis |
|-----------|-------------|------------|
| Temporal | 시간 축 랜덤 마스킹 | 시간 패턴 interpolation 학습 |
| Feature | 특성 채널 마스킹 | Cross-feature relationship 학습 |
| Spatial | 노드 랜덤 마스킹 | 공간적 관계 학습 |

### 5.2 Experiment D: Temporal Masking
> **목적**: 시간 축 마스킹으로 temporal continuity 학습

**Configuration:**
```python
AUGMENTATION_CONFIG = {
    'temporal_masking': {'mask_ratio': 0.15},  # 15% 시간 마스킹
}
```

**Results:**
| Metric | Value | vs Baseline |
|--------|-------|-------------|
| Test MAE | **11.7588** | -2.8% |
| MAE@h3 | 10.98 | |
| MAE@h6 | 11.78 | |
| MAE@h12 | 12.93 | |
| Incident MAE | 12.66 | |
| Normal MAE | 11.53 | |

**Insight:**
- Temporal masking이 시간 연속성 학습에 효과적
- 마스킹된 시간대의 값을 주변 context로부터 추론하도록 학습
- 기존 augmentation (jitter, scaling) 대비 더 명확한 학습 신호

---

### 5.3 Experiment E: Feature Masking ⭐ Best
> **목적**: 특성 채널 마스킹으로 cross-modal relationship 학습

**Configuration:**
```python
AUGMENTATION_CONFIG = {
    'feature_masking': {'mask_ratio': 0.3},  # 30% 특성 마스킹 (~1/3 features)
}
```

**Results:**
| Metric | Value | vs Baseline | vs Temporal |
|--------|-------|-------------|-------------|
| Test MAE | **11.7091** | **-3.2%** | -0.4% |
| MAE@h3 | 10.96 | | ✓ Better |
| MAE@h6 | 11.73 | | ✓ Better |
| MAE@h12 | 12.85 | | ✓ Better |
| Incident MAE | 12.60 | | ✓ Better |
| Normal MAE | 11.48 | | ✓ Better |

**Insight:**
- **Feature masking이 temporal masking보다 우수**
- Traffic 데이터의 특성:
  - flow ↔ time_of_day: 시간대별 교통량 패턴
  - flow ↔ day_of_week: 요일별 패턴
- Feature가 마스킹되면 다른 feature로부터 추론해야 함
- 이 과정에서 **cross-modal representation** 학습
- 예: tod가 마스킹되면 flow 패턴으로 시간대 추론

---

### 5.4 Experiment F: Spatial Masking
> **목적**: 노드 마스킹으로 spatial relationship 학습

**Configuration:**
```python
AUGMENTATION_CONFIG = {
    'spatial_masking': {'mask_ratio': 0.2},  # 20% 노드 마스킹
}
```

**Results (20% masking):**
| Metric | Freeze | Fine-tune | vs Baseline |
|--------|--------|-----------|-------------|
| Test MAE | **11.77** | 11.81 | -2.7% / -2.4% |

**Results (30% masking):**
| Metric | Freeze | Fine-tune | vs Baseline |
|--------|--------|-----------|-------------|
| Test MAE | 11.84 | 11.82 | -2.1% / -2.3% |

**Insight:**
- Spatial masking도 효과적이나 Feature masking보다는 덜 효과적
- **20% masking이 30%보다 약간 우수** (과도한 마스킹은 학습 신호 손실)
- Freeze가 fine-tune보다 약간 좋음 (spatial의 경우)
- 인접 노드로부터 마스킹된 노드 값을 추론하도록 학습
- Feature masking과 complementary한 정보 학습 가능성

---

## 6. Comprehensive Results Summary

### 6.1 Performance Comparison Table

| # | Model | Test MAE | Δ Baseline |
|---|-------|----------|------------|
| **1** | **Feature Masking (finetune)** ⭐ | **11.71** | **-3.2%** |
| 2 | Temporal Masking (finetune) | 11.76 | -2.8% |
| 3 | Spatial Masking 20% (freeze) | 11.77 | -2.7% |
| 4 | Spatial Masking 20% (finetune) | 11.81 | -2.4% |
| 5 | Spatial Masking 30% (finetune) | 11.82 | -2.3% |
| 6 | Spatial Masking 30% (freeze) | 11.84 | -2.1% |
| 7 | Scratch Temporal | 11.89 | -1.7% |
| 8 | STAEformer + Temporal Encoder | 11.91 | -1.6% |
| 9 | Frozen Encoder | 11.94 | -1.3% |
| - | **Baseline STAEformer** | **12.10** | - |

### 6.1.1 Masking 전략 비교

| Masking Type | Best MAE | 개선율 | 비고 |
|--------------|----------|--------|------|
| Feature | 11.71 | -3.2% | Cross-modal learning |
| Temporal | 11.76 | -2.8% | 시간 연속성 학습 |
| Spatial | 11.77 | -2.7% | 공간 관계 학습 |

### 6.2 Incident Analysis (Feature Masking)

| Condition | MAE | Gap |
|-----------|-----|-----|
| Normal | 11.48 | - |
| Incident | 12.60 | +1.12 |

**Insight**: Incident 상황에서도 준수한 성능. Gap이 크지 않아 모델이 비정상 상황에도 robust함.

### 6.3 Key Findings

1. **Pre-training은 효과적**: 모든 pre-training 방법이 baseline 초과
2. **Fine-tuning > Freezing**: Discriminative LR로 fine-tune이 최선 (단, spatial은 예외)
3. **Masking 전략 순위**: Feature > Temporal > Spatial
4. **Masking이 효과적인 augmentation**: Jitter/scaling보다 masking이 우수
5. **Spatial masking 특이점**: Freeze가 fine-tune보다 약간 좋음 (11.77 vs 11.81)

---

## 7. Key Insights & Lessons Learned

### 7.1 What Worked Well

#### ✅ Discriminative Learning Rates
```python
encoder_lr = 1e-5      # Pre-trained weights 보존
downstream_lr = 1e-3   # 빠른 adaptation
```
- 100배 차이가 적절
- Encoder의 learned representation을 파괴하지 않으면서 task adaptation

#### ✅ Feature Masking Augmentation
```python
augmentation = {'feature_masking': {'mask_ratio': 0.3}}
```
- Traffic data의 multi-modal 특성 활용
- Flow-temporal correlation 학습에 효과적

#### ✅ Cosine Annealing for Pre-training
- Pre-training에서 smooth한 LR decay
- Local minima 탈출에 도움

#### ✅ MultiStepLR for Fine-tuning
```python
milestones = [20, 25]  # 30 epoch 중
gamma = 0.1
```
- 후반부에 급격한 decay로 수렴 안정화
- Epoch 20 이후 성능 수렴 관찰됨

### 7.2 What Didn't Work

#### ❌ GAT-based Spatial Encoder
- Contrastive loss와 graph attention의 gradient 충돌
- 학습 불안정 (loss 증가)
- **해결 방안**: Pre-train GAT separately, then combine

#### ❌ High Learning Rate for Encoder
- Pre-trained weights 파괴
- Catastrophic forgetting 발생

### 7.3 Surprising Findings

1. **Feature > Temporal masking**: 직관과 달리 feature 차원 마스킹이 더 효과적
2. **Simple encoder works**: 2-layer transformer로도 충분한 representation 학습
3. **3 months is enough**: 3개월 데이터로도 의미 있는 pre-training 가능

---

## 8. Future Directions

### 8.1 Immediate Next Steps

#### 🔬 Combined Masking (진행 중)
```python
augmentation = {
    'temporal_masking': {'mask_ratio': 0.1},
    'feature_masking': {'mask_ratio': 0.2},
}
```
- 두 masking의 synergy 효과 검증
- 현재 실험 진행 중 (combined_v1, combined_v2)

#### 🔬 Feature + Spatial Combined
```python
augmentation = {
    'feature_masking': {'mask_ratio': 0.2},
    'spatial_masking': {'mask_ratio': 0.1},
}
```
- Feature (cross-modal) + Spatial (공간 관계) 결합
- 예상: 두 관점의 complementary learning

#### 🔬 Multi-dataset Pre-training
- 여러 county 데이터로 pre-train
- Cross-domain transfer 효과 검증

### 8.2 Architecture Improvements

#### 🔬 Stabilized GAT Encoder
```python
config = {
    'lr': 1e-5,
    'warmup_epochs': 5,
    'gradient_clip': 1.0,
}
```
- Spatial structure 활용
- 안정적인 학습 방법 필요
- 현재 실험 진행 중이나 성능 저조 (MAE ~14.9)

### 8.3 Training Strategies

#### 🔬 Curriculum Learning
1. Easy samples (normal traffic) → Hard samples (incidents)
2. Short sequences → Long sequences

#### 🔬 Multi-task Pre-training
- Contrastive + Reconstruction loss 결합
- MAE-style masked prediction

### 8.4 Evaluation Extensions

#### 🔬 Other Datasets
- METR-LA, PEMS-BAY로 일반화 검증
- ETT 등 다른 시계열 도메인 적용

#### 🔬 Few-shot Evaluation
- Pre-trained encoder의 few-shot learning 능력 평가
- 1-week, 2-week 데이터로 fine-tune

---

## 9. Appendix: Configuration Details

### 9.1 File Structure
```
baselines/ContextContrastive/
├── arch/
│   ├── __init__.py
│   ├── encoder.py              # Transformer encoder
│   └── pretrain_model.py       # Contrastive pre-training model
├── loss/
│   ├── __init__.py
│   └── contrastive.py          # NT-Xent loss
├── runner/
│   ├── __init__.py
│   └── finetuning_runner.py    # Discriminative LR runner
├── SAN_BERNARDINO/
│   ├── pretrain.py             # Original pre-training config
│   ├── pretrain_temporal_mask.py
│   ├── pretrain_feature_mask.py
│   ├── STAEformer_finetune.py
│   ├── STAEformer_finetune_temporal_mask.py
│   └── STAEformer_finetune_feature_mask.py
└── EXPERIMENTS.md              # This document
```

### 9.2 Checkpoint Locations
```
checkpoints/
├── ContextContrastive_baseline_3mo/              # Baseline STAEformer
├── ContextContrastive_pretrain_3mo/              # Original pre-training
├── ContextContrastive_temporal_mask_3mo/         # Temporal masking pre-train
├── ContextContrastive_feature_mask_3mo/          # Feature masking pre-train
├── ContextContrastive_spatial_mask_3mo/          # Spatial 20% pre-train
├── ContextContrastive_spatial_mask30_3mo/        # Spatial 30% pre-train
├── ContextContrastive_finetune_temporal_mask_3mo/
├── ContextContrastive_finetune_feature_mask_3mo/ # Best ⭐
├── ContextContrastive_finetune_spatial_mask_3mo/
├── ContextContrastive_finetune_spatial_mask30_3mo/
├── ContextContrastive_freeze_spatial_mask_3mo/   # Spatial freeze
├── ContextContrastive_freeze_spatial_mask30_3mo/
├── ContextContrastive_combined_v1_3mo/           # Combined (진행 중)
├── ContextContrastive_combined_v2_3mo/           # Combined (진행 중)
└── ContextContrastive_finetune_GAT_3mo/          # GAT (진행 중)
```

### 9.3 Running Experiments

#### Pre-training
```bash
# Temporal masking
python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_temporal_mask.py', gpus='0')"

# Feature masking
python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/pretrain_feature_mask.py', gpus='1')"
```

#### Fine-tuning
```bash
# After pre-training completes
python -c "from basicts import launch_training; launch_training('baselines/ContextContrastive/SAN_BERNARDINO/STAEformer_finetune_feature_mask.py', gpus='0')"
```

### 9.4 Key Hyperparameters Reference

| Parameter | Pre-training | Fine-tuning |
|-----------|--------------|-------------|
| Batch size | 16 | 16 |
| Epochs | 30 | 30 |
| Optimizer | Adam | Adam |
| Weight decay | 1e-4 | 3e-4 |
| LR (encoder) | 1e-3 | 1e-5 |
| LR (downstream) | - | 1e-3 |
| Temperature (contrastive) | 0.1 | - |
| Gradient clip | 5.0 | - |

---

---

## 10. Pre-training Baseline Comparisons (New Section)

### 10.1 Overview
> **목적**: 기존 representation learning 방법론들과 성능 비교

비교 대상 모델:
1. **GPT-ST**: Spatio-temporal GPT with adaptive masking
2. **STEP**: Pre-training Enhanced Spatial-temporal Graph Neural Network
3. **STMAE**: Spatio-Temporal Masked Autoencoder

모두 two-stage pre-training 방식:
- Stage 1: Self-supervised pre-training (masked reconstruction)
- Stage 2: Supervised fine-tuning (forecasting)

### 10.2 Experiment Status (2026-01-31)

| Model | Stage | Status | Notes |
|-------|-------|--------|-------|
| GPTST | Pretrain | ✅ Completed | MAE 117.85 (reconstruction) |
| GPTST | Finetune | ⏳ Queued | GPU 1 |
| STEP | Pretrain | 🏃 Running | Epoch 3/30, ETA ~8PM |
| STEP | Finetune | ⏳ Queued | GPU 0, waiting for pretrain |
| STMAE | Pretrain | ⏳ Queued | GPU 0 |
| STMAE | Finetune | ⏳ Queued | GPU 1, waiting for pretrain |

### 10.3 GPTST Pre-training Results

**Configuration:**
```python
MODEL_PARAM = {
    "hidden_dim": 64,
    "embed_dim": 16,
    "HS": 4,  # Spatial hyperedge heads
    "HT": 4,  # Temporal hyperedge heads
    "num_route": 3,  # Capsule routing
}
NUM_EPOCHS = 30
```

**Pre-training Results (Masked Reconstruction):**
| Metric | Value |
|--------|-------|
| Test MAE | 117.85 |
| RMSE | 154.88 |

Note: High MAE is expected for masked reconstruction task (predicting masked patches).

### 10.4 STEP Pre-training (In Progress)

**Configuration:**
```python
# TSFormer settings
PATCH_SIZE = 12
NUM_PATCHES = 168  # 7 days of 5-min data
LONG_HISTORY_LEN = 2016  # 12 * 168

MODEL_PARAM = {
    "tsformer_embed_dim": 96,
    "tsformer_num_heads": 4,
    "tsformer_mask_ratio": 0.75,
    "tsformer_encoder_depth": 4,
}
```

**Progress:**
- Epoch 3/30 running
- Current MAE: ~115 (reconstruction)
- ~7 min per epoch (2016 timestep input is slow)

### 10.5 Next Steps

1. Wait for all pretrain/finetune to complete
2. Collect final test_metrics.json for each model
3. Compare with baseline (MAE 12.10) and best feature masking (MAE 11.71)
4. Document insights about backbone-agnostic design

---

## Changelog

| Date | Experiment | Result |
|------|------------|--------|
| 2026-01-31 | GPTST Pretrain | MAE 117.85 (recon) |
| 2026-01-31 | STEP Pretrain | Running (Epoch 3/30) |
| 2026-01-31 | STMAE Pretrain | Queued |
| 2026-01-30 | Baseline STAEformer | MAE 12.10 |
| 2026-01-30 | Frozen encoder | MAE 11.94 |
| 2026-01-30 | Fine-tuned encoder | MAE 11.86 |
| 2026-01-30 | Temporal masking | MAE 11.76 |
| 2026-01-30 | Feature masking | MAE 11.71 ⭐ |
| 2026-01-31 | Spatial masking 20% (freeze) | MAE 11.77 |
| 2026-01-31 | Spatial masking 20% (finetune) | MAE 11.81 |
| 2026-01-31 | Spatial masking 30% (freeze) | MAE 11.84 |
| 2026-01-31 | Spatial masking 30% (finetune) | MAE 11.82 |
| 2026-01-31 | GAT finetune (진행 중) | MAE ~14.9 ❌ |

---

## 11. Pre-training Model Implementation Details (BasiCTS 적용)

> **목적**: 기존 pre-training 모델들(GPT-ST, STEP, STMAE)을 BasiCTS 프레임워크에 맞게 구현하면서 발생한 이슈와 해결 방법 문서화

### 11.1 Overview: Backbone-Agnostic Design

모든 pre-training 모델을 **backbone-agnostic** 방식으로 설계:
- Pre-trained encoder + 교체 가능한 downstream predictor
- String-based backbone registry로 pickle 안전성 확보
- Two-stage training: Pretrain → Finetune

```
baselines/
├── GPTST/           # GPT-ST 구현
├── STEP/            # STEP (TSFormer) 구현
├── STMAE/           # ST-MAE 구현
└── [각 모델]/
    ├── arch/        # 모델 아키텍처
    ├── runner/      # Pre-train / Fine-tune runner
    ├── loss/        # Custom loss functions
    └── *.py         # Config files
```

---

### 11.2 GPT-ST Implementation

#### 11.2.1 Architecture Overview

```python
# GPT-ST 핵심 구조
GPTSTModel
├── STHCN_Encoder (Spatio-Temporal Hypergraph Capsule Network)
│   ├── HyperGraphSpatial (HS heads)
│   ├── HyperGraphTemporal (HT heads)
│   └── Capsule Routing (num_route iterations)
├── STHCN_Decoder (Reconstruction)
└── AdaptiveMasking (curriculum learning)

GPTSTForForecasting (Wrapper for BasiCTS)
├── GPTSTModel (encoder)
├── Fusion layer (pretrain + input)
└── Prediction head
```

#### 11.2.2 Key Adaptations for BasiCTS

1. **Forward signature 통일**:
```python
def forward(self, history_data, future_data=None, batch_seen=None, epoch=None, train=True, **kwargs):
    # BasiCTS runner expects this signature
```

2. **Pretrained weight loading 지원**:
```python
class GPTSTForForecasting:
    def __init__(self, ..., pretrained_path=None, freeze_encoder=False):
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
        if freeze_encoder:
            self._freeze_encoder()
```

3. **Mode 분리**: `mode='pretrain'` vs `mode='finetune'`

#### 11.2.3 Errors and Fixes

**Error 1: `unexpected keyword argument 'pretrained_path'`**
```
TypeError: GPTSTForForecasting.__init__() got an unexpected keyword argument 'pretrained_path'
```
- **원인**: Config에서 `pretrained_path` 전달하지만 모델에서 지원 안함
- **해결**: `GPTSTForForecasting.__init__`에 `pretrained_path`, `freeze_encoder` 파라미터 추가
```python
# baselines/GPTST/arch/enhance_model.py
def __init__(self, ..., pretrained_path: str = None, freeze_encoder: bool = False):
    # ... existing init ...
    if pretrained_path is not None:
        self._load_pretrained(pretrained_path)
    if freeze_encoder:
        self._freeze_encoder()
```

---

### 11.3 STEP (TSFormer) Implementation

#### 11.3.1 Architecture Overview

```python
# TSFormer 핵심 구조 (Pre-training)
TSFormer
├── PatchEmbedding (시계열 → 패치)
├── PositionalEncoding (learnable)
├── MaskGenerator (75% random masking)
├── TransformerEncoder (4 layers)
├── TransformerDecoder (1 layer)
└── OutputLayer (reconstruction)

# STEP 전체 구조 (Fine-tuning)
STEP
├── TSFormer (pre-trained encoder)
├── DiscreteGraphLearning (adaptive adj)
└── GraphWaveNet (downstream predictor)
```

#### 11.3.2 Key Configuration

```python
# Pre-training config
PATCH_SIZE = 12       # 1시간 (5분 × 12)
NUM_PATCHES = 168     # 7일
INPUT_LEN = 2016      # 12 × 168 = 7일 히스토리
MASK_RATIO = 0.75     # 75% 마스킹

MODEL_PARAM = {
    "embed_dim": 96,
    "num_heads": 4,
    "encoder_depth": 4,
    "decoder_depth": 1,
}
```

#### 11.3.3 Critical Error: NaN (MAE=0.0000) ⚠️

**증상**:
```
train/MAE: 0.0000, train/RMSE: 0.0000
val/MAE: 0.0000, val/RMSE: 0.0000
```

**원인 분석**:
1. `TransformerLayers.forward()`에서 `src = src * math.sqrt(self.embed_dim)` (line 164)
2. `embed_dim=96` → `sqrt(96) ≈ 9.8` 배 스케일링
3. SAN_BERNARDINO 데이터: 값 범위 0-1318, 많은 노드가 std=1.0 (기본값)
4. ZScore 정규화 후에도 일부 노드에서 큰 값 유지
5. 스케일링 후 attention score 폭발 → softmax overflow → NaN

**해결 방법**: Pre-encoder LayerNorm 추가
```python
# baselines/STEP/arch/tsformer.py

class TSFormer(nn.Module):
    def __init__(self, ...):
        # ... existing code ...

        # Pre-norm layer to stabilize values before transformer
        # (prevents NaN from attention overflow)
        self.pre_encoder_norm = nn.LayerNorm(embed_dim)  # ← 추가

        self.encoder = TransformerLayers(...)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def encoding(self, x, mask=True):
        # ... patchify, positional encoding, masking ...

        # Pre-normalize before encoder to prevent NaN
        encoder_input = self.pre_encoder_norm(encoder_input)  # ← 추가

        hidden_states = self.encoder(encoder_input)
        # ...
```

**결과**:
- Before: `train/MAE: 0.0000` (NaN)
- After: `train/MAE: 127.15`, `val/MAE: 110.40` ✅

**교훈**:
1. Transformer에서 sqrt(d_model) 스케일링은 주의 필요
2. 입력 데이터 분포가 넓으면 pre-normalization 필수
3. Gradient clipping만으로는 forward pass NaN 해결 불가 (backprop이 아닌 forward에서 발생)

---

### 11.4 STMAE Implementation

#### 11.4.1 Architecture Overview

```python
# STMAE 핵심 구조
STMAE
├── FeatureMasking (temporal patches)
├── StructureMasking (graph edges via random walk)
├── Encoder (backbone-agnostic: AGCRN, GWNet 등)
├── FeatureDecoder (masked patch reconstruction)
└── StructureDecoder (edge reconstruction)

# Loss = FeatureLoss + StructureLoss
```

#### 11.4.2 Backbone Registry Pattern

STMAE는 다양한 backbone을 지원하도록 설계:

```python
# baselines/STMAE/arch/stmae_arch.py

BACKBONE_REGISTRY = {}

def get_backbone_class(backbone: Union[str, type]) -> type:
    """String → Class 변환 (pickle 안전)"""
    if backbone is None:
        return None
    if isinstance(backbone, str):
        if backbone in BACKBONE_REGISTRY:
            return BACKBONE_REGISTRY[backbone]
        if backbone == "AGCRN":
            from baselines.AGCRN.arch import AGCRN
            return AGCRN
        if backbone == "GWNet":
            from baselines.GWNet.arch import GWNet
            return GWNet
        if backbone == "STAEformer":
            from baselines.STAEformer.arch import STAEformer
            return STAEformer
        raise ValueError(f"Unknown backbone: {backbone}")
    return backbone
```

**Config 사용 예**:
```python
# ❌ 잘못된 방식 (pickle 에러)
MODEL_PARAM = {
    "backbone_class": AGCRN,  # Class reference → pickle 불가
}

# ✅ 올바른 방식 (pickle 안전)
MODEL_PARAM = {
    "backbone_class": "AGCRN",  # String → runtime에 resolve
}
```

#### 11.4.3 Errors and Fixes

**Error 1: `cannot pickle 'module' object`**
```
TypeError: cannot pickle 'module' object
```
- **원인**: RQ worker가 config를 pickle할 때 class reference 직렬화 실패
- **해결**: `get_backbone_class()` 함수로 string → class 변환

**Error 2: `cannot import name 'stmae_loss'`**
```
ImportError: cannot import name 'stmae_loss' from 'baselines.STMAE.loss'
```
- **원인**: `loss/__init__.py`에서 `stmae_loss` export 누락
- **해결**:
```python
# baselines/STMAE/loss/__init__.py
from .stmae_loss import (
    STMAELoss,
    StructureLoss,
    FeatureLoss,
    stmae_pretrain_loss,
    stmae_finetune_loss,
)

# Alias for compatibility
stmae_loss = stmae_pretrain_loss  # ← 추가

__all__ = [
    'STMAELoss', 'StructureLoss', 'FeatureLoss',
    'stmae_pretrain_loss', 'stmae_finetune_loss', 'stmae_loss',
]
```

---

### 11.5 Common Issues & Solutions

#### 11.5.1 Gradient Explosion

**증상**: Loss가 갑자기 NaN 또는 inf로 변함

**해결**:
```python
# Config에 gradient clipping 추가
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
```

**원본 구현 참고**:
- STEP: 원본에서 gradient clipping 사용 (`max_norm=5.0`)
- STMAE/AGCRN: 원본에서는 미사용, SAN_BERNARDINO에서는 필요

#### 11.5.2 Forward Pass NaN (Attention Overflow)

**증상**:
- Loss = 0.0000 또는 NaN
- Gradient clipping 적용해도 해결 안됨

**원인**:
- Transformer attention에서 softmax 입력값이 너무 커서 overflow
- 특히 `sqrt(d_model)` 스케일링이 문제

**해결**:
```python
# Transformer 입력 전에 LayerNorm 추가
self.pre_encoder_norm = nn.LayerNorm(embed_dim)

# encoding 함수에서
encoder_input = self.pre_encoder_norm(encoder_input)  # ← 추가
hidden_states = self.encoder(encoder_input)
```

#### 11.5.3 Pickle Serialization Issues (RQ)

**증상**:
```
TypeError: cannot pickle 'module' object
TypeError: cannot pickle 'type' object
```

**원인**: Config에 class reference 포함

**해결**:
1. Class reference 대신 string 사용
2. Runtime에 string → class 변환하는 registry pattern 구현

```python
# Config
backbone_class = "AGCRN"  # ✅ Not: AGCRN

# Model init
def __init__(self, ..., backbone_class):
    backbone_cls = get_backbone_class(backbone_class)
    self.backbone = backbone_cls(**backbone_params)
```

#### 11.5.4 Memory Issues

**증상**: CUDA out of memory

**해결**:
1. Batch size 줄이기: 32 → 16 → 8
2. Long sequence 모델 (STEP)은 메모리 많이 필요
   - INPUT_LEN=2016 (7일) → ~60GB GPU memory
3. 병렬 실험 시 GPU 여유 확인 필요

---

### 11.6 Implementation Checklist

새로운 pre-training 모델 추가 시 체크리스트:

- [ ] Forward signature 통일: `(history_data, future_data=None, batch_seen=None, epoch=None, **kwargs)`
- [ ] Mode 분리: `mode='pretrain'` / `mode='finetune'`
- [ ] Pretrained weight loading: `pretrained_path`, `freeze_encoder` 파라미터
- [ ] Backbone registry: String-based class resolution
- [ ] Gradient clipping: `CFG.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}`
- [ ] Input normalization: Transformer 사용 시 pre-encoder LayerNorm 고려
- [ ] Loss export: `__init__.py`에서 loss function export 확인
- [ ] Runner 구현: Pretrain runner + Finetune runner

---

### 11.7 File Locations Reference

```
baselines/GPTST/
├── arch/
│   ├── __init__.py
│   ├── gptst_arch.py        # GPTSTModel
│   ├── enhance_model.py     # GPTSTForForecasting (wrapper)
│   ├── sthcn.py             # STHCN encoder/decoder
│   └── modules.py           # HyperGraph layers
├── runner/
│   ├── gptst_pretrain_runner.py
│   └── gptst_finetune_runner.py
├── loss/
│   └── gptst_loss.py
├── SAN_BERNARDINO_pretrain.py
└── SAN_BERNARDINO_finetune.py

baselines/STEP/
├── arch/
│   ├── __init__.py
│   ├── tsformer.py          # TSFormer (patched with pre_encoder_norm)
│   ├── step_arch.py         # Full STEP model
│   ├── graphwavenet.py      # Downstream predictor
│   └── discrete_graph_learning.py
├── runner/
│   ├── tsformer_runner.py   # Pre-train runner
│   └── step_runner.py       # Fine-tune runner
├── loss/
│   └── step_loss.py
├── TSFormer_SAN_BERNARDINO.py    # Pre-train config
└── STEP_SAN_BERNARDINO.py        # Fine-tune config

baselines/STMAE/
├── arch/
│   ├── __init__.py
│   ├── stmae_arch.py        # Main STMAE (with backbone registry)
│   ├── masking.py           # Feature/Structure masking
│   └── decoders.py          # Feature/Structure decoders
├── runner/
│   ├── stmae_pretrain_runner.py
│   └── stmae_finetune_runner.py
├── loss/
│   └── stmae_loss.py        # Combined loss
├── SAN_BERNARDINO_pretrain.py
└── SAN_BERNARDINO_finetune.py
```

---

## Changelog (Updated)

| Date | Experiment | Result | Notes |
|------|------------|--------|-------|
| 2026-02-01 | TSFormer NaN Fix | ✅ Fixed | Added pre_encoder_norm |
| 2026-02-01 | TSFormer Pretrain (restart) | Running | Epoch 2/30, MAE 110.4 |
| 2026-02-01 | STMAE Finetune | Running | Epoch 15/30, MAE 12.6 |
| 2026-02-01 | GPTST Finetune | Queued | Waiting for GPU |
| 2026-01-31 | GPTST Pretrain | ✅ Completed | MAE 117.85 (recon) |
| 2026-01-31 | STMAE Pretrain | ✅ Completed | |
| 2026-01-31 | Baseline STAEformer | MAE 12.10 | |
| 2026-01-30 | Feature masking | MAE 11.71 ⭐ | Best so far |

---

---


---

## Mutual Information Analysis Research (2026-02-14)

### 목적
Baseline STAEformer의 내부 representation이 flow magnitude (scale) vs temporal pattern (shape)에 얼마나 의존하는지 측정하는 방법론 조사.

### 배경
- 모든 encoder 실험(contrastive, cross-variable)이 baseline 대비 개선 실패
- 가설: Baseline이 이미 scale-invariant representation을 학습하면 encoder가 도움이 안 될 수 있음
- 필요: Scale-dependence를 정량적으로 측정하는 방법

### 조사 방법론 (4가지)

#### 1. Linear Probing ⭐ 권장
**원리**: Frozen representation에서 선형 회귀로 mean_flow 예측 → R² 측정

**장점**:
- 구현 간단 (1-2시간)
- 해석 명확 (R² 점수)
- 계산 빠름 (분 단위)

**사용법**:
```python
# 1. Extract hidden representations (frozen model)
hiddens = model.encoder(inputs)  # (N, d_model)

# 2. Train linear probe
probe = nn.Linear(d_model, 1)
pred_mean_flow = probe(hiddens)
R2 = r2_score(true_mean_flow, pred_mean_flow)
```

**해석 기준**:
| R² | Scale-dependence | 의미 |
|---|---|---|
| > 0.8 | 높음 | Representation이 scale에 강하게 의존 |
| 0.5-0.8 | 중간 | 부분적 의존 |
| 0.2-0.5 | 낮음 | 약한 의존 |
| < 0.2 | 거의 없음 | Scale-invariant |

**Paper**: Alain & Bengio, 2016 - "Understanding Intermediate Layers Using Linear Classifier Probes"

---

#### 2. CKA (Centered Kernel Alignment)
**원리**: 같은 노드의 다른 연도(다른 scale) representation 간 유사도 측정

**장점**:
- 해석 명확 (0-1 유사도)
- 학습 불필요 (closed-form)
- Layer-wise 분석 가능

**사용법**:
```python
from ckatorch import cka

# 같은 노드, 다른 연도 representation 추출
repr_2022 = model.extract(node_i, year=2022)
repr_2023 = model.extract(node_i, year=2023)

# CKA similarity
similarity = cka(repr_2022, repr_2023, kernel='linear')
# High similarity (>0.8) → scale-invariant
# Low similarity (<0.5) → scale-dependent
```

**설치**: `pip install ckatorch`

**Paper**: Kornblith et al., 2019 - "Similarity of Neural Network Representations Revisited" (ICML)

---

#### 3. MINE (Mutual Information Neural Estimation)
**원리**: Neural network로 I(mean_flow; hidden_repr) 추정

**단점**:
- 구현 복잡 (2-3일)
- Statistics network 학습 필요
- 해석 어려움 (bits 절대값의 의미 불명확)

**사용 권장 상황**: 엄격한 information-theoretic 증명이 필요한 논문 작성 시

**Paper**: Belghazi et al., 2018 - "Mutual Information Neural Estimation" (ICML)

**구현**: gtegner/mine-pytorch (GitHub)

---

#### 4. Disentanglement Metrics (MIG, DCI)
**원리**: VAE latent의 disentanglement 측정 (factor별 독립성)

**단점**:
- Ground truth factors 정의 필요
- VAE 전용 설계 (forecasting과 misaligned)
- 계산 비용 높음 (3-5일)

**사용 권장 상황**: Generative model 연구, interpretability 논문

**Papers**:
- Chen et al., 2018 - "Isolating Sources of Disentanglement in VAEs" (NeurIPS)
- Eastwood & Williams, 2018 - "A Framework for Quantitative Evaluation"

---

### 권장 실험 프로토콜

#### Phase 1: Linear Probing (우선 실행)
```python
# Baseline STAEformer 체크포인트 로드
ckpt = 'checkpoints/STAEformer_5ch_unmasked/.../best.pth'
model.load_state_dict(torch.load(ckpt))
model.eval()

# Hidden representations 추출
hiddens_list = []
mean_flows_list = []

with torch.no_grad():
    for batch in test_loader:
        hidden = model.encoder(batch['input'])  # (B, T, N, d_model)
        hidden_pooled = hidden.mean(dim=1)  # (B, N, d_model)
        mean_flow = batch['input'][:, :, :, 0].mean(dim=1)  # (B, N)
        
        hiddens_list.append(hidden_pooled.cpu())
        mean_flows_list.append(mean_flow.cpu())

hiddens = torch.cat(hiddens_list, dim=0).reshape(-1, d_model)
mean_flows = torch.cat(mean_flows_list, dim=0).reshape(-1, 1)

# Train/test split for probe
X_train, X_test, y_train, y_test = train_test_split(
    hiddens.numpy(), mean_flows.numpy(), test_size=0.2, random_state=42
)

# Train linear probe
probe = nn.Linear(d_model, 1)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

for epoch in range(100):
    pred = probe(torch.FloatTensor(X_train))
    loss = F.mse_loss(pred, torch.FloatTensor(y_train))
    loss.backward()
    optimizer.step()

# Evaluate
probe.eval()
with torch.no_grad():
    y_pred = probe(torch.FloatTensor(X_test)).numpy()
    R2 = r2_score(y_test, y_pred)

print(f"[STAT:R2_mean_flow_probe] {R2:.4f}")

if R2 > 0.8:
    print("[FINDING] Baseline is highly scale-dependent (R² > 0.8)")
    print("[RECOMMENDATION] Try scale-aware pretraining or normalization")
elif R2 < 0.2:
    print("[FINDING] Baseline is already scale-invariant (R² < 0.2)")
    print("[LIMITATION] Encoder pretraining may not help (consistent with experiments)")
```

**예상 소요 시간**: 1-2 hours

---

#### Phase 2: CKA Analysis (R² > 0.5인 경우)
```python
from ckatorch import cka

# Extract layer-wise representations
layers = ['encoder_layer_0', 'encoder_layer_1', 'decoder_layer_0']
repr_dict = {}

for layer_name in layers:
    hiddens = extract_layer_activations(model, layer_name, test_loader)
    repr_dict[layer_name] = hiddens

# CKA heatmap (layer similarity)
cka_matrix = np.zeros((len(layers), len(layers)))
for i, layer_i in enumerate(layers):
    for j, layer_j in enumerate(layers):
        cka_matrix[i, j] = cka(repr_dict[layer_i], repr_dict[layer_j], kernel='linear')

# Visualize
plt.imshow(cka_matrix, cmap='viridis')
plt.colorbar()
plt.savefig('eda/cka_layer_similarity.png')

# Cross-year similarity (same node, different magnitude)
node_idx = 100  # Example functional node
repr_2022 = repr_dict['encoder_layer_0'][year_2022_mask, :, node_idx, :]
repr_2023 = repr_dict['encoder_layer_0'][year_2023_mask, :, node_idx, :]

cka_score = cka(repr_2022, repr_2023, kernel='linear')
print(f"[STAT:CKA_cross_year] {cka_score:.4f}")

if cka_score > 0.8:
    print("[FINDING] High cross-year similarity → scale-invariant")
else:
    print("[FINDING] Low similarity → representation changes with magnitude")
```

**예상 소요 시간**: 0.5 days

---

### 핵심 인사이트

1. **Linear probing이 첫 번째 선택**: 간단하고 빠르며 해석 명확
2. **CKA는 layer-wise 분석에 유용**: 어떤 layer에서 scale 정보가 encode되는지 파악
3. **MINE/Disentanglement는 overkill**: 논문 작성 시에만 필요
4. **실험 가설 검증**:
   - H0 (R² < 0.2): Baseline이 이미 scale-invariant → encoder 불필요
   - H1 (R² > 0.5): Scale-dependent → scale-aware pretraining이 도움될 수 있음

### 관련 자료

**Papers**:
- [MINE: Mutual Information Neural Estimation](https://arxiv.org/abs/1801.04062)
- [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)
- [Understanding Intermediate Layers Using Linear Classifier Probes](https://arxiv.org/pdf/1610.01644)
- [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810)

**Tools**:
- PyTorch: sklearn.metrics.r2_score
- CKA: `pip install ckatorch` (RistoAle97/centered-kernel-alignment)
- MINE: gtegner/mine-pytorch, mohith-sakthivel/mine-pytorch
- Disentanglement: YannDubs/disentangling-vae

**Report**: Full research report at `.omc/scientist/reports/20260214_144043_mutual_information_analysis_report.md`

---

## MICL v1: Missing-Invariant Contrastive Learning (2026-02-15)

### Motivation
Previous encoder experiments (contrastive, cross-variable, random projection) all failed to improve prediction accuracy. New hypothesis: instead of improving clean accuracy, target **robustness to missing pattern shifts** at test time. Train an encoder that produces representations invariant to missing data patterns via contrastive learning with missing pattern augmentation.

### Design
- **Asymmetric teacher-student**: teacher=original data (stop-gradient), student=augmented with additional missing patterns
- **SimSiam-style loss**: cosine similarity from predictor(z_student) → stopgrad(z_teacher)
- **MLPEncoder**: Pointwise MLP (8→64→32), registered in encoder registry
- **MissingAugmentation**: node_death, intermittent, block with rate Uniform(0.1, 0.5)
- **Observation-weighted loss**: per-node weight = mean of mask after augmentation
- **Downstream**: STAEformer with encoder output (32-dim) + tod/dow, fine-tuned end-to-end

### Pretraining
- Dataset: SAN_BERNARDINO_MASK (8ch), 30 epochs, batch_size=32
- SSL loss: 0.049 → 0.0008 (converged by epoch 2)
- Collapse check: NOT collapsed (z_std=5.0, cosine sim range 0.29-1.0 with real data)
- Checkpoint: `checkpoints/MICL_pretrain/SAN_BERNARDINO_MASK_30_12_12/f60dffe8.../`

### Downstream Results
- Overall MAE: 11.943 (baseline STAEformer 5ch: 12.10)
- Per-category masked MAE: functional=13.663, partial=5.926, major=10.473, dead=0.431
- Checkpoint: `checkpoints/MICL_downstream/SAN_BERNARDINO_MASK_30_12_12/b4684c0a.../`

### Phase 2: Missing Pattern Shift Robustness

**Clean MAE (functional, observed)**:

| Model | Clean MAE |
|---|---|
| **MICL downstream** | **11.804** |
| STAEformer 8ch mask | 11.869 |
| STAEformer 8ch mask-aware | 11.900 |
| STAEformer 5ch | 12.126 |
| STGCN 1ch | 13.337 |
| STGCN 8ch mask-aware | 13.509 |

**Node Death (% degradation on healthy nodes)**:

| Rate | 5ch | 8ch_mask | 8ch_mask_aware | STGCN_1ch | STGCN_8ch | **MICL** |
|---|---|---|---|---|---|---|
| r10 | +0.4% | +0.5% | +1.7% | +22.4% | +17.4% | +1.0% |
| r20 | +1.2% | +1.1% | +4.7% | +80.5% | +70.8% | +1.1% |
| r30 | +2.7% | +2.0% | +7.9% | +107% | +128% | **+1.5%** |
| r50 | +8.7% | +6.8% | +19.3% | +146% | +298% | **+5.5%** |

**Intermittent Missing (% degradation on healthy nodes)**:

| Rate | 5ch | 8ch_mask | 8ch_mask_aware | STGCN_1ch | STGCN_8ch | **MICL** |
|---|---|---|---|---|---|---|
| p10 | +32.2% | +74.6% | **+16.5%** | +81.8% | +59.1% | +18.9% |
| p20 | +65.0% | +106% | +38.0% | +174% | +107% | **+37.3%** |
| p30 | +101% | +115% | +64.1% | +225% | +145% | **+55.0%** |

### Key Findings
1. **Best clean MAE**: MICL achieves the lowest clean MAE (11.804) among all models
2. **Best node death robustness at high rates**: MICL #1 at r30 (+1.5%) and r50 (+5.5%), beating 8ch_mask (+2.0%, +6.8%)
3. **Best intermittent robustness at high rates**: MICL #1 at p20 (+37.3%) and p30 (+55.0%), beating mask-aware (+38.0%, +64.1%)
4. **Only model excelling at both**: Other models trade off node death vs intermittent robustness; MICL is consistently top-2 across all scenarios
5. **Contrastive pretraining with missing augmentation works**: Unlike previous encoder experiments that hurt clean accuracy, MICL improves both accuracy and robustness

### Code
- `baselines/MICL/arch/augmentation.py` - Missing pattern augmentation
- `baselines/MICL/arch/micl_arch.py` - MLPEncoder + MICLModel + micl_loss
- `baselines/MICL/runner/micl_runner.py` - MICLPretrainRunner
- `baselines/MICL/pretrain_config.py` - Pretraining config
- `baselines/MICL/downstream_config.py` - Downstream config

---

## MICL v2: Mask-Aware Downstream + Wider Augmentation (2026-02-15)

### Motivation
MICL v1이 clean accuracy와 robustness 모두 개선했지만, 두 가지 가설 검증:
1. **Mask-aware downstream loss**: downstream에서도 mask 채널을 활용하면 missing 패턴에 더 robust해지는가?
2. **Wider augmentation range**: pretraining augmentation rate를 U(0.01, 0.5)로 확장하면 low-rate corruption에도 더 강해지는가?

### Design Changes (vs v1)
- **Mask-aware loss**: `mask_aware_mae` loss - mask 채널(flow mask)을 사용해 observed values에만 loss 적용
- **Pass-through mask channels**: encoder output 뒤에 raw mask 채널 [3,4,5]를 그대로 concatenate → STAEformer가 직접 mask 정보 활용
- **input_dim 37**: encoder(32) + tod(1) + dow(1) + mask_flow(1) + mask_occ(1) + mask_speed(1) = 37 (v1은 34)
- **Wider augmentation**: rate ~ Uniform(0.01, 0.5) (v1은 0.1-0.5)

### Pretraining
- Same architecture as v1 (MLPEncoder 8→64→32)
- Augmentation rate: U(0.01, 0.5)
- 30 epochs, SSL loss converged
- Checkpoint: `checkpoints/MICL_pretrain_v2/SAN_BERNARDINO_MASK_30_12_12/85921292.../`

### Downstream Results
- Overall MAE: 11.924 (v1: 11.943, baseline 5ch: 12.10)
- Checkpoint: `checkpoints/MICL_downstream_v2/SAN_BERNARDINO_MASK_30_12_12/e50b21e2.../`

### Phase 2: Missing Pattern Shift Robustness

**Clean MAE comparison**:

| Model | Clean MAE |
|---|---|
| **MICL v2** | **11.699** |
| MICL v1 | 11.804 |
| STAEformer 8ch mask | 11.869 |
| STAEformer 8ch mask-aware | 11.900 |
| STAEformer 5ch | 12.126 |

**Node Death (% degradation on healthy nodes)**:

| Rate | 5ch | 8ch_mask | 8ch_mask_aware | MICL v1 | **MICL v2** |
|---|---|---|---|---|---|
| r10 | +0.4% | +0.5% | +1.7% | +1.0% | **+0.3%** |
| r20 | +1.2% | +1.1% | +4.7% | +1.1% | +1.0% |
| r30 | +2.7% | +2.0% | +7.9% | **+1.5%** | +3.9% |
| r50 | +8.7% | +6.8% | +19.3% | **+5.5%** | +7.5% |

**Intermittent Missing (% degradation on healthy nodes)**:

| Rate | 5ch | 8ch_mask | 8ch_mask_aware | MICL v1 | **MICL v2** |
|---|---|---|---|---|---|
| p10 | +32.2% | +74.6% | **+16.5%** | +18.9% | +32.5% |
| p20 | +65.0% | +106% | +38.0% | **+37.3%** | +75.5% |
| p30 | +101% | +115% | +64.1% | **+55.0%** | +127.0% |

### Key Findings
1. **Best clean MAE**: v2 (11.699) beats v1 (11.804) by 0.9% - pass-through mask channels help clean accuracy
2. **Better at low death rates**: v2 is #1 at r10 (+0.3% vs v1's +1.0%) - wider augmentation range helps low-rate scenarios
3. **Worse at high death rates**: v2 degrades at r30 (+3.9% vs v1's +1.5%) and r50 (+7.5% vs v1's +5.5%)
4. **Much worse intermittent robustness**: v2 degrades dramatically (p30: +127% vs v1's +55%) - **2.3x worse**
5. **Mask pass-through is a double-edged sword**: giving STAEformer direct mask access improves clean accuracy but creates **mask channel dependency** - when test-time masks differ from training distribution, the model breaks

### Analysis: Why v2 Fails on Intermittent
The pass-through mask channels let STAEformer learn to condition predictions on mask patterns during training. At test time:
- **Node death**: masks change for corrupted nodes only → limited distribution shift → v2 still OK at low rates
- **Intermittent**: masks change for ALL nodes randomly each timestep → massive distribution shift → v2 breaks catastrophically

This confirms: **robustness to missing pattern shifts requires mask-INVARIANT representations, not mask-AWARE predictions**. v1's approach (encode mask info into invariant representations via contrastive loss, then discard masks) is fundamentally superior for robustness.

### Conclusion
- v2's mask-aware approach: better clean accuracy, worse robustness (especially intermittent)
- v1's mask-invariant approach: slightly lower clean accuracy, much better robustness
- **MICL v1 remains the recommended model** for deployment where missing pattern shifts are expected

### Code
- `baselines/MICL/pretrain_config_v2.py` - Pretraining config (wider augmentation)
- `baselines/MICL/downstream_config_v2.py` - Downstream config (mask-aware loss + pass-through)
- `experiments/eval_missing_pattern_shift.py` - `MICLDownstreamV2Wrapper` + loader

---

## MICL v3 Ablation & Unified Baseline (2026-02-15)

### v3 Ablation Experiments
v1을 개선하기 위해 3가지 독립 변수를 각각 테스트:
- **v3a**: v2 pretrain (wider aug 0.01-0.5) + v1 downstream
- **v3b**: v1 pretrain + mask-aware loss (no pass-through)
- **v3c**: curriculum pretrain (max_rate 0.1→0.5) + v1 downstream

### v3 Results
| Experiment | Overall MAE | Clean MAE (functional, observed) |
|---|---|---|
| MICL v1 | 11.943 | 11.804 |
| MICL v3a | 11.829 | 11.916 |
| MICL v3b | 12.005 | 11.796 |
| MICL v3c | - | 11.831 |

**Per-node paired t-test**: All v3 vs v1 comparisons show p>0.5, Pearson r>0.91
→ **v3 변경사항은 통계적으로 v1과 구분 불가** (noise level)

**원인**: 30 epoch fine-tuning이 pretrain 차이를 overwite함. Pretrain condition 변경은 lasting effect 없음.

### Unified Baseline: 5ch + mask_aware loss
기존 baseline 문제점:
- `5ch`: SAN_BERNARDINO 데이터, masked_mae loss → missing과 real zero 구분 불가
- `8ch_mask_aware`: mask 채널을 모델 입력으로 사용 → 이것도 일종의 "method"

**통합 baseline** (`5ch_maskloss`): SAN_BERNARDINO_MASK 데이터, 5ch 입력 (mask 채널 미사용), mask_aware loss
- 모델은 mask를 보지 못함 (순수 baseline)
- Loss만 missing 값 무시 (clean supervision)

### MICL v1 vs Unified Baseline

| Scenario | 5ch_maskloss (baseline) | MICL v1 | MICL advantage |
|---|---|---|---|
| Clean MAE | 11.908 | 11.804 | -0.9% |
| death_r10 | +1.1% | +1.0% | 1.1x |
| death_r20 | +3.5% | +1.1% | **3.2x** |
| death_r30 | +8.3% | +1.5% | **5.4x** |
| death_r50 | +25.5% | +5.5% | **4.6x** |
| intermit_p10 | +26.9% | +18.9% | **1.4x** |
| intermit_p20 | +56.8% | +37.3% | **1.5x** |
| intermit_p30 | +88.5% | +55.0% | **1.6x** |

### Key Findings
1. **Clean MAE도 개선**: MICL v1이 baseline보다 0.9% 더 낮음
2. **Node death robustness 대폭 개선**: r20~r50에서 3-5x 더 robust
3. **Intermittent robustness 개선**: p10~p30에서 1.4-1.6x 더 robust
4. **Robustness 이점은 corruption rate이 높을수록 커짐**: 실제 배포 환경에서 더 큰 가치

### Conclusion
통합 baseline (5ch + mask_aware loss) 대비 MICL v1은 clean accuracy와 robustness 모두 개선.
MICL의 contrastive pretraining이 missing-invariant representation을 학습하여,
test time에 missing pattern이 변해도 예측 품질을 유지함.

### Code
- `baselines/STAEformer/SAN_BERNARDINO_5ch_maskloss.py` - Unified baseline config
- `baselines/MICL/downstream_config_v3a.py` / `v3b.py` / `v3c.py` - v3 configs
- `baselines/MICL/pretrain_config_v3c.py` - Curriculum pretrain config

---

*Last updated: 2026-02-15 (v3 ablation & unified baseline added)*
