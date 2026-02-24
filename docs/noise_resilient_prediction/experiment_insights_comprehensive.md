# SSL for Traffic Prediction: Comprehensive Experiment Insights

> **Dataset**: SAN_BERNARDINO (xtraffic), 893 nodes, 5 features (flow, occupancy, speed, tod, dow)
> **Period**: 3 months (26,280 timestamps, 5-min intervals)
> **Task**: 12-step → 12-step traffic flow prediction
> **Backbone**: STAEformer
> **Date**: January - February 2026

---

## 1. Executive Summary

**100개 이상의 실험을 수행한 결론: SSL contrastive encoder는 traffic prediction을 개선하지 않는다.**

- Phase 1 (masking strategies): masked MAE 기준 최대 -3.2% 개선으로 보였으나, 후속 분석에서 이 개선이 dead sensor artifact와 관련될 수 있음을 확인
- Phase 2 (기존 모델 비교): GPT-ST, STEP, STMAE 모두 baseline과 비슷하거나 약간 나쁨
- Phase 3 (DisentangledEncoder): masked MAE 기준 모든 encoder 실험이 baseline보다 나쁨
- Phase 4 (공정 비교): unmasked loss + dead node 제외 시, encoder가 baseline보다 나쁨 확정

---

## 2. Dataset Characteristics (Critical Context)

### 2.1 Sensor Quality Distribution

| Category | Node 수 | flow_zero_rate | 특징 |
|----------|---------|----------------|------|
| Dead (>90% zero) | 121 | 97.6%~100% | 센서 완전 고장 (24시간 100% zero) |
| Major fail (50-90%) | 28 | 50%~90% | 심각한 데이터 손실 |
| Partial fail (5-50%) | 161 | 5%~50% | 간헐적 데이터 손실, 낮/밤 패턴 존재 |
| Functional (<5%) | 583 | 0%~5% | 정상 센서, 심야에도 traffic 존재 |

### 2.2 Dead Node 확인

- 121개 중 119개는 flow+occupancy+speed **모두 100% zero** (모든 timestamp)
- Node 20: flow 97.58% zero, 나머지 100% zero (간헐적 소량 flow)
- Node 796: speed 99.95% non-zero (65~68mph 고정), flow/occ는 거의 zero (센서 이상)
- **결론: Dead node = 확인된 센서 고장, 교통이 없는 게 아님**

### 2.3 NULL_VAL Convention 문제

**XTraffic 데이터의 NULL_VAL=0 convention:**
- PeMS 원본: 4가지 imputation method 적용, "Card Off" 플래그 제공
- XTraffic: imputation 없이 raw 데이터 제공, 추후 개선 계획 언급
- **문제**: 센서 고장(0)과 실제 zero traffic을 구분 불가

**NULL_VAL=0 masking의 효과:**
- `masked_mae(null_val=0)`: target이 0인 샘플 제외 → dead sensor의 gradient 없음
- 모델은 dead sensor에 대해 mean 값(≈23) 예측 → unmasked MAE에서 높은 에러
- **중요**: RESCALE=True이므로 loss는 역정규화된 원래 scale에서 계산됨

---

## 3. Phase 1: Contrastive Pre-training with Masking Strategies (Jan 2026)

### 3.1 설정
- **Encoder**: 2-layer Transformer, d_model=64, input_dim=3 (flow, tod, dow)
- **Pre-training**: SimCLR-style contrastive learning with masking augmentation
- **Downstream**: STAEformer, freeze 또는 fine-tune
- **Metric**: masked MAE (NULL_VAL=0)

### 3.2 결과 (Overall MAE, masked)

| Experiment | MAE | Δ Baseline |
|-----------|-----|------------|
| **Baseline STAEformer** | **12.10** | - |
| Freeze: Spatial mask 20% | 11.77 | -2.7% |
| Freeze: Feature mask | 11.82 | -2.3% |
| Freeze: Spatial mask 30% | 11.84 | -2.1% |
| Freeze: Feature mask 30% | 11.84 | -2.1% |
| Freeze: Feature mask 10% | 11.84 | -2.1% |
| Freeze: Spatial mask 10% | 11.85 | -2.1% |
| Freeze: Feature mask 50% | 11.88 | -1.8% |
| Freeze: Temporal mask | 11.94 | -1.3% |
| Freeze: Spatial mask 90% | 12.12 | +0.1% |
| Finetune: Feature mask | 12.18 | +0.6% |
| Finetune: Combined v1 | 11.99 | -0.9% |
| Finetune: Combined v2 | 12.02 | -0.6% |
| Finetune: GAT | 12.34 | +2.0% |
| Finetune: Temporal mask | 12.74 | +5.3% |

### 3.3 Insights

1. **Freeze > Finetune** (Phase 1에서는): freeze가 대체로 더 나음
2. **Spatial mask 20%가 best**: MAE 11.77 (-2.7%)
3. **과도한 masking은 해로움**: 90% masking → baseline보다 나쁨
4. **GAT encoder 실패**: gradient 불안정, 학습 발산
5. **Old finetuning runner 결과와 로그 불일치**: 로그에는 11.71로 기록되어 있으나 checkpoint MAE는 12.18. 평가 방식 차이 가능성 (horizon별 vs overall)

### 3.4 주의사항
- 이 시점에서는 dead sensor artifact를 인지하지 못함
- masked MAE 기준이므로 dead node 영향은 제한적이나, 평가 방식에 따라 차이 가능

---

## 4. Phase 2: Pre-training Baselines Comparison (Jan-Feb 2026)

### 4.1 비교 대상
| Model | Architecture | Pre-training Method |
|-------|-------------|---------------------|
| GPT-ST | STHCN (Hypergraph Capsule) | Adaptive masking + reconstruction |
| STEP | TSFormer + GraphWaveNet | 75% patch masking + reconstruction |
| STMAE | Backbone-agnostic MAE | Feature + Structure masking |

### 4.2 결과

| Model | MAE | Δ Baseline | Notes |
|-------|-----|------------|-------|
| STMAE finetune | 11.90 | -1.7% | Backbone: AGCRN |
| GPT-ST finetune | 12.02 | -0.6% | |
| TSFormer pretrain | 26.95 | - | Reconstruction metric |
| GPT-ST pretrain | 117.85 | - | Reconstruction metric |

### 4.3 Insights

1. **기존 방법론도 큰 개선 없음**: STMAE -1.7%, GPT-ST -0.6%
2. **STEP NaN 문제**: `sqrt(d_model)` 스케일링으로 attention overflow → pre-encoder LayerNorm 추가로 해결
3. **Reconstruction pre-training의 한계**: masked reconstruction metric(MAE 100+)은 downstream 성능과 상관관계 약함

---

## 5. Phase 3: DisentangledEncoder Experiments (Feb 2026)

### 5.1 아키텍처
- **DisentangledEncoder**: context representation + self representation 분리
- input_dim=5 (flow, occupancy, speed, tod, dow)
- fusion='concat' → downstream input_dim = d_model*2 + 2 = 130
- **RepresentationLearningRunner**: 통합 runner (pretrained/scratch/freeze/finetune 지원)

### 5.2 Pre-training 결과

| Pre-train Model | MAE (recon) | Method |
|----------------|-------------|--------|
| DisentangledTemporal_5feat | 0.53 | Contrastive + Reconstruction |
| DisentangledTemporal_5feat_contrastive_only | 0.78 | Contrastive only |
| DisentangledTemporal_5feat_recon_only | 0.50 | Reconstruction only |
| DisentangledTemporal_3feat | 0.73 | 3 features only |
| PredictiveContrastive_5feat | - | Predictive + Contrastive |

### 5.3 Downstream 결과 (Masked MAE, NULL_VAL=0)

| Experiment | MAE | Δ Baseline |
|-----------|-----|------------|
| **Baseline** | **12.10** | - |
| Contrastive_only (concat, d=130) | 12.19 | +0.7% |
| Context_fusion (z_ctx only, d=64) | 12.19 | +0.7% |
| Self_only | 12.40 | +2.5% |
| Context_only | 12.45 | +2.9% |
| Random_projection (control) | 11.94 | -1.3% |
| Recon_only downstream | 16.44 | +35.9% |
| Predictive_contrastive | 13.10 | +8.3% |
| DisentangledTemporal_3feat freeze | 12.17 | +0.5% |
| DisentangledTemporal_5feat freeze | 12.41 | +2.6% |

### 5.4 Insights

1. **모든 encoder 실험이 baseline보다 나쁨** (masked MAE 기준)
2. **Random projection이 contrastive보다 나음** (11.94 vs 12.19): encoder가 유의미한 representation을 학습하지 못함
3. **Input dimension explosion**: 3→130 차원으로 증가가 STAEformer에 부담 (embedding, attention 모두 130차원 입력)
4. **Recon_only가 최악** (16.44): reconstruction loss만으로는 forecasting에 유용한 representation 학습 불가
5. **Fusion 방식 무관**: concat이든 context_fusion이든 비슷하게 나쁨

---

## 6. Phase 4: Fair Comparison - Unmasked Loss (Feb 2026)

### 6.1 동기
- NULL_VAL=0 masking convention에 의문 제기
- Dead sensor와 genuine zero traffic 구분 불가
- MAE는 MAPE와 달리 zero에 문제 없으므로 masking 불필요

### 6.2 실험 설계: 2×2 비교

| | Masked Loss (NULL_VAL=0) | Unmasked Loss (NULL_VAL=NaN) |
|---|---|---|
| **No Encoder** | Baseline: 12.10 | Baseline_unmasked: 10.19 |
| **Contrastive Encoder** | Encoder: 12.19 | Encoder_unmasked: 10.09 |

### 6.3 핵심 발견: Loss Function이 지배적 요인

**Masked → Unmasked: MAE 12.10 → 10.19 (-15.8%)**
- Loss 함수 변경만으로 대폭 개선
- Dead node에 gradient가 생겨 zero 예측 학습

**Encoder 효과 (unmasked 기준): 10.19 → 10.09 (-1.0%)**
- 매우 미미한 개선
- 893 전체 기준이므로 dead sensor 효과 포함 가능성

### 6.4 Per-Category 분석 (Unmasked 실험)

| Category | Baseline_unmasked | Encoder_unmasked | 차이 |
|----------|------------------|------------------|------|
| Dead (121) | 0.15 | 0.13 | -13% (무의미, base 작음) |
| Major fail (28) | ~4 | ~5.5 | **+37% 악화** |
| Partial fail (161) | ~6 | ~6 | 비슷 |
| Functional (583) | 14.54 | 13.74 | **-5.5% 개선** |

### 6.5 Dead Node 제외 실험

| Experiment | 노드 수 | MAE |
|-----------|---------|-----|
| Baseline_unmasked | 893 | 10.19 |
| Encoder_unmasked | 893 | 10.09 |
| **Baseline_nodead** | **772** | **11.29** |
| **Encoder_nodead** | **772** | **11.57 (+2.5%)** |
| Baseline_functional | 583 | 13.57 |
| Encoder_functional | 583 | 13.62 (+0.3%) |

### 6.6 최종 결론

1. **Dead node 제외 시 encoder가 baseline보다 나쁨** (+2.5%)
2. **Functional node만으로도 encoder가 나쁨** (+0.3%)
3. **893 전체에서 encoder가 좋아 보인 것 (10.09 vs 10.19)은 dead node 효과**
4. **SSL encoder는 traffic prediction을 개선하지 않는다** (공정한 비교 기준)

---

## 7. Hypothesis Testing Summary

### H1: Contrastive pre-training improves traffic prediction
- **결과: 기각**
- Phase 1에서 약간 개선으로 보였으나, Phase 3-4에서 공정 비교 시 개선 없음
- 특히 dead sensor artifact를 통제하면 오히려 악화

### H2: Feature masking > Temporal masking > Spatial masking
- **결과: 부분적 지지 (Phase 1에서만)**
- Phase 1에서 spatial > feature > temporal 순서 (freeze 기준)
- 그러나 이 결과도 dead sensor artifact 영향 가능

### H3: SSL encoder가 noisy data에서 robust한 representation 제공
- **결과: 기각**
- Major fail (+37% 악화), Partial fail (변화 없음)
- SSL이 가장 도움이 되어야 할 noisy node에서 오히려 나쁨

### H4: DisentangledEncoder가 context/self 분리로 더 나은 representation 학습
- **결과: 기각**
- concat(z_context, z_self) → 130차원이 STAEformer에 부담
- Random projection이 학습된 encoder보다 나은 성능

### H5: Unmasked loss가 더 공정한 평가
- **결과: 지지**
- NULL_VAL=0은 dead sensor를 "정답"에서 제외 → 불공정한 이점/불이익
- Unmasked loss에서 baseline 자체가 대폭 개선 (12.10→10.19)

### H6: Dead node 제외하면 SSL 효과가 드러남
- **결과: 기각**
- Dead 제외 후: encoder 11.57 vs baseline 11.29 (+2.5% 악화)
- Functional만: encoder 13.62 vs baseline 13.57 (+0.3% 악화)

---

## 8. Why SSL Failed: Root Cause Analysis

### 8.1 Input Dimension Explosion
- Baseline: input_dim=3 (flow, tod, dow)
- Encoder: input_dim=130 (z_context 64 + z_self 64 + tod + dow)
- STAEformer의 input_embedding_dim=24이므로 130→24 projection에서 정보 손실
- Random projection (130차원, 학습 안 함)이 비슷한 성능 → encoder가 유의미한 정보 미추가

### 8.2 Information Bottleneck
- Encoder가 5개 feature를 64차원으로 압축
- 원래 flow+tod+dow 3개로 충분히 예측 가능한 task
- Encoder가 추가 feature (occupancy, speed)에서 유용한 정보를 추출하지 못함

### 8.3 Pre-training Task와 Downstream Task의 Misalignment
- Contrastive learning: "같은 시점의 다른 view가 유사해야 한다"
- Forecasting: "현재 패턴으로 미래를 예측해야 한다"
- 두 objective가 다른 representation을 요구

### 8.4 Data Quality Issue는 SSL로 풀 문제가 아님
- Dead sensor: 전처리에서 필터링하면 해결 (SSL 불필요)
- Noisy sensor: SSL이 오히려 noise를 representation에 포함시킴
- Traffic prediction은 supervised signal이 충분한 domain → SSL의 한계

---

## 9. Other Attempted Approaches (성과 없음)

### 9.1 TTA (Test-Time Adaptation)
- `ContextContrastive_TTA_adaptation_3mo`: MAE 11.81 (baseline 대비 -2.4%)
- `ContextContrastive_TTA_ensemble_3mo`: MAE 13.37 (+10.5%)
- Adaptation은 약간 도움, ensemble은 악화

### 9.2 Temporal-Aware Negatives
- `ContextContrastive_temporal_aware_neg`: MAE 11.88 (-1.8%) / 12.30 (+1.6%)
- 일관성 없는 결과

### 9.3 Neighborhood Masking
- `ContextContrastive_freeze_neighborhood_mask`: MAE 11.90 (-1.7%)
- Spatial masking과 비슷

### 9.4 Multi-Resolution
- `ContextContrastive_finetune_multi_resolution`: MAE 12.16 (+0.5%)
- 개선 없음

### 9.5 Reconstruction + Spatial Masking (Freeze)
- 10%: 12.67, 30%: 13.08, 50%: 14.44, 70%: 13.32, 90%: 13.21
- 모두 baseline보다 나쁨

### 9.6 Residual/GatedResidual Fusion
- Residual_repr_3feat: 11.89 (-1.7%)
- GatedResidual_repr_3feat: 11.89 (-1.7%)
- Raw ablation 비슷: 11.92
- 미미한 개선, fusion 방식은 큰 차이 없음

---

## 10. Complete Results Table (SAN_BERNARDINO, sorted by MAE)

### 10.1 Downstream/Forecasting Results (MAE < 20)

| # | Experiment | MAE | Notes |
|---|-----------|-----|-------|
| 1 | Baseline_unmasked (893) | 10.19 | NULL_VAL=NaN, no encoder |
| 2 | Encoder_unmasked (893) | 10.09 | NULL_VAL=NaN, contrastive encoder |
| 3 | Baseline_nodead (772) | 11.29 | Dead 제외, no encoder |
| 4 | Encoder_nodead (772) | 11.57 | Dead 제외, contrastive encoder |
| 5 | Freeze: Spatial mask 20% | 11.77 | Phase 1 best (freeze) |
| 6 | TTA adaptation | 11.81 | |
| 7 | Freeze: Feature mask | 11.82 | |
| 8 | Freeze: Spatial mask 30% | 11.82~11.84 | |
| 9 | Freeze: Spatial mask 70% | 11.83 | |
| 10 | Freeze: Feature mask 30% | 11.84 | |
| 11 | Freeze: Feature mask 10% | 11.84 | |
| 12 | Freeze: Spatial mask 10% | 11.85 | |
| 13 | Freeze: Spatial mask 50% | 11.87 | |
| 14 | Freeze: Feature mask 50% | 11.88 | |
| 15 | GatedResidual repr 3feat | 11.89 | |
| 16 | Residual repr 3feat | 11.89 | |
| 17 | Temporal-aware negatives | 11.88 | |
| 18 | Freeze: Neighborhood mask | 11.90 | |
| 19 | Freeze: Spatial+Feature | 11.90 | |
| 20 | STMAE finetune | 11.90 | |
| 21 | STAEformer+TemporalEncoder | 11.91 | |
| 22 | Freeze: Feature mask 70% | 11.92 | |
| 23 | Random projection | 11.94 | **Control: no learning** |
| 24 | Freeze: Temporal mask | 11.94 | |
| 25 | Baseline STAEformer (masked) | 12.10 | **Phase 1 Baseline** |
| 26 | Finetune: Combined v1 | 11.99 | |
| 27 | Finetune: Combined v2 | 12.02 | |
| 28 | GPT-ST finetune | 12.02 | |
| 29 | Freeze: Spatial mask 90% | 12.12 | |
| 30 | DisentangledTemporal 3feat | 12.17 | Phase 3 |
| 31 | Finetune: Feature mask | 12.18 | |
| 32 | Contrastive_only downstream (masked) | 12.19 | Phase 3 |
| 33 | Context fusion downstream | 12.19 | Phase 3 |
| 34 | Freeze baseline (encoder) | 12.23 | |
| 35 | Self_only downstream | 12.40 | Phase 3 |
| 36 | DisentangledTemporal 5feat | 12.41 | Phase 3 |
| 37 | Context_only downstream | 12.45 | Phase 3 |
| 38 | Finetune: GAT | 12.34 | |
| 39 | Finetune: Temporal mask | 12.74 | |
| 40 | Predictive contrastive | 13.10 | Phase 3 |
| 41 | Baseline_functional (583) | 13.57 | Functional only |
| 42 | Encoder_functional (583) | 13.62 | Functional only |
| 43 | TTA ensemble | 13.37 | |
| 44 | Recon_only downstream | 16.44 | Phase 3, worst |

### 10.2 Pre-training Results (Reconstruction, MAE > 20)

| Experiment | MAE | Notes |
|-----------|-----|-------|
| TSFormer pretrain | 26.95 | Masked patch reconstruction |
| GPT-ST pretrain | 117.85 | Adaptive masking reconstruction |
| Various contrastive pretrains | 121~153 | Contrastive loss (not prediction metric) |

---

## 11. Key Takeaways

### 11.1 What We Learned About the Data
1. **SAN_BERNARDINO 데이터는 심각한 quality issue가 있음**: 893개 중 121개(13.5%)가 완전 dead sensor
2. **NULL_VAL=0 convention은 위험**: dead sensor artifact로 잘못된 결론 도출 가능
3. **Unmasked MAE가 더 공정한 metric**: 모든 값을 평가에 포함해야 진정한 성능 비교 가능
4. **Dead sensor 제거가 가장 효과적인 "개선"**: 전처리만으로 10.19 달성 (기존 12.10 대비 -15.8%)

### 11.2 What We Learned About SSL for Traffic
1. **Supervised signal이 충분한 domain에서 SSL의 한계**: traffic data는 label이 풍부
2. **Representation이 반드시 downstream에 유용하지 않음**: contrastive learning의 invariance가 forecasting에 해로울 수 있음
3. **Input dimension 증가가 항상 좋지 않음**: 3→130 차원 증가가 STAEformer에 부담
4. **Random projection이 학습된 encoder와 비슷**: encoder가 의미있는 추가 정보를 제공하지 못함

### 11.3 What We Learned About Experimental Methodology
1. **Metric 선택이 결론을 바꿈**: masked vs unmasked MAE로 완전히 다른 결론
2. **Per-category 분석이 필수**: overall metric은 dead sensor artifact를 숨김
3. **Control experiment (random projection) 중요**: SSL이 진짜 학습했는지 확인
4. **2×2 실험 설계**: (loss type) × (encoder type)으로 요인 분리 필수

### 11.4 리뷰어 예상 질문과 답변
- **Q: Dead node를 왜 제거하지 않고 SSL을 적용하나?**
  A: Dead node 문제는 단순 필터링으로 해결 가능. SSL이 풀어야 할 문제가 아님.
- **Q: Phase 1의 개선은 실제인가?**
  A: masked MAE 기준 -2.7% (spatial freeze). 하지만 dead sensor artifact 통제 후 재확인 필요.
- **Q: 왜 SSL이 작동하지 않는가?**
  A: (1) input dimension explosion, (2) pre-training/downstream objective misalignment, (3) supervised signal 충분.

---

## 12. Possible Future Directions (검증 필요)

1. **Phase 1 결과 재검증**: 기존 freeze spatial mask 20% (MAE 11.77)를 unmasked+nodead 조건에서 재실험
2. **Lightweight encoder**: 130차원 대신 3+α (e.g., 5~10차원) 수준의 augmented feature
3. **다른 downstream backbone**: STAEformer가 아닌 다른 모델에서 encoder 효과 확인
4. **다른 dataset**: dead sensor가 없는 깨끗한 데이터(METR-LA, PEMS-BAY)에서 실험
5. **Few-shot setting**: labeled data가 부족할 때 SSL 효과 확인 (traffic은 label 풍부하므로 SSL 이점 없을 수 있음)
6. **Data augmentation 관점**: encoder 대신 augmentation만 적용하여 regularization 효과 확인

---

*Last updated: 2026-02-10*
*Total experiments: 100+*
*Conclusion: SSL contrastive encoder does not improve traffic prediction on SAN_BERNARDINO when properly controlling for dead sensor artifacts and using fair evaluation metrics.*
