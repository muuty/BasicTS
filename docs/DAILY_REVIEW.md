# Daily Review

매일 작업이 끝날 때 `/daily-review`를 실행하면 Claude가 오늘의 리뷰를 정리해준다.
검토 후 수정하여 저장한다.

---

## 2025-02-25

### 오늘 한 일
- **Distributional proxy metric 분석 완료**: KL/JS divergence, Coverage Gap, MMD 추가 계산. Proxy vs MAE 상관 분석 (information_gain ρ=0.50, OT/Sinkhorn 유의하지 않음)
- **시각화 7종 생성**: correlation heatmap, radar chart, bar chart, KL scatter, temporal histogram, t-SNE (ToD/DoW coloring), nearest coreset distance
- **Redundancy paradox 분석**: graph_cut이 redundancy 최저이지만 MAE 최악인 이유 — spatial dominance in feature space, density 미보존
- **Phase B 실험 설계 및 제출**: 5개 모델(STGCN no-dropout, AGCRN, DCRNN, STID, STAEformer) deterministic training. 310 experiments → 62 SLURM jobs (batch-size=5) 제출
- **신규 config 생성**: DCRNN, STID SAN_BERNARDINO config + STGCN dropout=0 config

### 진전된 것
- Phase A proxy metric 분석이 완결됨 — "좋은 coreset = 분포 보존 (k_medoids)" 결론 도출
- Phase B 실험이 돌아가기 시작 — deterministic training으로 randomness 제거하고 5개 모델에서 검증

### 막힌 것 / 아쉬운 것
- Phase A의 training randomness가 너무 커서 (동일 coreset MAE 차이 1.17) 결론 내리기 어려웠음 → Phase B로 재실험 결정
- SLURM QOS 한도 (100 jobs) 때문에 처음에 batch-size=1로 제출 실패 → batch-size=5로 해결

### 배운 것
- **OT/Sinkhorn은 MAE를 예측하지 못함** (ρ=-0.25, p=0.09). Temporal/feature coverage가 더 강한 predictor
- **K_medoids ≈ implicit Wasserstein-1 minimizer**: density 보존이 핵심. "temporal coverage가 좋아서"가 아니라 "data distribution 자체를 보존해서" 성능이 좋음
- 5개 모델 모두 BatchNorm 미사용 (STAEformer만 LayerNorm) → BN→LN 수정 불필요

### 내일 할 것
- Phase B 실험 결과 수집 및 분석 (310개 완료 여부 확인)
- Deterministic training에서 seed간 MAE variance 확인 (randomness 제거 검증)
- Multi-model proxy metric vs MAE 상관 분석

---

