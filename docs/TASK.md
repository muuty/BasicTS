# Task Tracker

## Active Tasks

_현재 진행 중인 작업을 여기에 기록한다._

<!-- 형식:
### [Task Name]
- **상태**: 진행중 / 대기 / 완료
- **설명**: 무엇을 하는지
- **관련 파일**: 수정/생성할 파일
- **메모**: 진행 중 발견한 것들
-->

---

## Completed Tasks

### Phase A: Distance Screening
- **상태**: 진행중 (학습 결과 수집 중)
- **설명**: Coreset selection에 최적인 distance function 찾기
- **Config**: `experiments/config/phase_a_distance_screening.yaml`
- **체크리스트**: `docs/phase_a_experiment_checklist.md`
- **결과**: `experiments/result/phase_a_distance_screening.csv`
- **메모**:
  - Selection index 생성: 88/110 완료 (cosine pipeline 일부 미완)
  - Training: 220 runs 예정
  - 초기 결과: K-Center + combined @ 0.7이 안정적

### Coreset K-Medoids 기본 실험
- **상태**: 완료
- **설명**: K-Medoids로 ratio 0.1~1.0 sweep (4개 모델 × SAN_BERNARDINO)
- **Config**: `experiments/config/coreset.yaml`
- **결과**: `experiments/result/coreset.csv`, `experiments/result/coreset_san_bernardino_summary.csv`
- **핵심 발견**: 60~80% coreset이 full data와 동등하거나 더 좋음

### Experience Replay 실험
- **상태**: 완료
- **설명**: Gate + Replay 조합 실험 (STGCN, STAEformer, STGformer 등)
- **결과**: `experiments/result/experience_replay.csv`
- **핵심 발견**: STGCN에서만 효과적 (-22.7% MAE), STAEformer/STGformer는 효과 없음

### Node Identity 실험
- **상태**: 완료
- **설명**: Message passing이 incident signal을 희석시키는지 검증
- **결과**: `experiments/result/node_identity_exp.csv`
- **핵심 발견**: MP 제거/약화는 전반적 성능 저하

### Adjacency Matrix 실험
- **상태**: 완료
- **설명**: 4가지 인접 행렬 × 3개 모델 × 2개 데이터셋
- **Config**: `experiments/config/adjacency_phase1a_*.yaml`

---

## Backlog

_향후 할 작업들._

### Severity-aware Pre-training 구현
- **설명**: Context-aware contrastive pre-training (design.md 참조)
- **관련 파일**: `baselines/Encoder/`, `docs/design.md`
- **현황**: 설계 완료, 초기 구현 테스트 중

### Phase A 결과 분석 및 시각화
- **설명**: Distance screening 전체 결과 분석
- **관련 파일**: `experiments/result/phase_a_distance_screening.csv`

### 다른 데이터셋 (ALAMEDA, SACRAMENTO) 실험 확장
- **설명**: SAN_BERNARDINO에서 확인된 best setting을 다른 데이터셋에 적용

---

## Notes

- 실험 결과 수집: `python experiments/get_results.py --config <yaml> --metrics MAE RMSE`
- SLURM job 확인: `squeue -u $USER`
- 일일 작업 로그는 `docs/YYYY-MM-DD.md` 형식으로 기록 가능
