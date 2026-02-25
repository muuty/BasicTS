# Daily Log — Incident-Aware Traffic Forecasting

---

## 2026-02-26

### Done
- Coreset selection 110/110 완료 (all index files generated for SAN_BERNARDINO)
- Phase A training 제출: 220 runs → 44 SLURM batch jobs (batch-size=5)
- Facility Location selection method 구현 (`coreset/facility_location.py`)
  - Graph cut과 같은 similarity matrix 공유, greedy O(k·N) with `current_max` tracker
  - FL selection job 제출 (32 indices, Job 21498425)
  - FL training config 생성 (64 runs → 13 SLURM jobs, selection 완료 후 제출)
- Temporal distribution 분석 (`scripts/visualization/analyze_temporal_distribution.py`)
  - DOW/TOD KL divergence로 각 selection method의 temporal representativeness 측정
  - k_medoids가 가장 균일 (DOW_KL=0.0002), graph_cut/k_center가 temporal bias 큼
  - graph_cut_euclidean_030: 일요일 25%, 토요일 20% 과대선택 (RBF sigma 영향)
- `run_experiments.py` 가독성 개선: `run_num=batch[0][1]` → tuple unpacking, `parent_parts` robustness
- Discord token을 환경변수로 이동, git history rewrite (`git filter-branch`)
- Project notes를 `.research_notes/` 디렉토리로 이동

### Observations / Results
- Phase A training 4 jobs RUNNING, 40 PENDING (SLURM 동시 실행 제한)
- FL은 k-medoids와 동일 목적함수 (`min Σ dist`)를 greedy로 풀어 (1-1/e) 보장
  - 다만 FL은 similarity(RBF), k-medoids는 distance 직접 사용 → 비선형 변환 차이
- Graph cut temporal bias 원인: `column_sums` (information term)이 밀집 클러스터 선호 + RBF sigma 미튜닝

### Tomorrow
- FL selection 완료 확인 후 FL training 제출 (13 SLURM jobs)
- Phase A training 진행 모니터링
- Temporal distribution 분석을 FL에도 적용하여 k_medoids와 비교

---

## 2026-02-24

### Done
- Reorganized entire codebase into 7 logical git commits on `feature/incident-aware` branch
  - Removed vendored easytorch and legacy selection module
  - Cleaned up model architectures (removed gate integration, added STGformer)
  - Added incident-aware training infrastructure
  - Added experiment configs, coreset framework, orchestration, and documentation
- Set up project documentation system (CLAUDE.md, PROJECT.md, TASK.md, etc.)
- Phase A distance screening: ~135/220 training runs complete (up from ~87 earlier today)

### Observations / Results
- K-Center + combined distance @ ratio=0.7 remains most stable (MAE ~14.0-14.6)
- Graph Cut shows high seed sensitivity (MAE range 14.7-18.7) — may need more seeds
- Cosine pipeline k_medoids still running; graph_cut indices pending
- Some STGformer runs showing MAE=0.0 — needs investigation (possible early termination)

### Blockers
- SLURM QOSMaxJobsPerUserLimit caps at 4 concurrent jobs — Phase A completion is slow

### Tomorrow
- Monitor Phase A training progress, check for failed/zero-result runs
- Start preliminary analysis on completed L2 pipeline results
- Begin cosine graph_cut index computation if queue opens up
