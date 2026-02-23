# Claude Code Instructions

## Project Documentation
- **프로젝트 배경**: [docs/PROJECT_BACKGROUND.md](docs/PROJECT_BACKGROUND.md) — 연구 목표, 실험 히스토리, coreset 파이프라인
- **작업 추적**: [docs/TASK.md](docs/TASK.md) — 진행 중/완료/예정 작업 관리
- **아이디어**: [docs/IDEAS.md](docs/IDEAS.md) — 아이디어, 가설, 질문, 연결점
- **일일 리뷰**: [docs/DAILY_REVIEW.md](docs/DAILY_REVIEW.md) — `/daily-review`로 생성
- **주간 리뷰**: [docs/WEEKLY_REVIEW.md](docs/WEEKLY_REVIEW.md) — `/weekly-review`로 생성
- **Phase A 체크리스트**: [docs/phase_a_experiment_checklist.md](docs/phase_a_experiment_checklist.md)
- **설계 문서**: [docs/design.md](docs/design.md) — Severity-aware pre-training 설계

## Custom Commands
- `/daily-review` — 오늘의 작업을 정리하고 리뷰 초안 생성
- `/weekly-review` — 이번 주 성과/개선점을 정리하고 리뷰 초안 생성

## Environment
- Python 환경: `conda activate cuda` 필수 (easytorch, easydict, torch 등)
- SLURM 제출: `experiments/scripts/submit_job_cuda.sh` (gpu_cuda partition)

## Key Conventions
- Config 파일에서 import: `from baselines.X` (not `from other_baselines.X`)
- 3개월 데이터: `'data_range': (0, 24192)` in DATASET.PARAM
- 실험 실행: `python experiments/run_experiments.py --cfg <yaml> --arch=cuda`
- 결과 수집: `python experiments/get_results.py --config <yaml> --metrics MAE RMSE`
