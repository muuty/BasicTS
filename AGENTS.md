# Repository Guidelines

## Project Structure & Module Organization
- `basicts/`: core library (datasets, runners, models, scalers, metrics).
- `baselines/`: baseline model implementations and config files.
- `experiments/`: training, evaluation, inference, and RQ queue scripts.
- `datasets/`: dataset artifacts used by configs.
- `tests/`: `unittest` suite (files named `test_*.py`).
- `examples/`: example configs and scripts.
- `tutorial/`: usage docs (see `tutorial/getting_started.md`).
- `assets/`, `docs/`, `scripts/`: images, documentation, utilities.

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt`: install Python deps.
- `python experiments/train.py -c examples/regular_config.py -g 0`: train with an example config.
- `python experiments/train.py -c baselines/GWNet/METR-LA.py --gpus '0'`: run a baseline config.
- `python experiments/evaluate.py -cfg <CONFIG>.py -ckpt <CHECKPOINT>.pth -g 0`: evaluate a checkpoint.
- `python tests/run_all_test.py`: run all unit tests.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and `snake_case` for functions/vars.
- Linting tools: `pylint` (`.pylintrc`) and `isort` (`.isort.cfg`).
- Baselines may include external code; formatting is not mandatory there.

## Testing Guidelines
- Framework: `unittest`.
- Test discovery pattern: `tests/**/test_*.py`.
- Prefer targeted runs during development (e.g., `python -m unittest tests.basicts_test.metrics_test.test_mae`).

## Commit & Pull Request Guidelines
- Recent commits use a prefix + emoji, e.g. `feat: 🎸 ...`, `fix: 🐛 ...`, `chore: 🤖 ...`, `tests: 📏 ...`, `docs: ✏️ ...`, `style: 💄 ...`.
- Open PRs to `main`, link related issues, keep changes scoped, and allow maintainer edits.
- PR title format follows `<mark> <title>` (see `tutorial/contribution_guidelines.md`).

## Security & Reporting
- Security issues should not be filed publicly; follow `SECURITY.md`.

## Agent-Specific Notes
- Record experiment results in `docs/representation_learning_experiment_log.md` with date, name, purpose, results, insights.
- For metrics, use `checkpoints/**/test_metrics.json` and report `overall.MAE`.
- Prefer `gpus` argument (e.g., `gpus='0'`) over `CUDA_VISIBLE_DEVICES`.

## Research & Design Notes (Context-Contrastive)
- Goal: reduce performance variance and improve robustness to incidents via context-contrastive representation learning (see `docs/design.md`).
- Representation design: encode temporal window, then separate node vs. context views; exclude the last timestamp from context to avoid leakage.
- Training strategy: two-stage is preferred (pre-train contrastive, then fine-tune forecasting). End-to-end is an alternative with loss weighting.
- Key hyperparameters to keep in mind: `d_model=64`, `num_heads=4`, contrastive `temperature=0.1`, `contrast_weight` in `0.1–1.0`.
- Stability: use gradient clipping (`max_norm=5.0`), L2-normalize representations before contrastive loss, and consider early stopping (patience ~20).
- Evaluation extras: report MAE/RMSE/MAPE, variance across seeds, and robustness split (incident vs normal).
