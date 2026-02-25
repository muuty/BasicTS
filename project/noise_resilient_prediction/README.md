# Noise-Resilient Spatiotemporal Prediction

> Combining Denoising SSL with Noise Augmentation for Robust Traffic Forecasting

## Documentation

| File | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | **Start here** — 1-page summary of key findings and recommendations |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Detailed results by RQ, full per-scenario tables, cross-dataset analysis |
| [experiment_log.md](experiment_log.md) | Chronological experiment record (2026-02-09 onwards) |
| [experiment_log_archive.md](experiment_log_archive.md) | Archived: contrastive pre-training, MICL (failed approaches) |
| [noise_augmentation_ssl_results.md](noise_augmentation_ssl_results.md) | Superseded by EXPERIMENTS.md appendix; kept for reference |
| [attention_collapse_missing_values.md](attention_collapse_missing_values.md) | Standalone analysis: spatial attention + dead sensors |
| [noise_patterns_san_bernardino.md](noise_patterns_san_bernardino.md) | Real noise pattern analysis from SB data |
| [paper.md](paper.md) | Internal paper draft |

## Related Files (elsewhere in repo)

### Configs
- `baselines/STAEformer/SAN_BERNARDINO_5ch*.py` — SB STAEformer variants
- `baselines/STAEformer/CONTRA_COSTA_5ch*.py` — CC STAEformer variants
- `baselines/STGCN/SAN_BERNARDINO*.py` — STGCN variants

### Encoder Code
- `baselines/ContextContrastive/arch/denoising_encoder.py` — DenoisingEncoder (v1 + v2 residual)
- `baselines/ContextContrastive/arch/denoising_pretrain_model.py` — Pretrain wrapper
- `baselines/ContextContrastive/SAN_BERNARDINO/pretrain_denoising*.py` — SB pretrain configs
- `baselines/ContextContrastive/CONTRA_COSTA/pretrain_denoising.py` — CC pretrain config

### Runner
- `baselines/ContextContrastive/runner/noisy_representation_learning_runner.py` — Noise aug + encoder

### Evaluation
- `experiments/eval_noise_vulnerability.py` — Noise eval (SAN_BERNARDINO, 33 models)
- `experiments/eval_contra_costa.py` — Noise eval (CONTRA_COSTA, 4 models)
- `experiments/compare_datasets.py` — CC vs SB dataset structural comparison
- `experiments/noise_vulnerability_results/` — Raw results JSON
