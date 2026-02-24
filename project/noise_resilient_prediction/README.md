# Noise-Resilient Spatiotemporal Prediction

> Combining Denoising SSL with Noise Augmentation for Robust Traffic Forecasting

## Directory Structure

```
project/noise_resilient_prediction/
  README.md                          # This file - project overview
  noise_augmentation_ssl_results.md  # Comprehensive experiment results & analysis
  paper_story.md                     # Paper narrative (symlink → docs/)
  analysis/
    eval_noise_vulnerability.py      # Symlink → experiments/eval_noise_vulnerability.py
    noise_vulnerability_results/     # Symlink → experiments/noise_vulnerability_results/
```

## Related Files (elsewhere in repo)

### Configs
- `baselines/STAEformer/SAN_BERNARDINO_5ch.py` - Baseline
- `baselines/STAEformer/SAN_BERNARDINO_5ch_noisy.py` - + Noise augmentation
- `baselines/STAEformer/SAN_BERNARDINO_5ch_denoising.py` - + Denoising encoder v1
- `baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_noisy.py` - + v1 combo
- `baselines/STAEformer/SAN_BERNARDINO_5ch_denoising_v2_noisy.py` - + v2 combo
- `baselines/STGCN/SAN_BERNARDINO*.py` - STGCN variants (same naming)

### Encoder Code
- `baselines/ContextContrastive/arch/denoising_encoder.py` - DenoisingEncoder (v1 + v2 residual)
- `baselines/ContextContrastive/arch/denoising_pretrain_model.py` - Pretrain wrapper
- `baselines/ContextContrastive/SAN_BERNARDINO/pretrain_denoising.py` - v1 pretrain config
- `baselines/ContextContrastive/SAN_BERNARDINO/pretrain_denoising_v2.py` - v2 pretrain config

### Runner
- `baselines/ContextContrastive/runner/noisy_representation_learning_runner.py` - Noise aug + encoder

### Evaluation
- `experiments/eval_noise_vulnerability.py` - Full noise robustness evaluation
- `experiments/noise_vulnerability_results/results.json` - Raw results

### Prior Documentation
- `docs/noise_resilient_prediction/paper_story.md` - Paper story (pre-v2)
- `docs/noise_resilient_prediction/experiment_insights_comprehensive.md` - SSL experiment history
- `docs/noise_resilient_prediction/attention_collapse_missing_values.md` - Attention collapse analysis
