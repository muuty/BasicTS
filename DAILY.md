# Daily Log — Robust Spatio-Temporal Traffic Forecasting

---

## 2026-02-25

### Done
- Checked all experiment completion: all 8+ RQ experiments from Feb 18 finished (GPUs idle)
- Collected test_metrics.json from ~30 completed downstream experiments
- **Critical discovery**: "_noisy" suffix models inject noise during TRAINING only — test_metrics.json are clean test results, not noisy test
- Ran full noise vulnerability evaluation (gaussian/bias/stuck/drift) on all 33 models via `eval_noise_vulnerability.py`
  - Previously only 11 models had full eval; 22 were dead-only → now all 33 complete
- Implemented noise robustness auto-evaluation in `base_tsf_runner.test()` via `CFG.EVAL.NOISE_ROBUSTNESS = True`
  - Created `basicts/runners/noise_eval.py` with 5 noise injection functions
  - Modified `base_tsf_runner.py` to run noise eval after clean test and save to test_metrics.json
- Created `project/noise_resilient_prediction/EXPERIMENTS.md` — clean RQ1-5 results reference

### Key Results
- **RQ2**: Denoising v2 + noisy training = 70% degradation reduction (avg +12.6% vs baseline +41.8%)
- **RQ2**: Spillover nearly eliminated (20.3% → 0.1%) by all defense approaches
- **RQ3**: MLP-only encoder (A2) surprisingly competitive with full model (+13.4% vs +12.6%)
- **RQ4**: Common noise pretrain generalizes to unseen stuck noise; structural pretrain does NOT generalize
- **RQ5**: STGCN also benefits (53% reduction) — approach is model-agnostic
- **Negative**: Reliability estimation and plug&play approaches ineffective

### Observations
- Trade-off: best robustness (v2+noisy, +12.6%) costs +0.68 clean MAE; denoising-only is best balance (12.05 MAE, +22.0%)
- Bias noise is most damaging across all models (+24-76% degradation)
- Dead noise remains unsolved by all approaches (~+260% degradation)

### Tomorrow
- Update experiment log with full noise eval results
- Consider additional ablations (A1: spatial-only, missing from current set)
- CONTRA_COSTA downstream experiment (pretrain ready, no downstream yet)
- Start paper Section 4 (Experiments) draft using EXPERIMENTS.md as reference

---

## 2026-02-24

### Done
- Set up research project management files (PROJECT.md, TASK.md, DAILY.md, WEEKLY.md, IDEAS.md)
- Reviewed experiment status: RQ1-5 experiments launched ~Feb 18 still running on GPU 1
- Reviewed EPN few-shot adaptation project status — pausing to focus on noise-resilient paper

### Observations / Results
- Branch is hotfix/deepar (DeepAR runner bug fix merged)
- 8+ concurrent experiments were running as of Feb 18 — need to check which have completed
- Denoising encoder v2 best result: STAEformer 5ch MAE=12.13 (vs baseline 12.26), with up to 14x noise degradation reduction
- EPN results compiled: 1-5% MAE improvement at 7d budget, but mean_init competitive (beats EPN on ~48% nodes)

### Blockers
- GPU 1 is the only available GPU — experiment throughput bottlenecked
- Need to check which of the 8+ experiments from Feb 18 have completed before queuing new ones

### Tomorrow
- Check experiment completion status (A7, 1ch, 3ch, cross-noise, CONTRA_COSTA)
- Collect test_metrics.json from completed runs
- Queue A2/A3/B2 ablation downstream if A7 is done
