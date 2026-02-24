# Daily Log — Robust Spatio-Temporal Traffic Forecasting

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
