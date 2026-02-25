# Ideas — Incident-Aware Traffic Forecasting

## Experiment Ideas
- **Weighted combined distance** — current combined distance averages normalized temporal + spatial equally. Try learned or tuned weights (e.g., 0.7 spatial + 0.3 temporal) since spatial structure may matter more for incident propagation.
- **Curriculum coreset** — start training with full data, progressively switch to coreset. Hypothesis: early epochs need diversity, later epochs benefit from focused subset.
- **Adaptive ratio per model** — STGCN optimal at 0.6, STAEformer at 0.8 (from K-Medoids experiment). Use model-specific ratios instead of fixed 0.3/0.7.
- **Incident-weighted coreset selection** — modify facility location or graph cut objective to upweight timestamps near incidents. Current selection is incident-agnostic.
- **Ensemble distance** — combine L2 and cosine distances (rank fusion or normalized sum) instead of choosing one.
- **Proxy metric as selection criterion** — use OT divergence or FL objective to pick the best coreset without training, then only train the top-k candidates.

## Research Questions
- FL vs k-medoids: 같은 목적함수(facility location)를 greedy vs PAM으로 푸는 차이. FL이 (1-1/e) 보장이 있지만, RBF similarity 변환으로 인해 k-medoids(raw distance)와 다른 결과 가능. Cosine pipeline에서는 더 유사할 수 있음.
- Graph cut temporal bias: RBF sigma(median heuristic)와 lambda=1.0이 원인. Lambda tuning이나 sigma 조정으로 개선 가능한지 Phase B에서 검증.
- Temporal representativeness와 downstream performance의 상관관계: DOW/TOD KL이 낮으면 MAE도 좋은가?
- Why does 60-80% coreset beat full data? Is it noise reduction, implicit regularization, or removal of redundant normal-state samples that dilute incident signal?
- Does the optimal distance function change across datasets (SAN_BERNARDINO vs ALAMEDA vs SACRAMENTO)?
- Is Graph Cut's seed sensitivity a fundamental issue or fixable with better initialization?
- Can coreset selection and severity-aware pre-training stack? Or do they solve the same underlying problem (representation quality)?
- For experience replay: why does it work for STGCN but not STAEformer? Is it related to model capacity or attention mechanism?

## Future Directions
- **Cross-dataset coreset transfer** — compute indices on SAN_BERNARDINO, apply temporal pattern to ALAMEDA. If coreset captures universal traffic patterns, indices may transfer.
- **Online coreset update** — for streaming traffic data, incrementally update coreset indices instead of recomputing from scratch.
- **Multi-task objective** — jointly optimize overall MAE and incident-specific MAE with Pareto optimization instead of treating incident improvement as a secondary goal.
- **Extend to other anomaly types** — construction zones, weather events, special events (sports games). Incident metadata supports this.

## Uncertain / Risky Ideas
- **Incident-conditioned graph rewiring** — dynamically modify adjacency matrix when incident is detected at a node (reduce edge weights to incident node). Risk: requires incident detection at inference time.
- **GAN-based incident augmentation** — generate synthetic incident patterns instead of rule-based injection in severity-aware pre-training. Risk: mode collapse, training instability.
- **Self-supervised anomaly detection as auxiliary task** — train a binary classifier (normal/incident) alongside prediction. Use classifier confidence to modulate prediction. Risk: chicken-and-egg problem if incidents are rare.
- **Attention masking for incident nodes** — in STAEformer, mask attention from normal nodes to incident nodes during incident timestamps. Risk: requires knowing incident status at inference.
