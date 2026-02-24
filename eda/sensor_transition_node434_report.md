# Sensor Transition Analysis - Node 434

## Summary
**Target Node:** 434  
**Transition Day:** 63 (Week 9)  
**Split:** VAL (days 55-72)  
**Type:** PERMANENT failure

## Timeline

### Training Period (days 0-54)
- Node 434: **Functional** throughout training
- Mean flow: ~45 vehicles
- Zero rate: <0.5%
- Model learns Node 434 as a normal, functional sensor

### Validation Period (days 55-72)
- Days 55-62: Still functional (mean=46.32, zero_rate=0.00%)
- **Day 63: TRANSITION OCCURS**
- Days 63-72: Degraded (mean=19.02, zero_rate=57.24%)

### Test Period (days 73-91)
- Node 434: **Completely dead** (zero_rate=100.00%)
- Mean flow: 0.00
- No recovery observed

## Before/After Statistics

### Before Transition (days 42-62)
- Period: 21 days, entirely in TRAIN split
- Mean flow: 46.05
- Max flow: 123
- Zero rate: 0.00%
- Std: 23.66

### After Transition (days 63-83)
- Period: 21 days, spans VAL+TEST splits
- Mean flow: 6.34 (-86.2%)
- Max flow: 106
- Zero rate: 85.75%
- Std: 18.05

## Neighbor Analysis

**Total neighbors:** 867  
**Functional neighbors:** 623 (zero_rate < 10%, max > 50)

### Top 5 Neighbors by Adjacency Weight
1. Node 435: mean_flow=104.69, adj_weight=1.0000
2. Node 492: mean_flow=40.28, adj_weight=1.0000
3. Node 494: mean_flow=67.38, adj_weight=1.0000
4. Node 436: mean_flow=21.44, adj_weight=0.9998
5. Node 438: mean_flow=280.08, adj_weight=0.9998

## Spillover Analysis Strategy

### Experiment Design
1. **Baseline:** Train STAEformer on days 0-54 (all 893 nodes)
2. **Test:** Evaluate on days 73-91 
3. **Metric:** Per-node MAE on functional neighbors
4. **Compare:** With/without Node 434 in graph

### Expected Results
- **Hypothesis 1 (Attention Collapse):** 
  - Dead Node 434 attracts excessive attention
  - Functional neighbors show MAE increase
  - Evidence: Neighbors waste capacity on dead sensor

- **Hypothesis 2 (Robust Aggregation):**
  - Neighbors are unaffected by Node 434 failure
  - MAE remains constant
  - Evidence: Attention properly weights functional nodes

### Key Questions
1. Do Node 434's functional neighbors show MAE degradation in TEST?
2. Is the degradation correlated with adjacency weight to Node 434?
3. Does removing Node 434 from the graph improve neighbor predictions?

## Files Generated
- `eda/sensor_transition_node434.npy`: Full analysis results
- `eda/node434_functional_neighbors.npy`: Neighbor indices (top 20)

## Next Steps
1. Run baseline STAEformer on data_range=(0, 26280) [DONE in existing experiments]
2. Extract per-node MAE for Node 434's neighbors in test period
3. Train comparison model excluding Node 434
4. Compute spillover effect = MAE_with_434 - MAE_without_434
