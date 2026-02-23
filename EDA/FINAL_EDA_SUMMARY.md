# Final EDA Summary & Design Implications

## Key Numbers

| Metric | Value |
|--------|-------|
| Total incidents | 25,015 |
| **Severe incidents (z < -2)** | **335 (1.3%)** |
| Test set severe incidents | 120 |
| Average drop ratio (severe) | 69.2% |
| Unique nodes with severe | 147 / 893 |
| Mild incidents on same nodes | 10,891 |

## Critical Findings

### 1. Two Distinct Populations

```
Incident Data
├── Severe (1.3%, 335 cases)
│   ├── Actual flow drop: 69% average
│   ├── Zero flow: 35% of these
│   └── Recovery: ~15-30 minutes
│
└── Mild (98.7%, 24,680 cases)
    ├── No significant flow change
    └── z-score ≈ 0 (indistinguishable from normal)
```

### 2. Severe Incident Flow Pattern (CONFIRMED)

```
Time    | Flow   | Change
--------|--------|--------
t=-60m  | 123.5  | baseline
t=-30m  | 112.8  | -9% (early sign)
t=-15m  | 109.9  | -11%
t=0     | 94.5   | -23% (incident)
t=+15m  | 108.5  | recovery starts
t=+30m  | 113.6  | nearly recovered
```

**Pattern:** Gradual decline before → sharp drop at incident → recovery within 30min

### 3. Natural Contrastive Pairs Exist

- 147 nodes have **both** severe and mild incidents
- 10,891 mild incidents on these nodes
- **Same node, different severity = natural positive/negative pairs**

---

## Revised Design Recommendations

### 1. Focus on Severe Cases Only

**Evaluation:** Test only on 120 severe incidents in test set
- Current: MAE on all incident timestamps
- Proposed: MAE on **severe incident node at severe incident time**

### 2. Anomaly Injection - Match Real Patterns

```python
# OLD (unrealistic)
severity_levels = {
    1: (0.3, 0.5),  # too aggressive
    2: (0.5, 0.8),
    3: (0.8, 1.0),
}

# NEW (data-driven)
severity_levels = {
    1: (0.10, 0.25),  # mild: 10-25% drop
    2: (0.25, 0.50),  # moderate: 25-50% drop
    3: (0.50, 0.80),  # severe: 50-80% drop (matches 69% avg)
}

# Distribution (match real data)
severity_distribution = [0.60, 0.30, 0.10]  # mild:moderate:severe
```

### 3. Temporal Pattern in Augmentation

```python
def inject_anomaly(x, start_t, severity):
    """
    Match real pattern: gradual onset → peak → recovery
    """
    # Phase 1: Gradual onset (-3 to 0)
    for t in range(-3, 0):
        x[start_t + t] *= (1 - severity * (t + 4) / 4)

    # Phase 2: Peak drop (0 to +2)
    for t in range(0, 3):
        x[start_t + t] *= (1 - severity)

    # Phase 3: Recovery (+3 to +6)
    for t in range(3, 7):
        recovery = (t - 2) / 5
        x[start_t + t] *= (1 - severity * (1 - recovery))
```

### 4. Remove Neighbor Propagation

**Evidence:** Correlation between incident node and 1-hop neighbor = -0.02
**Action:** Remove `propagation_hops` and `propagation_decay` from design

### 5. Use Natural Pairs for Contrastive

Instead of synthetic augmentation alone:
```python
# Natural pairs from same node
positive = (node_i, time_normal)   # mild incident or normal
negative = (node_i, time_severe)   # severe incident

# This is more realistic than synthetic injection
```

---

## Updated Evaluation Strategy

### Primary Metric
```python
def evaluate_severe_incidents(model, test_loader, severe_incidents_df):
    """
    Evaluate ONLY on severe incident (node, time) pairs
    """
    severe_mae = []
    for _, row in severe_incidents_df.iterrows():
        node = row['sensor_idx']
        slot = row['incident_slot']

        pred = model_prediction[slot, node]
        true = ground_truth[slot, node]
        severe_mae.append(abs(pred - true))

    return np.mean(severe_mae)
```

### Secondary Metrics
1. Overall MAE (should not degrade significantly)
2. MAE on mild incidents (baseline)
3. MAE on non-incident timestamps (baseline)

### Success Criteria
- Severe incident MAE: **>20% improvement**
- Overall MAE: **<3% degradation**

---

## Files Created

| File | Purpose |
|------|---------|
| `severe_incidents.csv` | 335 severe incidents for evaluation |
| `eda_findings_summary.md` | Detailed findings |
| `FINAL_EDA_SUMMARY.md` | This summary |

---

## Next Steps

1. **Update design.md** with these findings
2. **Create severe_incident_evaluator.py** for targeted evaluation
3. **Simplify anomaly injection** (remove propagation, match real patterns)
4. **Consider using natural pairs** in addition to synthetic augmentation
