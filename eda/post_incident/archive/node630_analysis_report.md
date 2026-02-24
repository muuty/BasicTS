# Node 630 Analysis Report: Why It's The Worst Performing Node
Generated: 2026-02-12 14:16:26

## Executive Summary

Node 630 exhibits **108.00 MAE**, making it the worst-performing node in the SAN_BERNARDINO dataset - **10.9x worse** than the overall mean (9.92) and **97.5x worse** than the best functional nodes (1.02). Analysis reveals that node 630 is a **high-traffic, high-variance highway sensor** with extreme volatility that makes prediction extremely difficult.

**Key Finding**: Node 630 combines high traffic volume (446 vehicles/interval, 2.6x above average) with very high absolute volatility (26 vehicles/step average change, **23x higher than easy nodes**). While its coefficient of variation (0.45) is actually **lower** than typical nodes (0.60), the absolute magnitude of changes is what breaks the model.

## Node 630 Characteristics

### Traffic Statistics
| Metric | Node 630 | Median Functional | Easy Nodes (avg) | Overall |
|--------|----------|-------------------|------------------|---------|
| **MAE** | **108.00** | 11.08 | 1.02 | 9.92 |
| Mean Flow | 446.35 | 139.67 | 82.74 | 120.27 |
| Std Dev | 201.42 | 98.45 | 44.22 | 154.84 |
| Variance | 40,568 | 9,692 | 2,059 | - |
| CV | 0.451 | 0.704 | 0.547 | 0.596 |
| Zero Rate | 0.000 | 0.001 | 0.003 | - |

### Predictability Metrics
| Metric | Node 630 | Median Node | Easy Nodes | Interpretation |
|--------|----------|-------------|------------|----------------|
| Autocorr (lag-1) | 0.982 | 0.984 | **0.999** | Node 630 has **lower** short-term predictability |
| Autocorr (lag-12, 1hr) | 0.881 | 0.931 | - | Weaker hourly patterns |
| Autocorr (lag-288, 1day) | 0.879 | - | - | Moderate daily patterns |
| Avg Absolute Change | **25.91** | 12.48 | 1.12 | **23x higher volatility** than easy nodes |
| Max Single Change | 499.00 | - | - | Extreme spike events |

### Spatial Context
- **Neighbors**: 868 nodes (fully connected network)
- **Node 630 flow**: 3.74x neighbor average (446 vs 119)
- **Node 630 MAE**: 11.04x neighbor average (108 vs 9.79)
- **Conclusion**: Node 630 is a **spatial outlier** - high-traffic node in a network of lower-traffic sensors

## Why Prediction Fails

### 1. **High Absolute Volatility** (Primary Cause)
Node 630's average absolute change is **25.91 vehicles/step**, compared to:
- Median functional node: 12.48 (2.1x less)
- Easy nodes: 1.12 (23x less)

**Impact**: The model's 12-step ahead prediction must forecast through ~12 × 25.91 = 311 cumulative vehicles of change. Small errors compound rapidly.

### 2. **Lower Short-Term Autocorrelation**
Node 630's lag-1 autocorrelation (0.982) is **lower** than easy nodes (0.999), meaning:
- Easy nodes: flow at time t+1 is almost perfectly predicted by flow at time t
- Node 630: more variability between consecutive steps

**Impact**: STAEformer's temporal attention struggles when autocorrelation <0.99.

### 3. **High Traffic Volume** (Secondary Factor)
Node 630's mean flow (446) is **2.6x the functional average** (171). Higher volume means:
- Absolute errors scale linearly with traffic volume
- MAE=108 on mean=446 is actually 24% MAPE (not terrible)
- But MAE metric penalizes high-volume nodes

### 4. **Spatial Mismatch**
Node 630 is surrounded by much lower-traffic neighbors (119 avg). Spatial attention cannot help prediction because:
- Neighbor patterns don't transfer (flow magnitude mismatch)
- High heterogeneity in local network structure

## Top 10 Worst Nodes: Common Patterns

All worst nodes share similar characteristics:

| Characteristic | Worst 10 Avg | Overall Avg | Ratio |
|----------------|--------------|-------------|-------|
| Mean Flow | 439.34 | 171.32 | **2.56x** higher traffic |
| Variance | 42,180 | 11,648 | **3.62x** higher variance |
| Avg Change | 39.42 | 13.67 | **2.88x** higher volatility |
| CV | 0.481 | 0.596 | 0.81x (lower!) |
| Autocorr (lag-1) | 0.947 | 0.960 | Slightly worse |

**Key Pattern**: Worst nodes are **high-traffic highway sensors** with:
1. 2-3x higher traffic volume than average
2. 3-4x higher absolute variance
3. Similar or **lower** relative variability (CV)
4. But absolute changes break the model

### Correlation Analysis (Worst 10 Nodes)
Within the worst 10 nodes, **none of the standard metrics strongly correlate with MAE**:
- Flow vs MAE: r=0.346 (p=0.33) - weak, not significant
- Variance vs MAE: r=0.252 (p=0.48) - weak
- CV vs MAE: r=-0.148 (p=0.68) - no relationship
- Avg Change vs MAE: r=0.249 (p=0.49) - weak

**Interpretation**: Once a node is in the "hard" category, MAE differences are driven by factors beyond simple volume/variance.

## Distribution Analysis

### Node 630 Flow Distribution
- **Skewness**: 0.391 (slightly right-skewed, some high-flow periods)
- **Kurtosis**: -0.519 (platykurtic - flatter than normal, less extreme outliers)
- **Median**: 433.00 (close to mean 446, symmetric distribution)
- **IQR**: 231.00 (moderate spread)

### Comparison
- Median node skewness: -0.020 (nearly symmetric)
- Median node kurtosis: -1.548 (even flatter)

**Insight**: Node 630's distribution is relatively well-behaved (not heavy-tailed, not extreme outliers). The problem is **temporal volatility**, not anomalous values.

## Temporal Patterns

### Daily Profile
- **Node 630 variability**: σ=139.28 (mean daily std dev across time-of-day)
- **Median node variability**: σ=30.35
- **Ratio**: **4.59x higher intra-day variability**

Node 630 shows large fluctuations throughout the day, with peak variability (σ=197) during rush hours. The model struggles to learn consistent patterns when day-to-day variance is this high.

## Why Node 630 is Worst in BOTH Flow and Speed

Node 630 speed prediction MAE: **8.40 mph** (overall mean: 1.04 mph)
- **8.1x worse** than average speed prediction

**Explanation**: Flow and speed are coupled via fundamental traffic equations:
- Flow = Density × Speed
- High flow variance → high speed variance
- Congestion events cause simultaneous flow spikes and speed drops
- Model must predict coupled dynamics, amplifying errors

## Implications for Model Improvement

### 1. **Volume-Weighted Loss** (Recommended)
Current MAE treats all nodes equally. High-volume nodes like 630 dominate loss:
- Node 630: 1 node, contributes 108/893 = 12.1% of total MAE budget
- Bottom 90%: 803 nodes, contribute 88% of total MAE

**Solution**: Weight loss by `1 / sqrt(traffic_volume)` to balance contributions.

### 2. **Separate High-Traffic Models**
Train specialized models for high-volume nodes (>300 vehicles/interval):
- Different architecture (possibly deeper, more parameters)
- Higher tolerance for volatility
- Volume-specific normalization

### 3. **Volatility-Aware Attention**
Modify spatial attention to downweight high-volatility neighbors:
- Current: uniform neighbor aggregation
- Proposed: weight by `exp(-|volatility_i - volatility_j|)`

### 4. **Multi-Scale Temporal Modeling**
Node 630's autocorrelation drops faster than easy nodes:
- Add explicit multi-scale temporal branches (5-min, 1-hour, 1-day)
- Ensemble predictions at different time scales

### 5. **Robust Loss Function**
MAE is sensitive to outliers. Consider:
- Huber loss (L1 + L2 hybrid)
- Log-transformed targets for high-volume nodes
- Quantile regression (predict median instead of mean)

## Limitations

- **Single dataset**: Analysis based on 3 months of SAN_BERNARDINO data
- **No incident data**: Cannot rule out accidents/construction affecting node 630
- **No spatial coordinates**: Cannot assess if node 630 is a freeway merge/exit
- **No ground truth validation**: Sensor 630 could be malfunctioning (but no evidence of anomalies)

## Recommendations

1. **Prioritize high-traffic nodes in model design** - they drive overall error
2. **Use volume-weighted or log-scaled loss** to balance node contributions
3. **Investigate spatial heterogeneity modeling** - don't assume neighbors help
4. **Consider separate models for highway vs arterial sensors** (if metadata available)

---
*Analysis performed on SAN_BERNARDINO dataset (893 nodes, 26280 timesteps)*
*Node 630 MAE: 108.00 (flow), 8.40 (speed)*
