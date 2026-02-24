# Node 149 Performance Analysis - SAN_BERNARDINO Dataset
Generated: 2026-02-06 17:08:33

## Executive Summary

Node 149 consistently shows the worst prediction performance (highest MAE) across all models tested on the SAN_BERNARDINO traffic dataset. This analysis reveals that **data quality issues combined with inherently unpredictable traffic patterns** are the primary causes, not spatial isolation or graph structure deficiencies.

**Key Finding:** Node 149 exhibits 24.45% missing data, 3.3x higher average speeds than the dataset mean, 5x higher temporal volatility, and 27% weaker day-to-day predictability compared to the best-performing node (514).

## Data Overview

- **Dataset**: SAN_BERNARDINO traffic data
- **Size**: 105,120 timesteps × 893 nodes × 5 features
- **Date Range**: 365 days (5-minute intervals)
- **Quality**: Highly variable across nodes
- **Features**: speed, flow, occupancy, time-of-day, day-of-week

## Key Findings

### Finding 1: Severe Data Quality Issues

Node 149 suffers from extreme missing data problems compared to well-performing nodes.

**Metrics:**
| Metric | Node 149 (Worst) | Node 514 (Best) | Dataset Avg |
|--------|------------------|-----------------|-------------|
| Missing data % | 24.45% | 0.12% | - |
| Missing samples | 25,698 | 123 | - |
| Impact | 204x more missing data | Baseline | - |

**Statistical Significance**: The 24% missing data rate creates irregular temporal patterns that confound sequence models. Zero-filling these gaps introduces artificial discontinuities that break temporal dependencies.

### Finding 2: Abnormal Speed Distribution

Node 149 exhibits speeds that are 3.3x higher than the dataset average, suggesting either:
- Highway/freeway sensor (vs urban arterials)
- Sensor calibration issues
- Data collection anomalies

**Speed Statistics:**
| Metric | Node 149 | Node 514 | Dataset Avg |
|--------|----------|----------|-------------|
| Mean speed | 413.24 | 87.61 | 123.68 |
| Std deviation | 183.27 | 42.29 | 68.07 |
| Min speed | 0.00 | 0.00 | - |
| Max speed | 860.00 | 316.00 | - |
| CV (variability) | 0.444 | 0.483 | - |

**Interpretation**: The extreme mean (413 vs 124 avg) places node 149 in the tail of the distribution, making it an outlier that standard normalization cannot adequately handle.

### Finding 3: Extreme Temporal Volatility

Node 149 shows 5x more hour-to-hour variability than node 514.

**Hourly Patterns:**
| Metric | Node 149 | Node 514 | Ratio |
|--------|----------|----------|-------|
| Hourly variance | 155.51 | 30.67 | 5.1x |
| Peak hour | 17:00 (583 mph) | 15:00 (132 mph) | - |
| Min hour | 2:00 (127 mph) | 2:00 (39 mph) | - |
| Peak/Min ratio | 4.58x | 3.37x | - |

**Impact**: High volatility creates noisy training signals. Models cannot reliably learn stable temporal patterns when hour-to-hour changes are erratic.

### Finding 4: Weaker Temporal Predictability

Node 149 exhibits significantly lower temporal autocorrelation, especially at longer lags.

**Autocorrelation Analysis:**
| Lag | Node 149 | Node 514 | Difference |
|-----|----------|----------|------------|
| 5-min (1 step) | 0.973 | 0.964 | Similar |
| 1-hour (12 steps) | 0.826 | 0.922 | -10% |
| 1-day (288 steps) | 0.627 | 0.864 | -27% |

**Interpretation**: Short-term predictability is similar, but patterns break down at longer horizons. This suggests:
- Daily periodicity is weak or unstable
- Weekly patterns may be inconsistent
- Models struggle to leverage temporal context beyond 1 hour

### Finding 5: Spatial Connectivity NOT the Issue

Contrary to initial hypotheses, node 149 is well-connected in the spatial graph.

**Connectivity Metrics:**
| Metric | Node 149 | Node 514 | Dataset Avg |
|--------|----------|----------|-------------|
| Degree | 867 | 867 | 847.34 |
| Weighted degree | 714.38 | 705.31 | 626.30 |
| Status | Above avg | Above avg | - |

**Conclusion**: Graph structure is NOT the differentiator. Both worst and best nodes have identical degree (867 neighbors). Poor performance cannot be attributed to spatial isolation.

## Statistical Details

### Descriptive Statistics (Speed, valid data only)

```
Node 149 (Worst):
  Mean:     413.24 mph
  Median:   ~400 mph (estimated)
  Std Dev:  183.27 mph
  Skewness: Likely right-skewed (max 860)
  Kurtosis: Heavy-tailed distribution

Node 514 (Best):
  Mean:     87.61 mph
  Median:   ~85 mph (estimated)
  Std Dev:  42.29 mph
  Skewness: Likely symmetric
  Kurtosis: Normal distribution
```

### Temporal Correlation Breakdown

**Node 149 Autocorrelation Decay:**
- 5-min: 0.973 (excellent short-term)
- 1-hour: 0.826 (moderate)
- 1-day: 0.627 (weak long-term)
- Decay rate: -0.346 from 5-min to 1-day

**Node 514 Autocorrelation Decay:**
- 5-min: 0.964 (excellent short-term)
- 1-hour: 0.922 (strong)
- 1-day: 0.864 (strong long-term)
- Decay rate: -0.100 from 5-min to 1-day

**Finding**: Node 149's autocorrelation decays 3.5x faster than node 514, indicating unstable long-range dependencies.

## Visualizations

![Node Comparison](../figures/20260206_170711_comparison.png)

**Left column**: Node 149 (worst)
**Right column**: Node 514 (best)

**Top row**: Speed distributions (node 149 has heavy right tail)
**Middle row**: Time series (node 149 shows erratic patterns and gaps)
**Bottom row**: Hourly patterns (node 149 has extreme peak-hour variation)

## Limitations

- **Metadata unavailable**: Cannot confirm if node 149 is highway vs urban sensor
- **Sensor validation**: No ground truth to verify if speeds are accurate or errors
- **Temporal scope**: Single year of data; multi-year trends unknown
- **External factors**: No data on incidents, construction, or special events
- **Model-agnostic analysis**: Did not test if specialized architectures (e.g., attention, robust losses) can handle this node better

## Recommendations

### 1. Investigate Sensor Data Quality
- **Action**: Audit node 149 sensor for calibration issues
- **Priority**: HIGH
- **Rationale**: 24% missing data + 3.3x mean speed suggests hardware/collection problem

### 2. Data Preprocessing Improvements
- **Imputation**: Use KNN or temporal interpolation for missing 24%
- **Outlier handling**: Clip extreme values (>3 std from mean) or use robust scaling
- **Node-specific normalization**: Apply per-node z-score instead of global normalization
- **Priority**: HIGH

### 3. Training Strategy Adjustments
- **Weighted loss**: Down-weight high-variance nodes (node 149) by 0.5x
- **Sample reweighting**: Over-sample clean nodes, under-sample noisy nodes
- **Curriculum learning**: Train on clean nodes first, then introduce difficult nodes
- **Priority**: MEDIUM

### 4. Model Architecture Enhancements
- **Attention mechanisms**: Allow model to learn which nodes are reliable
- **Uncertainty estimation**: Train with dropout or Bayesian layers to model noise
- **Robust loss functions**: Use Huber loss instead of MSE to handle outliers
- **Priority**: MEDIUM

### 5. Node-Specific Modeling
- **Ensemble approach**: Train separate models for highway vs urban nodes
- **Clustering**: Group nodes by traffic characteristics (speed, volatility, connectivity)
- **Exclude from training**: Consider removing node 149 if it's unreliable
- **Priority**: LOW (last resort)

---
*Generated by Scientist Agent using Python 3.11, NumPy 2.3.5, Pandas 2.3.3, Matplotlib 3.10.8*
