# Related Work: Continual Traffic Forecasting on Expanding Networks

## Overview

This document surveys papers addressing continual/incremental learning for traffic forecasting, particularly when the sensor network **physically expands** (new nodes added over time). We focus on: (1) whether the method is model-agnostic, (2) experiment scenarios, and (3) how new nodes are handled.

---

## 1. TrafficStream (IJCAI 2021)

**Title:** TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning
**Authors:** Xu Chen, Junshan Wang, Kunqing Xie
**Venue:** IJCAI 2021
**Links:** [Paper](https://www.ijcai.org/proceedings/2021/498) | [arXiv](https://arxiv.org/abs/2106.06273) | [Code](https://github.com/AprLie/TrafficStream)

### Problem
First paper to address streaming traffic forecasting with expanding networks. Existing methods assume static networks; TrafficStream handles long-term evolution.

### Method
- **Architecture-dependent**: Built on GNN (not model-agnostic)
- **Two CL strategies**:
  1. Historical data replay (store and replay past data)
  2. Parameter smoothing (regularization to preserve old weights)
- **Pattern fusion**: JS-divergence-based algorithm to detect newly emerged traffic patterns, which are fused into the model

### Experiment Scenario
- **Dataset**: PEMS3-Stream (CalTrans PeMS District 3)
- **Time span**: 2011-2017 (7 yearly periods)
- **Network expansion**: Gradual sensor addition each year
- **Metrics**: MAE, RMSE, MAPE

### Key Findings
- Without consolidation → worst results (forgetting confirmed)
- Data replay > parameter smoothing (replay is stronger)
- Combining both yields best results
- Pioneer work; subsequent papers (PECPM, STKEC, EAC) all compare against it

### Limitations
- Not model-agnostic (GNN-specific)
- Requires storing historical data for replay
- Updates entire model (expensive)

---

## 2. STKEC / Knowledge Expansion and Consolidation (IEEE TITS 2023)

**Title:** Knowledge Expansion and Consolidation for Continual Traffic Prediction with Expanding Graphs
**Authors:** Binwu Wang, Yudong Zhang, Jiahao Shi, Pengkun Wang et al. (USTC)
**Venue:** IEEE Transactions on Intelligent Transportation Systems, 2023, Vol.24(7), pp.7190-7201
**Links:** [IEEE Xplore](https://ieeexplore.ieee.org/document/10101714/)

### Problem
Continual traffic prediction with expanding road networks under the continual learning paradigm.

### Method
- **Knowledge Expansion**: Detect evolved and new patterns from updated network
- **Knowledge Consolidation**: Bank preservation + pattern traceability to retain learned patterns without needing historical graph data
- Extends concepts from TrafficStream with prototype-based enhancements

### Experiment Scenario
- **Dataset**: PEMS3-Stream (same as TrafficStream)
- **Time span**: 2011-2017

### Key Findings
- Improved over TrafficStream by not requiring full historical data replay
- Pattern bank approach reduces memory requirements

---

## 3. PECPM (KDD 2023)

**Title:** Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction
**Authors:** Binwu Wang, Yudong Zhang, Xu Wang, Pengkun Wang et al. (USTC)
**Venue:** KDD 2023
**Links:** [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599463) | [PDF](http://home.ustc.edu.cn/~pengkun/files/Publications/KDD2023_2.pdf)

### Problem
Same as STKEC (journal extension). Continual traffic prediction on evolving/expanding graphs without access to historical graph data.

### Method
- **Pattern Bank**: Stores representative spatiotemporal patterns (size K=50)
- **Pattern Matching**: Retrieve best-matching patterns from bank to predict
- **Pattern Expansion**: Detect "conflict nodes" where patterns significantly changed → expand bank with new patterns
- **Pattern Consolidation**: Bank preservation + pattern traceability
- **Model-agnostic claim**: "deployed to multiple advanced spatiotemporal learning models to demonstrate wide applicability"

### Experiment Scenario
- **Dataset**: PEMS3-Stream (CalTrans PeMS District 3, July 10 - Aug 9, 2011-2017)
- **Input/Output**: 12 timesteps → 12 timesteps (15/30/60-min granularities)
- **Optimizer**: AdamW, lr=1e-3, batch=128, max 100 epochs + early stopping

### Key Findings
- Only fine-tunes on new nodes + conflict nodes (not all nodes) → 3x faster than retraining
- Better than SurSTG-Retrain (full retraining) in both performance and efficiency
- SurSTG-Static (no adaptation) performs worst
- SurSTG-Expand (fine-tune only new nodes) also insufficient

### Relevance to Our Work
- Pattern Bank concept is similar to our Pattern Bank (PB) approach
- Their K=50 representative patterns vs our K-means factorized embeddings
- Key difference: they use it for replay/consolidation, we explored it for embedding initialization

---

## 4. TFMoE (arXiv 2024)

**Title:** Continual Traffic Forecasting via Mixture of Experts
**Authors:** Sanghyun Lee, Junhyeok Park (presumed)
**Venue:** arXiv 2024
**Links:** [arXiv](https://arxiv.org/abs/2406.03140) | [OpenReview](https://openreview.net/forum?id=vJGKYWC8j8)

### Problem
Expanding traffic networks cause catastrophic forgetting. Retraining is inefficient.

### Method
- **MoE architecture**: Segment traffic data into homogeneous groups (regardless of geography), each with its own expert model
- **Expert model**: VAE reconstructor + Predictor (Diffusion Conv + 1D-Conv layers)
- **New node assignment**: Reconstruction probability — expert whose VAE best reconstructs the new sensor's 1-week data gets that sensor
- **Three anti-forgetting strategies**:
  1. Reconstructor-based knowledge consolidation loss
  2. Forgetting-resilient sampling (VAE generates synthetic past data)
  3. Reconstruction-based replay
- **Model-agnostic at predictor level**: "more advanced prediction models can be integrated"

### Experiment Scenario
- **Dataset**: PEMSD3-Stream
- **Baselines**: TrafficStream + others (details in appendix)

### Key Findings
- Generative replay (from VAE) avoids storing historical data
- Expert specialization prevents interference between different traffic patterns
- Each model handles a homogeneous group → less forgetting

### Relevance to Our Work
- New node assignment via reconstruction probability is clever but requires 1 week of data
- Our setting considers much smaller budgets (3h, 12h)
- MoE is a fundamentally different paradigm (multiple models vs single model adaptation)

---

## 5. EAC - Expand and Compress (ICLR 2025)

**Title:** Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting
**Authors:** (ICLR 2025)
**Links:** [arXiv](https://arxiv.org/abs/2410.12593) | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cb2266111eadcfa2c02187ace64e2183-Abstract-Conference.html)

### Problem
Existing continual ST forecasting methods optimize the entire STGNN → expensive and forgetting-prone.

### Method
- **Prompt tuning**: Freeze the base STGNN backbone, only learn per-node prompt vectors
- **Expand**: Each new period τ adds learnable prompt vectors P^(τ) for new nodes to a prompt pool
- **Compress**: Low-rank factorization P ≈ AB where A∈R^(n×k) (node-specific, k=6) and B∈R^(k×d) (shared)
- New periods only add small A^(τ) for new nodes; B stays constant
- Prompts are added element-wise to input node features
- **Model-agnostic**: YES - explicitly tested "universality" across spectral, spatial, recurrent, convolution, and attention-based STGNN backbones

### Experiment Scenario
- **PEMS-Stream**: 655→871 nodes, 7 periods (2011-2017), Traffic flow, Northern California
- **Air-Stream**: 1087→1202 nodes, 4 periods (2016-2019), Air quality, China
- **Energy-Stream**: 103→134 nodes, 4 periods (245 days), Wind power
- **Baselines**: Pretrain-ST, Retrain-ST, Online-ST variants, TrafficStream, STKEC, PECPM, TFMoE

### Key Results (PEMS-Stream)
| Method | MAE |
|---|---|
| EAC | ~13.9 |
| Online-ST-AN | 14.08 |
| TrafficStream | 14.22 |
| STKEC | 14.14 |

### Relevance to Our Work (HIGH)
- **Most relevant paper**: Also proposes to freeze backbone and only tune node-level parameters
- Prompt vectors are conceptually similar to node embeddings
- Low-rank factorization (Compress) parallels our Pattern Bank idea
- Key difference: EAC adds prompts to ALL nodes (including existing), we only adapt new node embeddings
- Key difference: EAC has full training data for each period, we study few-shot (3h-7d) budgets
- **Our advantage**: We enforce zero forgetting for existing nodes by gradient masking; EAC allows some drift via shared B matrix

---

## 6. STGNN-CL (Complex & Intelligent Systems, 2025)

**Title:** A Framework for Continual Learning in Real-Time Traffic Forecasting Utilizing Spatial-Temporal Graph Convolutional Recurrent Networks
**Venue:** Complex & Intelligent Systems, 2025
**Links:** [Springer](https://link.springer.com/article/10.1007/s40747-025-02049-7)

### Problem
Continual learning for persistent long-term traffic prediction.

### Method
- **CL techniques**: EWC, MAS, SI integrated into STGNN
- **Pattern Fusion**: KL-divergence to merge spatial and temporal patterns
- **Architecture**: GCN (spatial) + LSTM (temporal) + Pattern Fusion

### Experiment Scenario
- **Datasets**: PeMSD3, PeMSD4, PeMSD7, PeMSD8
- **Focus**: Temporal distribution shift (not explicitly node expansion)

### Relevance to Our Work
- Focuses more on temporal drift than network expansion
- EWC/MAS/SI are general CL techniques we could incorporate
- Less relevant since they don't study new node onboarding

---

## 7. STGNN for Expanding Traffic Network (DEXA 2024)

**Title:** Exploring of STGNN for Traffic Forecasting at Expanding Traffic Network
**Venue:** DEXA 2024 (Database and Expert Systems Applications)
**Links:** [Springer](https://link.springer.com/chapter/10.1007/978-3-031-68312-1_9)

### Problem
When traffic network expands, models need re-training but new nodes have insufficient data.

### Method
- **Transfer learning**: Fine-tune pre-trained model from before expansion
- Uses STGNN that can "store information about a city and consider long-term time series"

### Relevance to Our Work (MODERATE)
- Same problem setting as ours
- Simple fine-tuning baseline (similar to our full_model method)
- Less sophisticated than EAC or PECPM

---

## 8. XXLTraffic (arXiv 2024)

**Title:** XXLTraffic: Expanding and Extremely Long Traffic Dataset for Ultra-Dynamic Forecasting Challenges
**Links:** [arXiv](https://arxiv.org/abs/2406.12693)

### Overview
Not a method paper but a **dataset** contribution.
- **Source**: CalTrans PeMS, 9 districts
- **Scale**: Up to 23 years (2001-2024), up to 4,888 nodes per district
- **Network expansion**: Naturally captured (sensors added over years)
- **Gap forecasting**: Tests with 1-2 year gaps reveal severe domain shift
- All standard baselines perform poorly on long-gap forecasting

### Relevance to Our Work
- Potential future dataset for larger-scale evaluation
- Confirms that temporal distribution shift is a real, unsolved problem
- Complements the PEMS3-Stream used by most related work

---

## Comparison Table

| Paper | Venue | Model-Agnostic? | Dataset | Node Expansion | CL Approach | New Node Init | Few-Shot? |
|---|---|---|---|---|---|---|---|
| **TrafficStream** | IJCAI'21 | No (GNN-specific) | PEMS3-Stream | 2011-2017 yearly | Replay + Regularization | Not detailed | No |
| **STKEC** | TITS'23 | Partial | PEMS3-Stream | 2011-2017 yearly | Pattern Bank + Consolidation | Pattern matching | No |
| **PECPM** | KDD'23 | Yes (demonstrated) | PEMS3-Stream | 2011-2017 yearly | Pattern Bank + Conflict Detection | Pattern matching | No |
| **TFMoE** | arXiv'24 | Yes (predictor) | PEMSD3-Stream | Streaming | MoE + Generative Replay | Reconstruction prob. | No |
| **EAC** | ICLR'25 | Yes (tested 5 types) | PEMS/Air/Energy | Multi-year | Prompt Tuning (freeze backbone) | Learnable prompts | No |
| **STGNN-CL** | CIS'25 | No (GCN+LSTM) | PeMSD3/4/7/8 | No expansion | EWC/MAS/SI | N/A | No |
| **DEXA'24** | DEXA'24 | Partial | Not specified | Yes | Fine-tuning | Transfer learning | Mentioned |
| **Ours** | - | Yes (any STGNN) | SAN_BERNARDINO | Quarterly (Q1-Q4) | Embedding-only FT + Gradient Mask | Mean init | **Yes (3h-7d)** |

---

## Key Gaps in Existing Work (Our Contributions)

### 1. No Few-Shot / Low-Budget Study
All existing papers assume sufficient training data for each expansion period (typically months to a full year). **None study the few-shot scenario** where only hours to days of data are available for new nodes. This is our primary differentiator.

### 2. No Explicit Zero-Forgetting Guarantee
- TrafficStream/PECPM: Aim to minimize forgetting but allow some drift
- EAC: Shared B matrix can cause interference
- **Our gradient masking**: Mathematically guarantees zero forgetting for existing nodes (frozen embeddings + gradient mask)

### 3. Embedding-Centric Analysis Missing
No existing paper provides a systematic analysis of which model components are N-dependent across different STGNN architectures. Our analysis shows **node identity embeddings are the only N-dependent learned parameters** across 5 major STGNNs (STAEformer, STGCN, AGCRN, GWNet, DCRNN), motivating an embedding-only adaptation approach.

### 4. Real-World Expansion Scenario
Most papers use PEMS3-Stream (yearly expansion over 2011-2017). We use SAN_BERNARDINO with **quarterly** expansion (Q1:558→Q2:669→Q3:781→Q4:893 nodes), which is a more granular and realistic expansion scenario.

### 5. Budget-Performance Tradeoff
We explicitly study the **data budget vs. adaptation quality** tradeoff (3h, 12h, 1d, 7d, full), providing practical guidance for deployment.

---

## Most Relevant Comparisons

### EAC (ICLR 2025) - Most Similar
- Both: freeze backbone, tune node-level parameters only
- EAC: prompt vectors for all nodes; Ours: embeddings for new nodes only
- EAC: full data; Ours: few-shot budgets
- EAC: low-rank factorization; Ours: direct embedding learning
- **Our advantage**: Zero forgetting guarantee, few-shot study

### PECPM (KDD 2023) - Similar Pattern Bank Concept
- Both: pattern bank / prototype concept for node representation
- PECPM: conflict detection for evolved patterns; Ours: K-means factorization for initialization
- PECPM: full retraining on conflict nodes; Ours: embedding-only fine-tuning
- **Our advantage**: More parameter-efficient, explicit forgetting analysis

### TrafficStream (IJCAI 2021) - Pioneer Baseline
- TrafficStream: replay + regularization (classic CL)
- Ours: embedding-only approach (much more parameter-efficient)
- **Our advantage**: Model-agnostic, zero forgetting, few-shot capable
