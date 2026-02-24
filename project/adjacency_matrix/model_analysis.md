# Model Analysis: Adjacency Matrix Handling

## Summary Table

| Model | Cat | Graph Conv Type | Adj Input Method | Supports Multiple Adj | Adj Swap Ease | Code Location |
|-------|-----|----------------|------------------|-----------------------|---------------|---------------|
| **STGCN** | C1 | Chebyshev spectral | Constructor `gso`, baked into `ChebGraphConv` | No (single) | Hard | `other_baselines/STGCN/` |
| **STGODE** | C1 | ODE-based diffusion | Constructor `A_sp_hat`, `A_se_hat` (2 matrices) | Yes (spatial + semantic) | Hard | `other_baselines/STGODE/` |
| **GWNet** | C2 | Diffusion + adaptive | `self.supports` (list) + learned `nodevec1/nodevec2` | Yes (list) | **Easy** | `other_baselines/GWNet/` |
| **AGCRN** | C2 | Adaptive Chebyshev | No predefined adj; `node_embeddings` generate graph | N/A (fully learned) | N/A | `other_baselines/AGCRN/` |
| **MTGNN** | C2 | Mixprop diffusion | `predefined_A` + learned `graph_constructor` | Partial | **Easy** | `other_baselines/MTGNN/` |
| **STWave+** | C3 | Sparse attention + wavelet | `adj_gat` (k-NN) + `graphwave` (eigenvectors) | Partial | Medium | `other_baselines/STWave/` |
| **GMAN** | C3 | Attention-based | Not in codebase | - | - | 없음 |
| **STAEFormer** | C4 | Full attention | No predefined adj; attention is the adjacency | N/A | N/A | `baselines/STAEformer/` |
| **DGCRN** | C4 | Diffusion (dynamic + predefined) | `predefined_A` (list of 2) + dynamic per-step | Yes | Medium | `other_baselines/DGCRN/` |
| **DCRNN** | C1 | Diffusion (random walk) | `adj_mx` (list), stored in each `DCGRUCell` | Yes (list of supports) | Hard | `other_baselines/DCRNN/` |
| **DFDGCN** | C2+ | Diffusion + adaptive + dynamic freq | `supports` (list) + adaptive + dynamic | Yes | **Easy** | `other_baselines/DFDGCN/` |
| **D2STGNN** | C4 | ST-localized diffusion | `model_args['adjs']` + static hidden + dynamic | Yes | Medium | `other_baselines/D2STGNN/` |
| **MegaCRN** | - | Chebyshev (memory-derived) | No predefined adj; memory module generates graph | N/A | N/A | `other_baselines/MegaCRN/` |
| **StemGNN** | - | Spectral (Chebyshev + FFT) | No predefined adj; self-attention generates graph | N/A | N/A | `other_baselines/StemGNN/` |
| **BigST** | - | Linearized attention | Optional `supports`, but primary conv is implicit | N/A | N/A | `other_baselines/BigST/` |
| **WaveNet** | - | None (pure temporal) | No adjacency | N/A | N/A | `other_baselines/WaveNet/` |

## Detailed Architecture Notes

### STGCN (C1 - Static Predefined)
- **Key files**: `stgcn_arch.py`, `stgcn_layers.py`
- **Adj flow**: Config `load_adj()` → `MODEL_PARAM["adj_matrix"]` → `encoder(gso=adj_matrix)` → `STConvBlock.graph_conv.cheb_graph_conv.gso`
- **Graph conv**: `torch.einsum('hi,btij->bthj', self.gso, x)` with Chebyshev polynomial recurrence
- **Swap method**: Traverse `model.st_blocks[*].graph_conv.cheb_graph_conv.gso` and replace each

### STGODE (C1 - Static Predefined, Dual Graph)
- **Key files**: `stgode_arch.py`, `odegcn.py`
- **Adj flow**: Two separate adj matrices: `A_sp_hat` (spatial distance) and `A_se_hat` (semantic/DTW)
- **Graph conv**: `einsum('ij, kjlm->kilm', adj, x)` inside Neural ODE dynamics
- **Swap method**: Very hard — adj embedded in `ODEFunc` + `nn.Parameter(alpha)` sized to adj; changing shape requires model reconstruction
- **Note**: 이미 두 가지 adj를 비교하는 구조 (spatial vs semantic), 우리 연구에 좋은 baseline

### GWNet (C2 - Adaptive + Predefined)
- **Key files**: `gwnet_arch.py`
- **Adj flow**: `self.supports` (list of predefined) + `softmax(ReLU(nodevec1 @ nodevec2.T))` (adaptive)
- **Graph conv**: Power series diffusion over each support matrix
- **Swap method**: `model.supports = [new_adj_fwd, new_adj_bwd]` — **가장 쉬움**
- **실험 가능**: predefined만 / adaptive만 / 둘 다 / predefined 교체

### AGCRN (C2 - Fully Adaptive)
- **Key files**: `agcrn_arch.py`, `agcn.py`
- **Adj flow**: `self.node_embeddings` → `softmax(ReLU(E @ E.T))` → Chebyshev expansion
- **Graph conv**: Node-adaptive weights via `einsum('nd,dkio->nkio', embeddings, weights_pool)`
- **Swap method**: No predefined adj to swap. Potential: initialize `node_embeddings` from spatial info, or add predefined adj as additional input
- **실험 가능**: "no graph" baseline으로서 의미 있음

### MTGNN (C2 - Mixprop + Adaptive)
- **Key files**: `mtgnn_arch.py`, `mtgnn_layers.py`
- **Adj flow**: `predefined_A` (optional) + `graph_constructor` from learned embeddings
- **Graph conv**: `mixprop` — alpha-blended random walk: `h = alpha*x + (1-alpha)*A*h`
- **Swap method**: `model.predefined_A = new_adj` — **쉬움**

### STWave+ (C3 - Sparse Attention)
- **Key files**: `stwave_arch.py`
- **Adj flow**: `adj_gat` (k-NN indices for sparse attention) + `graphwave` (eigenvectors for positional encoding)
- **Graph conv**: Sparse Q-K-V attention using adj as neighbor mask
- **Swap method**: Update `.adj`, `.eigvec`, `.eigvalue` on each `dualEncoder` layer
- **실험 가능**: k-NN adj 교체, attention mask 방법 비교

### STAEFormer (C4 - Full Attention)
- **Key files**: `baselines/STAEformer/arch/staeformer_arch.py`
- **Adj flow**: No predefined adj. Spatial attention computes (N,N) attention weights per timestep
- **실험 가능**: attention bias, attention mask, graph-regularized attention 등 dynamic method 적용 대상
- **기존 실험**: credibility bias 실험 (pre-softmax bias) → MAE 12.178→12.151

### DGCRN (C4 - Dynamic + Predefined)
- **Key files**: `dgcrn_arch.py`, `dgcrn_layer.py`
- **Adj flow**: `predefined_A` (list of 2 transition matrices) + dynamic adj from hyper-GCN at each timestep
- **Graph conv**: Diffusion using both dynamic + predefined adj
- **Swap method**: Replace `model.predefined_A` — **medium difficulty**
- **실험 가능**: predefined 교체, predefined 제거 (dynamic만), predefined + dynamic 비율 조절

## Adjacency Swap Priority

실험 효율성 기준 우선순위:

### Tier 1 (Easy swap, 먼저 시작)
1. **GWNet** — `model.supports` 교체만으로 가능
2. **MTGNN** — `model.predefined_A` 교체
3. **DFDGCN** — GWNet과 동일 패턴

### Tier 2 (Medium, utility 함수 필요)
4. **DGCRN** — `predefined_A` 교체 + preprocessing
5. **STWave+** — nested layer 업데이트

### Tier 3 (Hard, 상당한 코드 수정)
6. **STGCN** — 모든 conv layer의 `gso` traverse 교체
7. **DCRNN** — 모든 `DCGRUCell._supports` 교체
8. **STGODE** — ODE 내부까지 수정 필요

### Special (Dynamic method 적용 대상)
9. **STAEFormer** — attention bias/mask 실험
10. **AGCRN** — "no predefined graph" baseline
