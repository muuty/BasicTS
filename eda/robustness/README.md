# Test-time Perturbation Analysis

**Research question**: 불량 센서 n개가 test time에 나머지 정상 센서의 예측 정확도에 미치는 영향은?

## 실험 설계

1. Clean 모델로 전체 test set 예측 (clean predictions)
2. n개 functional 노드를 선택하여 corruption 적용 (입력만 변경, 모델 가중치 고정)
3. 나머지 healthy 노드의 MAE 변화를 측정

### Corruption Types

| Type | 설명 | 현실 시나리오 |
|------|------|---------------|
| `flow_zero` | flow 채널만 0으로 고정 | 단일 채널 센서 고장 |
| `all_zero` | flow, occupancy, speed 모두 0 | 완전 센서 고장 |
| `noisy` | flow에 Gaussian noise 추가 (intensity × flow_std) | 센서 노화/간섭 |
| `spike` | 50% 타임스텝에 random spike 추가 | 간헐적 이상치 |

### 측정 지표

| 지표 | 정의 | 의미 |
|------|------|------|
| **System avg degradation** | healthy 전체 평균 MAE 변화 | 네트워크 전체 영향 |
| **Top-K by Δ** | prediction shift가 가장 큰 K개 노드의 MAE 변화 | 예측이 가장 많이 바뀐 곳의 정확도 |
| **Top-K by degradation** | MAE가 가장 악화된 K개 노드의 MAE 변화 | 가장 피해받은 곳의 심각도 |
| **Reach** | Δ 또는 degradation이 threshold 초과하는 노드 수 | 오염 전파 범위 |

### 파라미터

- **n_corrupt**: [5, 10, 20, 50] (corrupt할 노드 수)
- **noise intensity**: 0.1, 0.2, 0.3 (× flow_std ≈ 157.8)
- **seed**: 42 (노드 선택 + corruption 재현)

## 모델 비교

| 모델 | 구조 | 입력 | Clean MAE |
|------|------|------|-----------|
| STAEformer | Transformer (adaptive attention) | flow+occ+speed+tod+dow (5ch) | 12.06 |
| STGCN | Chebyshev GCN + temporal conv | flow only (1ch) | 13.19 |

## 파일 구조

```
eda/robustness/
├── run_perturbation.py                # 통합 실험 스크립트
├── README.md                          # 이 문서
│
├── staeformer_clean_mae.npy           # (893,) STAEformer per-node clean MAE
├── staeformer_per_node_delta.npy      # (24, 893) config별 per-node Δ
├── staeformer_per_node_degradation.npy# (24, 893) config별 per-node MAE degradation
├── staeformer_summary.json            # STAEformer 요약 통계
│
├── stgcn_clean_mae.npy                # (893,) STGCN per-node clean MAE
├── stgcn_per_node_delta.npy           # (24, 893) config별 per-node Δ
├── stgcn_per_node_degradation.npy     # (24, 893) config별 per-node MAE degradation
└── stgcn_summary.json                 # STGCN 요약 통계
```

### npy 파일 형식

- `*_clean_mae.npy`: shape `(893,)` — 노드별 clean masked MAE (target>0인 샘플만)
- `*_per_node_delta.npy`: shape `(24, 893)` — 24개 config × 893 노드, `|pred_corrupted - pred_clean|`의 시간/샘플 평균
- `*_per_node_degradation.npy`: shape `(24, 893)` — 24개 config × 893 노드, `MAE_corrupted - MAE_clean`

24개 config 순서는 `*_summary.json`의 `config_labels` 필드 참조.

### summary.json 구조

```json
{
  "model": "staeformer",
  "clean_mae_functional_avg": 12.0644,
  "n_functional": 745,
  "config_labels": ["flow_zero_n5", "flow_zero_n10", ...],
  "configs": {
    "flow_zero_n5": {
      "system_avg_delta": 0.226,
      "system_avg_degradation": 0.013,
      "by_delta_top5_degradation": {"mean": 0.051, "max": ..., "min": ...},
      "by_degrad_top5_degradation": {"mean": 0.132, "max": ..., "min": ...},
      "reach_delta_gt_0.5": 3,
      ...
    }
  }
}
```

## 실행

```bash
# STAEformer
python eda/robustness/run_perturbation.py --model staeformer --gpu 1

# STGCN
python eda/robustness/run_perturbation.py --model stgcn --gpu 1
```

## 새 모델 추가

`run_perturbation.py`의 `MODEL_REGISTRY`에 추가:

```python
MODEL_REGISTRY = {
    'new_model': {
        'loader': load_new_model,       # 모델 로드 함수
        'forward_features': [0],        # 입력 채널 인덱스
        'has_future': False,            # future_data 필요 여부
    },
}
```

## 핵심 발견

### STGCN은 센서 고장(flow_zero)에 20~27배 더 취약

| flow_zero n=50 | STAEformer | STGCN |
|----------------|-----------|-------|
| System avg | +0.095 | +2.551 |
| Top-5 by degradation | +0.764 | +16.269 |

### 모델별 취약 corruption type이 다름

| Corruption | STAEformer | STGCN |
|------------|-----------|-------|
| flow_zero/all_zero | by_Δ top-5 **개선** (-2.2) | by_Δ top-5 **악화** (+14.9) |
| noisy/spike | 악화 (spike +11.6) | by_Δ top-5 **개선** (-0.7) |

- **STGCN**: GCN message passing이 zero 값을 직접 전파 → zero-type에 취약. 반면 noise는 aggregation이 smoothing.
- **STAEformer**: Attention이 zero 패턴을 감지하여 무시 가능. 반면 "그럴듯한" noise/spike는 attention을 속임.
