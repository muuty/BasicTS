# Project Guidelines

## 실험 로그 관리
- 실험 결과, insight, 시도 이유는 `project/noise_resilient_prediction/experiment_log.md`에 기록
- 최종 결과 요약: `project/noise_resilient_prediction/RESULTS.md`
- 상세 실험 결과: `project/noise_resilient_prediction/EXPERIMENTS.md`
- 관련 논문 정리: `project/noise_resilient_prediction/LITERATURE.md`
- 새 실험 완료 시 반드시 문서 업데이트
- 형식: 날짜, 실험명, 목적, 결과, 인사이트

## 실험 결과 확인 방법
- 모델 성능 확인 시 `checkpoints/**/test_metrics.json` 파일 사용
- 로그 파싱하지 말 것 (토큰 낭비)
- **Overall MAE 사용** (`overall.MAE`), horizon별 MAE (h3, h6, h12) 아님
- 예시: `find checkpoints -name "test_metrics.json" -exec cat {} \;`

### Robustness 지표 (test_metrics.json)
`test_metrics.json`에 robustness 지표가 포함됨:
- **per_sample**: 샘플별 에러 분포
  - `worst_1pct_MAE`, `worst_5pct_MAE`, `worst_10pct_MAE`: 최악 K% 샘플의 평균 MAE
  - `max_MAE`, `median_MAE`, `std_MAE`: 통계량
- **per_node**: 노드별 에러 분포
  - `worst_1pct_MAE`, `worst_5pct_MAE`, `worst_10pct_MAE`: 최악 K% 노드의 평균 MAE
  - `std_MAE`: 노드간 에러 편차 (중요!)
  - `worst_node_idx`, `best_node_idx`: 최악/최고 노드 인덱스

## 데이터셋 위치
- `datasets/xtraffic/SAN_BERNARDINO/`
  - `data.dat`: 메인 데이터 (105120, 893, 5) - flow, occupancy, speed, tod, dow
  - `adj_mx.pkl`: adjacency matrix (893, 893)
  - 5분 간격, 288 steps/day

## 실험 실행
- Config 파일: `baselines/` 하위 디렉토리
- 단일 실행: `python -c "from basicts import launch_training; launch_training('path/to/config.py', gpus='0')"`

### 실험 프로세스 관리 (중요!)
- **반드시 `nohup`으로 실행**: 세션 종료 시에도 학습이 계속되도록 해야 함
  ```bash
  nohup bash -c 'source ~/.conda/etc/profile.d/conda.sh && conda activate basicts && python -c "
  from basicts import launch_training
  launch_training(\"config.py\", gpus=\"1\")
  "' > /tmp/exp_name.log 2>&1 &
  ```
- **학습 완료 대기 및 후속 작업**: 실험을 nohup으로 시작한 후, `sleep`과 로그 확인을 반복하며 완료를 감지하고 즉시 다음 작업(결과 분석, 후속 실험 실행 등)을 이어간다
  - 예상 완료 시간 파악: `grep "estimated.*finish" <log_path> | tail -1`
  - 완료 감지: `test_metrics.json` 생성 여부 확인
  - 패턴: `sleep <예상 남은 시간>` → 로그 확인 → 미완료 시 추가 sleep → 완료 시 결과 분석 및 다음 실험 진행
- **주의**: `launch_training`은 auto-resume을 지원하지 않음 — 중단 시 처음부터 재학습

### GPU 선택
- **`gpus` 파라미터 사용** (권장): `launch_training('config.py', gpus='1')` - GPU 1 사용
- `CUDA_VISIBLE_DEVICES` 사용하지 말 것 - `gpus` 파라미터가 내부적으로 처리함
- 여러 GPU: `gpus='0,1,2'`

## 실험 큐 시스템 (Ray)
Ray를 사용한 실험 관리. GPU가 비면 자동으로 다음 job 실행.

### 실험 추가
```bash
cd /data/pretrainingbasicts
source ~/.conda/etc/profile.d/conda.sh && conda activate basicts

# Queue에 실험 추가
python experiments/ray_queue.py add \
    baselines/.../config1.py \
    baselines/.../config2.py
```

### Queue 처리 시작
```bash
# 추가된 실험들 실행 (one-shot: 현재 pending 작업만 처리 후 종료)
python experiments/ray_queue.py start

# Watch 모드 (권장): 지속적으로 새 작업 감시, GPU 비면 자동 시작
python experiments/ray_queue.py start --watch

# GPU 개수 지정
python experiments/ray_queue.py start --watch --gpus 2

# Poll 간격 지정 (기본 30초)
python experiments/ray_queue.py start --watch --poll 60
```

**중요:** `--watch` 없이 실행하면 현재 pending 작업만 처리 후 종료됨.
지속적인 자동 스케줄링을 원하면 반드시 `--watch` 사용.

### 상태 확인
```bash
python experiments/ray_queue.py status
```

### 완료된 작업 정리
```bash
python experiments/ray_queue.py clear
```

### 특징
- **Persistent Queue**: `experiments/ray_queue_state.json`에 상태 저장
- **1 job per GPU**: Ray가 자동 관리
- **Watch 모드**: `--watch`로 지속적 감시, 새 작업 추가 시 자동 시작
- **자동 스케줄링**: GPU 비면 다음 job 실행
- **Zombie 없음**: Ray가 process lifecycle 관리
- **Dashboard**: `http://127.0.0.1:8265`

---
### [DEPRECATED] RQ 시스템
> ⚠️ RQ 기반 시스템은 zombie 프로세스, 불안정한 worker 등의 문제로 deprecated됨.
> `experiments/rq_*.py`, `experiments/start_worker.sh`는 더 이상 사용하지 말 것.

## 주요 모델 구조
- `baselines/STAEformer/`: Spatio-Temporal Adaptive Embedding Transformer
- `baselines/ContextContrastive/`: Contrastive pretraining 기반 모델

## Representation Learning Runner (중요!)
**반드시 `RepresentationLearningRunner`만 사용할 것!**

```
baselines/ContextContrastive/runner/representation_learning_runner.py
```

### 사용하지 말아야 할 레거시 Runner들
- ❌ `finetuning_runner.py` - 레거시, 사용 금지
- ❌ `pretrained_encoder_runner.py` - 레거시
- ❌ `scratch_encoder_runner.py` - 레거시

### Config 형식 (CFG.ENCODER)
```python
CFG.ENCODER = {
    'type': 'TransformerEncoder',  # 또는 'MaskedAutoEncoder', 'SpatioTemporalEncoder'
    'source': 'pretrained',        # 'pretrained' 또는 'scratch'
    'freeze': False,               # True: frozen encoder, False: fine-tune
    'ckpt_path': '...',            # source='pretrained'일 때 필수
    'lr': 1e-5,                    # encoder 학습률
    'd_model': 64,
    'num_layers': 2,
    'nhead': 4,
    'dropout': 0.1,
}
CFG.RUNNER = RepresentationLearningRunner
```

### 주의사항
- `CFG.PRETRAINED_ENCODER` 형식은 레거시 - 사용하지 말 것
- 새 실험 config 작성 시 반드시 `CFG.ENCODER` 형식 사용

## Scaler 설정 (중요!)
- **`norm_each_channel`은 반드시 `False`로 설정** (xtraffic/SAN_BERNARDINO 등 교통 데이터셋)
- `get_regular_settings(DATA_NAME)`에서 가져오면 자동으로 `False`
- `True`로 설정하면 per-node 정규화 → dead 센서(std=0)에서 scaler 불일치 발생
- 특히 cross-year evaluation 시 센서 생사 변동으로 치명적 오류 (MAE ~90)
- Config 수동 작성 시 `regular_settings['NORM_EACH_CHANNEL']` 사용 권장

```python
# GOOD - get_regular_settings 사용
regular_settings = get_regular_settings(DATA_NAME)
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']  # False

# BAD - 하드코딩
'norm_each_channel': True  # 절대 금지
```

## Conda 환경
- 환경 이름: `basicts`
- 활성화: `conda activate basicts`

## 코드 설계 원칙

### Fail Fast 원칙 (중요!)
**정상 실행 흐름에서 발생하지 않아야 하는 에러를 위한 방어 코드 금지.**

1. **Silent fallback 금지**: 오작동을 숨기는 대체 로직 추가하지 말 것
2. **명시적 에러 처리 최소화**: try-catch, ValueError 등으로 코드 복잡하게 만들지 말 것
3. **자연스러운 실패**: Python이 알아서 에러 던지도록 두기

**이유:**
- Config이 올바르면 에러가 발생하지 않아야 함
- 발생하지 않아야 하는 에러를 처리하느라 코드가 복잡해지면 안 됨
- Python의 기본 에러 메시지가 충분히 명확함

```python
# BAD - silent fallback
if adj_mx is None:
    adj_mx = torch.eye(num_nodes)  # 오작동 숨김

# BAD - 불필요한 명시적 에러 처리
if adj_mx is None:
    raise ValueError("adj_mx required")  # 코드만 복잡해짐

# GOOD - 그냥 사용 (None이면 Python이 알아서 에러)
adj_mx = self.adj_mx.to(device)  # self.adj_mx가 None이면 자연스럽게 실패
```

# Error Handling

커맨드 실행 중 에러가 발생하면:
1. 절대 그냥 멈추지 말 것
2. 에러 메시지를 분석하고 원인을 파악할 것
3. 수정 후 다시 실행할 것
4. 수정이 불가능하면 에러 내용과 원인을 반드시 보고할 것