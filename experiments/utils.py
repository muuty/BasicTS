import socket
import sys
import json
import re
import csv
import os
import threading
import datetime 
import time
import subprocess
from collections import deque
from pathlib import Path
import itertools
import hashlib
import select
from tensorboard import program
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple


EXPERIMENT_INFO_COLUMNS = [
    "exp_name",
    "dataset",
    "model",
    "learning_level",
    "num_clients",
    "pooling",
    "partition",
    "num_tokens",
]

# ExperimentInfo 클래스 제거

def _infer_info_from_path(checkpoint_dir: Path, metrics_path: Path) -> Dict[str, str]:
    rel = metrics_path.relative_to(checkpoint_dir)
    parts = list(rel.parts)

    training_model = parts[0] if len(parts) >= 1 else ""
    dataset_dir = parts[1] if len(parts) >= 2 else ""
    dataset = dataset_dir.split("_")[0] if dataset_dir else ""

    learning_level = _detect_learning_level(training_model)

    model = training_model.split("_")[-1] if training_model else ""
    exp_name = f"{training_model}_{dataset}".strip("_")
    
    return {
        "exp_name": exp_name,
        "dataset": dataset,
        "model": model,
        "learning_level": learning_level,
        "num_clients": "",
        "pooling": "",
        "partition": "",
        "num_tokens": "",
    }

@dataclass
class ExperimentRun:
    """실험 실행 단위를 나타내는 dataclass"""
    cfg_path: str  # 원본 config 파일 경로
    overrides: Dict[str, Any]  # 파라미터 오버라이드
    group_name: str  # 그룹 이름
    run_name: str  # 실행 이름
    actual_cfg_path: str = ""  # 실제 사용할 config 경로 (overrides가 있으면 생성된 파일)
    
    def __post_init__(self):
        """actual_cfg_path가 설정되지 않았으면 cfg_path 사용"""
        if not self.actual_cfg_path:
            self.actual_cfg_path = self.cfg_path


########################### Tensorboard Related Functions ###########################
def is_port_available(port: int) -> bool:
    """포트가 사용 가능한지 확인"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False


def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """사용 가능한 포트를 찾음"""
    for i in range(max_attempts):
        port = start_port + i
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find available port starting from {start_port}")

def start_tensorboard(logdir: str, port: int = 6006) -> str:
    global tb_instance
    logdir_path = str(Path(logdir).resolve())

    if not is_port_available(port):
        port = find_available_port(port)

    try:
        tb = program.TensorBoard()
        tb.configure(argv=["tensorboard", f"--logdir={logdir_path}", f"--port={port}"])
        url = tb.launch()
        tb_instance = tb
        time.sleep(1)
        return url
    except Exception as e:
        port = find_available_port(port + 1)
        tb = program.TensorBoard()
        tb.configure(argv=["tensorboard", f"--logdir={logdir_path}", f"--port={port}"])
        url = tb.launch()
        tb_instance = tb
        time.sleep(1)
        return url
########################### Tensorboard Related Functions ###########################


########################### Metrics Related Functions ###########################


########################### Constants ###########################
METRIC_KEYS = ["MAE", "RMSE", "MAPE"]
HORIZON_KEYS = ["overall", "horizon_3", "horizon_6", "horizon_12"]
HORIZON_SUFFIXES = {"overall": "", "horizon_3": "_h3", "horizon_6": "_h6", "horizon_12": "_h12"}
########################### Constants ###########################


def _extract_metrics(metrics_json: Dict[str, Any]) -> Dict[str, str]:
    """metrics_json에서 메트릭 추출"""
    row = {}
    
    def _fmt(v: Any) -> str:
        if isinstance(v, (int, float)):
            return f"{v:.4f}"
        return ""
    
    for horizon_key in HORIZON_KEYS:
        suffix = HORIZON_SUFFIXES[horizon_key]
        horizon_data = metrics_json[horizon_key]  # KeyError if missing
        
        for metric_key in METRIC_KEYS:
            row_key = f"{metric_key}{suffix}"
            row[row_key] = _fmt(horizon_data.get(metric_key))
    
    return row

LEARNING_LEVEL_MAP = {
    "hiersplit": "HierSplit",
    "split": "Split",
    "independent": "Independent",
    "federated": "Federated",
}

def _detect_learning_level(training_model: str) -> str:
    tm_lower = training_model.lower()
    for key, level in LEARNING_LEVEL_MAP.items():
        if key in tm_lower:
            return level
    return "Baseline"


def _infer_info_from_path(checkpoint_dir: Path, metrics_path: Path) -> Dict[str, str]:
    rel = metrics_path.relative_to(checkpoint_dir)
    parts = list(rel.parts)

    training_model = parts[0] if len(parts) >= 1 else ""
    dataset_dir = parts[1] if len(parts) >= 2 else ""
    dataset = dataset_dir.split("_")[0] if dataset_dir else ""

    learning_level = _detect_learning_level(training_model)

    model = training_model.split("_")[-1] if training_model else ""
    exp_name = f"{training_model}_{dataset}".strip("_")
    
    return {
        "exp_name": exp_name,
        "dataset": dataset,
        "model": model,
        "learning_level": learning_level,
        "num_clients": "",
        "pooling": "",
        "partition": "",
        "num_tokens": "",
    }


def _find_cfg_txt_for_metrics(checkpoint_dir: Path, metrics_path: Path) -> Optional[Path]:
    """test_metrics.json과 같은 run 디렉토리에 cfg.txt를 찾습니다."""
    cur = metrics_path.parent
    while True:
        candidate = cur / "cfg.txt"
        if candidate.exists():
            return candidate
        if cur == checkpoint_dir or cur.parent == cur:
            break
        cur = cur.parent
    return None

def _parse_cfg_txt(cfg_txt: str) -> Dict[str, str]:
    """cfg.txt에서 실험 정보 파싱"""
    info = {col: "" for col in EXPERIMENT_INFO_COLUMNS}
    
    patterns = [
        (r"DATASET\.NAME\s*[:=]\s*([A-Za-z0-9_-]+)", "dataset"),
        (r"MODEL\.NAME\s*[:=]\s*([A-Za-z0-9_]+)", "model"),
        (r"num_clients\s*[:=]\s*(\d+)", "num_clients"),
        (r"pooling_method\s*[:=]\s*[\"']?([A-Za-z0-9_]+)[\"']?", "pooling"),
        (r"num_tokens\s*[:=]\s*(\d+)", "num_tokens"),
    ]
    
    for pattern, field_name in patterns:
        m = re.search(pattern, cfg_txt)
        if m:
            info[field_name] = m.group(1)
    
    return info


def _process_single_metrics_file(
    checkpoint_dir: Path, 
    metrics_path: Path,
    metric_row_keys: List[str]
) -> Dict[str, str]:
    info = _infer_info_from_path(checkpoint_dir, metrics_path)
    
    if cfg_path := _find_cfg_txt_for_metrics(checkpoint_dir, metrics_path):
        cfg_info = _parse_cfg_txt(cfg_path.read_text(encoding="utf-8"))
        # cfg_info의 값으로 info 업데이트 (빈 값이 아닌 경우만)
        for key in EXPERIMENT_INFO_COLUMNS:
            if cfg_info.get(key):
                info[key] = cfg_info[key]
    
    row = {
        **info,
        "status": "Success",
        "duration": "",
        **{key: "" for key in metric_row_keys},
        **_extract_metrics(json.loads(metrics_path.read_text(encoding="utf-8"))),
    }
    return row


def collect_rows_from_checkpoints(checkpoint_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not checkpoint_dir.exists():
        return rows

    # 기본 row 키 생성
    metric_row_keys = [f"{metric}{suffix}" for metric in METRIC_KEYS for suffix in HORIZON_SUFFIXES.values()]

    for metrics_path in sorted(checkpoint_dir.rglob("test_metrics.json")):
        rows.append(_process_single_metrics_file(checkpoint_dir, metrics_path, metric_row_keys))

    return rows
########################### Metrics Related Functions ###########################


########################### Parameterization Related Functions ###########################
def _format_param_str(overrides: Dict[str, Any], sep: str = "=") -> str:
    """파라미터를 문자열로 포맷 (예: 'lr=0.01_batch32')"""
    if not overrides:
        return ""
    return "_".join(
        f"{k.split('.')[-1]}{sep}{v}" 
        for k, v in sorted(overrides.items())
    )


def _to_python_literal(value: Any) -> str:
    """Python 리터럴 문자열로 변환"""
    return f'"{value}"' if isinstance(value, str) else repr(value)


def generate_config_file(
    template_path: str, 
    overrides: Dict[str, Any], 
    output_dir: Path
) -> Tuple[str, str]:
    """템플릿 config에서 파라미터를 오버라이드한 임시 config 파일 생성"""
    param_suffix = _format_param_str(overrides)
    param_hash = hashlib.md5(param_suffix.encode()).hexdigest()[:8]
    
    template = Path(template_path)
    output_path = output_dir / f"{template.stem}_{param_hash}.py"
    
    override_block = "\n".join([
        "\n# === Auto-generated parameter overrides ===",
        f"# Template: {template_path}",
        f"# Params: {param_suffix}",
        *(f"CFG.{k} = {_to_python_literal(v)}" for k, v in overrides.items()),
    ])
    
    output_path.write_text(
        template.read_text(encoding='utf-8') + override_block,
        encoding='utf-8'
    )
    
    return str(output_path), param_suffix


def expand_experiments(
    config: Dict,
    generated_config_dir: Path,
) -> List[ExperimentRun]:
    """
    experiments 정의를 개별 실행 단위로 확장 (grid 조합)
    파라미터 오버라이드가 있으면 config 파일을 미리 생성
    
    Returns:
        List[ExperimentRun]: 확장된 실험 실행 리스트
    """
    all_runs = []
    
    for exp in config['runs']:
        group_name = exp['name']
        params = exp.get('params', {})
        
        # 파라미터 조합 생성
        combinations = (
            [dict(zip(params.keys(), combo)) for combo in itertools.product(*params.values())]
            if params else [{}]
        )
        
        for cfg_path in exp['configs']:
            cfg_name = Path(cfg_path).stem
            
            for overrides in combinations:
                param_str = _format_param_str(overrides)
                suffix = f"_{param_str}" if param_str else ""
                run_name = f"{group_name}_{cfg_name}{suffix}"
                
                # 파라미터 오버라이드가 있으면 config 파일 생성
                if overrides:
                    actual_cfg_path, _ = generate_config_file(cfg_path, overrides, generated_config_dir)
                else:
                    actual_cfg_path = cfg_path
                
                all_runs.append(ExperimentRun(
                    cfg_path=cfg_path,
                    overrides=overrides,
                    group_name=group_name,
                    run_name=run_name,
                    actual_cfg_path=actual_cfg_path,
                ))
    
    return all_runs
########################### Parameterization Related Functions ###########################


def stream_with_spinner(proc: subprocess.Popen, log_f, epoch_pattern: re.Pattern) -> None:
    """stdout을 읽으며 1초마다 스피너와 에포크 진행 상황을 표시"""
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0
    current_epoch = '?'
    total_epoch = '?'
    last_tick = time.time()

    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line == '':
                if proc.poll() is not None:
                    break
                continue

            log_f.write(line)
            log_f.flush()

            match = epoch_pattern.search(line)
            if match:
                current_epoch = match.group(1)
                total_epoch = match.group(2) or '?'

            spinner_idx = (spinner_idx + 1) % len(spinner)
            print(f"\r   {spinner[spinner_idx]} Epoch {current_epoch}/{total_epoch}", end='', flush=True)
            last_tick = time.time()
        else:
            now = time.time()
            if now - last_tick >= 1.0:
                spinner_idx = (spinner_idx + 1) % len(spinner)
                print(f"\r   {spinner[spinner_idx]} Epoch {current_epoch}/{total_epoch}", end='', flush=True)
                last_tick = now


# Metrics Related Functions 섹션에 추가
def save_rows_to_csv(rows: List[Dict[str, str]], csv_path: Path) -> None:
    # 첫 번째 row에서 모든 키 추출
    fieldnames = list(rows[0].keys())
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(cfg_path: str, gpu: str, log_dir: Path, exp_name: str, log_file: Path) -> Tuple[int, Path]:
    """단일 실험 실행 (순차 실행용)"""
    with open(log_file, 'w', encoding='utf-8') as f:
        proc = create_experiment_process(cfg_path, gpu, exp_name)
        epoch_pattern = re.compile(r'[Ee]poch[:\s]+(\d+)[/\s]*(\d+)?')
        stream_with_spinner(proc, f, epoch_pattern)
        proc.wait()
        print()
    
    return proc.returncode, log_file


def create_experiment_process(
    cfg_path: str,
    gpu: str,
    exp_name: str,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """
    실험 프로세스 생성 (공통 로직)
    
    Args:
        cfg_path: config 파일 경로
        gpu: GPU ID (프로세스 입장에서는 0)
        exp_name: 실험 이름
        env: 환경 변수 (None이면 현재 환경 사용)
    
    Returns:
        subprocess.Popen 객체
    """
    cmd = [
        sys.executable,
        'experiments/run_experiment.py',
        '-c', cfg_path,
        '-g', gpu,
        '-n', exp_name
    ]
    
    proc_env = os.environ.copy() if env is None else env.copy()
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=proc_env
    )
    
    return proc

########################### Experiment Execution Helpers ###########################
def print_experiment_header(
    exp_name: str,
    tensorboard_url: str,
    checkpoint_dir: Path,
    total_runs: int,
    results_dir: Path,
    num_workers: int,
    gpus: List[str],
) -> None:
    """실험 시작 헤더 출력"""
    print("=" * 70)
    print(f"🚀 {exp_name}")
    print(f"📊 TensorBoard: {tensorboard_url}")
    print(f"📋 Total: {total_runs} runs")
    print(f"📁 Results: {results_dir}")
    if num_workers > 1:
        print(f"⚡ Parallel: {num_workers} workers, GPUs: {gpus}")
    else:
        print(f"⚡ Sequential: 1 worker, GPU: {gpus[0]}")
    print("=" * 70)


def print_experiment_summary(
    rows: List[Dict[str, str]],
    tensorboard_url: str,
    results_dir: Path,
    csv_path: Path,
) -> None:
    """실험 완료 요약 출력"""
    success = sum(1 for r in rows if r.get("status") == "Success")
    
    print("\n" + "=" * 70)
    print(f"📊 Results: {success}/{len(rows)} succeeded")
    print(f"📊 TensorBoard: {tensorboard_url}")
    print(f"📁 Logs: {results_dir}")
    print(f"📄 Results CSV: {csv_path}")
    print("=" * 70)
########################### Experiment Execution Helpers ###########################




EPOCH_PATTERN = re.compile(r'Epoch\s+(\d+)\s*/\s*(\d+)')

SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

# Thread-safe 상태 관리
status_lock = threading.Lock()
process_status: Dict[int, Dict] = {}

# 출력 관리
_last_status_lines = 0

def _resolve_visible_gpu_token(gpu_id: str) -> str:
    """CUDA_VISIBLE_DEVICES 환경에서 GPU ID 해석"""
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not parent_cvd:
        return gpu_id
    
    visible_list = [x.strip() for x in parent_cvd.split(',') if x.strip() != '']
    if not visible_list:
        return gpu_id
    
    if gpu_id.isdigit():
        idx = int(gpu_id)
        if 0 <= idx < len(visible_list):
            return visible_list[idx]
    
    return gpu_id


def _extract_epoch(line: str) -> tuple:
    """Epoch 추출"""
    match = EPOCH_PATTERN.search(line)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _stream_reader(proc: subprocess.Popen, log_f, run_name: str, pid: int) -> None:
    """
    stdout을 읽어서 로그 파일에 쓰고, 상태만 업데이트
    (출력은 하지 않음 - 메인 스레드에서 담당)
    """
    current_epoch = '?'
    total_epoch = '?'
    
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0.5)
        
        if ready:
            line = proc.stdout.readline()
            if line == '':
                if proc.poll() is not None:
                    break
                continue
            
            log_f.write(line)
            log_f.flush()
            
            # Epoch 추출
            epoch, total = _extract_epoch(line)
            if epoch:
                current_epoch = epoch
                total_epoch = total
        
        # 상태 업데이트 (출력 X, 데이터만)
        with status_lock:
            if pid in process_status:
                process_status[pid]['current_epoch'] = current_epoch
                process_status[pid]['total_epoch'] = total_epoch
        
        if proc.poll() is not None:
            break


def _clear_status_lines() -> None:
    """이전에 출력한 상태 줄들을 지움"""
    global _last_status_lines
    
    if _last_status_lines > 0:
        # 커서를 위로 이동하고 지우기
        sys.stdout.write(f"\033[{_last_status_lines}A")
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        _last_status_lines = 0


def _print_status() -> None:
    """현재 상태를 출력 (메인 스레드에서만 호출)"""
    global _last_status_lines
    
    with status_lock:
        if not process_status:
            _clear_status_lines()
            return
        
        # 이전 출력 지우기
        _clear_status_lines()
        
        # 새 상태 출력
        lines = []
        for pid, status in process_status.items():
            spinner = SPINNER[status.get('spinner_idx', 0)]
            line = f"{spinner} {status['run_name']}: Epoch {status['current_epoch']}/{status['total_epoch']}"
            lines.append(line)
        
        for line in lines:
            print(line)
        
        sys.stdout.flush()
        _last_status_lines = len(lines)


def _print_message(message: str) -> None:
    """상태 영역을 지우고 메시지 출력 후 상태 다시 표시"""
    _clear_status_lines()
    print(message)
    sys.stdout.flush()
    _print_status()


def run_experiments_parallel(
    runs: list,
    gpus: list,
    exp_name: str,
    results_dir: Path,
) -> None:
    global process_status, _last_status_lines
    
    process_status = {}
    _last_status_lines = 0
    running_processes = {}
    
    run_queue = deque(runs)
    total_runs = len(runs)
    num_workers = len(gpus)
    spinner_idx = 0
    
    try:
        while (run_queue or running_processes):
            # ─────────────────────────────────────────────────────────
            # 1. 완료된 작업 정리
            # ─────────────────────────────────────────────────────────
            ended_pids = []
            for pid, proc_info in list(running_processes.items()):
                proc, f, run, log_file, reader_thread = proc_info
                
                if proc.poll() is not None:
                    reader_thread.join(timeout=2.0)
                    f.close()
                    
                    # 상태에서 제거
                    with status_lock:
                        process_status.pop(pid, None)
                    
                    duration = time.time() - proc.start_time
                    status = "Success" if proc.returncode == 0 else "Failed"
                    icon = "✅" if status == "Success" else "❌"
                    
                    if status == "Failed":
                        msg = f"{icon} {run.run_name} ({status}, {duration:.0f}s) | Log: {log_file}"
                    else:
                        msg = f"{icon} {run.run_name} ({status}, {duration:.0f}s)"
                    
                    _print_message(msg)
                    ended_pids.append(pid)
            
            for pid in ended_pids:
                del running_processes[pid]
            
            # ─────────────────────────────────────────────────────────
            # 2. 새 작업 시작
            # ─────────────────────────────────────────────────────────
            while len(running_processes) < num_workers and run_queue:
                run = run_queue.popleft()
                job_idx = total_runs - len(run_queue) - 1
                gpu_id = gpus[job_idx % len(gpus)]
                
                _print_message(f"▶ Starting: {run.run_name} (GPU {gpu_id})")
                
                # GPU 환경 설정
                gpu_token = _resolve_visible_gpu_token(gpu_id)
                env = os.environ.copy()
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_token)
                
                # 로그 파일
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cfg_name = Path(run.cfg_path).stem
                log_file = results_dir / f"{run.group_name}_{cfg_name}_{timestamp}.log"
                f = open(log_file, 'w', encoding='utf-8')
                
                # 프로세스 생성
                full_exp_name = f"{exp_name}/{run.run_name}"
                proc = create_experiment_process(
                    cfg_path=run.actual_cfg_path,
                    gpu='0',
                    exp_name=full_exp_name,
                    env=env
                )
                proc.start_time = time.time()
                
                # 상태 초기화
                with status_lock:
                    process_status[proc.pid] = {
                        'run_name': run.run_name,
                        'spinner_idx': 0,
                        'current_epoch': '?',
                        'total_epoch': '?',
                    }
                
                # Reader 스레드 시작 (출력 X, 상태 업데이트만)
                reader_thread = threading.Thread(
                    target=_stream_reader,
                    args=(proc, f, run.run_name, proc.pid),
                    daemon=True
                )
                reader_thread.start()
                
                running_processes[proc.pid] = (proc, f, run, log_file, reader_thread)
            
            # ─────────────────────────────────────────────────────────
            # 3. Spinner 업데이트 & 상태 출력 (메인 스레드에서만)
            # ─────────────────────────────────────────────────────────
            spinner_idx = (spinner_idx + 1) % len(SPINNER)
            
            with status_lock:
                for pid in process_status:
                    process_status[pid]['spinner_idx'] = spinner_idx
            
            _print_status()
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        _clear_status_lines()
        print("\n[STOP] Interrupted, cleaning up...")
        
        for pid, (proc, f, _, _, reader_thread) in running_processes.items():
            if proc.poll() is None:
                proc.kill()
            f.close()
            reader_thread.join(timeout=1.0)
        
        raise
    
    finally:
        _clear_status_lines()