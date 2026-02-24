#!/usr/bin/env python
"""
Parametrized experiment runner with Discord notifications.

Usage:
    # Basic usage (includes Discord notifications and automatic job completion monitoring)
    python experiments/run_experiments.py --cfg=experiments/configs/exp.yaml --arch=cuda
    python experiments/run_experiments.py --cfg=experiments/configs/exp.yaml --dry-run
    python experiments/run_experiments.py --cfg=experiments/configs/exp.yaml --test-run

    # Disable Discord notifications
    python experiments/run_experiments.py --cfg=experiments/configs/exp.yaml --arch=cuda --no-discord

Note: Discord credentials are configured as constants in this file. Update DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID below.
"""

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.train import prepare_and_launch
from experiments.discord_notifier import DiscordNotifier

# Discord Configuration - Set via environment variables (e.g. in .bashrc)
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")

def generate_config(
    original_path: str,
    overrides: Dict[str, Any],
    output_dir: Path,
) -> Tuple[str, str]:
    """오버라이드가 적용된 새 config 파일 생성. (config_path, hash) 반환."""
    original = Path(original_path)
    
    with open(original, 'r') as f:
        content = f.read()
    
    # 오버라이드 추가
    override_lines = ["\n# === Auto-generated overrides ==="]
    added_easy_dicts = set()  # 이미 추가된 EasyDict 선언 추적
    
    for key, value in overrides.items():
        # 중첩된 키 경로 파싱 (예: "EXPERIENCE_REPLAY.CAPACITY_RATIO")
        key_parts = key.split('.')
        
        # 중간 경로에 대해 EasyDict 선언이 없으면 추가
        for i in range(1, len(key_parts)):
            parent_path = '.'.join(key_parts[:i])
            easy_dict_line = f"CFG.{parent_path} = EasyDict()"
            # content에 없고, override_lines에도 아직 추가되지 않았으면 추가
            if easy_dict_line not in content and parent_path not in added_easy_dicts:
                override_lines.append(easy_dict_line)
                added_easy_dicts.add(parent_path)
        
        # 실제 값 할당
        override_lines.append(f"CFG.{key} = {repr(value)}")
    
    new_content = content + "\n".join(override_lines) + "\n"
    
    # 파일명 충돌 방지:
    # - 서로 다른 경로의 config가 같은 stem을 가질 수 있으므로(예: */SAN_BERNARDINO/SAN_BERNARDINO.py),
    #   stem 앞에 상위 디렉토리명(최대 2단)을 포함해 generated 파일명이 겹치지 않도록 한다.
    # - run_name은 별도 로직을 사용하므로 여기서는 generated config 파일명만 안전하게 만든다.
    override_hash = hashlib.md5(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]
    # 파일명 고유성: 상위 디렉토리 최대 2단을 접두사로 사용
    # e.g. baselines/STGCN/SAN_BERNARDINO/SAN_BERNARDINO.py -> STGCN__SAN_BERNARDINO__SAN_BERNARDINO
    relative = original.resolve().relative_to(Path.cwd())
    parent_parts = list(relative.parent.parts)
    parent_suffix = "__".join(parent_parts[-2:]) if parent_parts else "root"
    safe_prefix = f"{parent_suffix}__{original.stem}"
    output_path = output_dir / f"{safe_prefix}_{override_hash}.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(new_content)
    
    return str(output_path), override_hash


def expand_experiments(config: Dict, generated_dir: Path) -> List[Tuple[str, str]]:
    """실험 정의를 (config_path, run_name) 리스트로 확장."""
    all_runs = []
    
    for exp in config['runs']:
        group_name = exp['name']
        params = exp.get('params', {})
        
        # 파라미터 조합 생성
        if params:
            keys = list(params.keys())
            combinations = [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]
        else:
            combinations = [{}]
        
        for cfg_path in exp['configs']:
            cfg_stem = Path(cfg_path).stem
            
            for overrides in combinations:
                if overrides:
                    actual_path, override_hash = generate_config(cfg_path, overrides, generated_dir)
                    run_name = f"{group_name}_{cfg_stem}_{override_hash}"
                else:
                    actual_path = cfg_path
                    run_name = f"{group_name}_{cfg_stem}"
                
                # SLURM job name은 공백을 허용하지 않으므로 언더스코어로 치환
                run_name = run_name.replace(' ', '_')
                
                all_runs.append((actual_path, run_name))
    
    return all_runs


def extract_job_id(sbatch_output: str) -> Optional[str]:
    """Extract job ID from sbatch output.

    Args:
        sbatch_output: Output from sbatch command

    Returns:
        Job ID if found, None otherwise
    """
    match = re.search(r'Submitted batch job (\d+)', sbatch_output)
    return match.group(1) if match else None


def submit_job_and_get_id(submit_script: str, cfg_path: str, exp_name: str, run_num: int) -> Optional[str]:
    """Submit a job and return its job ID.

    Args:
        submit_script: Path to submit script
        cfg_path: Config file path
        exp_name: Experiment name
        run_num: Run number

    Returns:
        Job ID if successful, None otherwise
    """
    try:
        result = subprocess.run(
            ["bash", submit_script, cfg_path, exp_name, str(run_num)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            job_id = extract_job_id(result.stdout)
            if job_id:
                print(f"[SUBMITTED] {exp_name} run{run_num} -> Job ID: {job_id}")
                return job_id
            else:
                print(f"[ERROR] Failed to extract job ID from: {result.stdout}")
        else:
            print(f"[ERROR] Failed to submit job: {result.stderr}")

    except subprocess.TimeoutExpired:
        print(f"[ERROR] Timeout submitting job for {exp_name} run{run_num}")
    except Exception as e:
        print(f"[ERROR] Exception submitting job: {e}")

    return None


def test_run(cfg_path: str, run_name: str, gpus: str = "0", exp: str = None, run: int = None):
    """로컬에서 직접 실행하여 테스트."""
    print(f"\n[TEST RUN] {cfg_path} (exp={exp}, run={run})")
    prepare_and_launch(cfg_path, gpus, run, exp)


def submit_notification_job(exp_name: str, job_ids: List[str],
                            start_time: float, arch: str):
    """Submit a Discord notification job that fires after all experiment jobs complete."""
    try:
        result = subprocess.run(
            ["bash", "experiments/scripts/submit_notification_job.sh",
             exp_name, ",".join(job_ids), str(start_time), arch],
            capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            notify_id = extract_job_id(result.stdout)
            if notify_id:
                print(f"Notification job submitted: {notify_id}")
    except Exception as e:
        print(f"Warning: Failed to submit notification job: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--arch", type=str, choices=["cuda", "rocm"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without submitting")
    parser.add_argument("--test-run", action="store_true", help="Run locally without job submission")
    parser.add_argument("--gpus", type=str, default="0", help="GPU ids for test-run (default: 0)")
    parser.add_argument("--total-runs", type=int, default=1, help="Number of times to repeat each experiment (default: 1)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of configs per SLURM job (default: 1)")
    parser.add_argument("--no-discord", action="store_true", help="Disable Discord notifications")
    args = parser.parse_args()

    # Validation
    if not args.arch and not args.dry_run and not args.test_run:
        parser.error("--arch is required unless --dry-run is specified")

    # 1. Load & expand
    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)
        exp_name = Path(args.cfg).stem

    generated_dir = Path("experiments/generated")
    runs = expand_experiments(config, generated_dir)
    total_jobs = len(runs) * args.total_runs

    print(f"Expanded to {len(runs)} runs x {args.total_runs} repeats = {total_jobs} total")

    # Initialize Discord notifier if enabled
    notifier = None
    if not args.no_discord:
        if DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID:
            try:
                notifier = DiscordNotifier(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
                print("Discord notifications enabled")
            except Exception as e:
                print(f"Warning: Failed to initialize Discord notifier: {e}")
        else:
            print("Warning: Discord credentials not set. Export DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in your environment.")

    # 2. Flatten all (config_path, run_num) pairs
    all_configs = [(cfg_path, run_num)
                   for cfg_path, _ in runs
                   for run_num in range(1, args.total_runs + 1)]

    # 3. Group into batches
    batch_size = args.batch_size
    batches = [all_configs[i:i + batch_size]
               for i in range(0, len(all_configs), batch_size)]

    print(f"{len(all_configs)} configs -> {len(batches)} SLURM jobs (batch_size={batch_size})")

    # 4. Execute
    if args.test_run:
        cfg_path, run_num = all_configs[0]
        test_run(cfg_path, "", args.gpus, exp_name, run_num)

    elif args.dry_run:
        for batch_num, batch in enumerate(batches, 1):
            print(f"  Batch {batch_num}: {len(batch)} configs")
            for cfg_path, _ in batch:
                print(f"    {Path(cfg_path).name}")

    else:
        start_time = time.time()
        if notifier:
            notifier.send_experiment_start_notification(exp_name, total_jobs)

        # Write batch files and submit
        batch_dir = Path("experiments/batches")
        batch_dir.mkdir(parents=True, exist_ok=True)
        submit_script = f"experiments/scripts/submit_batch_job_{args.arch}.sh"
        job_ids = []

        for batch_num, batch in enumerate(batches, 1):
            batch_file = batch_dir / f"{exp_name}_batch{batch_num:03d}.txt"
            with open(batch_file, 'w') as f:
                for cfg_path, _ in batch:
                    f.write(f"{cfg_path}\n")

            _, first_run_num = batch[0]
            job_id = submit_job_and_get_id(
                submit_script, str(batch_file), exp_name, first_run_num)
            if job_id:
                job_ids.append(job_id)
                print(f"  -> batch {batch_num}: {len(batch)} configs")

        # Discord completion notification
        if job_ids and not args.no_discord:
            submit_notification_job(exp_name, job_ids, start_time, args.arch)
            print(f"\nAll {len(job_ids)} batch jobs submitted.")
        elif not job_ids:
            print("No jobs were successfully submitted.")


if __name__ == "__main__":
    main()