#!/usr/bin/env python
"""
Ray-based experiment queue system.

Usage:
    # Add experiments to queue
    python experiments/ray_queue.py add config1.py config2.py

    # Start processing queue (auto-schedules on available GPUs)
    python experiments/ray_queue.py start
    python experiments/ray_queue.py start --gpus 2

    # Check status
    python experiments/ray_queue.py status

    # Clear completed jobs
    python experiments/ray_queue.py clear

Features:
    - 1 job per GPU (Ray manages this automatically)
    - Auto-scheduling: next job starts when GPU becomes free
    - Persistent state: survives restarts
    - Dashboard: http://127.0.0.1:8265
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

STATE_FILE = Path(__file__).parent / "ray_queue_state.json"


def load_state():
    """Load queue state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"pending": [], "running": [], "completed": [], "failed": []}


def save_state(state):
    """Save queue state to file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def add_jobs(config_paths):
    """Add config files to the queue."""
    state = load_state()

    added = []
    for config_path in config_paths:
        # Normalize path
        config_path = str(Path(config_path).resolve())

        if not Path(config_path).exists():
            print(f"⚠️  Config not found: {config_path}")
            continue

        # Check if already in queue
        all_configs = (
            [j["config"] for j in state["pending"]] +
            [j["config"] for j in state["running"]] +
            [j["config"] for j in state["completed"]] +
            [j["config"] for j in state["failed"]]
        )

        if config_path in all_configs:
            print(f"⚠️  Already in queue: {config_path}")
            continue

        job = {
            "id": f"job_{int(time.time()*1000)}_{len(state['pending'])}",
            "config": config_path,
            "added_at": datetime.now().isoformat(),
        }
        state["pending"].append(job)
        added.append(config_path)
        print(f"✅ Added: {config_path}")

    save_state(state)
    print(f"\n📋 Queue: {len(state['pending'])} pending, {len(state['running'])} running")
    return added


def show_status():
    """Show current queue status."""
    state = load_state()

    print("\n" + "="*60)
    print("📊 RAY QUEUE STATUS")
    print("="*60)

    print(f"\n⏳ PENDING ({len(state['pending'])}):")
    for job in state["pending"]:
        config_name = Path(job["config"]).name
        print(f"   • {config_name}")

    print(f"\n🚀 RUNNING ({len(state['running'])}):")
    for job in state["running"]:
        config_name = Path(job["config"]).name
        gpu = job.get("gpu", "?")
        started = job.get("started_at", "?")[:19]
        print(f"   • {config_name} (GPU {gpu}, started {started})")

    print(f"\n✅ COMPLETED ({len(state['completed'])}):")
    for job in state["completed"][-5:]:  # Show last 5
        config_name = Path(job["config"]).name
        print(f"   • {config_name}")
    if len(state["completed"]) > 5:
        print(f"   ... and {len(state['completed'])-5} more")

    print(f"\n❌ FAILED ({len(state['failed'])}):")
    for job in state["failed"][-5:]:
        config_name = Path(job["config"]).name
        error = job.get("error", "unknown")[:50]
        print(f"   • {config_name}: {error}")

    print("\n" + "="*60)


def clear_completed():
    """Clear completed and failed jobs from state."""
    state = load_state()
    n_completed = len(state["completed"])
    n_failed = len(state["failed"])
    state["completed"] = []
    state["failed"] = []
    save_state(state)
    print(f"🧹 Cleared {n_completed} completed and {n_failed} failed jobs")


def recover_state():
    """Recover state by checking actual experiment completion status.

    This fixes orphaned 'running' jobs when main process was killed.
    Checks for test_metrics.json to determine if experiment completed.
    """
    state = load_state()

    if not state["running"]:
        print("✅ No running jobs to recover")
        return

    print(f"🔍 Checking {len(state['running'])} running job(s)...")

    recovered = []
    still_running = []

    for job in state["running"]:
        config_path = job["config"]

        # Parse config to find checkpoint directory
        checkpoint_dir = find_checkpoint_dir(config_path)

        if checkpoint_dir is None:
            print(f"   ⚠️  Cannot determine checkpoint dir for: {Path(config_path).name}")
            still_running.append(job)
            continue

        # Check if test_metrics.json exists (indicates completion)
        test_metrics = Path(checkpoint_dir) / "test_metrics.json"

        # Also check if process is still running
        is_process_running = check_process_running(config_path)

        if test_metrics.exists():
            job["finished_at"] = datetime.now().isoformat()
            state["completed"].append(job)
            recovered.append(job)
            print(f"   ✅ Completed: {Path(config_path).name}")
        elif is_process_running:
            still_running.append(job)
            print(f"   🚀 Still running: {Path(config_path).name}")
        else:
            # Process not running, no metrics = failed or killed
            job["finished_at"] = datetime.now().isoformat()
            job["error"] = "Process terminated without completion (recovered)"
            state["failed"].append(job)
            recovered.append(job)
            print(f"   ❌ Failed (no process): {Path(config_path).name}")

    state["running"] = still_running
    save_state(state)

    print(f"\n📊 Recovery complete: {len(recovered)} jobs recovered, {len(still_running)} still running")


def find_checkpoint_dir(config_path):
    """Find the checkpoint directory for a config file."""
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        cfg = config_module.CFG

        # Build checkpoint path similar to EasyTorch
        model_name = getattr(cfg, 'MODEL', {}).get('NAME', 'Unknown')
        dataset_name = getattr(cfg, 'DATASET', {}).get('NAME', 'Unknown')

        # Get hash from TRAIN settings
        import hashlib
        train_cfg = str(getattr(cfg, 'TRAIN', {}))
        cfg_hash = hashlib.md5(train_cfg.encode()).hexdigest()

        base_dir = f"checkpoints/{model_name}/{dataset_name}/{cfg_hash}"

        if Path(base_dir).exists():
            return base_dir

        # Try to find any matching directory
        pattern = f"checkpoints/{model_name}/{dataset_name}/*"
        from glob import glob
        matches = glob(pattern)
        if matches:
            # Return most recently modified
            return max(matches, key=os.path.getmtime)

    except Exception as e:
        pass

    return None


def check_process_running(config_path):
    """Check if a process for this config is still running."""
    import subprocess

    config_name = Path(config_path).name

    result = subprocess.run(
        ["pgrep", "-f", config_name],
        capture_output=True, text=True
    )

    return result.returncode == 0


def start_queue(num_gpus=None, watch=False, poll_interval=30):
    """Start processing the queue with Ray.

    Args:
        num_gpus: Number of GPUs to use (auto-detect if None)
        watch: If True, keep running and watch for new jobs
        poll_interval: Seconds between checks in watch mode
    """
    try:
        import ray
    except ImportError:
        print("❌ Ray not installed. Install with: pip install ray")
        return

    # Detect available GPUs
    if num_gpus is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
        else:
            num_gpus = 1

    print(f"🚀 Starting Ray queue with {num_gpus} GPU(s)")
    print(f"📊 Dashboard: http://127.0.0.1:8265")
    if watch:
        print(f"👀 Watch mode: polling every {poll_interval}s for new jobs")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus, dashboard_host="0.0.0.0")

    @ray.remote(num_gpus=1)
    def run_experiment(config_path, job_id):
        """Run a single experiment (uses 1 GPU)."""
        import subprocess
        import os

        # Get assigned GPU from Ray
        gpu_ids = ray.get_gpu_ids()
        gpu_id = int(gpu_ids[0]) if gpu_ids else 0

        # Update state to running
        state = load_state()
        for job in state["pending"]:
            if job["id"] == job_id:
                job["gpu"] = gpu_id
                job["started_at"] = datetime.now().isoformat()
                state["running"].append(job)
                state["pending"].remove(job)
                break
        save_state(state)

        # Run the experiment via conda run (ensures correct environment)
        project_root = Path(__file__).parent.parent
        # Convert absolute path to relative path (easytorch requires relative paths)
        config_rel = Path(config_path).relative_to(project_root)
        python_code = f"""
import sys, os
os.chdir('{project_root}')
sys.path.insert(0, '{project_root}')
from basicts import launch_training
launch_training('{config_rel}', gpus='{gpu_id}')
"""
        cmd = [
            "conda", "run", "-n", "basicts", "--no-capture-output",
            "python", "-c", python_code
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))

        # Update state
        state = load_state()
        for job in state["running"]:
            if job["id"] == job_id:
                job["finished_at"] = datetime.now().isoformat()
                state["running"].remove(job)
                if result.returncode == 0:
                    state["completed"].append(job)
                else:
                    job["error"] = result.stderr[-500:] if result.stderr else "Unknown error"
                    state["failed"].append(job)
                break
        save_state(state)

        return result.returncode == 0

    # Main loop (runs once if not watch mode, continuously if watch mode)
    futures = []  # List of (job_id, future) tuples for currently running jobs
    submitted_ids = set()  # Track submitted job IDs to prevent duplicates

    try:
        while True:
            # Check for new pending jobs
            state = load_state()

            # Only submit new jobs if we have free GPU slots
            num_running = len(futures)
            num_free_gpus = num_gpus - num_running

            # Filter out already submitted jobs (file state may lag behind)
            pending_jobs = [j for j in state["pending"] if j["id"] not in submitted_ids]

            if pending_jobs and num_free_gpus > 0:
                # Only submit up to num_free_gpus jobs
                jobs_to_submit = pending_jobs[:num_free_gpus]
                print(f"\n📋 Found {len(pending_jobs)} pending, {num_free_gpus} GPU(s) free, submitting {len(jobs_to_submit)}...")

                for job in jobs_to_submit:
                    future = run_experiment.remote(job["config"], job["id"])
                    futures.append((job["id"], future))
                    submitted_ids.add(job["id"])
                    print(f"   📤 Submitted: {Path(job['config']).name}")

            elif not futures:
                if watch:
                    # No pending jobs and no running jobs - wait for new jobs
                    pass
                else:
                    print("📭 Queue is empty. Add jobs with: python experiments/ray_queue.py add <config.py>")
                    return

            # Process completed futures
            if futures:
                done_ids = []
                for job_id, future in futures:
                    try:
                        ready, _ = ray.wait([future], timeout=0.1)
                        if ready:
                            success = ray.get(future)
                            status = "✅" if success else "❌"
                            print(f"   {status} Finished: {job_id}")
                            done_ids.append(job_id)
                    except Exception as e:
                        print(f"   ❌ Error: {job_id} - {e}")
                        done_ids.append(job_id)

                futures = [(jid, f) for jid, f in futures if jid not in done_ids]

            # Exit condition for non-watch mode
            if not watch and not futures:
                print("\n🎉 All jobs completed!")
                show_status()
                return

            # Wait before next poll
            time.sleep(poll_interval if watch else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Jobs continue in background.")
        print("   Check status with: python experiments/ray_queue.py status")


def main():
    parser = ArgumentParser(description="Ray-based experiment queue")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # add command
    add_parser = subparsers.add_parser("add", help="Add experiments to queue")
    add_parser.add_argument("configs", nargs="+", help="Config file paths")

    # start command
    start_parser = subparsers.add_parser("start", help="Start processing queue")
    start_parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs to use")
    start_parser.add_argument("--watch", action="store_true", help="Keep running and watch for new jobs")
    start_parser.add_argument("--poll", type=int, default=30, help="Poll interval in seconds for watch mode")

    # status command
    subparsers.add_parser("status", help="Show queue status")

    # clear command
    subparsers.add_parser("clear", help="Clear completed jobs")

    args = parser.parse_args()

    if args.command == "add":
        add_jobs(args.configs)
    elif args.command == "start":
        start_queue(args.gpus, watch=args.watch, poll_interval=args.poll)
    elif args.command == "status":
        show_status()
    elif args.command == "clear":
        clear_completed()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
