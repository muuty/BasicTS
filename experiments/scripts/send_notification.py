#!/usr/bin/env python
"""
Standalone script to send Discord notification after all SLURM jobs complete.
This script is designed to be run as a SLURM job with --dependency=afterany.

Usage:
    python send_notification.py --exp-name EXP_NAME --job-ids JOB1,JOB2,JOB3 --start-time TIMESTAMP
"""

import argparse
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple
import requests


# Discord Configuration - Set via environment variables (e.g. in .bashrc)
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")


def get_job_final_status(job_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Get final status and exit code of a completed job using sacct.

    Returns:
        Tuple of (status, exit_code) or (None, None) if not found
    """
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, "-o", "State,ExitCode", "--noheader", "-P"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            # Get the main job status (first line, not the .batch or .extern)
            for line in lines:
                if '|' in line:
                    parts = line.split('|')
                    status = parts[0].strip()
                    exit_code = parts[1].strip() if len(parts) > 1 else "N/A"
                    return status, exit_code
    except Exception as e:
        print(f"Error getting status for job {job_id}: {e}")

    return None, None


def collect_job_stats(job_ids: List[str]) -> Dict[str, int]:
    """Collect statistics from all jobs.

    Returns:
        Dictionary with total, completed, failed counts
    """
    success_states = {"COMPLETED"}
    failure_states = {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}

    completed = 0
    failed = 0

    for job_id in job_ids:
        status, exit_code = get_job_final_status(job_id)

        if status in success_states:
            completed += 1
        elif status in failure_states:
            failed += 1
        else:
            # Unknown or still running (shouldn't happen with afterany dependency)
            print(f"Job {job_id} has unexpected status: {status}")
            failed += 1

    return {
        "total": len(job_ids),
        "completed": completed,
        "failed": failed
    }


def send_discord_notification(
    exp_name: str,
    stats: Dict[str, int],
    duration_hours: float
) -> bool:
    """Send completion notification to Discord.

    Returns:
        True if successful, False otherwise
    """
    if stats["failed"] == 0:
        title = "All Experiments Completed Successfully"
        color = 0x27ae60  # Green
        emoji = ":white_check_mark:"
    else:
        title = "Experiments Completed with Errors"
        color = 0xe74c3c  # Red
        emoji = ":warning:"

    embed = {
        "title": title,
        "description": f"Experiment batch **{exp_name}** has finished",
        "color": color,
        "fields": [
            {"name": "Total Jobs", "value": str(stats["total"]), "inline": True},
            {"name": "Completed", "value": str(stats["completed"]), "inline": True},
            {"name": "Failed", "value": str(stats["failed"]), "inline": True},
            {"name": "Duration", "value": f"{duration_hours:.1f} hours", "inline": True}
        ],
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }

    message = f"{emoji} Experiment batch **{exp_name}** completed: {stats['completed']}/{stats['total']} successful"

    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    headers = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "content": message,
        "embeds": [embed]
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Discord notification sent successfully")
        return True
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Send Discord notification for completed experiments")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name")
    parser.add_argument("--job-ids", type=str, required=True, help="Comma-separated list of job IDs")
    parser.add_argument("--start-time", type=float, required=True, help="Start timestamp (Unix time)")
    args = parser.parse_args()

    job_ids = [jid.strip() for jid in args.job_ids.split(',') if jid.strip()]

    print(f"Collecting stats for {len(job_ids)} jobs...")
    stats = collect_job_stats(job_ids)

    duration_hours = (time.time() - args.start_time) / 3600

    print(f"Results: {stats['completed']}/{stats['total']} successful, {stats['failed']} failed")
    print(f"Duration: {duration_hours:.1f} hours")

    send_discord_notification(args.exp_name, stats, duration_hours)


if __name__ == "__main__":
    main()
