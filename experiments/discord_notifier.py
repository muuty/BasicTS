#!/usr/bin/env python
"""Discord notification utility for experiment completion."""

import time
from typing import Optional, Dict, Any
import requests


class DiscordNotifier:
    """Discord bot notification utility."""

    def __init__(self, bot_token: str, channel_id: str):
        """Initialize Discord notifier.

        Args:
            bot_token: Discord bot token
            channel_id: Discord channel ID
        """
        if not bot_token:
            raise ValueError("Discord bot token is required")
        if not channel_id:
            raise ValueError("Discord channel ID is required")

        self.bot_token = bot_token
        self.channel_id = channel_id

        self.base_url = "https://discord.com/api/v10"
        self.headers = {
            "Authorization": f"Bot {self.bot_token}",
            "Content-Type": "application/json"
        }

    def send_message(self, message: str, embed: Optional[Dict[str, Any]] = None) -> bool:
        """Send message to Discord channel.

        Args:
            message: Text message to send
            embed: Optional embed data

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.base_url}/channels/{self.channel_id}/messages"
        data = {"content": message}
        if embed:
            data["embeds"] = [embed]

        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")
            return False

    def send_experiment_start_notification(self, exp_name: str, total_jobs: int) -> bool:
        """Send notification when experiments start.

        Args:
            exp_name: Experiment name
            total_jobs: Total number of jobs submitted

        Returns:
            True if successful, False otherwise
        """
        embed = {
            "title": "🚀 Experiments Started",
            "description": f"Experiment batch **{exp_name}** has been submitted to SLURM",
            "color": 0x3498db,  # Blue
            "fields": [
                {"name": "Total Jobs", "value": str(total_jobs), "inline": True},
                {"name": "Status", "value": "Running", "inline": True}
            ],
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }

        message = f"🚀 Started experiment batch: **{exp_name}** ({total_jobs} jobs)"
        return self.send_message(message, embed)

    def send_experiment_completion_notification(
        self,
        exp_name: str,
        total_jobs: int,
        completed_jobs: int,
        failed_jobs: int,
        duration_hours: float
    ) -> bool:
        """Send notification when all experiments complete.

        Args:
            exp_name: Experiment name
            total_jobs: Total number of jobs
            completed_jobs: Number of completed jobs
            failed_jobs: Number of failed jobs
            duration_hours: Total duration in hours

        Returns:
            True if successful, False otherwise
        """
        if failed_jobs == 0:
            title = "✅ All Experiments Completed Successfully"
            color = 0x27ae60  # Green
            emoji = "✅"
        else:
            title = "⚠️ Experiments Completed with Errors"
            color = 0xe74c3c  # Red
            emoji = "⚠️"

        embed = {
            "title": title,
            "description": f"Experiment batch **{exp_name}** has finished",
            "color": color,
            "fields": [
                {"name": "Total Jobs", "value": str(total_jobs), "inline": True},
                {"name": "Completed", "value": str(completed_jobs), "inline": True},
                {"name": "Failed", "value": str(failed_jobs), "inline": True},
                {"name": "Duration", "value": f"{duration_hours:.1f} hours", "inline": True}
            ],
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }

        message = f"{emoji} Experiment batch **{exp_name}** completed: {completed_jobs}/{total_jobs} successful"
        return self.send_message(message, embed)


