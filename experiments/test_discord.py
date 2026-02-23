#!/usr/bin/env python
"""Test script for Discord notification functionality."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from experiments.discord_notifier import DiscordNotifier

# Discord Configuration - Update these constants with your Discord bot credentials
DISCORD_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
DISCORD_CHANNEL_ID = "YOUR_CHANNEL_ID_HERE"


def test_discord_bot():
    """Test Discord bot functionality."""
    if DISCORD_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("Error: DISCORD_BOT_TOKEN not configured")
        print("Update DISCORD_BOT_TOKEN constant in this file")
        return False

    if DISCORD_CHANNEL_ID == "YOUR_CHANNEL_ID_HERE":
        print("Error: DISCORD_CHANNEL_ID not configured")
        print("Update DISCORD_CHANNEL_ID constant in this file")
        return False

    try:
        notifier = DiscordNotifier(DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID)
        print("Testing Discord notification...")

        # Test start notification
        success = notifier.send_experiment_start_notification("test_experiment", 5)
        if success:
            print("✅ Start notification sent successfully!")
        else:
            print("❌ Failed to send start notification")
            return False

        # Test completion notification
        success = notifier.send_experiment_completion_notification(
            "test_experiment",
            total_jobs=5,
            completed_jobs=4,
            failed_jobs=1,
            duration_hours=2.5
        )
        if success:
            print("✅ Completion notification sent successfully!")
        else:
            print("❌ Failed to send completion notification")
            return False

        print("🎉 All tests passed! Discord notifications are working correctly.")
        return True

    except Exception as e:
        print(f"❌ Error testing Discord bot: {e}")
        return False


if __name__ == "__main__":
    test_discord_bot()