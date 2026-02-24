"""GPU Monitor - Discord 알림

GPU가 하나라도 idle 상태가 되면 Discord로 알림을 보냄.
10분마다 체크.

Usage:
    nohup python experiments/gpu_monitor.py &
"""
import subprocess
import sys
import time
import json
import urllib.request

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_ID = os.environ.get("DISCORD_CHANNEL_ID", "")
CHECK_INTERVAL = 600  # 10분


def get_gpu_status():
    """nvidia-smi로 GPU 상태 조회."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        gpus.append({
            "index": int(parts[0]),
            "name": parts[1],
            "mem_used": int(parts[2]),
            "mem_total": int(parts[3]),
            "util": int(parts[4]),
        })
    return gpus


def send_discord(message):
    """Discord Bot API로 메시지 전송."""
    url = f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages"
    data = json.dumps({"content": message}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "DiscordBot (https://github.com/gpu-monitor, 1.0)",
    })
    try:
        urllib.request.urlopen(req)
        print(f"[Discord] 전송 완료: {message[:80]}")
    except Exception as e:
        print(f"[Discord] 전송 실패: {e}")


def main():
    notified_idle = set()   # idle 알림 보낸 GPU
    was_all_busy = True     # 이전에 모든 GPU가 busy였는지
    print(f"GPU Monitor 시작 (체크 간격: {CHECK_INTERVAL}s)", flush=True)

    while True:
        gpus = get_gpu_status()
        idle_gpus = [g for g in gpus if g["util"] == 0 and g["mem_used"] < 100]
        busy_gpus = [g for g in gpus if g not in idle_gpus]
        all_busy = len(idle_gpus) == 0

        # idle GPU 중 아직 알림 안 보낸 것
        new_idle = [g for g in idle_gpus if g["index"] not in notified_idle]
        if new_idle:
            lines = [f"🖥️ **GPU {g['index']}** ({g['name']}) - IDLE (mem: {g['mem_used']}MB/{g['mem_total']}MB)"
                     for g in new_idle]
            busy_lines = [f"  GPU {g['index']}: util {g['util']}%, mem {g['mem_used']}MB"
                          for g in busy_gpus]

            msg = "⚡ **GPU Available!**\n" + "\n".join(lines)
            if busy_lines:
                msg += "\n\n📊 Other GPUs:\n" + "\n".join(busy_lines)

            send_discord(msg)
            for g in new_idle:
                notified_idle.add(g["index"])

        # 모든 GPU가 다시 채워졌을 때 알림
        if all_busy and not was_all_busy:
            lines = [f"  GPU {g['index']}: util {g['util']}%, mem {g['mem_used']}MB"
                     for g in busy_gpus]
            send_discord("✅ **All GPUs Busy!**\n" + "\n".join(lines))

        was_all_busy = all_busy

        # busy로 돌아간 GPU는 notified에서 제거 (다시 idle 되면 재알림)
        for g in busy_gpus:
            notified_idle.discard(g["index"])

        ts = time.strftime("%H:%M:%S")
        status = ", ".join(f"GPU{g['index']}:{g['util']}%" for g in gpus)
        print(f"[{ts}] {status}", flush=True)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
