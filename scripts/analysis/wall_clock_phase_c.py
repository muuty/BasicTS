"""Aggregate Phase C training wall-clock from training_log_*.log files.

Walks `checkpoints/phase_c_*` recursively, extracts (model, dataset, method,
ratio, seed) plus first/last log timestamps, and writes summary CSV.

Usage:
    conda activate cuda
    python scripts/analysis/wall_clock_phase_c.py
"""
from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_ROOT = REPO_ROOT / "checkpoints"
OUT_CSV = REPO_ROOT / "experiments" / "result" / "analysis" / "phase_c_wall_clock.csv"

PHASE_DIRS = ["phase_c_method_comparison", "phase_c_extra_ratios", "phase_c_dcrnn_no_cl"]

# 2026-03-03 00:46:21,523
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
# coreset_indices/SAN_BERNARDINO/k_medoids_euclidean_070_seed123.json
INDEX_RE = re.compile(
    r"coreset_indices/(?P<ds>[A-Z_]+)/(?P<method>[a-z_]+)_(?P<dist>[a-z]+)_(?P<ratio>\d{3})_seed(?P<seed>\d+)\.json"
)
# Train dataset length: 10145
DSLEN_RE = re.compile(r"Train dataset length:\s*(\d+)")


EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
COMPLETED_RE = re.compile(r"Test (metrics|results) saved to")


def parse_log(path: Path) -> dict | None:
    try:
        with path.open(errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None

    start_ts: dt.datetime | None = None
    end_ts: dt.datetime | None = None
    method = ratio = seed = ds = dist = None
    train_len = None
    epochs_total = None
    epochs_seen = 0
    completed = False

    for line in lines:
        em = EPOCH_RE.search(line)
        if em:
            epochs_seen += 1
            epochs_total = int(em.group(2))
        if COMPLETED_RE.search(line):
            completed = True
        m = TS_RE.match(line)
        if m:
            try:
                ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if start_ts is None:
                start_ts = ts
            end_ts = ts
        if method is None:
            mi = INDEX_RE.search(line)
            if mi:
                ds = mi.group("ds")
                method = mi.group("method")
                dist = mi.group("dist")
                ratio = int(mi.group("ratio")) / 100.0
                seed = int(mi.group("seed"))
        if train_len is None:
            md = DSLEN_RE.search(line)
            if md:
                train_len = int(md.group(1))

    if start_ts is None or end_ts is None:
        return None
    duration = (end_ts - start_ts).total_seconds()

    # If no selection JSON found in log -> probably full-data run.
    if method is None:
        method = "full"
        ratio = 1.0
        seed = -1
        dist = "-"

    # Model and dataset come from path:
    # checkpoints/<phase>/<MODEL>/xtraffic/<DS>_XX_12_12/<run>/<hash>/training_log_*.log
    parts = path.parts
    try:
        model = parts[parts.index("checkpoints") + 2]
    except (ValueError, IndexError):
        model = "?"
    if ds is None:
        for p in parts:
            if p.startswith(("SAN_BERNARDINO", "CONTRA_COSTA")):
                ds = p.split("_")[0] + "_" + p.split("_")[1]
                break

    return {
        "phase": parts[parts.index("checkpoints") + 1],
        "model": model,
        "dataset": ds or "?",
        "method": method,
        "distance": dist or "-",
        "ratio": ratio,
        "seed": seed if seed is not None else -1,
        "train_len": train_len,
        "epochs_total": epochs_total,
        "epochs_seen": epochs_seen,
        "completed": completed,
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "wall_seconds": duration,
        "log_path": str(path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    rows: list[dict] = []
    for phase in PHASE_DIRS:
        root = CKPT_ROOT / phase
        if not root.exists():
            continue
        for log_path in root.rglob("training_log_*.log"):
            row = parse_log(log_path)
            if row is None:
                continue
            rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "phase", "model", "dataset", "method", "distance", "ratio", "seed",
        "train_len", "epochs_total", "epochs_seen", "completed",
        "start", "end", "wall_seconds", "log_path",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")
    if rows:
        # Quick summary
        from statistics import mean, median
        per_ratio: dict = {}
        for r in rows:
            per_ratio.setdefault(r["ratio"], []).append(r["wall_seconds"])
        for ratio, values in sorted(per_ratio.items(), key=lambda kv: kv[0] or 0):
            print(f"  ratio {ratio}: n={len(values):4d} median={median(values):7.1f}s mean={mean(values):7.1f}s")


if __name__ == "__main__":
    main()
