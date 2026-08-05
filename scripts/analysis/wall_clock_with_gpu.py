"""Annotate Phase C wall-clock rows with the GPU type they ran on.

Strategy:
1. Pull `sacct` rows for every Phase C SLURM job (Start, End, NodeList).
2. Map NodeList -> GPU type using the cluster's known node assignments.
3. For each training_log start timestamp, find the unique sacct entry whose
   [Start, End] window contains it. Attach the GPU type to that row.

Outputs:
    experiments/result/analysis/phase_c_wall_clock_gpu.csv
"""
from __future__ import annotations

import csv
import datetime as dt
import re
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WC_CSV = REPO / "experiments" / "result" / "analysis" / "phase_c_wall_clock.csv"
OUT_CSV = REPO / "experiments" / "result" / "analysis" / "phase_c_wall_clock_gpu.csv"

# Cluster node -> GPU model. Built from `sinfo -p gpu_{cuda,rocm} --Format=NodeList,Gres`.
NODE_TO_GPU = {
    # CUDA partition
    "bun003": "A100",
    "bun068": "A100",
    "bun004": "A100-80GB",
    "bun005": "A100-80GB",
    **{f"bun{n:03d}": "H100" for n in (71, 72, 73, 74, 75, 76, 116)},
    **{f"bun{n:03d}": "L40" for n in (77, 78, 79, 80, 81, 82)},
    "bun124": "L40s",
    "bun125": "L40s",
    # ROCm partition
    "bun001": "MI210",
    "bun002": "MI210",
    "bun070": "MI210",
    "bun145": "MI300X",
    "bun146": "MI300X",
}


def fetch_sacct() -> pd.DataFrame:
    """Pull all Phase C jobs from sacct as a DataFrame."""
    cmd = [
        "sacct",
        "-S", "2026-02-01",
        "-E", "2026-04-30",
        "--format=JobID,JobName,Partition,NodeList,State,Start,End,Elapsed",
        "--parsable2",
        "-X",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = [line.split("|") for line in out.stdout.strip().splitlines()]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df[df.JobName.str.contains("phase_c", case=False, na=False)]
    df = df[df.State.isin(["COMPLETED", "TIMEOUT", "FAILED", "RUNNING"])]
    df = df[df.Start != "Unknown"]
    df = df[df.Start != "None"]
    df["Start_dt"] = pd.to_datetime(df.Start, errors="coerce")
    df["End_dt"] = pd.to_datetime(df.End, errors="coerce")
    df = df.dropna(subset=["Start_dt", "End_dt"])
    df["NodeList"] = df.NodeList.fillna("")
    df["GPU"] = df.NodeList.map(lambda n: NODE_TO_GPU.get(n.strip(), "UNKNOWN"))
    return df.reset_index(drop=True)


def annotate() -> None:
    sacct = fetch_sacct()
    print(f"sacct: {len(sacct)} Phase C job rows")
    print("GPU distribution:", sacct.GPU.value_counts().to_dict())

    wc = pd.read_csv(WC_CSV)
    wc["start_dt"] = pd.to_datetime(wc["start"])

    # Build sorted index of sacct windows for fast lookup.
    sacct_sorted = sacct.sort_values("Start_dt").reset_index(drop=True)

    def lookup_gpu(ts: dt.datetime) -> tuple[str, str, str]:
        """Return (job_id, node, gpu) of the sacct row whose window contains ts."""
        cands = sacct_sorted[(sacct_sorted.Start_dt <= ts) & (sacct_sorted.End_dt >= ts)]
        if cands.empty:
            return ("", "", "UNMATCHED")
        row = cands.iloc[0]
        return (str(row.JobID), str(row.NodeList), str(row.GPU))

    job_ids: list[str] = []
    nodes: list[str] = []
    gpus: list[str] = []
    for ts in wc.start_dt:
        jid, node, gpu = lookup_gpu(ts.to_pydatetime())
        job_ids.append(jid)
        nodes.append(node)
        gpus.append(gpu)
    wc["slurm_job_id"] = job_ids
    wc["slurm_node"] = nodes
    wc["gpu"] = gpus

    wc.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(wc)} rows)")
    print("Annotated GPU distribution:", wc.gpu.value_counts().to_dict())


if __name__ == "__main__":
    annotate()
