"""
Extract severe incidents (flow drop z-score < -2) for any Xtraffic dataset.

Generalization of EDA/severe_incident_analysis.py to both SB and CC. A severe
incident is one where observed flow at the incident (time, sensor) is > 2σ
below the time-of-day baseline (sampled from other days at the same TOD).

Output: EDA/severe_incidents_{DATASET}.csv

Usage
-----
  conda activate cuda && python scripts/analysis/extract_severe_incidents.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATASETS = ['SAN_BERNARDINO', 'CONTRA_COSTA']
OUT_DIR = REPO / 'EDA'

Z_THRESHOLD = -2.0
MIN_BASELINE_FLOW = 20.0
STEPS_PER_DAY = 288


def find_severe_incidents(dataset: str) -> pd.DataFrame:
    base = REPO / 'datasets/xtraffic' / dataset
    desc = json.load(open(base / 'desc.json'))
    shape = tuple(desc['shape'])
    data = np.memmap(base / 'data.dat', dtype=np.float32, mode='r', shape=shape)

    incidents = pd.read_csv(base / 'incident_metadata_2023.csv')
    print(f'\n[{dataset}] data {shape}, incidents {len(incidents)}')

    records = []
    flow_idx = 0
    n_days = shape[0] // STEPS_PER_DAY

    for _, row in incidents.iterrows():
        node = int(row['sensor_idx'])
        slot = int(row['incident_slot'])
        if slot >= shape[0] or node >= shape[1] or node < 0:
            continue
        tod = slot % STEPS_PER_DAY
        incident_day = slot // STEPS_PER_DAY

        # baseline from other days at same TOD, skipping ±1 around incident day
        base_slots = [day * STEPS_PER_DAY + tod for day in range(n_days)
                      if abs(day - incident_day) > 1
                      and 0 <= day * STEPS_PER_DAY + tod < shape[0]]
        if len(base_slots) < 10:
            continue
        flow_at   = float(data[slot, node, flow_idx])
        base      = data[np.array(base_slots), node, flow_idx]
        base_mean = float(base.mean())
        base_std  = float(base.std())
        if base_std <= 0:
            continue
        z = (flow_at - base_mean) / base_std
        if z < Z_THRESHOLD and base_mean > MIN_BASELINE_FLOW:
            records.append({**row.to_dict(),
                            'flow_at_incident': flow_at,
                            'baseline_mean':    base_mean,
                            'baseline_std':     base_std,
                            'z_score':          z,
                            'drop_ratio':       (base_mean - flow_at) / base_mean,
                            'hour':             tod // 12})
    df = pd.DataFrame(records)
    # count incidents within 3-month training window (data_range=(0, 24192))
    in_window = df[df['incident_slot'] < 24192]
    print(f'  severe incidents (z<{Z_THRESHOLD}, baseline>{MIN_BASELINE_FLOW}): {len(df)}')
    print(f'  within 3-month window [0, 24192): {len(in_window)}')
    return df


def main() -> None:
    for ds in DATASETS:
        df = find_severe_incidents(ds)
        out = OUT_DIR / f'severe_incidents_{ds}.csv'
        df.to_csv(out, index=False)
        print(f'  wrote {out} ({len(df)} rows)')


if __name__ == '__main__':
    main()
