"""
Build CONTRA_COSTA incident metadata matching the SAN_BERNARDINO format.

Rationale
---------
SB had `datasets/xtraffic/SAN_BERNARDINO/incident_metadata_2023.csv` already,
but CC was missing. The source data exists in the XTraffic repo: incidents are
matched to stations in `match_incidents.csv`, and incident type/description
come from `incidents_sel.csv`. We filter to CC stations and compute the same
slot derivatives as SB.

Sources
-------
- /home/uqtyu7/github/XTraffic/process/match_incidents.csv
    columns: station_id, Abs PM_x, incident_id, Abs PM_y, dt, dis
- /home/uqtyu7/github/XTraffic/causal_analysis/data/incidents_sel.csv
    columns (relevant): `Incident Id`, type, DESCRIPTION
- datasets/xtraffic/CONTRA_COSTA/metadata.csv
    (maps station_id -> sensor_idx via `idx` column)

Slot formulas (verified against SB)
-----------------------------------
- incident_slot = floor((incident_time - 2023-01-01T00:00) / 5min)
- post_incident_start_slot = incident_slot + 1
- input_start_slot  = incident_slot - 11    (INPUT_LEN=12 ending at incident)
- input_end_slot    = incident_slot + 1
- output_start_slot = incident_slot + 1
- output_end_slot   = incident_slot + 7     (SB uses 6 post-incident steps)

Output
------
- datasets/xtraffic/CONTRA_COSTA/incident_metadata_2023.csv
  with columns matching SB format.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MATCH_CSV = Path('/home/uqtyu7/github/XTraffic/process/match_incidents.csv')
INCIDENTS_SEL = Path('/home/uqtyu7/github/XTraffic/causal_analysis/data/incidents_sel.csv')
CC_METADATA = REPO / 'datasets/xtraffic/CONTRA_COSTA/metadata.csv'
OUT_CSV = REPO / 'datasets/xtraffic/CONTRA_COSTA/incident_metadata_2023.csv'

REF_TIME = datetime(2023, 1, 1, 0, 0, 0)
SLOT_SECONDS = 5 * 60


def compute_slot(t: pd.Timestamp) -> int:
    return int((t - pd.Timestamp(REF_TIME)).total_seconds() // SLOT_SECONDS)


def main() -> None:
    cc_meta = pd.read_csv(CC_METADATA)
    station_to_idx = dict(zip(cc_meta['station_id'], cc_meta['idx']))
    cc_stations = set(cc_meta['station_id'])
    print(f'CC stations: {len(cc_stations)}')

    match = pd.read_csv(MATCH_CSV)
    cc_match = match[match['station_id'].isin(cc_stations)].copy()
    print(f'CC incident-sensor pairs: {len(cc_match)} '
          f'(out of {len(match)} total in match_incidents.csv)')

    inc_sel = pd.read_csv(INCIDENTS_SEL)
    inc_sel = inc_sel.rename(columns={'Incident Id': 'incident_id',
                                      'DESCRIPTION': 'incident_description',
                                      'type': 'incident_type'})
    # Keep just what we need
    inc_sel = inc_sel[['incident_id', 'incident_type', 'incident_description']]
    inc_sel = inc_sel.drop_duplicates('incident_id')

    merged = cc_match.merge(inc_sel, on='incident_id', how='left')

    # Convert datetime — match_incidents uses MM/DD/YYYY HH:MM:SS
    merged['incident_time'] = pd.to_datetime(merged['dt'], errors='coerce')
    before = len(merged)
    merged = merged.dropna(subset=['incident_time'])
    if before != len(merged):
        print(f'Dropped {before - len(merged)} rows with unparseable dt')

    merged['sensor_idx'] = merged['station_id'].map(station_to_idx).astype('Int64')
    merged['incident_slot'] = merged['incident_time'].apply(compute_slot)
    merged['post_incident_start_slot'] = merged['incident_slot'] + 1
    merged['input_start_slot']  = merged['incident_slot'] - 11
    merged['input_end_slot']    = merged['incident_slot'] + 1
    merged['output_start_slot'] = merged['incident_slot'] + 1
    merged['output_end_slot']   = merged['incident_slot'] + 7
    merged['distance'] = merged['dis']

    out = merged[[
        'incident_id', 'station_id', 'sensor_idx',
        'incident_time', 'incident_slot', 'post_incident_start_slot',
        'input_start_slot', 'input_end_slot',
        'output_start_slot', 'output_end_slot',
        'distance', 'incident_type', 'incident_description',
    ]].copy()
    out['incident_time'] = out['incident_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    out = out.sort_values(['incident_slot', 'sensor_idx']).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False)
    print(f'\nWrote {len(out)} rows to {OUT_CSV}')
    print('Sample:')
    print(out.head(3).to_string(index=False))
    # Sanity vs SB
    n_in_3mo = (out['incident_slot'] < 24192).sum()
    print(f'\nIncidents within 3-month window [0, 24192): {n_in_3mo}')


if __name__ == '__main__':
    main()
