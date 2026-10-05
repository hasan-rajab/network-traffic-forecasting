"""SQLite aggregation of selected raw records and a tested analytics warehouse.

Raw-country rows are aggregated in SQL before time-grid filling. The warehouse
contains the selected modelling subset, never a claim that all raw rows train ML.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITY = ['sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
SQL_ROOT = Path(__file__).resolve().parents[1] / 'sql'


def aggregate_selected_chunks(chunks, cells):
    """Stream selected country-code rows into SQL, returning grouped rows + scope."""
    raw_rows = selected_rows = 0
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE selected_raw(cell_id INTEGER, timestamp_ms INTEGER, '
                   'sms_in REAL, sms_out REAL, call_in REAL, call_out REAL, internet REAL)')
        for chunk in chunks:
            raw_rows += len(chunk)
            subset = chunk.loc[chunk.cell_id.isin(cells), ['cell_id', 'timestamp_ms', *ACTIVITY]].copy()
            subset[ACTIVITY] = subset[ACTIVITY].fillna(0.0)
            if not np.isfinite(subset.to_numpy(dtype=float)).all():
                raise ValueError('Non-finite values in selected raw records')
            if (subset[ACTIVITY] < 0).any().any():
                raise ValueError('Negative activity in selected raw records')
            selected_rows += len(subset)
            db.executemany('INSERT INTO selected_raw VALUES (?,?,?,?,?,?,?)',
                           subset.itertuples(index=False, name=None))
        sums = ', '.join(f'SUM({col}) AS {col}' for col in ACTIVITY)
        grouped = pd.read_sql_query(
            f'SELECT cell_id, timestamp_ms, {sums} FROM selected_raw '
            'GROUP BY cell_id, timestamp_ms ORDER BY cell_id, timestamp_ms', db)
    grouped.attrs['selected_source_rows'] = selected_rows
    return grouped, raw_rows


def build_warehouse(traffic, selection, database, quality=None):
    """Atomically refresh the subset warehouse. Null load stays null, not zero."""
    if traffic.duplicated(['cell_id', 'timestamp']).any():
        raise ValueError('Duplicate cell/timestamp in hourly traffic')
    if selection.cell_id.duplicated().any():
        raise ValueError('Duplicate cell dimensions')
    if set(traffic.cell_id) - set(selection.cell_id):
        raise ValueError('Traffic cell missing from selected-cell dimension')
    values = pd.to_numeric(traffic.internet, errors='raise')
    present = values.dropna()
    if (present < 0).any() or not np.isfinite(present.to_numpy()).all():
        raise ValueError('Invalid hourly activity')
    times = pd.to_datetime(traffic.timestamp, utc=True)
    if not (times == times.dt.floor('h')).all():
        raise ValueError('Warehouse input must be hourly')
    if not selection.traffic_band.isin(['low', 'medium', 'high']).all():
        raise ValueError('Unknown traffic stratum')
    path = Path(database); path.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3's executescript commits a pending transaction, so execute DDL before
    # the data transaction. Existing tables/views survive a failed data refresh.
    with sqlite3.connect(path) as db:
        db.executescript((SQL_ROOT / 'schema.sql').read_text())
        db.execute('PRAGMA foreign_keys = ON')
        with db:
            db.execute('DELETE FROM fact_hourly'); db.execute('DELETE FROM dim_cell')
            db.execute('DELETE FROM run_metadata')
            db.executemany('INSERT INTO dim_cell VALUES (?, ?, ?)',
                           [(int(r.cell_id), r.traffic_band, float(r.first_window_internet_total))
                            for r in selection.itertuples()])
            db.executemany('INSERT INTO fact_hourly VALUES (?, ?, ?)',
                           [(int(cid), int(ts.timestamp()), None if pd.isna(v) else float(v))
                            for cid, ts, v in zip(traffic.cell_id, times, values)])
            metadata = {'warehouse_rows': len(traffic), 'warehouse_cells': int(traffic.cell_id.nunique()),
                        'source_quality': quality or {}, 'scope': 'selected hourly modelling subset'}
            db.execute('INSERT INTO run_metadata VALUES (?, ?)', ('scope', json.dumps(metadata)))
        for model in sorted((SQL_ROOT / 'models').glob('*.sql')):
            db.executescript(model.read_text())
        checks = warehouse_checks(db)
        if any(checks.values()):
            raise ValueError(f'Warehouse checks failed: {checks}')
    return metadata


def warehouse_checks(db):
    """SQL data contracts; every query returns a count of invalid rows."""
    checks = {
        'orphan_cells': 'SELECT COUNT(*) FROM fact_hourly f LEFT JOIN dim_cell d USING(cell_id) WHERE d.cell_id IS NULL',
        'invalid_activity': 'SELECT COUNT(*) FROM fact_hourly WHERE internet < 0',
        'duplicate_hours': 'SELECT COUNT(*) FROM (SELECT cell_id, timestamp_s FROM fact_hourly GROUP BY 1,2 HAVING COUNT(*) > 1)',
        'unknown_strata': "SELECT COUNT(*) FROM dim_cell WHERE traffic_band NOT IN ('low','medium','high')",
    }
    return {name: int(db.execute(query).fetchone()[0]) for name, query in checks.items()}


def export_marts(database, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as db:
        for view in ['mart_daily_load', 'mart_cell_trends']:
            pd.read_sql_query(f'SELECT * FROM {view}', db).to_csv(out / f'{view}.csv', index=False)
        checks = warehouse_checks(db)
        (out / 'sql_quality_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic', default='data/processed/traffic.parquet')
    parser.add_argument('--selection', default='results/selected_cells.csv')
    parser.add_argument('--quality', default='results/data_quality.json')
    parser.add_argument('--database', default='data/processed/analytics.sqlite')
    parser.add_argument('--outdir', default='results')
    args = parser.parse_args()
    read = pd.read_csv if Path(args.traffic).suffix == '.csv' else pd.read_parquet
    quality = json.loads(Path(args.quality).read_text()) if Path(args.quality).exists() else {}
    metadata = build_warehouse(read(args.traffic), pd.read_csv(args.selection), args.database, quality)
    export_marts(args.database, args.outdir)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
