import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.sql_analytics import aggregate_selected_chunks, build_warehouse, export_marts


def test_raw_sql_sums_country_rows_across_chunks_and_excludes_unselected_cells():
    columns = ['cell_id', 'timestamp_ms', 'country_code', 'sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
    chunks = [pd.DataFrame([[1, 1000, 39, 1, 2, 3, 4, 10], [9, 1000, 39, 0, 0, 0, 0, 999]], columns=columns),
              pd.DataFrame([[1, 1000, 44, 2, 1, 1, 0, 20], [1, 2000, 39, 0, 0, 0, 0, np.nan]], columns=columns)]
    result, scanned = aggregate_selected_chunks(iter(chunks), {1})
    assert scanned == 4 and result.attrs['selected_source_rows'] == 3
    assert result.internet.tolist() == [30.0, 0.0]
    assert result.sms_in.tolist() == [3.0, 0.0]


def fixture_data():
    traffic = pd.DataFrame({'cell_id': [1, 1, 1, 2],
        'timestamp': pd.to_datetime(['2020-01-01T00:00Z', '2020-01-01T01:00Z', '2020-01-01T03:00Z', '2020-01-01T00:00Z']),
        'internet': [10., 20., 90., np.nan]})
    cells = pd.DataFrame({'cell_id': [1, 2], 'traffic_band': ['low', 'high'], 'first_window_internet_total': [30., 0.]})
    return traffic, cells


def test_sql_windows_are_past_only_gap_aware_and_nulls_are_not_zero(tmp_path):
    traffic, cells = fixture_data(); path = tmp_path/'warehouse.sqlite'
    build_warehouse(traffic, cells, path)
    assert export_marts(path, tmp_path) == dict.fromkeys(['orphan_cells', 'invalid_activity', 'duplicate_hours', 'unknown_strata'], 0)
    with sqlite3.connect(path) as db:
        trend = db.execute('SELECT prior_24h_mean, hourly_change FROM mart_cell_trends WHERE cell_id=1 ORDER BY timestamp_utc').fetchall()
        assert trend == [(None, None), (10., 10.), (15., None)]
        assert db.execute("SELECT missing_hours, total_activity FROM mart_daily_load WHERE traffic_band='high'").fetchone() == (1, None)
    build_warehouse(traffic, cells, path)  # refresh is idempotent
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM fact_hourly').fetchone()[0] == 4


@pytest.mark.parametrize('case', ['duplicate', 'orphan', 'negative', 'infinity', 'subhour'])
def test_rejects_invalid_input_without_overwriting_existing_warehouse(tmp_path, case):
    traffic, cells = fixture_data(); path = tmp_path/'warehouse.sqlite'
    build_warehouse(traffic, cells, path)
    invalid = traffic.copy()
    if case == 'duplicate': invalid = pd.concat([invalid, invalid.iloc[:1]])
    if case == 'orphan': invalid.loc[0, 'cell_id'] = 999
    if case == 'negative': invalid.loc[0, 'internet'] = -1
    if case == 'infinity': invalid.loc[0, 'internet'] = np.inf
    if case == 'subhour': invalid.loc[0, 'timestamp'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError): build_warehouse(invalid, cells, path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT SUM(internet) FROM fact_hourly').fetchone()[0] == 120
