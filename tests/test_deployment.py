import json
import sqlite3

import pandas as pd
import pytest

from scripts.serve import validate_artifacts


def artifacts(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    pd.DataFrame({"cell_id": [1, 1], "timestamp": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
                  "internet": [1.0, 2.0]}).to_parquet(tmp_path / "data/processed/traffic.parquet")
    pd.DataFrame({"cell_id": [1], "prediction": [1.1], "actual": [1.0]}).to_parquet(tmp_path / "results/forecast_predictions.parquet")
    (tmp_path / "results/data_quality.json").write_text(json.dumps({"hourly_rows": 2, "selected_cells": 1}))
    (tmp_path / "results/evidence_report.json").write_text(json.dumps({"forecast_comparisons": [{"horizon_h": 1}], "anomaly_event_counts": [{"tp": 1}]}))
    with sqlite3.connect(tmp_path / "data/processed/analytics.sqlite") as db:
        for table in ("mart_daily_load", "mart_cell_trends"):
            db.execute(f"CREATE TABLE {table} (n INTEGER)")
            db.execute(f"INSERT INTO {table} VALUES (1)")
    return tmp_path


def test_empty_deployment_cannot_serve_a_successful_dashboard(tmp_path):
    with pytest.raises(ValueError, match="Missing generated artifacts"):
        validate_artifacts(tmp_path)


def test_valid_outputs_pass_and_inconsistent_denominators_are_rejected(tmp_path):
    root = artifacts(tmp_path)
    assert validate_artifacts(root) == {"hourly_rows": 2, "selected_cells": 1}
    (root / "results/data_quality.json").write_text('{"hourly_rows": 200000000, "selected_cells": 30}')
    with pytest.raises(ValueError, match="denominators"):
        validate_artifacts(root)


def test_corrupt_forecast_is_rejected(tmp_path):
    root = artifacts(tmp_path)
    pd.DataFrame({"cell_id": [1], "prediction": [float("inf")], "actual": [1.0]}).to_parquet(root / "results/forecast_predictions.parquet")
    with pytest.raises(ValueError, match="non-finite"):
        validate_artifacts(root)
