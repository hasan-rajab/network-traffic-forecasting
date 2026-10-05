"""Validate saved, real-data artifacts before starting the public dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import sys

import pandas as pd
import numpy as np


def validate_artifacts(root: Path) -> dict:
    required = (
        "data/processed/traffic.parquet", "data/processed/analytics.sqlite",
        "results/data_quality.json", "results/evidence_report.json",
        "results/forecast_predictions.parquet",
    )
    missing = [p for p in required if not (root / p).is_file()]
    if missing:
        raise ValueError("Missing generated artifacts: " + ", ".join(missing))
    traffic = pd.read_parquet(root / required[0], columns=["cell_id", "timestamp", "internet"])
    quality = json.loads((root / "results/data_quality.json").read_text())
    evidence = json.loads((root / "results/evidence_report.json").read_text())
    if traffic.empty or not traffic["internet"].notna().any():
        raise ValueError("Traffic output is empty or has no observed activity")
    if traffic.duplicated(["cell_id", "timestamp"]).any():
        raise ValueError("Traffic output has duplicate cell/timestamp keys")
    if quality.get("hourly_rows") != len(traffic) or quality.get("selected_cells") != traffic.cell_id.nunique():
        raise ValueError("Data-quality denominators do not match the served traffic")
    if not evidence.get("forecast_comparisons") or not evidence.get("anomaly_event_counts"):
        raise ValueError("Forecast comparison or anomaly event denominators are missing")
    predictions = pd.read_parquet(root / "results/forecast_predictions.parquet",
                                  columns=["cell_id", "prediction", "actual"])
    if predictions.empty or not np.isfinite(predictions[["prediction", "actual"]].to_numpy()).all():
        raise ValueError("Forecast predictions are empty or contain non-finite values")
    if not predictions.cell_id.isin(traffic.cell_id.unique()).all():
        raise ValueError("Forecast predictions refer to cells outside the served traffic")
    database_path = (root / "data/processed/analytics.sqlite").resolve()
    with sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Analytics warehouse integrity check failed")
        for table in ("mart_daily_load", "mart_cell_trends"):
            if db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
                raise ValueError(f"Analytics warehouse has an empty {table}")
    return {"hourly_rows": len(traffic), "selected_cells": int(traffic.cell_id.nunique())}


def main() -> None:
    root = Path(os.environ.get("NETWORK_ARTIFACT_DIR", ".")).resolve()
    try:
        summary = validate_artifacts(root)
        port = int(os.environ.get("PORT", "8501"))
        if not 1024 <= port <= 65535:
            raise ValueError("PORT must be between 1024 and 65535")
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        sys.exit(f"Dashboard deployment blocked: {exc}. Generate and mount the real pipeline outputs first.")
    print(json.dumps({"artifacts_validated": summary}), flush=True)
    os.execvp("streamlit", ["streamlit", "run", "app/dashboard.py",
              "--server.address=0.0.0.0", f"--server.port={port}",
              "--server.headless=true", "--browser.gatherUsageStats=false"])


if __name__ == "__main__":
    main()
