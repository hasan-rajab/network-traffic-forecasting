from __future__ import annotations

import numpy as np


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.abs(y) + np.abs(p)
    return float(np.mean(np.where(den == 0, 0.0, 2 * np.abs(y - p) / den)) * 100)


def mase_scale(train, season: int = 24) -> float:
    train = np.asarray(train, float)
    return float(np.mean(np.abs(train[season:] - train[:-season])))


def metric_bundle(y, p, scale: float) -> dict:
    m = mae(y, p)
    return {"mae": m, "rmse": rmse(y, p), "smape": smape(y, p), "mase": float(m / scale)}


def generate_portfolio_outputs(config_path: str = "config.yaml"):
    import json, sys
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import yaml
    import pandas as _pd, numpy as _np, sklearn, statsmodels, lightgbm, torch, pyarrow, matplotlib

    with open(config_path, "r", encoding="utf-8") as f: cfg = yaml.safe_load(f)
    root = Path("results"); Path("figures").mkdir(exist_ok=True)
    env = {"python": sys.version.split()[0], "pandas": _pd.__version__, "numpy": _np.__version__,
           "scikit-learn": sklearn.__version__, "statsmodels": statsmodels.__version__, "lightgbm": lightgbm.__version__,
           "torch": torch.__version__, "pyarrow": pyarrow.__version__, "matplotlib": matplotlib.__version__}
    (root / "environment_versions.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    fs = pd.read_csv(root / "forecast_summary.csv"); am = pd.read_csv(root / "anomaly_metrics.csv"); cs = pd.read_csv(root / "capacity_summary.csv")
    lg = fs[fs.model == "lightgbm_global"].set_index("horizon_h")
    best = am[am.breakdown == "global"].sort_values("f1", ascending=False).iloc[0]
    dq = json.loads((root / "data_quality.json").read_text(encoding="utf-8"))
    cv = pd.DataFrame([{
        "forecast_model": "lightgbm_global", "cells": int(lg.loc[1, "cells"]), "rolling_folds": int(cfg["forecast"]["folds"]),
        "mae_1h": lg.loc[1, "mae_mean"], "mae_improvement_1h_vs_naive24_pct": lg.loc[1, "mae_improvement_vs_naive24_pct"],
        "mae_24h": lg.loc[24, "mae_mean"], "mae_improvement_24h_vs_naive24_pct": lg.loc[24, "mae_improvement_vs_naive24_pct"],
        "anomaly_detector": best.detector, "anomaly_event_f1": best.f1, "anomaly_event_precision": best.precision,
        "anomaly_event_recall": best.recall, "capacity_precision": cs.iloc[0].precision, "capacity_recall": cs.iloc[0].recall,
        "capacity_f1": cs.iloc[0].f1, "raw_rows_scanned": dq["raw_rows_scanned"],
    }])
    cv.to_csv(root / "cv_metrics.csv", index=False)

    preds = pd.read_parquet(root / "forecast_predictions.parquet")
    selected = pd.read_csv(root / "selected_cells.csv"); cid = int(selected.sort_values("first_window_internet_total").iloc[-1].cell_id)
    ex = preds[(preds.model == "lightgbm_global") & (preds.horizon_h == 24) & (preds.fold == int(cfg["forecast"]["folds"])) & (preds.cell_id == cid)].sort_values("target_time")
    fig, ax = plt.subplots(figsize=(11, 4)); ax.plot(ex.target_time, ex.actual, label="actual"); ax.plot(ex.target_time, ex.prediction, label="+24h LightGBM")
    ax.legend(); ax.set_title(f"24-hour forecast example — cell {cid}, final fold"); ax.set_ylabel("Internet activity"); fig.tight_layout(); fig.savefig("figures/forecast_example.png", dpi=160); plt.close(fig)


def _main_cli():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args(); generate_portfolio_outputs(args.config)


if __name__ == "__main__":
    _main_cli()
