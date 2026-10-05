from pathlib import Path
import json
import sqlite3
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Network Traffic Capacity Planning", layout="wide")
st.title("Network Traffic Forecasting, Anomaly Detection & Capacity Planning")
st.markdown("[← Portfolio](https://nexusmind-production-3da9.up.railway.app/portfolio)")
st.write("Explore historical demand, compare short and long forecast horizons, and test operational planning scenarios.")
artifact_root = Path(os.environ.get("NETWORK_ARTIFACT_DIR", "."))

quality_path = artifact_root / "results/data_quality.json"
if quality_path.exists():
    quality = json.loads(quality_path.read_text())
    st.info(f"Source scope: {quality['raw_rows_scanned']:,} raw country-code rows scanned; "
            f"{quality['selected_cells']} selected cells over {quality['source_days']} days; "
            f"{quality['hourly_rows']:,} hourly modelling rows. Raw scans are not ML sample size.")

evidence_path = artifact_root / "results/evidence_report.json"
if evidence_path.exists():
    evidence = json.loads(evidence_path.read_text())
    st.subheader("Forecast comparison and uncertainty — all evaluated cells")
    st.dataframe(pd.DataFrame(evidence["forecast_comparisons"]), use_container_width=True)
    st.caption("Paired 24-hour block bootstrap within folds; interpret each horizon separately. "
               "A small point gain or interval containing zero is not a strong planning advantage.")
    st.subheader("Anomaly event denominators — held-out injected labels")
    st.dataframe(pd.DataFrame(evidence["anomaly_event_counts"]), use_container_width=True)

traffic_path = artifact_root / "data/processed/traffic.parquet"
if not traffic_path.exists():
    st.warning("Run `make all` first. The dashboard only displays generated real-data outputs.")
    st.stop()

df = pd.read_parquet(traffic_path)
cells = sorted(df.cell_id.unique())
cell = st.selectbox("Network cell", cells)
horizon = st.radio("Forecast horizon", [1, 24], format_func=lambda value: f"+{value} hour{'s' if value > 1 else ''}", horizontal=True)
st.caption("The strongest measured advantage is at +1h. The +24h point improvement is 4.19%, but its paired MAE-difference interval includes zero.")

warehouse_path = artifact_root / "data/processed/analytics.sqlite"
if warehouse_path.exists():
    with sqlite3.connect(warehouse_path.resolve().as_uri() + "?mode=ro", uri=True) as database:
        st.subheader("SQL daily load, segmentation and coverage")
        daily = pd.read_sql_query("SELECT * FROM mart_daily_load ORDER BY date_utc, traffic_band", database)
        bands = st.multiselect("Traffic segments", sorted(daily.traffic_band.unique()), default=sorted(daily.traffic_band.unique()))
        daily = daily[daily.traffic_band.isin(bands)]
        st.dataframe(daily, use_container_width=True)
        st.download_button("Download selected daily SQL aggregates", daily.to_csv(index=False), "telecom-daily-segments.csv", "text/csv")
        if not daily.empty:
            st.line_chart(daily.groupby("date_utc")["total_activity"].sum())
        st.subheader("SQL cell trends — prior-only 24-hour history")
        trends = pd.read_sql_query("SELECT * FROM mart_cell_trends WHERE cell_id=? ORDER BY timestamp_utc", database, params=(int(cell),))
        st.dataframe(trends, use_container_width=True)
        st.download_button("Download this cell's SQL trends", trends.to_csv(index=False), f"telecom-cell-{cell}-trends.csv", "text/csv")
        with st.expander("Inspect SQL transformations"):
            st.write("Daily segmentation joins the hourly fact to cell bands. Prior-only rolling history uses elapsed-time windows and LAG; missing hours stay explicit.")
            for model in ["01_daily_load.sql", "02_cell_trends.sql"]:
                path = Path(__file__).resolve().parents[1] / "sql/models" / model
                st.code(path.read_text(), language="sql")

hist = df[df.cell_id == cell].sort_values("timestamp")
st.subheader("History")
st.line_chart(hist.set_index("timestamp")[["internet"]])

forecast_path = artifact_root / "results/forecast_predictions.parquet"
if forecast_path.exists():
    fp = pd.read_parquet(forecast_path)
    f = fp[(fp.model == "lightgbm_global") & (fp.horizon_h == horizon) & (fp.cell_id == cell)].sort_values("target_time")
    if len(f):
        st.subheader(f"+{horizon}h rolling-origin forecast")
        chart = f.set_index("target_time")[["actual", "prediction"]]
        st.line_chart(chart)
        st.download_button("Download selected evaluated forecasts", f.to_csv(index=False), f"telecom-cell-{cell}-{horizon}h.csv", "text/csv")

anomaly_path = artifact_root / "results/anomaly_test_predictions.parquet"
if anomaly_path.exists():
    a = pd.read_parquet(anomaly_path); a = a[a.cell_id == cell].sort_values("target_time")
    if len(a):
        st.subheader("Injected-anomaly evaluation view")
        st.caption("Injected anomalies are synthetic labels on real held-out traffic; they are not real operator incidents.")
        cols = ["target_time", "injected_actual", "prediction", "robust_z_flag", "isolation_forest_flag"]
        st.dataframe(a[cols], use_container_width=True)

capacity_path = artifact_root / "results/capacity_predictions.parquet"
if capacity_path.exists():
    c = pd.read_parquet(capacity_path); c = c[c.cell_id == cell].sort_values("target_time")
    if len(c):
        st.subheader("Capacity-proxy warnings")
        multiplier = st.slider("Capacity scenario multiplier", 0.5, 1.5, 1.0, 0.05)
        latest = c[["target_time", "prediction", "capacity_proxy", "predicted_breach", "actual_breach"]].copy()
        latest["scenario_proxy"] = latest.capacity_proxy * multiplier
        latest["scenario_predicted_breach"] = latest.prediction > latest.scenario_proxy
        st.metric("Evaluated periods above scenario proxy", int(latest.scenario_predicted_breach.sum()))
        st.dataframe(latest, use_container_width=True)
        st.line_chart(latest.set_index("target_time")[["prediction", "scenario_proxy"]])
        st.download_button("Download capacity scenario", latest.to_csv(index=False), f"telecom-cell-{cell}-capacity-scenario.csv", "text/csv")
        st.caption("Capacity is the training-period 95th percentile proxy, not real engineered operator capacity.")
        st.caption("The capacity evaluation uses the saved +24h planning outputs. The multiplier changes a hypothetical threshold; it does not alter the forecast or measured model performance.")
