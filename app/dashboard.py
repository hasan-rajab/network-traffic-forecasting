from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Network Traffic Capacity Planning", layout="wide")
st.title("Network Traffic Forecasting, Anomaly Detection & Capacity Planning")

traffic_path = Path("data/processed/traffic.parquet")
if not traffic_path.exists():
    st.warning("Run `make all` first. The dashboard only displays generated real-data outputs.")
    st.stop()

df = pd.read_parquet(traffic_path)
cells = sorted(df.cell_id.unique())
cell = st.selectbox("Network cell", cells)

hist = df[df.cell_id == cell].sort_values("timestamp")
st.subheader("History")
st.line_chart(hist.set_index("timestamp")[["internet"]])

forecast_path = Path("results/forecast_predictions.parquet")
if forecast_path.exists():
    fp = pd.read_parquet(forecast_path)
    f = fp[(fp.model == "lightgbm_global") & (fp.horizon_h == 24) & (fp.cell_id == cell)].sort_values("target_time")
    if len(f):
        st.subheader("+24h rolling-origin forecast")
        chart = f.set_index("target_time")[["actual", "prediction"]]
        st.line_chart(chart)

anomaly_path = Path("results/anomaly_test_predictions.parquet")
if anomaly_path.exists():
    a = pd.read_parquet(anomaly_path); a = a[a.cell_id == cell].sort_values("target_time")
    if len(a):
        st.subheader("Injected-anomaly evaluation view")
        st.caption("Injected anomalies are synthetic labels on real held-out traffic; they are not real operator incidents.")
        cols = ["target_time", "injected_actual", "prediction", "robust_z_flag", "isolation_forest_flag"]
        st.dataframe(a[cols], use_container_width=True)

capacity_path = Path("results/capacity_predictions.parquet")
if capacity_path.exists():
    c = pd.read_parquet(capacity_path); c = c[c.cell_id == cell].sort_values("target_time")
    if len(c):
        st.subheader("Capacity-proxy warnings")
        latest = c[["target_time", "prediction", "capacity_proxy", "predicted_breach", "actual_breach"]]
        st.dataframe(latest, use_container_width=True)
        st.caption("Capacity is the training-period 95th percentile proxy, not real engineered operator capacity.")
