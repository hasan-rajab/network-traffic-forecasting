from pathlib import Path
import json
import pandas as pd

p=Path("README.md"); text=p.read_text(encoding="utf-8")
fs=pd.read_csv("results/forecast_summary.csv"); am=pd.read_csv("results/anomaly_metrics.csv"); cs=pd.read_csv("results/capacity_summary.csv"); dq=json.loads(Path("results/data_quality.json").read_text())
def r(x,n=3): return f"{float(x):.{n}f}"
lg=fs[fs.model=="lightgbm_global"].set_index("horizon_h"); tcn=fs[fs.model=="temporal_cnn_global"].set_index("horizon_h") if (fs.model=="temporal_cnn_global").any() else None
best=am[am.breakdown=="global"].sort_values("f1",ascending=False).iloc[0]
types=am[(am.detector==best.detector)&(am.breakdown=="type")][["group","precision","recall","f1","detection_delay_h"]]
block=f"""<!-- AUTO_RESULTS_START -->
## Measured results

The measured run scanned **{dq['raw_rows_scanned']:,} original raw rows** across {dq['source_days']} days and selected {dq['selected_cells']} cells. Missing 10-minute intervals were **{dq['missing_pct_mean']:.1f}% mean / {dq['missing_pct_max']:.1f}% max** before short-gap filling.

### Forecasting

| Model | Horizon | Cells | MAE | RMSE | sMAPE | MASE | MAE vs 24h naive |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
for _,x in fs.iterrows():
    block+=f"| {x.model} | {int(x.horizon_h)}h | {int(x.cells)} | {r(x.mae_mean)} | {r(x.rmse_mean)} | {r(x.smape_mean)}% | {r(x.mase_mean)} | {r(x.mae_improvement_vs_naive24_pct,2)}% |\n"
block+=f"""
On all 30 cells, LightGBM reduced MAE by **{lg.loc[1,'mae_improvement_vs_naive24_pct']:.2f}% at +1h** and **{lg.loc[24,'mae_improvement_vs_naive24_pct']:.2f}% at +24h** versus the 24-hour seasonal-naive baseline.
"""
if tcn is not None:
    block+=f"The Temporal CNN did **not** beat LightGBM: TCN MAE was {tcn.loc[1,'mae_mean']:.2f} at +1h and {tcn.loc[24,'mae_mean']:.2f} at +24h, versus LightGBM {lg.loc[1,'mae_mean']:.2f} and {lg.loc[24,'mae_mean']:.2f}.\n"
block+=f"""
### Anomaly detection

Best test detector: **{best.detector}** — event precision **{best.precision:.3f}**, recall **{best.recall:.3f}**, F1 **{best.f1:.3f}**, mean detected-event delay **{best.detection_delay_h:.2f} h**.

{types.to_markdown(index=False)}

These labels are synthetic anomalies injected into held-out **real** traffic. They are not operator incident labels.

### Capacity proxy

Using the training-only 95th-percentile proxy, +24h LightGBM produced **{cs.iloc[0].precision:.3f} precision**, **{cs.iloc[0].recall:.3f} recall**, and **{cs.iloc[0].f1:.3f} F1** over {int(cs.iloc[0].n_predictions):,} predictions. Low recall is a real limitation of this capacity-warning formulation.

<!-- AUTO_RESULTS_END -->"""
start="<!-- AUTO_RESULTS_START -->"; end="<!-- AUTO_RESULTS_END -->"
text=text.split(start)[0]+block+text.split(end)[1]
p.write_text(text,encoding="utf-8")
print(block)
