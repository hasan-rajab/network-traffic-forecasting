from __future__ import annotations
import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    x=df.sort_values(["cell_id","timestamp"]).copy()
    g=x.groupby("cell_id",sort=False)
    for lag in (1,24,168):
        x[f"lag_{lag}"]=g["internet"].shift(lag)
    shifted=g["internet"].shift(1)
    for w in (24,168):
        x[f"roll_mean_{w}"]=shifted.groupby(x.cell_id).transform(lambda s:s.rolling(w,min_periods=w).mean())
        x[f"roll_std_{w}"]=shifted.groupby(x.cell_id).transform(lambda s:s.rolling(w,min_periods=w).std())
    x["hour"]=x.timestamp.dt.hour
    x["dow"]=x.timestamp.dt.dayofweek
    events={str(e["date"]) for e in config.get("holiday_events",[])}
    x["holiday"]=x.timestamp.dt.strftime("%Y-%m-%d").isin(events).astype("int8")
    return x
