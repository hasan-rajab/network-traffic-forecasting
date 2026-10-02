import pandas as pd

def make_features(df: pd.DataFrame, target="internet", lags=(1,24,168), rolls=(24,168)):
    x=df.sort_values(["cell_id","timestamp"]).copy()
    grp=x.groupby("cell_id",group_keys=False)
    for lag in lags: x[f"lag_{lag}"]=grp[target].shift(lag)
    for w in rolls:
        shifted=grp[target].shift(1)
        x[f"roll_mean_{w}"]=shifted.groupby(x["cell_id"]).rolling(w,min_periods=max(2,w//4)).mean().reset_index(level=0,drop=True)
        x[f"roll_std_{w}"]=shifted.groupby(x["cell_id"]).rolling(w,min_periods=max(2,w//4)).std().reset_index(level=0,drop=True)
    x["hour"]=x.timestamp.dt.hour; x["dow"]=x.timestamp.dt.dayofweek
    return x
