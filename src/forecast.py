from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from src.features import make_features
from src.evaluate import metric_bundle, mase_scale

def rolling_origin_windows(times, folds=4, test_hours=72):
    t=np.array(sorted(pd.Series(times).drop_duplicates()))
    need=folds*test_hours
    if len(t)<need: raise ValueError(f"Need at least {need} timestamps")
    start=len(t)-need
    return [t[start+i*test_hours:start+(i+1)*test_hours] for i in range(folds)]

def run_forecasting(config_path="config.yaml"):
    with open(config_path) as f: cfg=yaml.safe_load(f)
    root=Path("results"); root.mkdir(exist_ok=True)
    df=pd.read_parquet(Path(cfg["data"]["processed_dir"])/"traffic.parquet")
    df=make_features(df,cfg)
    rows=[]; preds=[]; fold_rows=[]
    feature_base=["lag_1","lag_24","lag_168","roll_mean_24","roll_std_24","roll_mean_168","roll_std_168","hour","dow","holiday","cell_id"]
    for h in cfg["forecast"]["horizons"]:
        w=df.copy(); g=w.groupby("cell_id",sort=False)
        w["target"]=g.internet.shift(-h)
        w["target_time"]=w.timestamp+pd.Timedelta(hours=h)
        w["naive24"]=g.internet.shift(24-h); w["naive168"]=g.internet.shift(168-h)
        w=w.dropna(subset=feature_base+["target","naive24","naive168"]).copy()
        windows=rolling_origin_windows(w.timestamp,cfg["forecast"]["folds"],cfg["forecast"]["test_hours_per_fold"])
        for fold,ts in enumerate(windows,1):
            test_start=pd.Timestamp(ts[0]); test=w[w.timestamp.isin(ts)].copy()
            train=w[w.target_time<test_start].copy()
            stats=train.groupby("cell_id").internet.agg(["mean","std"]).rename(columns={"mean":"cell_train_mean","std":"cell_train_std"})
            train=train.join(stats,on="cell_id"); test=test.join(stats,on="cell_id")
            feats=feature_base+["cell_train_mean","cell_train_std"]
            hist=df[df.timestamp<test_start]
            scales={int(cid):mase_scale(s.internet.to_numpy(),24) for cid,s in hist.groupby("cell_id")}
            for model,col in [("seasonal_naive_24","naive24"),("seasonal_naive_168","naive168")]:
                for cid,tg in test.groupby("cell_id"):
                    m=metric_bundle(tg.target,tg[col],scales[int(cid)])
                    rows.append({"model":model,"horizon_h":h,"fold":fold,"cell_id":int(cid),"n":len(tg),**m})
                    for _,r in tg.iterrows():
                        preds.append({"model":model,"horizon_h":h,"fold":fold,"cell_id":int(cid),"origin":r.timestamp,"target_time":r.target_time,"actual":r.target,"prediction":r[col]})
            p=cfg["forecast"]["lightgbm"]
            mdl=LGBMRegressor(n_estimators=p["n_estimators"],learning_rate=p["learning_rate"],num_leaves=p["num_leaves"],subsample=p["subsample"],colsample_bytree=p["colsample_bytree"],random_state=cfg["seed"],n_jobs=2,verbosity=-1)
            mdl.fit(train[feats],train.target); test["prediction"]=mdl.predict(test[feats])
            for cid,tg in test.groupby("cell_id"):
                m=metric_bundle(tg.target,tg.prediction,scales[int(cid)])
                rows.append({"model":"lightgbm_global","horizon_h":h,"fold":fold,"cell_id":int(cid),"n":len(tg),**m})
                for _,r in tg.iterrows():
                    preds.append({"model":"lightgbm_global","horizon_h":h,"fold":fold,"cell_id":int(cid),"origin":r.timestamp,"target_time":r.target_time,"actual":r.target,"prediction":r.prediction})
            fold_rows.append({"horizon_h":h,"fold":fold,"train_rows":len(train),"test_rows":len(test),"train_target_end":str(train.target_time.max()),"test_start":str(test.timestamp.min()),"test_end":str(test.timestamp.max())})
    fm=pd.DataFrame(rows); fp=pd.DataFrame(preds)
    fm.to_csv(root/"forecast_metrics.csv",index=False); fp.to_parquet(root/"forecast_predictions.parquet",index=False)
    pd.DataFrame(fold_rows).to_csv(root/"forecast_folds.csv",index=False)
    agg=fm.groupby(["model","horizon_h"]).agg(cells=("cell_id","nunique"),cell_folds=("mae","size"),mae_mean=("mae","mean"),mae_std=("mae","std"),rmse_mean=("rmse","mean"),rmse_std=("rmse","std"),smape_mean=("smape","mean"),smape_std=("smape","std"),mase_mean=("mase","mean"),mase_std=("mase","std")).reset_index()
    n=agg[agg.model=="seasonal_naive_24"][["horizon_h","mae_mean"]].rename(columns={"mae_mean":"naive24_mae"})
    agg=agg.merge(n,on="horizon_h",how="left"); agg["mae_improvement_vs_naive24_pct"]=100*(agg.naive24_mae-agg.mae_mean)/agg.naive24_mae
    agg.to_csv(root/"forecast_summary.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args(); run_forecasting(a.config)
if __name__=="__main__": main()
