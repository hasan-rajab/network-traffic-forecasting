from __future__ import annotations
import argparse, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.features import make_features
from src.evaluate import metric_bundle, mase_scale

def rolling_origin_windows(times, folds=4, test_hours=72):
    t=np.array(sorted(pd.Series(times).drop_duplicates()))
    need=folds*test_hours
    if len(t)<need: raise ValueError(f"Need at least {need} timestamps")
    start=len(t)-need
    return [t[start+i*test_hours:start+(i+1)*test_hours] for i in range(folds)]

def _append_metrics(rows,preds,model,h,fold,tg,pred_col,scale):
    m=metric_bundle(tg.target,tg[pred_col],scale)
    rows.append({"model":model,"horizon_h":h,"fold":fold,"cell_id":int(tg.cell_id.iloc[0]),"n":len(tg),**m})
    for _,r in tg.iterrows():
        preds.append({"model":model,"horizon_h":h,"fold":fold,"cell_id":int(r.cell_id),"origin":r.timestamp,"target_time":r.target_time,"actual":r.target,"prediction":r[pred_col]})

class TemporalCNN(nn.Module):
    def __init__(self,ncells,channels=12,emb=6):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(6,channels,3,padding=1),nn.ReLU(),nn.Conv1d(channels,channels,3,padding=2,dilation=2),nn.ReLU())
        self.emb=nn.Embedding(ncells,emb); self.head=nn.Sequential(nn.Linear(channels+emb,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x,c):
        z=self.conv(x.transpose(1,2))[:,:,-1]
        return self.head(torch.cat([z,self.emb(c)],1)).squeeze(1)

def _tcn_arrays(df,cells,h,test_times,test_start,seq,events):
    c2i={int(c):i for i,c in enumerate(cells)}
    Xtr=[];Ctr=[];Ytr=[];Xte=[];Cte=[];Yte=[];meta=[]
    test_set=set(pd.to_datetime(test_times))
    for cid in cells:
        g=df[df.cell_id==cid].sort_values("timestamp").reset_index(drop=True); y=g.internet.to_numpy(float); times=g.timestamp
        mask=times<test_start; mu=float(y[mask].mean()); sd=float(y[mask].std()); sd=sd if sd>1e-8 else 1.0
        hist=y[mask]; scale=mase_scale(hist,24)
        hr=times.dt.hour.to_numpy(); dw=times.dt.dayofweek.to_numpy(); hol=times.dt.strftime("%Y-%m-%d").isin(events).astype(float).to_numpy()
        f=np.c_[(y-mu)/sd,np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),np.sin(2*np.pi*dw/7),np.cos(2*np.pi*dw/7),hol].astype("float32")
        for i in range(seq-1,len(g)-h):
            origin=times.iloc[i]; target_time=times.iloc[i+h]; window=f[i-seq+1:i+1]
            if target_time<test_start:
                Xtr.append(window); Ctr.append(c2i[int(cid)]); Ytr.append((y[i+h]-mu)/sd)
            elif origin in test_set:
                Xte.append(window); Cte.append(c2i[int(cid)]); Yte.append(y[i+h]); meta.append((int(cid),origin,target_time,scale,mu,sd))
    return np.stack(Xtr),np.array(Ctr),np.array(Ytr,dtype="float32"),np.stack(Xte),np.array(Cte),np.array(Yte,dtype="float32"),meta

def run_forecasting(config_path="config.yaml"):
    with open(config_path) as f: cfg=yaml.safe_load(f)
    root=Path("results"); root.mkdir(exist_ok=True)
    raw=pd.read_parquet(Path(cfg["data"]["processed_dir"])/"traffic.parquet").sort_values(["cell_id","timestamp"]).reset_index(drop=True)
    df=make_features(raw,cfg)
    rows=[]; preds=[]; fold_rows=[]
    base=["lag_1","lag_24","lag_168","roll_mean_24","roll_std_24","roll_mean_168","roll_std_168","hour","dow","holiday","cell_id"]
    windows_by_h={}
    for h in cfg["forecast"]["horizons"]:
        w=df.copy(); g=w.groupby("cell_id",sort=False)
        w["target"]=g.internet.shift(-h); w["target_time"]=w.timestamp+pd.Timedelta(hours=h)
        w["naive24"]=g.internet.shift(24-h); w["naive168"]=g.internet.shift(168-h)
        w=w.dropna(subset=base+["target","naive24","naive168"]).copy()
        windows=rolling_origin_windows(w.timestamp,cfg["forecast"]["folds"],cfg["forecast"]["test_hours_per_fold"]); windows_by_h[h]=windows
        for fold,ts in enumerate(windows,1):
            test_start=pd.Timestamp(ts[0]); test=w[w.timestamp.isin(ts)].copy(); train=w[w.target_time<test_start].copy()
            st=train.groupby("cell_id").internet.agg(["mean","std"]).rename(columns={"mean":"cell_train_mean","std":"cell_train_std"})
            train=train.join(st,on="cell_id"); test=test.join(st,on="cell_id"); feats=base+["cell_train_mean","cell_train_std"]
            hist=raw[raw.timestamp<test_start]; scales={int(cid):mase_scale(s.internet.to_numpy(),24) for cid,s in hist.groupby("cell_id")}
            for model,col in [("seasonal_naive_24","naive24"),("seasonal_naive_168","naive168")]:
                for cid,tg in test.groupby("cell_id"): _append_metrics(rows,preds,model,h,fold,tg,col,scales[int(cid)])
            p=cfg["forecast"]["lightgbm"]
            mdl=LGBMRegressor(n_estimators=p["n_estimators"],learning_rate=p["learning_rate"],num_leaves=p["num_leaves"],subsample=p["subsample"],colsample_bytree=p["colsample_bytree"],random_state=cfg["seed"],n_jobs=2,verbosity=-1)
            mdl.fit(train[feats],train.target); test["prediction"]=mdl.predict(test[feats])
            for cid,tg in test.groupby("cell_id"): _append_metrics(rows,preds,"lightgbm_global",h,fold,tg,"prediction",scales[int(cid)])
            fold_rows.append({"horizon_h":h,"fold":fold,"train_rows":len(train),"test_rows":len(test),"train_target_end":str(train.target_time.max()),"test_start":str(test.timestamp.min()),"test_end":str(test.timestamp.max())})

    # ETS: 2 cells per traffic stratum
    warnings.filterwarnings("ignore"); sel=pd.read_csv(root/"selected_cells.csv"); rng=np.random.default_rng(cfg["seed"]); ets_cells=[]
    for band in ["low","medium","high"]:
        ids=sel[sel.traffic_band==band].cell_id.to_numpy(); ets_cells.extend(map(int,np.sort(rng.choice(ids,size=cfg["forecast"]["ets_cells_per_band"],replace=False))))
    for h,windows in windows_by_h.items():
        for fold,ts in enumerate(windows,1):
            test_start=pd.Timestamp(ts[0]); test_end=pd.Timestamp(ts[-1])
            for cid in ets_cells:
                hist=raw[(raw.cell_id==cid)&(raw.timestamp<test_start)].set_index("timestamp").internet.astype(float)
                actual=raw[(raw.cell_id==cid)&(raw.timestamp>=test_start+pd.Timedelta(hours=h))&(raw.timestamp<=test_end+pd.Timedelta(hours=h))].sort_values("timestamp").internet.to_numpy()
                fit=ExponentialSmoothing(hist,trend="add",damped_trend=True,seasonal="add",seasonal_periods=24,initialization_method="estimated").fit(optimized=True,use_brute=False)
                fc=np.asarray(fit.forecast(len(ts)+h),float)[h:h+len(ts)]
                scale=mase_scale(hist.to_numpy(),24); m=metric_bundle(actual,fc,scale)
                rows.append({"model":"ets_additive_24","horizon_h":h,"fold":fold,"cell_id":cid,"n":len(actual),**m})
                for t,a,pv in zip(pd.date_range(test_start+pd.Timedelta(hours=h),periods=len(ts),freq="h"),actual,fc):
                    preds.append({"model":"ets_additive_24","horizon_h":h,"fold":fold,"cell_id":cid,"origin":t-pd.Timedelta(hours=h),"target_time":t,"actual":a,"prediction":pv})

    # Global Temporal CNN
    deep=cfg["forecast"]["deep"]; cells=sorted(raw.cell_id.unique()); events={str(e["date"]) for e in cfg.get("holiday_events",[])}
    alltimes=np.array(sorted(raw.timestamp.unique())); seq=int(deep["sequence_length"])
    for h in cfg["forecast"]["horizons"]:
        valid=alltimes[seq-1:len(alltimes)-h]; windows=rolling_origin_windows(valid,cfg["forecast"]["folds"],cfg["forecast"]["test_hours_per_fold"])
        for fold,ts in enumerate(windows,1):
            test_start=pd.Timestamp(ts[0]); Xtr,Ctr,Ytr,Xte,Cte,Yte,meta=_tcn_arrays(raw,cells,h,ts,test_start,seq,events)
            torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"]); torch.set_num_threads(2)
            model=TemporalCNN(len(cells),int(deep["channels"]),int(deep["cell_embedding_dim"])); opt=torch.optim.Adam(model.parameters(),lr=float(deep["learning_rate"])); lossfn=nn.MSELoss()
            loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(Ctr).long(),torch.from_numpy(Ytr)),batch_size=int(deep["batch_size"]),shuffle=True,generator=torch.Generator().manual_seed(cfg["seed"]))
            model.train()
            for _ in range(int(deep["epochs"])):
                for xb,cb,yb in loader:
                    opt.zero_grad(); loss=lossfn(model(xb,cb),yb); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad(): pn=model(torch.from_numpy(Xte),torch.from_numpy(Cte).long()).numpy()
            pp=np.array([pn[i]*m[5]+m[4] for i,m in enumerate(meta)])
            temp=pd.DataFrame([{"cell_id":m[0],"timestamp":m[1],"target_time":m[2],"scale":m[3],"target":float(Yte[i]),"prediction":float(pp[i])} for i,m in enumerate(meta)])
            for cid,tg in temp.groupby("cell_id"):
                mm=metric_bundle(tg.target,tg.prediction,float(tg.scale.iloc[0])); rows.append({"model":"temporal_cnn_global","horizon_h":h,"fold":fold,"cell_id":int(cid),"n":len(tg),**mm})
                for _,r in tg.iterrows(): preds.append({"model":"temporal_cnn_global","horizon_h":h,"fold":fold,"cell_id":int(cid),"origin":r.timestamp,"target_time":r.target_time,"actual":r.target,"prediction":r.prediction})

    fm=pd.DataFrame(rows); fp=pd.DataFrame(preds)
    fm.to_csv(root/"forecast_metrics.csv",index=False); fp.to_parquet(root/"forecast_predictions.parquet",index=False); pd.DataFrame(fold_rows).to_csv(root/"forecast_folds.csv",index=False)
    agg=fm.groupby(["model","horizon_h"]).agg(cells=("cell_id","nunique"),cell_folds=("mae","size"),mae_mean=("mae","mean"),mae_std=("mae","std"),rmse_mean=("rmse","mean"),rmse_std=("rmse","std"),smape_mean=("smape","mean"),smape_std=("smape","std"),mase_mean=("mase","mean"),mase_std=("mase","std")).reset_index()
    n=agg[agg.model=="seasonal_naive_24"][["horizon_h","mae_mean"]].rename(columns={"mae_mean":"naive24_mae"}); agg=agg.merge(n,on="horizon_h",how="left")
    agg["mae_improvement_vs_naive24_pct"]=100*(agg.naive24_mae-agg.mae_mean)/agg.naive24_mae; agg.to_csv(root/"forecast_summary.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args(); run_forecasting(a.config)
if __name__=="__main__": main()
