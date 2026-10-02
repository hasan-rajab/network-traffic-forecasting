from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

COLS=["cell_id","timestamp_ms","country_code","sms_in","sms_out","call_in","call_out","internet"]
NUM=["sms_in","sms_out","call_in","call_out","internet"]

def load_config(path="config.yaml"):
    with open(path) as f: return yaml.safe_load(f)

def read_daily(path: str|Path) -> pd.DataFrame:
    df=pd.read_csv(path,sep="\t",header=None,names=COLS,dtype={"cell_id":"int32","timestamp_ms":"int64","country_code":"Int32"})
    for c in NUM: df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0.0)
    return df.groupby(["cell_id","timestamp_ms"],as_index=False)[NUM].sum()

def choose_cells(raw_files, n_cells=30, seed=42):
    totals={}
    for p in raw_files[:min(7,len(raw_files))]:
        d=read_daily(p).groupby("cell_id")["internet"].sum()
        for k,v in d.items(): totals[k]=totals.get(k,0.0)+float(v)
    s=pd.Series(totals,name="traffic").sort_index()
    q=pd.qcut(s.rank(method="first"),3,labels=["low","medium","high"])
    rng=np.random.default_rng(seed); chosen=[]
    base=n_cells//3
    for band in ["low","medium","high"]:
        ids=s.index[q==band].to_numpy(); take=base+(1 if len(chosen)<n_cells%3 else 0)
        chosen.extend(rng.choice(ids,size=min(take,len(ids)),replace=False).tolist())
    return sorted(map(int,chosen))

def build_processed(config_path="config.yaml"):
    cfg=load_config(config_path); raw=sorted(Path(cfg["data"]["raw_dir"]).glob("sms-call-internet-mi-*.txt"))
    if not raw: raise FileNotFoundError("No real Telecom Italia raw files found under data/raw/")
    cells=choose_cells(raw,cfg["data"]["n_cells"],cfg["seed"])
    frames=[]
    for p in raw:
        d=read_daily(p); frames.append(d[d.cell_id.isin(cells)])
    x=pd.concat(frames,ignore_index=True)
    x["timestamp"]=pd.to_datetime(x.timestamp_ms,unit="ms",utc=True).dt.tz_convert(cfg["data"]["timezone"])
    out=[]; missing=[]
    for cid,g in x.groupby("cell_id"):
        g=g.set_index("timestamp")[NUM].resample(cfg["data"]["resample_freq"]).sum(min_count=1)
        miss=float(g["internet"].isna().mean())
        g[NUM]=g[NUM].ffill(limit=int(cfg["data"]["short_gap_limit"]))
        g["cell_id"]=cid; missing.append({"cell_id":cid,"missing_pct_before_fill":miss*100}); out.append(g.reset_index())
    df=pd.concat(out,ignore_index=True).sort_values(["cell_id","timestamp"])
    Path(cfg["data"]["processed_dir"]).mkdir(parents=True,exist_ok=True)
    df.to_parquet(Path(cfg["data"]["processed_dir"])/"traffic.parquet",index=False)
    pd.DataFrame(missing).to_csv("results/missingness.csv",index=False)
    pd.DataFrame({"cell_id":cells}).to_csv("results/selected_cells.csv",index=False)
    print(json.dumps({"raw_files":len(raw),"rows":len(df),"cells":len(cells),"start":str(df.timestamp.min()),"end":str(df.timestamp.max())},indent=2))
    return df

if __name__=="__main__": build_processed()
