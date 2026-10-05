from __future__ import annotations
import argparse, json, urllib.parse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

COLS=["cell_id","timestamp_ms","country_code","sms_in","sms_out","call_in","call_out","internet"]
ACT=["sms_in","sms_out","call_in","call_out","internet"]

def load_config(path="config.yaml"):
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def date_range(start,end):
    a=date.fromisoformat(start); b=date.fromisoformat(end)
    while a<=b:
        yield a.isoformat(); a+=timedelta(days=1)

def mirror_url(day,cfg):
    fn=f"data/sms-call-internet-mi-{day}.txt"
    ref=cfg["source"]["transport_mirror"]
    return f"https://www.kaggle.com/api/v1/datasets/download/{ref}/"+urllib.parse.quote(fn,safe="")

def stream_chunks(day,cfg):
    r=requests.get(mirror_url(day,cfg),stream=True,timeout=90); r.raise_for_status(); r.raw.decode_content=True
    first=True
    for ch in pd.read_csv(r.raw,sep="\t",header=None,names=COLS,chunksize=int(cfg["data"]["chunksize"])):
        if first:
            if len(ch.columns)!=8: raise ValueError(f"Unexpected raw schema for {day}: {len(ch.columns)} columns")
            first=False
        yield ch

def day_totals(day,cfg):
    totals=defaultdict(float); n=0
    for ch in stream_chunks(day,cfg):
        n+=len(ch); z=ch.groupby("cell_id")["internet"].sum(min_count=1)
        for cid,v in z.items():
            if pd.notna(v): totals[int(cid)]+=float(v)
    return totals,n

def aggregate_selected_day(day,cfg,cells):
    from src.sql_analytics import aggregate_selected_chunks
    return aggregate_selected_chunks(stream_chunks(day,cfg), cells)

def select_cells(cfg):
    days=list(date_range(cfg["data"]["start_date"],cfg["data"]["end_date"]))[:int(cfg["data"]["stratification_days"])]
    totals=defaultdict(float)
    for d in days:
        z,_=day_totals(d,cfg)
        for cid,v in z.items(): totals[cid]+=v
    s=pd.Series(totals,name="first_window_internet_total").sort_index()
    bands=pd.qcut(s.rank(method="first"),3,labels=["low","medium","high"])
    rng=np.random.default_rng(int(cfg["seed"])); n=int(cfg["data"]["n_cells"]); per=n//3; chosen=[]; bandmap={}
    for band in ["low","medium","high"]:
        ids=s.index[bands==band].to_numpy(); picks=np.sort(rng.choice(ids,size=per,replace=False))
        chosen.extend(map(int,picks))
        for cid in picks: bandmap[int(cid)]=band
    chosen=sorted(chosen)
    return chosen,pd.DataFrame({"cell_id":chosen,"traffic_band":[bandmap[c] for c in chosen],"first_window_internet_total":[s[c] for c in chosen]})

def build_processed(config_path="config.yaml"):
    cfg=load_config(config_path); outdir=Path(cfg["data"]["processed_dir"]); outdir.mkdir(parents=True,exist_ok=True); Path("results").mkdir(exist_ok=True)
    cells,selection=select_cells(cfg); frames=[]; raw_rows=0; selected_source_rows=0; days=list(date_range(cfg["data"]["start_date"],cfg["data"]["end_date"]))
    for d in days:
        z,n=aggregate_selected_day(d,cfg,set(cells)); frames.append(z); raw_rows+=n; selected_source_rows+=z.attrs["selected_source_rows"]; print(f"{d}: scanned {n:,} raw rows")
    ten=pd.concat(frames,ignore_index=True).drop_duplicates(["cell_id","timestamp_ms"]).sort_values(["cell_id","timestamp_ms"])
    ten["timestamp"]=pd.to_datetime(ten.timestamp_ms,unit="ms",utc=True).dt.tz_convert(cfg["data"]["timezone"])
    filled=[]; miss=[]
    for cid,g in ten.groupby("cell_id"):
        g=g.set_index("timestamp")[ACT].sort_index()
        idx=pd.date_range(g.index.min(),g.index.max(),freq="10min",tz=cfg["data"]["timezone"])
        x=g.reindex(idx); before=int(x.internet.isna().sum())
        x[ACT]=x[ACT].ffill(limit=int(cfg["data"]["short_gap_limit_10min"]))
        after=int(x.internet.isna().sum()); x["cell_id"]=cid; x.index.name="timestamp"; filled.append(x.reset_index())
        miss.append({"cell_id":int(cid),"expected_10min_intervals":len(x),"missing_before":before,"missing_pct_before_fill":100*before/len(x),"missing_after_short_ffill":after})
    full10=pd.concat(filled,ignore_index=True)
    hourly=[]
    for cid,g in full10.groupby("cell_id"):
        h=g.set_index("timestamp")[ACT].resample(cfg["data"]["resample_freq"]).sum(min_count=1); h["cell_id"]=cid; hourly.append(h.reset_index())
    hourly=pd.concat(hourly,ignore_index=True).sort_values(["cell_id","timestamp"])
    full10.to_parquet(outdir/"traffic_10min_selected.parquet",index=False); hourly.to_parquet(outdir/"traffic.parquet",index=False)
    m=pd.DataFrame(miss); m.to_csv("results/missingness.csv",index=False); selection.to_csv("results/selected_cells.csv",index=False)
    summary={"source_days":len(days),"raw_rows_scanned":int(raw_rows),"selected_source_rows":int(selected_source_rows),"aggregation_engine":"SQLite GROUP BY across country-code rows","selected_cells":len(cells),"ten_min_rows":len(full10),"hourly_rows":len(hourly),"start":str(hourly.timestamp.min()),"end":str(hourly.timestamp.max()),"missing_pct_mean":float(m.missing_pct_before_fill.mean()),"missing_pct_max":float(m.missing_pct_before_fill.max()),"remaining_missing_after_fill":int(m.missing_after_short_ffill.sum())}
    Path("results/data_quality.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return hourly

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args(); build_processed(a.config)
if __name__=="__main__": main()

