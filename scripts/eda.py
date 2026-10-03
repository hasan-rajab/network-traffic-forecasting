from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml

def main(config_path):
    with open(config_path) as f: cfg=yaml.safe_load(f)
    df=pd.read_parquet(Path(cfg["data"]["processed_dir"])/"traffic.parquet"); Path("figures").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
    dist=df.groupby("cell_id").internet.agg(["count","mean","std","min","median","max"]); dist.to_csv("results/cell_distribution.csv")
    hour=df.assign(hour=df.timestamp.dt.hour).groupby("hour").internet.agg(["mean","std"]); hour.to_csv("results/hourly_seasonality.csv")
    week=df.assign(dow=df.timestamp.dt.dayofweek).groupby("dow").internet.agg(["mean","std"]); week.to_csv("results/weekly_seasonality.csv")
    daily=df.set_index("timestamp").groupby("cell_id").internet.resample("1D").sum().reset_index(); daily.to_csv("results/daily_cell_activity.csv",index=False)
    events=[]
    for e in cfg["holiday_events"]:
        d=pd.Timestamp(e["date"],tz=cfg["data"]["timezone"]); v=daily[daily.timestamp==d].internet
        events.append({"date":e["date"],"event":e["name"],"scope":e["scope"],"cells_observed":len(v),"mean_daily_internet":float(v.mean()),"median_daily_internet":float(v.median())})
    pd.DataFrame(events).to_csv("results/event_days.csv",index=False)
    plots=[
        ("daily_activity.png",daily.groupby("timestamp").internet.mean(),"Mean daily internet activity — selected cells","Internet activity"),
        ("hourly_seasonality.png",hour["mean"],"Hourly seasonality","Mean internet activity"),
        ("weekly_seasonality.png",week["mean"],"Weekly seasonality (0=Mon)","Mean internet activity"),
    ]
    for fn,s,title,ylab in plots:
        fig,ax=plt.subplots(figsize=(10,4)); s.plot(ax=ax,marker="o" if len(s)<30 else None); ax.set_title(title); ax.set_ylabel(ylab); fig.tight_layout(); fig.savefig("figures/"+fn,dpi=160); plt.close(fig)
    m=pd.read_csv("results/missingness.csv"); fig,ax=plt.subplots(figsize=(9,4)); ax.bar(m.cell_id.astype(str),m.missing_pct_before_fill); ax.set_title("Missing 10-minute intervals by cell"); ax.set_ylabel("Missing (%)"); ax.tick_params(axis="x",rotation=90,labelsize=6); fig.tight_layout(); fig.savefig("figures/missingness.png",dpi=160); plt.close(fig)
    order=sorted(df.cell_id.unique()); data=[df[df.cell_id==c].internet.values for c in order]; fig,ax=plt.subplots(figsize=(12,5)); ax.boxplot(data,showfliers=False); ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order,rotation=90,fontsize=6); ax.set_title("Hourly internet-activity distributions by cell"); fig.tight_layout(); fig.savefig("figures/cell_distributions.png",dpi=160); plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args(); main(a.config)
