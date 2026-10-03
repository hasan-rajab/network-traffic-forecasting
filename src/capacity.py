from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def run_capacity(config_path="config.yaml"):
    with open(config_path) as f: cfg=yaml.safe_load(f)
    root=Path("results"); preds=pd.read_parquet(root/"forecast_predictions.parquet")
    p=preds[(preds.model=="lightgbm_global")&(preds.horizon_h==24)].copy()
    traffic=pd.read_parquet(Path(cfg["data"]["processed_dir"])/"traffic.parquet")
    rows=[]; frames=[]
    for fold,g in p.groupby("fold"):
        hist=traffic[traffic.timestamp<g.origin.min()]
        caps=hist.groupby("cell_id").internet.quantile(cfg["capacity"]["percentile"]).to_dict()
        x=g.copy(); x["capacity_proxy"]=x.cell_id.map(caps); x["predicted_breach"]=x.prediction>x.capacity_proxy; x["actual_breach"]=x.actual>x.capacity_proxy; frames.append(x)
        for cid,c in x.groupby("cell_id"):
            tp=int((c.predicted_breach&c.actual_breach).sum()); fp=int((c.predicted_breach&~c.actual_breach).sum()); fn=int((~c.predicted_breach&c.actual_breach).sum()); tn=int((~c.predicted_breach&~c.actual_breach).sum())
            pr=tp/(tp+fp) if tp+fp else np.nan; rc=tp/(tp+fn) if tp+fn else np.nan; f1=2*pr*rc/(pr+rc) if np.isfinite(pr) and np.isfinite(rc) and pr+rc else np.nan
            rows.append({"fold":int(fold),"cell_id":int(cid),"n":len(c),"capacity_proxy":float(c.capacity_proxy.iloc[0]),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":pr,"recall":rc,"f1":f1,"actual_breaches":int(c.actual_breach.sum()),"predicted_breaches":int(c.predicted_breach.sum())})
    detail=pd.DataFrame(rows); allp=pd.concat(frames,ignore_index=True); detail.to_csv(root/"capacity_metrics.csv",index=False); allp.to_parquet(root/"capacity_predictions.parquet",index=False)
    tp,fp,fn,tn=detail[["tp","fp","fn","tn"]].sum(); pr=tp/(tp+fp); rc=tp/(tp+fn); f1=2*pr*rc/(pr+rc)
    pd.DataFrame([{"capacity_percentile":cfg["capacity"]["percentile"],"n_predictions":len(allp),"actual_breaches":int(allp.actual_breach.sum()),"predicted_breaches":int(allp.predicted_breach.sum()),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn),"precision":pr,"recall":rc,"f1":f1}]).to_csv(root/"capacity_summary.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args(); run_capacity(a.config)
if __name__=="__main__": main()
