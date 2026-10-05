from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest

DUR={"spike":1,"drop":3,"level_shift":6,"daily_pattern_shift":12}

def inject_events(frame: pd.DataFrame, history: pd.DataFrame, config: dict, seed: int):
    out=frame.copy(); out["injected_actual"]=out.actual.astype(float); out["event_id"]=""
    ac=config["anomaly"]; combos=[(t,m) for t in ac["types"] for m in ac["magnitudes_std"] for _ in range(ac["replicates_per_type_magnitude"])]
    rng=np.random.default_rng(seed); rng.shuffle(combos)
    cells=np.array(sorted(out.cell_id.unique()))
    if len(cells) < len(combos):
        raise ValueError(f"Need {len(combos)} cells for one injected event per cell; got {len(cells)}")
    rng.shuffle(cells); chosen=cells[:len(combos)]
    scale=history.groupby("cell_id").internet.std().to_dict(); events=[]
    for eid,(cid,(typ,mag)) in enumerate(zip(chosen,combos),1):
        sub=out[out.cell_id==cid].sort_values("target_time"); dur=DUR[typ]; start=int(rng.integers(4,len(sub)-dur-3)); idx=sub.index[start:start+dur]
        vals=out.loc[idx,"injected_actual"].to_numpy(float); s=float(scale[int(cid)])
        if typ=="spike": vals[0]+=mag*s
        elif typ=="drop": vals=np.maximum(0,vals-mag*s)
        elif typ=="level_shift": vals=vals+mag*s
        else:
            phase=np.linspace(0,2*np.pi,dur,endpoint=False)+np.pi/2; vals=np.maximum(0,vals+mag*s*np.sin(phase))
        out.loc[idx,"injected_actual"]=vals; out.loc[idx,"event_id"]=f"E{eid:02d}"
        tt=out.loc[idx,"target_time"].sort_values()
        events.append({"event_id":f"E{eid:02d}","cell_id":int(cid),"type":typ,"magnitude_std":mag,"start":tt.iloc[0],"end":tt.iloc[-1],"duration_h":dur})
    out["residual_injected"]=out.injected_actual-out.prediction
    return out,pd.DataFrame(events)

def robust_scores(history,current,res_col,window=168):
    vals={}
    for cid,cur in current.groupby("cell_id"):
        hist=list(history[history.cell_id==cid].sort_values("target_time").residual.astype(float))
        for i,r in cur.sort_values("target_time").iterrows():
            w=np.asarray(hist[-window:]); med=np.median(w); mad=np.median(np.abs(w-med)); mad=mad if mad>1e-8 else 1e-8
            vals[i]=abs(0.6745*(float(r[res_col])-med)/mad); hist.append(float(r[res_col]))
    return pd.Series(vals).reindex(current.index)

def _pred_events(frame,flag):
    out=[]
    for cid,g in frame.sort_values(["cell_id","target_time"]).groupby("cell_id"):
        t=list(g[g[flag]].target_time)
        if not t: continue
        s=p=t[0]
        for q in t[1:]:
            if q-p>pd.Timedelta(hours=1): out.append((int(cid),s,p)); s=q
            p=q
        out.append((int(cid),s,p))
    return out

def event_metrics(frame,events,flag):
    pe=_pred_events(frame,flag); matched=set(); tp=fp=0; delays=[]
    for cid,s,e in pe:
        cand=events[(events.cell_id==cid)&(events.start<=e)&(events.end>=s)&(~events.event_id.isin(matched))]
        if len(cand):
            r=cand.iloc[0]; matched.add(r.event_id); tp+=1
            f=frame[(frame.cell_id==cid)&frame[flag]&(frame.target_time>=r.start)&(frame.target_time<=r.end)].target_time
            if len(f): delays.append((f.min()-r.start)/pd.Timedelta(hours=1))
        else: fp+=1
    fn=len(events)-tp; pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f1=2*pr*rc/(pr+rc) if pr+rc else 0
    return {"precision":pr,"recall":rc,"f1":f1,"detection_delay_h":float(np.mean(delays)) if delays else np.nan,"tp_events":tp,"fp_events":fp,"fn_events":fn,"true_events":len(events)}

def run_anomaly(config_path="config.yaml"):
    with open(config_path) as f: cfg=yaml.safe_load(f)
    root=Path("results"); pr=pd.read_parquet(root/"forecast_predictions.parquet")
    p=pr[(pr.model=="lightgbm_global")&(pr.horizon_h==1)].sort_values(["fold","cell_id","target_time"]).copy(); p["residual"]=p.actual-p.prediction
    train=p[p.fold<cfg["anomaly"]["validation_fold"]].copy(); val0=p[p.fold==cfg["anomaly"]["validation_fold"]].copy(); test0=p[p.fold==cfg["anomaly"]["test_fold"]].copy()
    traffic=pd.read_parquet(Path(cfg["data"]["processed_dir"])/"traffic.parquet")
    val,evv=inject_events(val0,traffic[traffic.timestamp<val0.target_time.min()],cfg,cfg["anomaly"]["injection_seed_validation"])
    test,evt=inject_events(test0,traffic[traffic.timestamp<test0.target_time.min()],cfg,cfg["anomaly"]["injection_seed_test"])
    val["rz_score"]=robust_scores(train,val,"residual_injected",cfg["anomaly"]["residual_window"])
    test_hist=pd.concat([train,val0],ignore_index=True); test["rz_score"]=robust_scores(test_hist,test,"residual_injected",cfg["anomaly"]["residual_window"])
    st=train.groupby("cell_id").residual.agg(med="median",mad=lambda s:np.median(np.abs(s-np.median(s)))).reset_index(); st["mad"]=st.mad.replace(0,1e-8)
    def feat(x,col):
        z=x.merge(st,on="cell_id",how="left"); q=0.6745*(z[col]-z.med)/z.mad; hr=z.target_time.dt.hour; dw=z.target_time.dt.dayofweek
        return np.c_[q,np.abs(q),np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),np.sin(2*np.pi*dw/7),np.cos(2*np.pi*dw/7)]
    iso=IsolationForest(n_estimators=200,random_state=cfg["seed"],n_jobs=2).fit(feat(train,"residual"))
    val["if_score"]=-iso.score_samples(feat(val,"residual_injected")); test["if_score"]=-iso.score_samples(feat(test,"residual_injected"))
    chosen={}; throws=[]
    for name,score,cands in [("robust_z","rz_score",np.linspace(1.5,8,27)),("isolation_forest","if_score",np.quantile(val.if_score,np.linspace(.6,.995,40)))]:
        best=None
        for th in np.unique(cands):
            val["_f"]=val[score]>=th; m=event_metrics(val,evv,"_f"); r={"detector":name,"threshold":float(th),**m}
            if best is None or (r["f1"],r["precision"])>(best["f1"],best["precision"]): best=r
        chosen[name]=best["threshold"]; throws.append(best)
    pd.DataFrame(throws).to_csv(root/"anomaly_thresholds.csv",index=False)
    rows=[]
    for name,score in [("robust_z","rz_score"),("isolation_forest","if_score")]:
        flag=f"{name}_flag"; test[flag]=test[score]>=chosen[name]; rows.append({"detector":name,"breakdown":"global","group":"all",**event_metrics(test,evt,flag)})
        for typ in cfg["anomaly"]["types"]:
            cells=evt[evt.type==typ].cell_id; sub=test[test.cell_id.isin(cells)]; rows.append({"detector":name,"breakdown":"type","group":typ,**event_metrics(sub,evt[evt.type==typ],flag)})
        for mag in cfg["anomaly"]["magnitudes_std"]:
            cells=evt[evt.magnitude_std==mag].cell_id; sub=test[test.cell_id.isin(cells)]; rows.append({"detector":name,"breakdown":"magnitude","group":str(mag),**event_metrics(sub,evt[evt.magnitude_std==mag],flag)})
    pd.DataFrame(rows).to_csv(root/"anomaly_metrics.csv",index=False); evt.to_csv(root/"injected_anomaly_events_test.csv",index=False); test.to_parquet(root/"anomaly_test_predictions.parquet",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); a=ap.parse_args(); run_anomaly(a.config)
if __name__=="__main__": main()

