from __future__ import annotations
import numpy as np, pandas as pd

def robust_z(residual, window=168):
    s=pd.Series(residual); med=s.rolling(window,min_periods=max(10,window//4)).median(); mad=(s-med).abs().rolling(window,min_periods=max(10,window//4)).median(); return 0.6745*(s-med)/(mad.replace(0,np.nan))

def inject_anomalies(values, seed=42):
    rng=np.random.default_rng(seed); x=np.asarray(values,float).copy(); labels=np.zeros(len(x),dtype=int)
    if len(x)<50: return x,labels,[]
    std=np.nanstd(x); events=[]
    specs=[("spike",5,3), ("drop",15,3), ("level_shift",25,12), ("daily_pattern_shift",40,min(24,len(x)-40))]
    for typ,start,length in specs:
        if start+length>len(x): continue
        mag=float(rng.choice([2,3,5])); labels[start:start+length]=1
        if typ=="spike": x[start]+=mag*std
        elif typ=="drop": x[start:start+length]=np.maximum(0,x[start:start+length]-mag*std)
        elif typ=="level_shift": x[start:start+length]+=mag*std
        else: x[start:start+length]=x[start:start+length][::-1]
        events.append({"type":typ,"start":start,"length":length,"magnitude":mag})
    return x,labels,events
