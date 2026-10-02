from __future__ import annotations
import numpy as np

def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def smape(y,p):
    y=np.asarray(y);p=np.asarray(p); d=np.abs(y)+np.abs(p); return float(np.mean(np.where(d==0,0,2*np.abs(p-y)/d))*100)
def mase(y,p,train,season=24):
    train=np.asarray(train); denom=np.mean(np.abs(train[season:]-train[:-season])); return float(mae(y,p)/denom) if denom else float('nan')
