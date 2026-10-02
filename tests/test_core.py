import numpy as np, pandas as pd
from src.features import make_features
from src.forecast import rolling_folds
from src.anomaly import inject_anomalies
from src.evaluate import mae,rmse,smape

def test_features_no_current_target_leakage():
    t=pd.date_range('2020-01-01',periods=200,freq='h',tz='UTC'); d=pd.DataFrame({'cell_id':1,'timestamp':t,'internet':np.arange(200.)})
    x=make_features(d)
    assert x.loc[10,'lag_1']==9
    assert x.loc[30,'roll_mean_24'] < x.loc[30,'internet']

def test_split_ordering():
    t=pd.date_range('2020-01-01',periods=24*40,freq='h')
    for tr,te in rolling_folds(t,4,24*20,24*2): assert max(tr)<min(te)

def test_injection_reproducible():
    x=np.arange(100.,dtype=float); a=inject_anomalies(x,7); b=inject_anomalies(x,7)
    assert np.allclose(a[0],b[0]); assert np.array_equal(a[1],b[1]); assert a[2]==b[2]

def test_metrics_toy():
    y=np.array([1.,2.,3.]); p=np.array([1.,4.,2.])
    assert mae(y,p)==1.0
    assert round(rmse(y,p),6)==round(np.sqrt(5/3),6)
    assert smape(y,y)==0.0
