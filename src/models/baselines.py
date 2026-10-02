import numpy as np

def seasonal_naive(series, horizon=1, lag=24):
    if len(series) < lag+horizon: return np.full(horizon,np.nan)
    base=np.asarray(series)[-lag:]
    return np.resize(base,horizon)
