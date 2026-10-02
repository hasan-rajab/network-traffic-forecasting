from __future__ import annotations
import numpy as np, pandas as pd

def rolling_folds(times, n_folds=4, min_train=24*21, test_size=24*3):
    t=np.array(sorted(pd.Series(times).drop_duplicates()))
    need=min_train+n_folds*test_size
    if len(t)<need: raise ValueError(f"Need at least {need} timestamps, got {len(t)}")
    start=len(t)-n_folds*test_size
    out=[]
    for i in range(n_folds):
        cut=start+i*test_size; out.append((t[:cut],t[cut:cut+test_size]))
    return out
