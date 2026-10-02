import numpy as np

def capacity_proxy(train, percentile=.95): return float(np.nanquantile(train,percentile))
def breach_flags(pred, capacity): return np.asarray(pred)>capacity
