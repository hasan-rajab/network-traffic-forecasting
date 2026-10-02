from lightgbm import LGBMRegressor

def make_model(seed=42, **kwargs):
    return LGBMRegressor(random_state=seed,verbosity=-1,**kwargs)
