from __future__ import annotations

from lightgbm import LGBMRegressor


def build_lightgbm(config: dict, seed: int) -> LGBMRegressor:
    p = config["forecast"]["lightgbm"]
    return LGBMRegressor(
        n_estimators=int(p["n_estimators"]),
        learning_rate=float(p["learning_rate"]),
        num_leaves=int(p["num_leaves"]),
        subsample=float(p["subsample"]),
        colsample_bytree=float(p["colsample_bytree"]),
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )
