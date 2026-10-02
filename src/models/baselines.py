from __future__ import annotations

import pandas as pd


def add_seasonal_naive_columns(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()
    g = x.groupby("cell_id", sort=False)
    x["seasonal_naive_24"] = g["internet"].shift(24 - horizon)
    x["seasonal_naive_168"] = g["internet"].shift(168 - horizon)
    return x
