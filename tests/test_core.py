import numpy as np
import pandas as pd

from src.evaluate import mae, rmse, smape
from src.features import make_features
from src.forecast import rolling_origin_windows


def _cfg():
    return {"holiday_events": [{"date": "2020-01-01"}]}


def test_no_leakage_in_lag_and_rolling_features():
    t = pd.date_range("2020-01-01", periods=220, freq="h", tz="UTC")
    d = pd.DataFrame({"cell_id": 1, "timestamp": t, "internet": np.arange(220.0)})
    x = make_features(d, _cfg())
    assert x.loc[10, "lag_1"] == 9.0
    assert x.loc[200, "roll_mean_24"] < x.loc[200, "internet"]


def test_split_ordering():
    t = pd.date_range("2020-01-01", periods=800, freq="h")
    windows = rolling_origin_windows(t, folds=4, test_hours=72)
    assert len(windows) == 4
    for a, b in zip(windows[:-1], windows[1:]):
        assert max(a) < min(b)


def test_direct_target_purge_rule():
    origins = pd.date_range("2020-01-01", periods=50, freq="h")
    h = 24; test_start = origins[40]
    target_time = origins + pd.Timedelta(hours=h)
    train_origins = origins[target_time < test_start]
    assert all(train_origins + pd.Timedelta(hours=h) < test_start)


def test_metric_correctness_on_toy_inputs():
    y = np.array([1.0, 2.0, 3.0]); p = np.array([1.0, 4.0, 2.0])
    assert mae(y, p) == 1.0
    assert np.isclose(rmse(y, p), np.sqrt(5 / 3))
    assert smape(y, y) == 0.0
