import numpy as np
import pandas as pd

from src.anomaly import inject_events


def test_anomaly_injection_reproducibility():
    times = pd.date_range("2013-12-03", periods=72, freq="h", tz="Europe/Rome")
    rows = []
    for cid in range(1, 31):
        for i, t in enumerate(times):
            rows.append({"cell_id": cid, "target_time": t, "actual": 100 + i, "prediction": 100 + i})
    frame = pd.DataFrame(rows)
    hist = pd.DataFrame({"cell_id": np.repeat(np.arange(1, 31), 100),
                         "timestamp": list(pd.date_range("2013-11-01", periods=100, freq="h", tz="Europe/Rome")) * 30,
                         "internet": np.tile(np.linspace(80, 120, 100), 30)})
    cfg = {"anomaly": {"types": ["spike", "drop", "level_shift", "daily_pattern_shift"],
                       "magnitudes_std": [2, 3, 5], "replicates_per_type_magnitude": 2}}
    a, ea = inject_events(frame, hist, cfg, 42); b, eb = inject_events(frame, hist, cfg, 42)
    assert np.allclose(a.injected_actual, b.injected_actual)
    pd.testing.assert_frame_equal(ea, eb)
