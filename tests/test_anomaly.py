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



def test_event_metrics_keep_sample_size_and_false_alerts_on_unlabelled_cells():
    from src.anomaly import event_metrics
    times = pd.date_range('2026-01-01', periods=5, freq='h', tz='UTC')
    frame = pd.DataFrame({'cell_id': [1]*5+[2]*5,
                          'target_time': list(times)*2,
                          'flag': [False,True,True,False,True, True,False,False,False,False]})
    events = pd.DataFrame([{'event_id':'one','cell_id':1,'start':times[1],'end':times[2]}])
    result = event_metrics(frame, events, 'flag')
    assert (result['true_events'],result['tp_events'],result['fp_events'],result['fn_events']) == (1,1,2,0)
    assert result['recall'] == 1 and result['f1'] == .5


def test_one_truth_event_is_not_counted_twice_when_alerts_fragment():
    from src.anomaly import event_metrics
    times = pd.date_range('2026-01-01', periods=5, freq='h', tz='UTC')
    frame = pd.DataFrame({'cell_id':1,'target_time':times,'flag':[True,False,True,False,False]})
    events = pd.DataFrame([{'event_id':'one','cell_id':1,'start':times[0],'end':times[3]}])
    result = event_metrics(frame, events, 'flag')
    assert result['tp_events']==1 and result['fp_events']==1 and result['fn_events']==0
