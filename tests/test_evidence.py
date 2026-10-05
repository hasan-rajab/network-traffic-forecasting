import pandas as pd
import pytest

from src.evidence import anomaly_count_report, paired_forecast_report


def predictions():
    rows = []
    for fold in [1, 2]:
        for h in [1, 24]:
            for t in pd.date_range(f'2020-01-{fold*3:02d}', periods=48, freq='h', tz='UTC'):
                for cell in [1, 2]:
                    for model, value in [('lightgbm_global', 99), ('seasonal_naive_24', 98)]:
                        rows.append(dict(model=model, fold=fold, horizon_h=h, cell_id=cell,
                                         origin=t, target_time=t+pd.Timedelta(hours=h), actual=100, prediction=value))
    return pd.DataFrame(rows)


def test_bootstrap_retains_cells_together_and_compares_both_horizons():
    result = paired_forecast_report(predictions(), resamples=200)
    assert [r['horizon_h'] for r in result] == [1, 24]
    for r in result:
        assert r['paired_predictions'] == 192 and r['time_blocks'] == 4
        assert r['mae_reduction_pct'] == 50 and r['mae_difference_ci95'] == [1., 1.]
    assert result == paired_forecast_report(predictions(), resamples=200)


@pytest.mark.parametrize('case', ['missing_baseline', 'duplicate', 'different_actual', 'bad_horizon'])
def test_forecast_pairing_rejects_misleading_comparisons(case):
    p = predictions()
    if case == 'missing_baseline': p = p.drop(p[p.model=='seasonal_naive_24'].index[0])
    if case == 'duplicate': p = pd.concat([p, p.iloc[:1]])
    if case == 'different_actual': p.loc[p[p.model=='seasonal_naive_24'].index[0], 'actual'] = 101
    if case == 'bad_horizon': p['target_time'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError): paired_forecast_report(p, resamples=200)


def test_anomaly_report_displays_sample_size_and_exact_confusion_counts():
    frame = pd.DataFrame([dict(detector='example', breakdown='global', tp_events=2, fp_events=1,
                               fn_events=1, true_events=3, precision=2/3, recall=2/3, f1=2/3)])
    r = anomaly_count_report(frame)[0]
    assert r['true_events'] == 3 and r['predicted_events'] == 3
    frame.loc[0, 'true_events'] = 9
    with pytest.raises(ValueError): anomaly_count_report(frame)
