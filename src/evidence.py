"""Evidence from run artifacts: paired forecast uncertainty and anomaly counts.

No historical CV percentage or rounded F1 is used to reconstruct observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['horizon_h', 'fold', 'cell_id', 'origin', 'target_time']


def paired_forecast_report(predictions, model='lightgbm_global', baseline='seasonal_naive_24',
                           seed=42, resamples=2000, block_hours=24):
    """Pair the same observations, then resample time blocks within each fold.

    Each block holds ALL cells together. The conditional interval describes this
    short evaluation window; it does not measure unseen markets or future drift.
    """
    if resamples < 100 or block_hours < 1:
        raise ValueError('Use at least 100 resamples and positive block hours')
    required = set(KEY + ['model', 'actual', 'prediction'])
    if not required.issubset(predictions):
        raise ValueError(f'Missing forecast fields: {sorted(required-set(predictions))}')
    p = predictions.copy()
    for col in ['origin', 'target_time']:
        p[col] = pd.to_datetime(p[col], utc=True, errors='raise')
    rows = []
    for horizon in sorted(p.loc[p.model == model, 'horizon_h'].unique()):
        m = p[(p.model == model) & (p.horizon_h == horizon)]
        b = p[(p.model == baseline) & (p.horizon_h == horizon)]
        if m.duplicated(KEY).any() or b.duplicated(KEY).any():
            raise ValueError('Duplicate forecast keys; pairing would multiply rows')
        paired = m.merge(b, on=KEY, how='outer', suffixes=('_model', '_baseline'),
                         indicator=True, validate='one_to_one')
        if not paired['_merge'].eq('both').all():
            raise ValueError('Model and baseline coverage differ; compare the same observations')
        num = paired[['actual_model', 'actual_baseline', 'prediction_model', 'prediction_baseline']].to_numpy(float)
        if not np.isfinite(num).all() or not np.allclose(num[:, 0], num[:, 1], rtol=0, atol=1e-10):
            raise ValueError('Non-finite forecasts or mismatched target values')
        if not (paired.target_time > paired.origin).all():
            raise ValueError('Targets must be strictly after origins')
        if not ((paired.target_time-paired.origin).dt.total_seconds() == int(horizon)*3600).all():
            raise ValueError('Forecast horizon does not match target timestamps')
        paired['model_error'] = np.abs(num[:, 0] - num[:, 2])
        paired['baseline_error'] = np.abs(num[:, 0] - num[:, 3])
        anchor = paired.groupby('fold').target_time.transform('min')
        paired['block'] = ((paired.target_time-anchor).dt.total_seconds() // (block_hours*3600)).astype(int)
        blocks = paired.groupby(['fold', 'block']).agg(
            model_error=('model_error', 'sum'), baseline_error=('baseline_error', 'sum'),
            n=('model_error', 'size')).reset_index()
        rng = np.random.default_rng(seed)
        model_sums = np.zeros(resamples); base_sums = np.zeros(resamples); sizes = np.zeros(resamples)
        for _, group in blocks.groupby('fold'):
            values = group[['model_error', 'baseline_error', 'n']].to_numpy(float)
            sample = values[rng.integers(0, len(values), size=(resamples, len(values)))].sum(axis=1)
            model_sums += sample[:, 0]; base_sums += sample[:, 1]; sizes += sample[:, 2]
        differences = (base_sums-model_sums)/sizes
        lo, hi = np.quantile(differences, [.025, .975])
        model_mae = float(paired.model_error.mean()); baseline_mae = float(paired.baseline_error.mean())
        rows.append({'model': model, 'baseline': baseline, 'horizon_h': int(horizon),
                     'paired_predictions': len(paired), 'cells': int(paired.cell_id.nunique()),
                     'folds': int(paired.fold.nunique()), 'time_blocks': len(blocks),
                     'block_hours': block_hours, 'bootstrap_resamples': resamples, 'seed': seed,
                     'model_mae': model_mae, 'baseline_mae': baseline_mae,
                     'mae_reduction_pct': 100*(baseline_mae-model_mae)/baseline_mae if baseline_mae else None,
                     'mae_difference': baseline_mae-model_mae,
                     'mae_difference_ci95': [float(lo), float(hi)],
                     'interval_excludes_zero_in_model_favour': bool(lo > 0)})
    if not rows:
        raise ValueError(f'No predictions for {model}')
    return rows


def anomaly_count_report(metrics):
    results = []
    global_rows = metrics[metrics.breakdown == 'global']
    if global_rows.detector.duplicated().any() or global_rows.empty:
        raise ValueError('Need one global anomaly row per detector')
    for r in global_rows.to_dict('records'):
        counts = {}
        for key in ['tp_events', 'fp_events', 'fn_events', 'true_events']:
            value = float(r[key])
            if not np.isfinite(value) or value < 0 or value != int(value):
                raise ValueError(f'Invalid event count: {key}')
            counts[key] = int(value)
        tp, fp, fn = counts['tp_events'], counts['fp_events'], counts['fn_events']
        if counts['true_events'] != tp+fn:
            raise ValueError('True-event count must equal TP + FN')
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        for key, expected in [('precision', precision), ('recall', recall), ('f1', f1)]:
            if not np.isclose(float(r[key]), expected, atol=.00051, rtol=0):
                raise ValueError(f'{key} disagrees with event counts')
        results.append({'detector': r['detector'], **counts, 'predicted_events': tp+fp,
                        'precision': precision, 'recall': recall, 'f1': f1,
                        'label_source': 'synthetic perturbations in held-out real traffic',
                        'threshold_selection': 'validation fold; test labels not used for threshold tuning'})
    return results


def write_evidence(root='results', config_path='config.yaml'):
    import yaml
    root = Path(root)
    config = yaml.safe_load(Path(config_path).read_text())
    quality = json.loads((root/'data_quality.json').read_text())
    forecasts = paired_forecast_report(pd.read_parquet(root/'forecast_predictions.parquet'))
    anomalies = anomaly_count_report(pd.read_csv(root/'anomaly_metrics.csv'))
    scope = {key: quality.get(key) for key in ['raw_rows_scanned', 'selected_source_rows',
             'selected_cells', 'source_days', 'ten_min_rows', 'hourly_rows', 'start', 'end']}
    result = {'scope': scope, 'forecast_comparisons': forecasts, 'anomaly_event_counts': anomalies,
              'notes': ['Raw records scanned are distinct from selected country-code rows and hourly ML rows.',
                        'Paired intervals resample whole 24-hour blocks within folds and retain cross-cell dependence.',
                        'Four folds over a short window give conditional uncertainty, not production significance.',
                        'Report each horizon separately; a small +24h gain is not a large long-horizon advantage.',
                        'Capacity uses a training-period percentile proxy, not actual operator limits.'],
              'configured_anomaly_events': len(config['anomaly']['types'])*len(config['anomaly']['magnitudes_std'])*config['anomaly']['replicates_per_type_magnitude']}
    (root/'evidence_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results'); parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args(); print(json.dumps(write_evidence(args.results, args.config), indent=2))


if __name__ == '__main__':
    main()
