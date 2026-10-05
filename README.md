# Network Traffic Forecasting & Anomaly Detection

A small, rigorous telecom ML portfolio project framed as **network capacity planning**: forecast per-cell load, detect unusual traffic behavior, and warn when forecast demand is above a capacity proxy.

## Problem and motivation

Operators need short-horizon demand estimates for planning, operations, and congestion risk. This project treats Telecom Italia's historical Milan activity weights as a network-load proxy and tests three questions:

1. How much can a global model improve on daily/weekly seasonal-naive forecasts?
2. Can forecast residuals detect injected outage/spike/pattern events in held-out real traffic?
3. Can +24h forecasts identify future load above a per-cell planning threshold?

## Dataset and license

Canonical dataset: **Telecommunications - SMS, Call, Internet - MI**, Telecom Italia Big Data Challenge, Harvard Dataverse DOI **10.7910/DVN/EGZHFV**. The source release is licensed under **ODbL 1.0**.

Original raw schema is tab-separated with eight fields: cell/square ID, Unix epoch milliseconds, country code, SMS-in, SMS-out, call-in, call-out, and internet activity. Rows are at 10-minute resolution and multiple country-code rows can occur for a cell/timestamp; the loader therefore sums across country codes before modeling.

### Transport note

During the measured run, Harvard Dataverse returned DNS/403 failures from the available execution environments. The project therefore fetched the original daily raw-file names through the public Kaggle mirror `dkgmgo/telecom-italia-milan`. A byte-range validation confirmed the expected 8-column, 10-minute, country-code-level structure before execution. The mirror is a transport path; Harvard Dataverse remains the canonical source and ODbL license reference.

The measured portfolio run uses **2013-11-01 through 2013-12-08 (38 days)** to keep CPU/runtime scope small while retaining weekly seasonality and the Dec 7–8 event window. Raw files are never committed to Git.

## How to run

```bash
make setup
make all
```

`make all` downloads/streams the real raw daily files, produces `data/processed/traffic.parquet`, runs EDA, forecasting, anomaly evaluation, capacity analysis, README result generation, and tests. No synthetic traffic dataset is used. Synthetic data appears **only as anomaly perturbations injected into held-out real observations** for detector evaluation.

To launch the dashboard after the pipeline:

```bash
make dashboard
```

## Methodology

### Data preparation

- Aggregate selected country-code rows per cell and 10-minute timestamp using SQLite SQL across streaming chunk boundaries.
- Use internet activity as the main target; retain SMS/call activity as optional columns.
- Select 30 cells with seed 42: 10 each from low, medium, and high first-week traffic strata.
- Reindex every selected cell to the expected 10-minute grid, report missingness, and forward-fill only gaps of at most two 10-minute intervals.
- Resample to hourly for the default experiment.

### SQL analytics engineering and statistical evidence

Selected raw country-code rows are now aggregated in **SQLite SQL**, with source scans, retained rows and modelling rows tracked separately. A tested SQLite warehouse exposes daily load/coverage by stratum and prior-only rolling trends using CTEs, joins and time-based window functions. Run `make analytics` after data preparation.

`make evidence` produces paired +1h/+24h forecast comparisons with 24-hour block-bootstrap intervals, exact coverage checks, and anomaly **true/predicted event counts plus TP/FP/FN**. `make all` runs both additions automatically. Counts are computed from run artifacts; historical CV percentages are not hard-coded. Read [the evidence guide](docs/ANALYTICS_EVIDENCE.md) for scope, methodology, reproducibility and limitations.

### EDA and event checks

The config explicitly records All Saints Day (Nov 1), Saint Ambrose Day in Milan (Dec 7), and Immaculate Conception (Dec 8). Real-date detector flags are reported only as qualitative evidence; the dataset has no incident labels.

### Forecasting

Four **expanding rolling-origin folds**, each with a 72-hour evaluation block. Direct +1h and +24h targets use a purge rule: every training target timestamp must be strictly earlier than the test fold start. Features use only information available at forecast time.

Models:

- 24h and 168h seasonal-naive baselines.
- Additive Holt-Winters/ETS with 24h seasonality on six stratified cells (runtime-scoped baseline).
- One global LightGBM across 30 cells using lags, rolling mean/std, hour, weekday, holiday flag, cell ID, and training-only cell statistics.
- One global PyTorch Temporal CNN using a 168-hour sequence, training-only per-cell normalization, calendar channels, and cell embeddings.

Metrics: MAE, RMSE, sMAPE, and MASE with a 24-hour seasonal scale. `results/forecast_metrics.csv` contains cell/fold-level metrics; `forecast_summary.csv` is the generated aggregate.

### Anomaly detection

No ground-truth anomalies exist. Fold 3 is used to inject validation anomalies and select thresholds; fold 4 is untouched until final test evaluation. Fixed-seed anomalies include spikes, drops/outages, level shifts, and daily-pattern shifts at 2σ, 3σ, and 5σ magnitudes.

Detectors:

- rolling robust residual z-score using median/MAD;
- Isolation Forest on normalized residuals plus cyclical time features.

The optional autoencoder is intentionally omitted to keep scope small; the PyTorch requirement is already exercised by the forecasting TCN.

### Capacity planning

Per-cell capacity is **assumed** to be the 95th percentile of load observed before each fold. This is not real operator capacity. The +24h LightGBM forecast is flagged when it exceeds that training-only proxy, then compared with actual proxy breaches.

<!-- AUTO_RESULTS_START -->
## Measured results

Run `make all` to regenerate this section strictly from `results/*.csv`.
<!-- AUTO_RESULTS_END -->

## Limitations

- Activity values are anonymized Telecom Italia activity weights, not Mbps/Gbps or RAN resource counters.
- The selected 30 cells are a deterministic portfolio subset, not the whole Milan grid.
- Anomaly F1 uses synthetic labels injected into real held-out observations; it does not measure real incident detection accuracy.
- Capacity is a percentile proxy, not engineered capacity. Low breach recall is reported rather than hidden.
- ETS is evaluated on only six cells and should not be ranked directly against full-cell model averages without that coverage caveat.
- The public raw mirror preserves the expected source schema, but the execution environment could not independently checksum it against Harvard's files because Harvard access was blocked.

## Future work

- Probabilistic forecasts and calibrated prediction intervals.
- Spatial graph features between neighboring cells.
- Federated training with cells or base-station regions treated as nodes.
- Real operator capacity counters and labeled incident/alarm streams.
- Drift monitoring and periodic model retraining.


## Production deployment

Deployment configuration and launch requirements are documented in [docs/PRODUCTION.md](docs/PRODUCTION.md). The deployment has not yet been verified live.
# Interactive public demo

[Explore Telecom analytics](https://telecom-analytics-production.up.railway.app) ·
[Portfolio](https://nexusmind-production-3da9.up.railway.app/portfolio)

Choose a network cell and +1h/+24h forecast, filter traffic segments, inspect SQL, download aggregate
data and adjust the hypothetical capacity threshold. The page distinguishes the 200M+ raw source
scans from the 27,360 hourly modelling rows and shows injected-event denominators and uncertainty.
Capacity thresholds are training-percentile proxies; scenarios do not establish real operator capacity.
