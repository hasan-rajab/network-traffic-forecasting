# SQL analytics and defensible forecasting evidence

## Decision question

Estimate short-horizon demand, inspect unusual traffic and prioritize cells for investigation. Compare forecasting accuracy at each horizon before relying on a model for planning. The capacity threshold is a training-period percentile proxy; the project does not measure engineered capacity or financial savings.

## Data scope and SQL

The full public source has country-code-level, 10-minute activity records. `src.data.aggregate_selected_day` filters the 30 previously selected cells, then calls `src.sql_analytics.aggregate_selected_chunks`. SQLite `GROUP BY cell_id, timestamp_ms` sums selected country-code rows **across chunk boundaries**. Python/pandas handles streaming, cell selection, missingness and time-grid resampling.

Keep these quantities separate in `data_quality.json`:

| Field | Meaning |
|---|---|
| `raw_rows_scanned` | Original rows across the configured daily files, before cell filtering |
| `selected_source_rows` | Country-code rows retained for the selected cells, before aggregation |
| `ten_min_rows` | Selected-cell grid after aggregation/reindexing/filling |
| `hourly_rows` | Selected hourly modelling dataset |
| `selected_cells`, `source_days` | Spatial and temporal modelling scope |

The first-week stratification pass reads some files again; it does not increase the reported unique source-row total. A scan of 200M raw records is not a claim that the model trains on 200M examples. Historical runs made before the SQL change used pandas aggregation; do not retrospectively describe them as SQL runs.

## Warehouse and marts

After processing, run `make analytics`. This refreshes `data/processed/analytics.sqlite`, then exports:

- `mart_daily_load.csv`: activity, coverage and missing hours by UTC date and first-week traffic stratum.
- `mart_cell_trends.csv`: prior-24-hour mean, observed-hour count, hourly change and deviation from historical load.
- `sql_quality_checks.json`: foreign-key coverage, uniqueness, valid strata and nonnegative activity checks.

The SQL models demonstrate CTEs, aggregations, inner/left joins and window functions. `RANGE` uses elapsed seconds, so a gap is not silently treated as the previous hour. Rolling statistics exclude the current/future observation. Missing load remains `NULL`, distinct from zero activity. The warehouse contains the selected subset, not all source records. These are explicit SQL models and data contracts, not a claim of dbt, Looker or GCP experience.

The Streamlit dashboard includes warehouse marts, source-to-subset scope, per-horizon paired uncertainty and event confusion counts. CSV exports can be opened in BI tools without requiring a proprietary service.

## Forecast uncertainty

Run `make evidence` after the forecasting/anomaly outputs exist. The report pairs model and seasonal-naive forecasts on the exact horizon, fold, cell, origin and target timestamp. Duplicate keys, missing baseline coverage, different actuals and invalid horizons fail rather than quietly altering the comparison.

Within each fold, bootstrap whole 24-hour blocks, retaining all cells together. Export point-weighted MAEs, paired observation counts, block counts, the 95% interval for baseline MAE minus model MAE, and separate +1h/+24h reductions. A small +24h point gain is not evidence of a comparable long-horizon advantage; an interval containing zero should be described as inconclusive for this window.

This is a conditional uncertainty analysis of a short historical evaluation window. Few blocks, shared training periods and longer serial dependence limit inference; it is not an A/B experiment, a production guarantee or proof of business savings.

## Anomaly denominators

`evidence_report.json` and `cv_metrics.csv` retain true events, predicted events, TP, FP and FN. The report checks that precision/recall/F1 agree with those counts. Thresholds remain selected on fold 3 and evaluated on fold 4. The exporter names Isolation Forest explicitly rather than choosing the best detector using test F1.

The default configuration requests **24 injected events** (4 types × 3 magnitudes × 2 replicates), one per distinct cell. Insufficient cell coverage now fails instead of silently truncating the experiment. This is a configured sample size, not an independently recovered denominator for an old rounded CV score. Cite the actual generated run report for measured denominators.

No original measured prediction/metric CSVs are committed in the current repository. The new code does not reconstruct historical counts from rounded 0.667/0.833 scores. To reproduce those results, obtain the original run artifacts or rerun `make all`; preserve the configuration, environment versions, data quality, forecast metrics/predictions, threshold CSV, injected-event CSV and evidence JSON together.

## Validation and interview discussion

Tests cover cross-chunk SQL aggregation, scope counts, null semantics, time gaps, prior-only windows, idempotent warehouse refresh, invalid inputs, exact forecast pairing, block structure and event-count consistency. Existing feature-leakage and purge tests remain active.

Defensible implementation wording: “Implemented SQL aggregation and a tested SQLite analytics warehouse with CTEs, joins and time-based window functions; added paired forecast uncertainty and explicit anomaly-event denominators.” Combine this with independently evidenced historical record/model metrics only after identifying the relevant run.
