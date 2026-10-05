# Telecom dashboard deployment

Status: live on Railway, with HTTPS and startup artifact validation.

Dashboard: https://telecom-analytics-production.up.railway.app

The real-data regeneration run completed successfully:
https://github.com/hasan-rajab/network-traffic-forecasting/actions/runs/37289212554
Its derived serving bundle was committed at
`3c502e4ead42efb60ec6eeb7889a589bbc9f16bb`.
The data audit distinguishes 200,908,858 raw country-code rows scanned from
618,755 selected source rows and 27,360 hourly modelling rows across 30 cells
and 38 days. The raw scan count is not ML sample size.

The regenerated LightGBM one-hour MAE reduction is 21.68% against the
seasonal-naive baseline. The 24-hour point reduction is 4.19%, but its paired
24-hour block-bootstrap interval for the MAE difference includes zero.
Isolation Forest evaluation uses 24 injected held-out events: 20 true
positives, 16 false positives and 4 false negatives (event F1 0.667, recall
0.833). These are simulated labels on real traffic, not operator incidents.

The manual `bootstrap.yml` workflow now regenerates the full real-data pipeline,
validates its forecast and anomaly denominators, and commits only the derived
serving bundle on the deployment branch. It cannot publish to `main`. Run it
on `codex/data-analytics-evidence-2026-10-05`; the Docker image then serves the
validated files from `/app/deployment/artifacts`. The bundle records checksums,
the source commit, workflow run and ODbL data attribution. Raw records and model
binaries are excluded. This workflow does not assert historical CV percentages;
use the regenerated evidence report for actual results.

Deploy the `codex/data-analytics-evidence-2026-10-05` branch. Railway reads
`railway.json`, builds the root Dockerfile and checks `/_stcore/health`.
The image runs only the Streamlit serving dependencies. Training remains an
offline job using the original `requirements.txt` and `make all`.

## Bundled data and optional external artifacts

The default deployment already contains validated outputs under
`/app/deployment/artifacts`; it does not need a runtime data download or a
persistent volume. `deployment/artifacts/provenance.json` records the source
run and file checksums.

For a separate regenerated bundle, mount the real pipeline outputs at
`/app/artifacts`:

```
/app/artifacts/data/processed/traffic.parquet
/app/artifacts/data/processed/analytics.sqlite
/app/artifacts/results/data_quality.json
/app/artifacts/results/evidence_report.json
/app/artifacts/results/forecast_predictions.parquet
```

Include `anomaly_test_predictions.parquet` and `capacity_predictions.parquet`
under `results/` for the anomaly and capacity views. Set
`NETWORK_ARTIFACT_DIR=/app/artifacts`. The app reads the SQLite warehouse in
read-only mode. It uses Railway's `PORT`, defaulting to 8501 elsewhere.

Run `python scripts/serve.py` against these mounted outputs. Startup rejects
missing files, empty marts, inconsistent cell/row denominators and invalid
forecast predictions before the web process can become healthy.

Regenerate replacement artifacts with `make all` and package them using
`python scripts/package_artifacts.py`, or dispatch the guarded real-data
workflow on the deployment branch. Do not substitute synthetic traffic while
retaining the real-data record count or evaluation metrics.

## Verification and rollback

After launch, verify the health route, open the dashboard, select multiple
cells and inspect the forecast comparisons, SQL marts and event denominators.
The production-image workflow also runs `scripts/check_dashboard.py` inside
the exact serving image. It exercises the first and last cell, checks for
rendering errors and verifies that SQL trend rows follow the selected cell.
Record the deployed commit and artifact provenance. Roll back the image and
matching data artifacts together; do not mix evidence from different runs.
