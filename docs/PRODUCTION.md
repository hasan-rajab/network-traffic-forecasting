# Telecom dashboard deployment

Status: deployment configuration prepared; no live deployment has been verified.

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

## Data required before launch

Mount the real pipeline outputs at `/app/artifacts`:

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

The historical processed data and prediction files are not in this repository.
They must be recovered from the original run or regenerated with `make all`.
Do not replace them with synthetic data while retaining the historical
200,908,858-record claim or MAE metrics.

## Verification and rollback

After launch, verify the health route, open the dashboard, select multiple
cells and inspect the forecast comparisons, SQL marts and event denominators.
Record the deployed commit and artifact provenance. Roll back the image and
matching data artifacts together; do not mix evidence from different runs.
