# Network Traffic Forecasting and Anomaly Detection

Portfolio project framed as telecom capacity planning using the **Telecom Italia Milan Big Data Challenge** telecommunications activity dataset.

## Dataset
Source: Harvard Dataverse DOI `10.7910/DVN/EGZHFV`, *Telecommunications - SMS, Call, Internet - MI* (Telecom Italia, 2015). The release contains 62 daily files from 2013-11-01 through 2014-01-01. Raw rows are tab-separated: square/cell id, Unix epoch milliseconds, country code, SMS-in, SMS-out, call-in, call-out, internet activity. Country-code rows are aggregated per cell/timestamp before modeling.

License: **Open Database License (ODbL) 1.0**. Raw data is not committed to Git.

## Reproducibility
- Fixed seed: 42
- Pinned dependencies in `requirements.txt`
- One entry point: `make all`
- Real data only. No synthetic substitute is used for forecasting. Synthetic anomalies are injected only into held-out real observations for detector evaluation and are explicitly labeled as such.

## Data placement
Put the real Harvard files in `data/raw/` with original names such as `sms-call-internet-mi-2013-11-01.txt`.

## Methodology
The loader aggregates across country codes, selects 30 cells reproducibly across low/medium/high traffic strata, resamples hourly, measures missingness, and forward-fills only short gaps. Forecast validation uses expanding-window rolling origin splits. Planned models are seasonal naive (24h/168h), ETS on a smaller cell subset, one global LightGBM model, and one global PyTorch LSTM. Anomaly evaluation uses fixed-seed injected spike, drop/outage, level-shift, and daily-pattern-shift events. Capacity is a proxy equal to each cell's training-period 95th percentile, **not real operator capacity**.

## Limitations
This is a portfolio analysis of historical activity weights, not operator traffic in Mbps/Gbps. There are no ground-truth anomaly labels, so quantitative anomaly metrics rely on synthetic events injected into held-out real data. The capacity threshold is an analytical proxy and must not be interpreted as engineered network capacity.

## Future work
Federated or decentralized training with cells treated as nodes; probabilistic forecast intervals; graph-based spatial models; calibration against real RAN/core capacity counters.

<!-- AUTO_RESULTS_START -->
## Results

Results are generated only after real-data execution.
<!-- AUTO_RESULTS_END -->
