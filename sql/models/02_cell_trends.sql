DROP VIEW IF EXISTS mart_cell_trends;
CREATE VIEW mart_cell_trends AS
WITH trends AS (
    SELECT cell_id, timestamp_s, internet,
        -- RANGE is elapsed seconds, so a missing hour is not a previous row.
        AVG(internet) OVER (PARTITION BY cell_id ORDER BY timestamp_s
             RANGE BETWEEN 86400 PRECEDING AND 3600 PRECEDING) AS prior_24h_mean,
        COUNT(internet) OVER (PARTITION BY cell_id ORDER BY timestamp_s
             RANGE BETWEEN 86400 PRECEDING AND 3600 PRECEDING) AS prior_24h_observed,
        LAG(internet) OVER (PARTITION BY cell_id ORDER BY timestamp_s) AS previous_row_load,
        LAG(timestamp_s) OVER (PARTITION BY cell_id ORDER BY timestamp_s) AS previous_timestamp_s
    FROM fact_hourly
)
SELECT t.cell_id, d.traffic_band, datetime(t.timestamp_s, 'unixepoch') AS timestamp_utc,
       t.internet, t.prior_24h_mean, t.prior_24h_observed,
       CASE WHEN t.timestamp_s - t.previous_timestamp_s = 3600
            THEN t.internet - t.previous_row_load END AS hourly_change,
       CASE WHEN t.prior_24h_mean > 0 THEN
            100.0 * (t.internet - t.prior_24h_mean) / t.prior_24h_mean END AS change_vs_prior_24h_pct
FROM trends t LEFT JOIN dim_cell d USING (cell_id);
