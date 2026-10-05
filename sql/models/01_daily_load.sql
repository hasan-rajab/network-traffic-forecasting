DROP VIEW IF EXISTS mart_daily_load;
CREATE VIEW mart_daily_load AS
WITH hourly AS (
    SELECT cell_id, date(timestamp_s, 'unixepoch') AS date_utc, internet
    FROM fact_hourly
)
SELECT h.date_utc, d.traffic_band, COUNT(DISTINCT h.cell_id) AS cells,
       COUNT(*) AS grid_hours, COUNT(h.internet) AS observed_hours,
       SUM(h.internet IS NULL) AS missing_hours,
       SUM(h.internet) AS total_activity, AVG(h.internet) AS mean_activity,
       MAX(h.internet) AS peak_activity,
       1.0 * COUNT(h.internet) / COUNT(*) AS coverage_rate
FROM hourly h
JOIN dim_cell d USING (cell_id)
GROUP BY h.date_utc, d.traffic_band;
