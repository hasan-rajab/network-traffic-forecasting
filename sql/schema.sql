PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS dim_cell (
    cell_id INTEGER PRIMARY KEY,
    traffic_band TEXT NOT NULL CHECK (traffic_band IN ('low', 'medium', 'high')),
    first_window_internet_total REAL NOT NULL CHECK (first_window_internet_total >= 0)
);
CREATE TABLE IF NOT EXISTS fact_hourly (
    cell_id INTEGER NOT NULL REFERENCES dim_cell(cell_id),
    timestamp_s INTEGER NOT NULL,
    internet REAL CHECK (internet >= 0),
    PRIMARY KEY (cell_id, timestamp_s)
);
CREATE TABLE IF NOT EXISTS run_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
