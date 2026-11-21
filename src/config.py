import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
REPORTS_DIR = PROJECT_ROOT / "reports"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
LOGS_DIR = PROJECT_ROOT / "logs"

# Target Data Paths
CLEANED_PARQUET_PATH = PROCESSED_DATA_DIR / "cleaned_trips.parquet"
STATION_HOURLY_PARQUET_PATH = PROCESSED_DATA_DIR / "station_hourly_demand.parquet"
STATION_HOTSPOTS_CSV_PATH = PROCESSED_DATA_DIR / "station_hotspots.csv"
MODEL_METRICS_PATH = REPORTS_DIR / "model_evaluation_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "gbt_feature_importance.json"

# Data Cleaning Thresholds
MIN_TRIP_DURATION_MIN = 1.0       # Ignore accidental unlocks under 1 minute
MAX_TRIP_DURATION_MIN = 1440.0    # 24 hours (filter out unreturned/stolen bikes)

# Geographic Bounding Box for NYC Metropolitan Area
NYC_BOUNDS = {
    "min_lat": 40.50,
    "max_lat": 41.05,
    "min_lng": -74.25,
    "max_lng": -73.65
}

# Machine Learning Chronological Split Boundaries
# Full cleaned dataset spans 2026-07-31 to 2026-08-31
# July 31 contains incomplete data (1,423 trips across NYC vs 202,163 on Aug 1).
# August 01 serves as the 24-hour warm-up window to initialize valid lag_1h and lag_24h features.
TRAIN_START_DATE = "2026-08-02" # Train: Aug 02 - Aug 23 (22 days, 79,200 rows)
TRAIN_END_DATE = "2026-08-23"   # Train: Aug 02 - Aug 23 (22 days, 79,200 rows)
VAL_END_DATE = "2026-08-27"     # Validation: Aug 24 - Aug 27 (4 days, 14,400 rows)
# Test: Aug 28 - Aug 31 (4 days, 14,400 rows)

# Machine Learning Station Selection Configuration
TOP_N_STATIONS = 150

def get_top_n_stations() -> int:
    """
    Returns and validates the configured number of top transit stations for ML modeling.
    Must be a positive integer.
    """
    global TOP_N_STATIONS
    if not isinstance(TOP_N_STATIONS, int) or TOP_N_STATIONS <= 0:
        raise ValueError(
            f"Invalid configuration: TOP_N_STATIONS must be a positive integer, got: {TOP_N_STATIONS}"
        )
    return TOP_N_STATIONS
