import glob
import os
from pathlib import Path
from typing import Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame

from src.config import DATASET_DIR
from src.logger import logger

EXPECTED_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual"
]

def load_raw_dataset(spark: SparkSession, dataset_dir: Path = DATASET_DIR) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Ingests all Citi Bike CSV files in the dataset directory using PySpark.
    Validates column presence and returns the raw DataFrame with ingestion statistics.
    Includes robust validation for missing directory, empty files, and schema mismatches.
    """
    if not dataset_dir.exists():
        msg = f"Dataset directory not found: {dataset_dir.resolve()}. Please ensure raw Citi Bike CSV files are placed in 'dataset/'."
        logger.error(msg)
        raise FileNotFoundError(msg)

    csv_pattern = str(dataset_dir / "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        msg = f"No CSV files found in {dataset_dir.resolve()}. Please place the 6 monthly Citi Bike CSV files in 'dataset/'."
        logger.error(msg)
        raise FileNotFoundError(msg)
    
    # Check for empty files
    empty_files = [f for f in csv_files if os.path.getsize(f) == 0]
    if empty_files:
        logger.warning(f"Found empty CSV files that will be ignored: {empty_files}")

    total_size_mb = sum(os.path.getsize(f) for f in csv_files) / (1024 * 1024)
    logger.info(f"[Ingestion] Found {len(csv_files)} CSV files ({total_size_mb:.2f} MB total). Loading into PySpark...")

    try:
        df_raw = spark.read.csv(csv_pattern, header=True, inferSchema=False)
    except Exception as e:
        logger.error(f"Failed to read CSV files via PySpark: {e}")
        raise

    # Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        msg = f"Dataset schema mismatch! Missing required columns: {missing_cols}. Found columns: {df_raw.columns}"
        logger.error(msg)
        raise ValueError(msg)

    raw_count = df_raw.count()
    if raw_count == 0:
        msg = "Ingested dataset contains 0 records! Verify that CSV files are not empty."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"[Ingestion] Successfully ingested {raw_count:,} raw records across {len(csv_files)} files.")

    stats = {
        "num_files": len(csv_files),
        "total_size_mb": round(total_size_mb, 2),
        "total_records": raw_count,
        "columns": df_raw.columns,
        "files": [os.path.basename(f) for f in csv_files]
    }
    
    return df_raw, stats

if __name__ == "__main__":
    from src.spark_session import get_spark_session
    spark = get_spark_session("Data_Ingestion_Test")
    df, stats = load_raw_dataset(spark)
    logger.info(f"Ingestion Test Successful: {stats['total_records']:,} rows loaded.")
    spark.stop()
