import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import (
    CLEANED_PARQUET_PATH,
    MIN_TRIP_DURATION_MIN,
    MAX_TRIP_DURATION_MIN,
    NYC_BOUNDS,
    REPORTS_DIR
)
from src.logger import logger

def clean_and_transform_trips(df_raw: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Cleans raw Citi Bike trip records:
    1. Casts types (timestamps and numeric coordinates).
    2. Drops duplicate ride records.
    3. Filters invalid/missing start station IDs, names, and coordinates.
    4. Computes trip duration and filters short false unlocks (<1 min) and lost bikes (>24 hr).
    5. Filters coordinate outliers outside the NYC metropolitan area.
    6. Adds temporal derived features: pickup_date, pickup_hour, day_of_week, day_name, is_weekend, month, day_of_month.
    """
    raw_count = df_raw.count()
    logger.info(f"[Cleaning] Starting data cleaning on {raw_count:,} raw records...")

    t0 = time.time()

    # 1. Deduplicate by ride_id
    df_dedup = df_raw.dropDuplicates(["ride_id"])
    dedup_count = df_dedup.count()
    duplicates_removed = raw_count - dedup_count

    # 2. Type casting and timestamp conversion with millisecond safety
    df_typed = df_dedup \
        .withColumn("t_start", F.to_timestamp("started_at")) \
        .withColumn("t_end", F.to_timestamp("ended_at")) \
        .withColumn("start_lat_d", F.col("start_lat").cast("double")) \
        .withColumn("start_lng_d", F.col("start_lng").cast("double")) \
        .withColumn("end_lat_d", F.col("end_lat").cast("double")) \
        .withColumn("end_lng_d", F.col("end_lng").cast("double"))

    # 3. Calculate trip duration in minutes
    df_with_duration = df_typed.withColumn(
        "trip_duration_min",
        F.round((F.unix_timestamp("t_end") - F.unix_timestamp("t_start")) / 60.0, 2)
    )

    # 4. Construct validation filter
    valid_stations_condition = (
        F.col("start_station_name").isNotNull() &
        (F.col("start_station_name") != "") &
        F.col("start_station_id").isNotNull() &
        (F.col("start_station_id") != "")
    )

    valid_coords_condition = (
        F.col("start_lat_d").isNotNull() &
        F.col("start_lng_d").isNotNull() &
        (F.col("start_lat_d") >= NYC_BOUNDS["min_lat"]) &
        (F.col("start_lat_d") <= NYC_BOUNDS["max_lat"]) &
        (F.col("start_lng_d") >= NYC_BOUNDS["min_lng"]) &
        (F.col("start_lng_d") <= NYC_BOUNDS["max_lng"])
    )

    valid_duration_condition = (
        F.col("t_start").isNotNull() &
        F.col("t_end").isNotNull() &
        (F.col("trip_duration_min") >= MIN_TRIP_DURATION_MIN) &
        (F.col("trip_duration_min") <= MAX_TRIP_DURATION_MIN)
    )

    cleaning_filter = valid_stations_condition & valid_coords_condition & valid_duration_condition

    df_filtered = df_with_duration.filter(cleaning_filter)

    # 5. Temporal Feature Engineering
    df_clean = df_filtered \
        .withColumn("pickup_date", F.to_date("t_start")) \
        .withColumn("pickup_hour", F.hour("t_start")) \
        .withColumn("day_of_week", F.dayofweek("t_start")) \
        .withColumn("day_name", F.date_format("t_start", "E")) \
        .withColumn(
            "is_weekend",
            F.when((F.col("day_of_week") == 1) | (F.col("day_of_week") == 7), 1.0).otherwise(0.0)
        ) \
        .withColumn("month", F.month("t_start")) \
        .withColumn("day_of_month", F.dayofmonth("t_start")) \
        .select(
            "ride_id",
            "rideable_type",
            F.col("t_start").alias("started_at"),
            F.col("t_end").alias("ended_at"),
            "start_station_name",
            "start_station_id",
            "end_station_name",
            "end_station_id",
            F.col("start_lat_d").alias("start_lat"),
            F.col("start_lng_d").alias("start_lng"),
            F.col("end_lat_d").alias("end_lat"),
            F.col("end_lng_d").alias("end_lng"),
            "member_casual",
            "trip_duration_min",
            "pickup_date",
            "pickup_hour",
            "day_of_week",
            "day_name",
            "is_weekend",
            "month",
            "day_of_month"
        )

    clean_count = df_clean.count()
    removed_count = raw_count - clean_count
    retention_rate = (clean_count / raw_count) * 100 if raw_count > 0 else 0.0
    elapsed_time = round(time.time() - t0, 2)

    logger.info(f"[Cleaning] Raw records: {raw_count:,}")
    logger.info(f"[Cleaning] Cleaned records: {clean_count:,}")
    logger.info(f"[Cleaning] Removed records: {removed_count:,}")
    logger.info(f"[Cleaning] Data retention rate: {retention_rate:.2f}% (completed in {elapsed_time}s)")

    stats = {
        "raw_count": raw_count,
        "duplicates_removed": duplicates_removed,
        "cleaned_count": clean_count,
        "removed_count": removed_count,
        "retention_rate_pct": round(retention_rate, 2),
        "min_duration_filter_min": MIN_TRIP_DURATION_MIN,
        "max_duration_filter_min": MAX_TRIP_DURATION_MIN,
        "nyc_bounds": NYC_BOUNDS,
        "elapsed_seconds": elapsed_time
    }

    return df_clean, stats

def save_cleaned_data(df: DataFrame, target_path: Path = CLEANED_PARQUET_PATH) -> None:
    """
    Saves cleaned PySpark DataFrame to compressed Parquet format.
    """
    target_str = str(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Storage] Writing cleaned data to Parquet at {target_str}...")
    try:
        df.write.mode("overwrite").parquet(target_str)
        logger.info(f"[Storage] Successfully saved Parquet data to {target_str}.")
    except Exception as e:
        logger.error(f"[Storage] Failed to save Parquet data: {e}")
        raise

def run_etl_pipeline(spark) -> Dict[str, Any]:
    """
    Executes the complete ETL pipeline from ingestion to Parquet storage.
    """
    from src.data_ingestion import load_raw_dataset

    logger.info("=== STARTING PYSPARK ETL PIPELINE ===")
    t_start = time.time()

    df_raw, ingestion_stats = load_raw_dataset(spark)
    df_clean, cleaning_stats = clean_and_transform_trips(df_raw)
    save_cleaned_data(df_clean, CLEANED_PARQUET_PATH)

    total_elapsed = round(time.time() - t_start, 2)
    summary = {
        "ingestion": ingestion_stats,
        "cleaning": cleaning_stats,
        "total_pipeline_elapsed_seconds": total_elapsed
    }

    # Save summary audit report to JSON
    report_file = REPORTS_DIR / "etl_audit_report.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"[ETL Audit] Saved pipeline audit summary to {report_file}")
    logger.info(f"=== PYSPARK ETL PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed}s ===")

    return summary

if __name__ == "__main__":
    from src.spark_session import get_spark_session
    spark = get_spark_session("ETL_Pipeline_Runner")
    summary = run_etl_pipeline(spark)
    spark.stop()
