import json
from pathlib import Path
from typing import Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.config import (
    CLEANED_PARQUET_PATH,
    STATION_HOTSPOTS_CSV_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR
)

def compute_station_hotspots(spark: SparkSession, df: DataFrame) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Computes station-level demand statistics:
    - Total pickups
    - Latitude and Longitude
    - Average hourly demand
    - Average trip duration
    - Demand Classification (High, Medium, Low based on 80th and 30th percentiles)
    """
    print("[Hotspot Analysis] Aggregating station-level statistics...")

    # Distinct days in dataset
    num_days = df.select("pickup_date").distinct().count()
    total_system_hours = max(num_days * 24.0, 1.0)

    station_agg = df.groupBy("start_station_id", "start_station_name") \
        .agg(
            F.count("ride_id").alias("total_pickups"),
            F.round(F.avg("start_lat"), 6).alias("latitude"),
            F.round(F.avg("start_lng"), 6).alias("longitude"),
            F.round(F.count("ride_id") / total_system_hours, 2).alias("avg_hourly_demand"),
            F.round(F.avg("trip_duration_min"), 2).alias("avg_trip_duration_min")
        )

    # Compute percentiles for reproducible classification:
    # High Demand: Top 20% (>= 80th percentile)
    # Medium Demand: 30th to 80th percentile
    # Low Demand: Bottom 30% (< 30th percentile)
    quantiles = station_agg.approxQuantile("total_pickups", [0.30, 0.80], 0.001)
    p30, p80 = quantiles[0], quantiles[1]
    print(f"[Hotspot Analysis] Demand thresholds: Low < {p30:.0f} pickups, Medium: {p30:.0f}-{p80:.0f}, High >= {p80:.0f}")

    station_classified = station_agg.withColumn(
        "demand_classification",
        F.when(F.col("total_pickups") >= p80, "High Demand")
         .when(F.col("total_pickups") >= p30, "Medium Demand")
         .otherwise("Low Demand")
    ).orderBy(F.col("total_pickups").desc())

    total_stations = station_classified.count()
    print(f"[Hotspot Analysis] Successfully classified {total_stations:,} unique pickup stations.")

    # Top 10, Top 20, and Bottom 10
    top_10 = [row.asDict() for row in station_classified.limit(10).collect()]
    top_20 = [row.asDict() for row in station_classified.limit(20).collect()]
    bottom_10 = [row.asDict() for row in station_classified.orderBy(F.col("total_pickups").asc()).limit(10).collect()]

    class_counts = [
        row.asDict() for row in station_classified.groupBy("demand_classification").count().collect()
    ]

    summary = {
        "total_unique_stations": total_stations,
        "total_system_days": num_days,
        "classification_thresholds": {
            "p30_low_medium_boundary": p30,
            "p80_medium_high_boundary": p80
        },
        "classification_breakdown": class_counts,
        "top_10_stations": top_10,
        "top_20_stations": top_20,
        "bottom_10_stations": bottom_10
    }

    return station_classified, summary

def save_hotspot_data(station_df: DataFrame) -> None:
    """
    Saves station hotspots to both CSV (for dashboard/GIS) and Parquet.
    """
    csv_path = str(STATION_HOTSPOTS_CSV_PATH)
    parquet_path = str(PROCESSED_DATA_DIR / "station_hotspots.parquet")

    # Export single CSV for dashboard mapping
    pandas_df = station_df.toPandas()
    pandas_df.to_csv(csv_path, index=False)
    print(f"[Hotspot Storage] Saved station hotspots CSV to {csv_path}")

    station_df.write.mode("overwrite").parquet(parquet_path)
    print(f"[Hotspot Storage] Saved station hotspots Parquet to {parquet_path}")

def execute_hotspot_analysis(spark: SparkSession) -> Dict[str, Any]:
    """
    Runs station hotspot analysis and writes output tables.
    """
    print(f"[Hotspot Analysis] Reading cleaned data from {CLEANED_PARQUET_PATH}...")
    df = spark.read.parquet(str(CLEANED_PARQUET_PATH))
    station_df, summary = compute_station_hotspots(spark, df)
    save_hotspot_data(station_df)

    out_file = REPORTS_DIR / "hotspot_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"[Hotspot Analysis] Saved hotspot summary report to {out_file}")

    return summary

if __name__ == "__main__":
    from src.spark_session import get_spark_session
    spark = get_spark_session("Hotspot_Analysis_Module")
    execute_hotspot_analysis(spark)
    spark.stop()
