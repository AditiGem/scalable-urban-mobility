import json
from pathlib import Path
from typing import Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.config import CLEANED_PARQUET_PATH, PROCESSED_DATA_DIR, REPORTS_DIR

def run_temporal_analysis(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive temporal analysis using Spark SQL and DataFrame aggregations.
    """
    df.createOrReplaceTempView("trips")

    print("\n--- A1. Hourly Trip Distribution ---")
    hourly_df = spark.sql("""
        SELECT 
            pickup_hour,
            COUNT(*) AS total_trips,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_share
        FROM trips
        GROUP BY pickup_hour
        ORDER BY pickup_hour
    """)
    hourly_data = [row.asDict() for row in hourly_df.collect()]

    print("--- A2. Weekday vs Weekend Comparison ---")
    weekday_weekend_df = spark.sql("""
        SELECT 
            CASE WHEN is_weekend = 1.0 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
            COUNT(*) AS total_trips,
            COUNT(DISTINCT pickup_date) AS distinct_days,
            ROUND(COUNT(*) / COUNT(DISTINCT pickup_date), 0) AS avg_daily_trips,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY is_weekend
    """)
    ww_data = [row.asDict() for row in weekday_weekend_df.collect()]

    print("--- A3. Day of Week Demand Distribution ---")
    dow_df = spark.sql("""
        SELECT 
            day_of_week,
            day_name,
            COUNT(*) AS total_trips,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY day_of_week, day_name
        ORDER BY day_of_week
    """)
    dow_data = [row.asDict() for row in dow_df.collect()]

    print("--- A4. Time of Day Diurnal Buckets ---")
    diurnal_df = spark.sql("""
        SELECT 
            CASE 
                WHEN pickup_hour BETWEEN 7 AND 9 THEN 'Morning Rush (07-09)'
                WHEN pickup_hour BETWEEN 10 AND 15 THEN 'Midday (10-15)'
                WHEN pickup_hour BETWEEN 16 AND 19 THEN 'Evening Rush (16-19)'
                ELSE 'Night / Early Morning (20-06)'
            END AS time_period,
            COUNT(*) AS total_trips,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_share,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY 1
        ORDER BY total_trips DESC
    """)
    diurnal_data = [row.asDict() for row in diurnal_df.collect()]

    print("--- A5. Daily Demand Trend ---")
    daily_df = spark.sql("""
        SELECT 
            pickup_date,
            day_name,
            COUNT(*) AS total_trips,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY pickup_date, day_name
        ORDER BY pickup_date
    """)
    daily_data = [row.asDict() for row in daily_df.collect()]

    return {
        "hourly": hourly_data,
        "weekday_weekend": ww_data,
        "day_of_week": dow_data,
        "diurnal_buckets": diurnal_data,
        "daily_trend": daily_data
    }

def run_rider_and_bike_analysis(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """
    Analyzes member vs. casual rider behavior and bike type utilization.
    """
    df.createOrReplaceTempView("trips")

    print("\n--- B1. Rider Category Breakdown ---")
    rider_df = spark.sql("""
        SELECT 
            member_casual,
            COUNT(*) AS total_trips,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_share,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY member_casual
    """)
    rider_data = [row.asDict() for row in rider_df.collect()]

    print("--- B2. Rider Hourly Demand Pattern ---")
    rider_hourly_df = spark.sql("""
        SELECT 
            pickup_hour,
            member_casual,
            COUNT(*) AS trip_count
        FROM trips
        GROUP BY pickup_hour, member_casual
        ORDER BY pickup_hour, member_casual
    """)
    rider_hourly_data = [row.asDict() for row in rider_hourly_df.collect()]

    print("--- B3. Bike Type Distribution & Durations ---")
    bike_df = spark.sql("""
        SELECT 
            rideable_type,
            COUNT(*) AS total_trips,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS pct_share,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY rideable_type
    """)
    bike_data = [row.asDict() for row in bike_df.collect()]

    print("--- B4. Rider Category vs Bike Type Cross-Preference ---")
    cross_df = spark.sql("""
        SELECT 
            member_casual,
            rideable_type,
            COUNT(*) AS trip_count,
            ROUND(AVG(trip_duration_min), 2) AS avg_duration_min
        FROM trips
        GROUP BY member_casual, rideable_type
        ORDER BY member_casual, rideable_type
    """)
    cross_data = [row.asDict() for row in cross_df.collect()]

    return {
        "rider_summary": rider_data,
        "rider_hourly": rider_hourly_data,
        "bike_summary": bike_data,
        "rider_bike_cross": cross_data
    }

def execute_demand_analysis(spark: SparkSession) -> Dict[str, Any]:
    """
    Loads cleaned parquet data and runs all analytical modules.
    Saves aggregated metrics to processed_data and reports.
    """
    print(f"[Demand Analysis] Reading cleaned data from {CLEANED_PARQUET_PATH}...")
    df = spark.read.parquet(str(CLEANED_PARQUET_PATH))
    
    temporal_results = run_temporal_analysis(spark, df)
    rider_bike_results = run_rider_and_bike_analysis(spark, df)

    full_analysis = {
        "temporal_analysis": temporal_results,
        "rider_and_bike_analysis": rider_bike_results
    }

    # Save to reports/analytics_summary.json for dashboard & report use
    out_file = REPORTS_DIR / "analytics_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        # Convert date objects to string for JSON serialization
        json.dump(full_analysis, f, indent=4, default=str)
    print(f"[Demand Analysis] Saved full analytical metrics to {out_file}")

    return full_analysis

if __name__ == "__main__":
    from src.spark_session import get_spark_session
    spark = get_spark_session("Demand_Analysis_Module")
    execute_demand_analysis(spark)
    spark.stop()
