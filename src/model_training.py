import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
    RandomForestRegressionModel
)
from pyspark.ml.evaluation import RegressionEvaluator

from src.config import (
    CLEANED_PARQUET_PATH,
    PROCESSED_DATA_DIR,
    STATION_HOURLY_PARQUET_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    FEATURE_IMPORTANCE_PATH,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VAL_END_DATE,
    get_top_n_stations
)
from src.logger import logger

FEATURE_COLS = [
    "pickup_hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "latitude",
    "longitude",
    "lag_1h_demand",
    "lag_24h_demand"
]

def create_station_hour_demand_dataset(spark: SparkSession, top_n: int = None) -> DataFrame:
    """
    Builds a rigorous station-hourly demand dataset:
    1. Selects the top active stations using TRAINING-PERIOD data only (TRAIN_START_DATE to TRAIN_END_DATE).
       This strictly avoids lookahead data leakage in station selection.
       Groups by start_station_id (the unique physical dock identifier) to prevent station name ambiguity.
    2. Constructs a full hourly time grid across August (31 days * 24 hours).
       July 31 boundary data is excluded due to incomplete source coverage (1,423 trips across NYC vs 202,163 on Aug 01).
       August 01 is used as the 24-hour warm-up window to initialize valid lag_1h and lag_24h features.
    3. Performs a cross-join between stations and time grid to form the observation skeleton.
    4. Left joins actual trip counts and fills non-pickup hours with zero (demand = 0).
    5. Computes temporal, cyclical, and strictly past lagged demand features (lag 1h and lag 24h).
    6. Drops the August 01 warm-up window so all modeling rows (Aug 02 to Aug 31) have 100% complete lags.
    """
    if top_n is None:
        top_n = get_top_n_stations()

    if not CLEANED_PARQUET_PATH.exists():
        msg = f"Cleaned trips dataset not found at {CLEANED_PARQUET_PATH}. Run ETL pipeline first: python -m src.data_cleaning"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"[Modeling Data] Loading cleaned trips from {CLEANED_PARQUET_PATH}...")
    df_trips = spark.read.parquet(str(CLEANED_PARQUET_PATH))

    # Filter to August data (July 31 has truncated boundary coverage of only 1,423 trips system-wide)
    df_aug_trips = df_trips.filter(F.col("pickup_date") >= "2026-08-01")

    # 1. Identify top active stations using TRAINING data only (TRAIN_START_DATE <= date <= TRAIN_END_DATE)
    # Using start_station_id as primary key to prevent station-name collisions
    logger.info(f"[Modeling Data] Selecting top {top_n} active transit stations based on training period ({TRAIN_START_DATE} to {TRAIN_END_DATE})...")
    df_train_trips = df_aug_trips.filter(
        (F.col("pickup_date") >= TRAIN_START_DATE) & (F.col("pickup_date") <= TRAIN_END_DATE)
    )

    top_stations = df_train_trips.groupBy("start_station_id") \
        .agg(
            F.count("ride_id").alias("train_period_trips"),
            F.first("start_station_name").alias("start_station_name"),
            F.round(F.avg("start_lat"), 6).alias("latitude"),
            F.round(F.avg("start_lng"), 6).alias("longitude")
        ) \
        .orderBy(F.col("train_period_trips").desc()) \
        .limit(top_n) \
        .cache()

    station_count = top_stations.count()
    logger.info(f"[Modeling Data] Successfully identified {station_count} top active stations from training period.")

    # 2. Extract unique calendar dates in August (31 days: Aug 01 to Aug 31)
    # August 01 acts as the 24-hour warm-up period to construct uncorrupted lags for August 02 onwards
    dates_df = df_aug_trips.select("pickup_date").distinct()
    hours_df = spark.createDataFrame([(h,) for h in range(24)], ["pickup_hour"])
    time_grid = dates_df.crossJoin(hours_df)

    # 3. Create station-time grid (Skeleton: 150 stations * 31 days * 24 hours = 111,600 rows)
    station_skeleton = top_stations.select(
        "start_station_id", "start_station_name", "latitude", "longitude"
    ).crossJoin(time_grid)

    # 4. Aggregate actual pickups for these top stations across the full timeline
    actual_demand = df_aug_trips.join(
        top_stations.select("start_station_id"),
        on="start_station_id",
        how="inner"
    ).groupBy(
        "start_station_id", "pickup_date", "pickup_hour"
    ).agg(
        F.count("ride_id").alias("actual_pickups")
    )

    # 5. Left join skeleton with actual pickups and fill zero-demand hours
    demand_grid = station_skeleton.join(
        actual_demand,
        on=["start_station_id", "pickup_date", "pickup_hour"],
        how="left"
    ).na.fill({"actual_pickups": 0}) \
     .withColumnRenamed("actual_pickups", "demand")

    # 6. Feature Engineering
    demand_df = demand_grid \
        .withColumn("day_of_week", F.dayofweek("pickup_date")) \
        .withColumn("day_of_month", F.dayofmonth("pickup_date")) \
        .withColumn(
            "is_weekend",
            F.when((F.col("day_of_week") == 1) | (F.col("day_of_week") == 7), 1.0).otherwise(0.0)
        ) \
        .withColumn("hour_sin", F.round(F.sin(2.0 * math.pi * F.col("pickup_hour") / 24.0), 4)) \
        .withColumn("hour_cos", F.round(F.cos(2.0 * math.pi * F.col("pickup_hour") / 24.0), 4)) \
        .withColumn("date_str", F.col("pickup_date").cast("string"))

    # Compute strictly past lagged features using Window partitioned by unique station ID
    # Order chronologically by pickup_date and pickup_hour without crossing station boundaries
    station_time_window = Window.partitionBy("start_station_id").orderBy("pickup_date", "pickup_hour")

    # Drop August 01 warm-up period: from August 02 onwards, lag_1h and lag_24h are 100% complete and verified
    demand_with_lags = demand_df \
        .withColumn("lag_1h_demand", F.lag("demand", 1).over(station_time_window)) \
        .withColumn("lag_24h_demand", F.lag("demand", 24).over(station_time_window)) \
        .filter(F.col("date_str") >= TRAIN_START_DATE) \
        .na.drop(subset=["lag_1h_demand", "lag_24h_demand"])

    # Cast integer fields to double for MLlib compatibility
    final_modeling_df = demand_with_lags \
        .withColumn("demand", F.col("demand").cast("double")) \
        .withColumn("pickup_hour", F.col("pickup_hour").cast("double")) \
        .withColumn("day_of_week", F.col("day_of_week").cast("double")) \
        .withColumn("day_of_month", F.col("day_of_month").cast("double")) \
        .withColumn("is_weekend", F.col("is_weekend").cast("double")) \
        .withColumn("lag_1h_demand", F.col("lag_1h_demand").cast("double")) \
        .withColumn("lag_24h_demand", F.col("lag_24h_demand").cast("double"))

    total_rows = final_modeling_df.count()
    logger.info(f"[Modeling Data] Total station-hour observations with zero-demand and lag features: {total_rows:,}")

    # Save to parquet for fast reuse in downstream modules and dashboard
    final_modeling_df.write.mode("overwrite").parquet(str(STATION_HOURLY_PARQUET_PATH))
    logger.info(f"[Modeling Data] Saved station-hour modeling dataset to {STATION_HOURLY_PARQUET_PATH}")

    return final_modeling_df

def extract_and_save_feature_importance(model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts feature importances from the trained Spark MLlib tree-based model (Random Forest / GBT),
    pairs them with feature names, sorts descending, and saves to JSON.
    """
    importances = model.featureImportances.toArray().tolist()
    if len(importances) != len(feature_names):
        logger.warning(f"Feature count mismatch: {len(importances)} weights vs {len(feature_names)} names.")

    importance_list = []
    for name, score in zip(feature_names, importances):
        importance_list.append({
            "feature": name,
            "importance": round(float(score), 4)
        })

    importance_list.sort(key=lambda x: x["importance"], reverse=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_IMPORTANCE_PATH, "w", encoding="utf-8") as f:
        json.dump(importance_list, f, indent=4)

    logger.info(f"[Feature Importance] Saved feature importances to {FEATURE_IMPORTANCE_PATH}")
    for item in importance_list:
        logger.info(f"  {item['feature']:20s}: {item['importance']:.4f}")

    return importance_list

def train_and_evaluate_models(spark: SparkSession, modeling_df: DataFrame) -> Dict[str, Any]:
    """
    Executes chronological model training and evaluation:
    - Train split: TRAIN_START_DATE <= date <= TRAIN_END_DATE (Aug 02 - Aug 23, 22 days = 79,200 rows)
    - Val split: date > TRAIN_END_DATE and date <= VAL_END_DATE (Aug 24 - Aug 27, 4 days = 14,400 rows)
    - Test split: date > VAL_END_DATE (Aug 28 - Aug 31, 4 days = 14,400 rows)
    - Total modeling sample: 108,000 station-hours across 150 stations (Aug 02 - Aug 31, 30 days * 24h)

    Evaluates:
    1. Linear Regression (Baseline)
    2. Decision Tree Regressor
    3. Random Forest Regressor
    4. Gradient-Boosted Trees (GBT) Regressor

    Note on Testing Paradigm:
    Evaluated under a strictly chronological rolling 1-hour-ahead protocol.
    For each target hour t, lag_1h_demand is the actual observed pickups at t-1 and lag_24h_demand is at t-24.
    No future demand values are accessed.
    """
    feature_cols = FEATURE_COLS

    logger.info(f"[Feature Engineering] Assembling feature vector with {len(feature_cols)} features: {feature_cols}")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data_assembled = assembler.transform(modeling_df)

    # Chronological Split
    train_df = data_assembled.filter(
        (F.col("date_str") >= TRAIN_START_DATE) & (F.col("date_str") <= TRAIN_END_DATE)
    ).cache()
    val_df = data_assembled.filter(
        (F.col("date_str") > TRAIN_END_DATE) & (F.col("date_str") <= VAL_END_DATE)
    ).cache()
    test_df = data_assembled.filter(F.col("date_str") > VAL_END_DATE).cache()

    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()

    logger.info(f"[Model Training] Split sizes -> Train: {train_count:,} | Val: {val_count:,} | Test: {test_count:,}")

    evaluator_rmse = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="rmse")
    evaluator_mae = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="mae")
    evaluator_r2 = RegressionEvaluator(labelCol="demand", predictionCol="prediction", metricName="r2")

    models_to_train = {
        "Linear Regression (Baseline)": LinearRegression(featuresCol="features", labelCol="demand", maxIter=50, regParam=0.01),
        "Decision Tree": DecisionTreeRegressor(featuresCol="features", labelCol="demand", maxDepth=6, seed=42),
        "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="demand", numTrees=25, maxDepth=7, seed=42),
        "Gradient-Boosted Trees": GBTRegressor(featuresCol="features", labelCol="demand", maxIter=25, maxDepth=5, seed=42)
    }

    results = {}
    best_val_rmse = float("inf")
    best_model_name = None
    best_model_obj = None

    logger.info("--- Model Training & Validation Performance ---")
    for name, model_cls in models_to_train.items():
        t_m0 = time.time()
        logger.info(f"Training {name}...")
        fitted_model = model_cls.fit(train_df)
        
        # Predict on validation set
        val_preds = fitted_model.transform(val_df)
        val_rmse = evaluator_rmse.evaluate(val_preds)
        val_mae = evaluator_mae.evaluate(val_preds)
        val_r2 = evaluator_r2.evaluate(val_preds)
        t_elapsed = round(time.time() - t_m0, 2)

        logger.info(f"  {name:28s} -> Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f} ({t_elapsed}s)")

        results[name] = {
            "validation": {
                "rmse": round(val_rmse, 4),
                "mae": round(val_mae, 4),
                "r2": round(val_r2, 4)
            }
        }

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_name = name
            best_model_obj = fitted_model

    logger.info(f"[Model Selection] Best Model based on Validation RMSE: {best_model_name}")

    # Evaluate best model and baseline on TEST set
    test_eval_summary = {}
    for name, fitted_model in [
        (best_model_name, best_model_obj),
        ("Linear Regression (Baseline)", models_to_train["Linear Regression (Baseline)"].fit(train_df))
    ]:
        test_preds = fitted_model.transform(test_df).cache()
        test_rmse = evaluator_rmse.evaluate(test_preds)
        test_mae = evaluator_mae.evaluate(test_preds)
        test_r2 = evaluator_r2.evaluate(test_preds)

        test_eval_summary[name] = {
            "test_rmse": round(test_rmse, 4),
            "test_mae": round(test_mae, 4),
            "test_r2": round(test_r2, 4)
        }
        if name in results:
            results[name]["test"] = test_eval_summary[name]

    logger.info("--- Final Test Set Evaluation (Chronological Test Window: Aug 28 - Aug 31) ---")
    for name, m in test_eval_summary.items():
        logger.info(f"  {name:28s} -> Test RMSE: {m['test_rmse']:.4f} | Test MAE: {m['test_mae']:.4f} | Test R2: {m['test_r2']:.4f}")

    # Save best model to disk
    best_model_dir = str(MODELS_DIR / "best_demand_model")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_obj.write().overwrite().save(best_model_dir)
    logger.info(f"[Model Storage] Saved best model ({best_model_name}) to {best_model_dir}")

    # Extract feature importance if model supports it (Random Forest / GBT / Decision Tree)
    if hasattr(best_model_obj, "featureImportances"):
        extract_and_save_feature_importance(best_model_obj, feature_cols)

    # Save a sample of test predictions for dashboard visualization
    test_sample_path = PROCESSED_DATA_DIR / "test_predictions_sample.parquet"
    best_test_preds = best_model_obj.transform(test_df)
    best_test_preds.select(
        "start_station_name", "pickup_date", "pickup_hour", "demand",
        F.round("prediction", 2).alias("predicted_demand")
    ).write.mode("overwrite").parquet(str(test_sample_path))
    logger.info(f"[Model Storage] Saved test predictions sample to {test_sample_path}")

    # Save comprehensive metrics report
    metrics_payload = {
        "best_model": best_model_name,
        "feature_columns": feature_cols,
        "split_counts": {
            "train": train_count,
            "validation": val_count,
            "test": test_count
        },
        "model_comparison": results,
        "test_evaluation": test_eval_summary
    }

    with open(REPORTS_DIR / "model_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)
    logger.info(f"[Report] Saved model evaluation metrics to {REPORTS_DIR / 'model_evaluation_metrics.json'}")

    return metrics_payload

def execute_machine_learning_pipeline(spark: SparkSession, force_recompute_grid: bool = True) -> Dict[str, Any]:
    """
    Orchestrates the entire MLlib demand modeling workflow.
    Defaults to force_recompute_grid=True to ensure training-based station selection is applied.
    """
    logger.info("=== STARTING SPARK MLLIB DEMAND PREDICTION PIPELINE ===")
    t0 = time.time()
    if not force_recompute_grid and STATION_HOURLY_PARQUET_PATH.exists():
        logger.info(f"[Modeling Data] Loading existing station-hour demand dataset from {STATION_HOURLY_PARQUET_PATH}...")
        modeling_df = spark.read.parquet(str(STATION_HOURLY_PARQUET_PATH))
    else:
        modeling_df = create_station_hour_demand_dataset(spark, top_n=get_top_n_stations())

    metrics = train_and_evaluate_models(spark, modeling_df)
    total_time = round(time.time() - t0, 2)
    logger.info(f"=== SPARK MLLIB DEMAND PREDICTION COMPLETED SUCCESSFULLY IN {total_time}s ===")
    return metrics

if __name__ == "__main__":
    from src.spark_session import get_spark_session
    spark = get_spark_session("Model_Training_Pipeline")
    execute_machine_learning_pipeline(spark, force_recompute_grid=True)
    spark.stop()
