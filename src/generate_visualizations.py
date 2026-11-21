import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.config import (
    VISUALIZATIONS_DIR,
    REPORTS_DIR,
    STATION_HOTSPOTS_CSV_PATH,
    PROCESSED_DATA_DIR,
    FEATURE_IMPORTANCE_PATH
)
from src.logger import logger

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.autolayout": True
})

def generate_all_charts():
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Visualizations] Generating publication-quality charts in {VISUALIZATIONS_DIR}...")

    # Load reports with error handling
    try:
        with open(REPORTS_DIR / "etl_audit_report.json", "r", encoding="utf-8") as f:
            etl_report = json.load(f)
        with open(REPORTS_DIR / "analytics_summary.json", "r", encoding="utf-8") as f:
            analytics = json.load(f)
        with open(REPORTS_DIR / "hotspot_summary.json", "r", encoding="utf-8") as f:
            hotspots = json.load(f)
        with open(REPORTS_DIR / "model_evaluation_metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"[Visualizations] Required report JSON missing: {e}. Run analysis and model training first.")
        raise

    # 1. Data Cleaning Summary
    fig, ax = plt.subplots(figsize=(8, 4))
    raw_cnt = etl_report["cleaning"]["raw_count"]
    clean_cnt = etl_report["cleaning"]["cleaned_count"]
    rem_cnt = etl_report["cleaning"]["removed_count"]

    stages = ["Raw Ingested Records", "Cleaned & Validated Trips", "Removed Outliers / Nulls"]
    values = [raw_cnt, clean_cnt, rem_cnt]
    colors = ["#2b5c8f", "#2ca02c", "#d62728"]
    bars = ax.barh(stages, values, color=colors, height=0.55)
    ax.set_xlabel("Number of Records")
    ax.set_title("Data Cleaning and Ingestion Pipeline Summary")
    for bar, val in zip(bars, values):
        pct = (val / raw_cnt) * 100
        ax.text(val + raw_cnt * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.2f}%)", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, raw_cnt * 1.25)
    plt.savefig(VISUALIZATIONS_DIR / "01_data_cleaning_summary.png", dpi=300)
    plt.close()

    # 2. Hourly Trip Distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hourly_df = pd.DataFrame(analytics["temporal_analysis"]["hourly"])
    ax.bar(hourly_df["pickup_hour"], hourly_df["total_trips"], color="#1f77b4", edgecolor="#0e4d79", alpha=0.85)
    ax.set_xlabel("Hour of Day (0 - 23)")
    ax.set_ylabel("Total Pickups")
    ax.set_title("System-Wide Diurnal Bike Pickup Distribution")
    ax.set_xticks(range(0, 24))
    ax.axvspan(7, 9, color="orange", alpha=0.2, label="Morning Peak (07-09)")
    ax.axvspan(16, 19, color="green", alpha=0.15, label="Evening Peak (16-19)")
    ax.legend()
    plt.savefig(VISUALIZATIONS_DIR / "02_hourly_trip_distribution.png", dpi=300)
    plt.close()

    # 3. Weekday vs Weekend Demand
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ww_df = pd.DataFrame(analytics["temporal_analysis"]["weekday_weekend"])
    bars = ax.bar(ww_df["day_type"], ww_df["avg_daily_trips"], color=["#1f77b4", "#ff7f0e"], width=0.45)
    ax.set_ylabel("Average Trips Per Day")
    ax.set_title("Daily Mobility Demand: Weekdays vs. Weekends")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,.0f} trips/day",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(ww_df["avg_daily_trips"]) * 1.18)
    plt.savefig(VISUALIZATIONS_DIR / "03_weekday_vs_weekend_demand.png", dpi=300)
    plt.close()

    # 4. Top 10 Busiest Stations
    fig, ax = plt.subplots(figsize=(10, 5))
    top10_df = pd.DataFrame(hotspots["top_10_stations"]).sort_values("total_pickups", ascending=True)
    bars = ax.barh(top10_df["start_station_name"], top10_df["total_pickups"], color="#3b528b")
    ax.set_xlabel("Total Trip Departures (August 2026)")
    ax.set_title("Top 10 High-Traffic Citi Bike Pickup Stations")
    for bar, val in zip(bars, top10_df["total_pickups"]):
        ax.text(val + max(top10_df["total_pickups"]) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8.5)
    ax.set_xlim(0, max(top10_df["total_pickups"]) * 1.15)
    plt.savefig(VISUALIZATIONS_DIR / "04_top10_busiest_stations.png", dpi=300)
    plt.close()

    # 5. Rider Category Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    rider_df = pd.DataFrame(analytics["rider_and_bike_analysis"]["rider_summary"])
    ax1.pie(rider_df["total_trips"], labels=rider_df["member_casual"].str.capitalize(),
            autopct="%1.1f%%", colors=["#ff9896", "#aec7e8"], startangle=140, explode=(0.05, 0))
    ax1.set_title("Trip Volume Share by Rider Type")

    bars = ax2.bar(rider_df["member_casual"].str.capitalize(), rider_df["avg_duration_min"], color=["#e377c2", "#17becf"], width=0.45)
    ax2.set_ylabel("Average Trip Duration (minutes)")
    ax2.set_title("Average Ride Duration by Rider Type")
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f} min", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylim(0, max(rider_df["avg_duration_min"]) * 1.2)
    plt.savefig(VISUALIZATIONS_DIR / "05_rider_category_comparison.png", dpi=300)
    plt.close()

    # 6. Bike-Type Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bike_df = pd.DataFrame(analytics["rider_and_bike_analysis"]["bike_summary"])
    bars = ax.bar(bike_df["rideable_type"].str.replace("_", " ").str.title(),
                  bike_df["total_trips"], color=["#3182bd", "#9ecae1"], width=0.45)
    ax.set_ylabel("Total Trips")
    ax.set_title("Bike Technology Utilization: Electric vs. Classic")
    for bar in bars:
        h = bar.get_height()
        pct = (h / bike_df["total_trips"].sum()) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(bike_df["total_trips"]) * 1.18)
    plt.savefig(VISUALIZATIONS_DIR / "06_bike_type_distribution.png", dpi=300)
    plt.close()

    # 7. Station Demand Classification
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cls_df = pd.DataFrame(hotspots["classification_breakdown"]).sort_values("count", ascending=False)
    colors = {"High Demand": "#d62728", "Medium Demand": "#ff7f0e", "Low Demand": "#2ca02c"}
    bars = ax.bar(cls_df["demand_classification"], cls_df["count"],
                  color=[colors.get(c, "#1f77b4") for c in cls_df["demand_classification"]], width=0.45)
    ax.set_ylabel("Number of Stations")
    ax.set_title("Station Network Demand Tier Classification")
    for bar in bars:
        h = bar.get_height()
        pct = (h / cls_df["count"].sum()) * 100
        ax.annotate(f"{h:,} ({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, max(cls_df["count"]) * 1.18)
    plt.savefig(VISUALIZATIONS_DIR / "07_station_demand_classification.png", dpi=300)
    plt.close()

    # 8. Geographic Pickup Hotspots Scatter Map
    fig, ax = plt.subplots(figsize=(8, 9))
    stations_df = pd.read_csv(STATION_HOTSPOTS_CSV_PATH)
    color_map = {"High Demand": "red", "Medium Demand": "orange", "Low Demand": "green"}
    for cat in ["Low Demand", "Medium Demand", "High Demand"]:
        subset = stations_df[stations_df["demand_classification"] == cat]
        alpha = 0.4 if cat == "Low Demand" else (0.6 if cat == "Medium Demand" else 0.9)
        size = 10 if cat == "Low Demand" else (25 if cat == "Medium Demand" else 65)
        ax.scatter(subset["longitude"], subset["latitude"], c=color_map[cat],
                   label=f"{cat} ({len(subset):,})", alpha=alpha, s=size, edgecolors="none")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geographic Pickup Hotspots across New York City")
    ax.legend(loc="upper left")
    plt.savefig(VISUALIZATIONS_DIR / "08_geographic_hotspots.png", dpi=300)
    plt.close()

    # 9. Actual vs. Predicted Demand by Hour (from test sample)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    test_sample_path = PROCESSED_DATA_DIR / "test_predictions_sample.parquet"
    test_preds_df = pd.read_parquet(test_sample_path)
    hourly_comp = test_preds_df.groupby("pickup_hour")[["demand", "predicted_demand"]].mean().reset_index()

    ax.plot(hourly_comp["pickup_hour"], hourly_comp["demand"], marker="o", color="#1f77b4", linewidth=2.5, label="Actual Test Demand")
    ax.plot(hourly_comp["pickup_hour"], hourly_comp["predicted_demand"], marker="s", linestyle="--", color="#d62728", linewidth=2.2, label="Random Forest Predictions")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Pickups Per Station-Hour")
    ax.set_title("Chronological Holdout (Aug 28-31): Actual vs. Predicted Demand by Hour")
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.savefig(VISUALIZATIONS_DIR / "09_actual_vs_predicted_demand.png", dpi=300)
    plt.close()

    # 10. Comprehensive Model Comparison (Validation Benchmark + Final Test Evaluation)
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [2, 1]})
    model_res = metrics["model_comparison"]
    model_names = list(model_res.keys())
    val_rmse = [model_res[m]["validation"]["rmse"] for m in model_names]
    val_mae = [model_res[m]["validation"]["mae"] for m in model_names]
    val_r2 = [model_res[m]["validation"]["r2"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.28

    rects1 = ax_val.bar(x - width, val_rmse, width, label="Val RMSE (trips/hr)", color="#4575b4")
    rects2 = ax_val.bar(x, val_mae, width, label="Val MAE (trips/hr)", color="#74add1")
    rects3 = ax_val.bar(x + width, val_r2, width, label="Val R² Score", color="#f46d43")

    ax_val.set_ylabel("Metric Value")
    ax_val.set_xticks(x)
    short_names = [m.replace(" (Baseline)", "\n(Baseline)").replace("Gradient-Boosted Trees", "GBT") for m in model_names]
    ax_val.set_xticklabels(short_names, fontsize=9)
    ax_val.set_title("Model Selection: Validation Set Benchmark (Aug 24-27)")
    ax_val.legend(loc="upper right")
    ax_val.set_ylim(0, max(val_rmse) * 1.25)

    # Annotate RMSE values
    for rect in rects1:
        h = rect.get_height()
        ax_val.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Test evaluation subplot
    test_res = metrics["test_evaluation"]
    test_models = list(test_res.keys())
    t_rmse = [test_res[m]["test_rmse"] for m in test_models]
    t_r2 = [test_res[m]["test_r2"] for m in test_models]
    x_t = np.arange(len(test_models))
    w_t = 0.35

    ax_test.bar(x_t - w_t/2, t_rmse, w_t, label="Test RMSE", color="#313695")
    ax_test.bar(x_t + w_t/2, t_r2, w_t, label="Test R²", color="#d73027")
    ax_test.set_xticks(x_t)
    ax_test.set_xticklabels([m.replace(" (Baseline)", "\n(Baseline)").replace("Gradient-Boosted Trees", "GBT\n(Selected)").replace("Random Forest", "Random Forest\n(Selected)") for m in test_models], fontsize=9)
    ax_test.set_title("Final Holdout Evaluation (Aug 28-31)")
    ax_test.legend(loc="upper right")
    ax_test.set_ylim(0, max(t_rmse) * 1.25)

    for i, (rmse_val, r2_val) in enumerate(zip(t_rmse, t_r2)):
        ax_test.annotate(f"RMSE: {rmse_val:.2f}\nR²: {r2_val:.3f}",
                         xy=(x_t[i] - w_t/2, rmse_val), xytext=(0, 4),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.savefig(VISUALIZATIONS_DIR / "10_model_performance_comparison.png", dpi=300)
    plt.close()

    # 11. Feature Importance Visualization
    if FEATURE_IMPORTANCE_PATH.exists():
        with open(FEATURE_IMPORTANCE_PATH, "r", encoding="utf-8") as f:
            feat_imp = json.load(f)

        imp_df = pd.DataFrame(feat_imp).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(imp_df["feature"], imp_df["importance"], color="#2ca02c", height=0.6)
        ax.set_xlabel("Relative Split Importance (Spark MLlib Tree Ensemble)")
        ax.set_title("Machine Learning Feature Importance Profile\n(Note: Shows predictive contribution, not causal mechanism)")
        for bar, val in zip(bars, imp_df["importance"]):
            pct = val * 100
            ax.text(val + max(imp_df["importance"]) * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f} ({pct:.1f}%)", va="center", fontsize=8.5, fontweight="bold")
        ax.set_xlim(0, max(imp_df["importance"]) * 1.18)
        plt.savefig(VISUALIZATIONS_DIR / "11_rf_feature_importance.png", dpi=300)
        plt.close()
        logger.info(f"[Visualizations] Generated 11_rf_feature_importance.png")

    logger.info(f"[Visualizations] Successfully saved all publication charts to {VISUALIZATIONS_DIR}.")

if __name__ == "__main__":
    generate_all_charts()
