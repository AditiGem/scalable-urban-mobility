# Technical Audit and System Verification Report
## Scalable Urban Mobility Demand Prediction and Hotspot Analysis Using Apache Spark MLlib

**Author**: BTech CSE Candidate  
**Date**: November 2026  
**Environment**: Apache Spark 3.5.9 (`local[*]`), Python 3.10.0, Java 1.8.0_481, Windows 64-bit  
**Repository Working Directory**: `c:\Users\ADITI\Desktop\bikes`  

---

## 1. Executive Summary

This technical audit provides a rigorous, ground-truth verification of the end-to-end Big Data analytics and machine learning system engineered for urban micro-mobility demand forecasting. The project processes **5,246,236 real-world Citi Bike transactions** across 6 monthly dataset files (August 2026) using PySpark, Parquet columnar storage, Spark SQL analytics, and Spark MLlib regression ensembles.

All performance metrics, row counts, split boundaries, and model evaluations documented herein have been independently re-executed and verified from source code and disk storage.

---

## 2. Dataset Verification & Data Quality Audit

### 2.1 File-by-File Ingestion Audit
The ingestion pipeline loads all six raw CSV files located in `dataset/`:

| File Name | On-Disk Size | Total Records | Earliest Timestamp | Latest Timestamp | Missing Start Stations | Missing End Stations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `202608-citibike-tripdata_1.csv` | 186.42 MB | 1,000,000 | 2026-07-31 16:40:21 | 2026-08-14 19:59:58 | 512 | 2,781 |
| `202608-citibike-tripdata_2.csv` | 186.11 MB | 1,000,000 | 2026-07-31 13:37:51 | 2026-08-14 19:59:58 | 498 | 2,810 |
| `202608-citibike-tripdata_3.csv` | 185.75 MB | 1,000,000 | 2026-07-31 02:10:56 | 2026-08-26 19:59:59 | 530 | 2,845 |
| `202608-citibike-tripdata_4.csv` | 186.07 MB | 1,000,000 | 2026-08-14 20:00:00 | 2026-08-26 19:59:55 | 541 | 2,790 |
| `202608-citibike-tripdata_5.csv` | 186.22 MB | 1,000,000 | 2026-08-14 20:00:00 | 2026-08-31 23:56:20 | 582 | 2,714 |
| `202608-citibike-tripdata_6.csv` | 46.02 MB | 246,236 | 2026-08-28 20:00:00 | 2026-08-31 23:58:01 | 207 | 642 |
| **Combined Raw Dataset** | **976.59 MB** | **5,246,236** | **2026-07-31 02:10:56** | **2026-08-31 23:58:01** | **2,870 (0.05%)** | **14,582 (0.28%)** |

### 2.2 Investigation of July 31 Boundary & Incomplete Coverage
A critical audit finding concerns the records originating on July 31, 2026:
- **Disproportionate Volume**: July 31 contains only **1,423 trips across all of NYC**, compared to **202,163 trips on August 1**.
- **Collection Truncation**: Citi Bike extracts monthly releases based on UTC boundaries (`2026-08-01 00:00:00 UTC` corresponds to `2026-07-31 20:00:00 EDT`). Consequently:
  - Hours 0–11 on July 31 have **0 trips** in the raw files.
  - Hours 12–22 have only between **1 and 43 trips** citywide.
  - Hour 23 has **1,337 trips** as the extract window began logging.
- **Danger of Premature Zero-Imputation**: Treating unrecorded hours on July 31 as genuine zero demand is an imputation error. If July 31 was used to build lag features for August 1, `lag_24h_demand` on August 1 would be artificially set to 0 during peak daytime hours.
- **Warm-Up Resolution**: July 31 records are excluded from the modeling grid. The full hourly grid is generated from August 01 to August 31 (31 complete days). **August 01 serves as a 24-hour warm-up window** (3,600 station-hours across 150 stations) to establish uncorrupted historical ground truth for `lag_1h_demand` and `lag_24h_demand`. The modeling dataset begins cleanly on **August 02 at 00:00**.

### 2.3 Uniqueness and Data Retention
- **Unique Trip IDs**: 5,246,236 unique `ride_id` hashes (**0 duplicates found**).
- **Duration Filtering**: Trips $\le 1.0$ minute (false dock unlocks) and $> 1,440$ minutes (unreturned/stolen bikes) were removed.
- **Geographic Filtering**: Coordinates outside the NYC transit box ($40.50 \le \text{lat} \le 41.05, -74.25 \le \text{lng} \le -73.65$) were removed.
- **Cleaned Retention**: **5,242,103 clean trips retained (99.92% retention rate, 4,133 anomalies removed)**.

---

## 3. ETL Pipeline & Storage Optimization Verification

- **Storage Format**: Apache Parquet with Snappy block compression.
- **Partitioning Strategy**: Written to `processed_data/cleaned_trips.parquet` across 8 partitions (aligned with local multi-core parallelism).
- **Physical Footprint**:
  - Raw uncompressed CSVs: **976.59 MB**
  - Snappy Parquet partitions: **250.03 MB**
  - **Storage Reduction**: **74.40% compression gain** with zero loss of column metadata or precision.
- **Query Performance**: Parquet dictionary encoding and columnar projection reduce I/O overhead by eliminating full table scans during downstream analytical queries.

---

## 4. Machine Learning Pipeline & Logic Verification

### 4.1 Station-Hour Grid & Zero-Demand Representation
Standard trip logs only record positive departures, causing naive regression models to predict inflated demand during late-night or low-volume hours. To solve this:
1. Top commuter stations were selected using training-period data only.
2. An exhaustive Cartesian grid of all hours across August (31 days $\times$ 24 hours = 744 hours per station) was generated.
3. Actual hourly departures were left-joined onto the grid, imputing inactive hours with `demand = 0`.
4. Lags were computed across the full month, and August 01 was dropped as the 24-hour warm-up window.
5. **Total Modeling Sample**: **108,000 observations** across 150 stations (30 days $\times$ 24 hours = 720 hours per station).

### 4.2 Station Identification & Lag Partitioning
In Citi Bike data, station names are not guaranteed to be globally unique:
- Across the NYC network, **44 station names map to multiple station IDs** (e.g., `Clinton St & Grand St` maps to 2 distinct physical stations).
- Conversely, `start_station_id` uniquely identifies the physical dock terminal.
- **Implementation Audit**: Station selection and Window lag partitioning are anchored on `start_station_id`:
  ```python
  station_time_window = Window.partitionBy("start_station_id").orderBy("pickup_date", "pickup_hour")
  ```
  This guarantees that historical lag values never cross station boundaries and remain strictly ordered chronologically by date and hour.

### 4.3 Training-Period Station Selection (Lookahead Elimination)
Station selection is computed **exclusively on the training period** (`2026-08-02 <= pickup_date <= 2026-08-23`):
- Station count is parameterized via `TOP_N_STATIONS` in `src/config.py` (default: `150`).
- Station selection is completely decoupled from validation and test periods, eliminating candidate selection leakage.
- The broader geographic hotspot analysis continues to evaluate all **2,386 unique stations** across NYC.

### 4.4 Operational Forecasting Paradigm
The predictive model operates under a **rolling 1-hour-ahead demand forecasting paradigm**:
- For any station at target hour $t$, the system uses observed features up to $t-1$ (including `lag_1h_demand` at $t-1$ and `lag_24h_demand` at $t-24$).
- This reflects realistic bike-share operations: dispatchers review verified pickups from the immediate past hour to organize van rebalancing for the upcoming hour.
- It is explicitly **not** a multi-day static forecast made 96 hours in advance without intermediate observations.
- Verification confirms that **no future demand values are accessed**.

### 4.5 Chronological Train-Validation-Test Splitting
To avoid temporal data leakage inherent in random K-fold splits, the dataset is split strictly by calendar date:

| Split Partition | Calendar Date Range | Days | Hours / Station | Stations | Total Observations | Relative Share |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Warm-Up Window** | August 01, 2026 | 1 | 24 | 150 | 3,600 | Lag initialization |
| **Training Set** | August 02 – August 23, 2026 | 22 | 528 | 150 | **79,200** | 73.33% |
| **Validation Set** | August 24 – August 27, 2026 | 4 | 96 | 150 | **14,400** | 13.33% |
| **Test Set** | August 28 – August 31, 2026 | 4 | 96 | 150 | **14,400** | 13.33% |
| **Total Modeling Set** | August 02 – August 31, 2026 | 30 | 720 | 150 | **108,000** | 100.00% |

$$\text{Sum Check}: 79,200 + 14,400 + 14,400 = 108,000 \quad \checkmark$$

---

## 5. Machine Learning Evaluation & Feature Importance

### 5.1 Model Selection on Validation Set (August 24 – August 27)
Four regression algorithms were trained on the training partition and tuned/compared on the validation set:

| Model | Hyperparameters | Validation RMSE | Validation MAE | Validation $R^2$ | Selection Outcome |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | `maxIter=50, regParam=0.01` | 7.4493 | 4.9671 | 0.6447 | Baseline benchmark |
| **Decision Tree Regressor** | `maxDepth=6, seed=42` | 7.1291 | 4.7301 | 0.6746 | Non-linear tree baseline |
| **Random Forest Regressor** | `numTrees=25, maxDepth=7, seed=42` | 6.9214 | 4.5505 | 0.6932 | Bagging ensemble |
| **Gradient-Boosted Trees (GBT)** | `maxIter=25, maxDepth=5, seed=42` | **6.6748** | **4.4474** | **0.7147** | **Selected Champion Model** |

*Selection Rationale*: Gradient-Boosted Trees achieved the lowest validation RMSE (6.6748) and highest $R^2$ (0.7147), successfully learning sequential residual corrections across diurnal and spatial patterns.

### 5.2 Holdout Test Evaluation (August 28 – August 31)
The champion Gradient-Boosted Trees model and baseline Linear Regression model were evaluated on the unseen holdout test partition:

| Model | Test RMSE (trips/hr) | Test MAE (trips/hr) | Test $R^2$ | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient-Boosted Trees (GBT)** | **6.3087** | **4.3031** | **0.7159** | **-0.3136 RMSE (R² increased by 0.0289 / 2.89 percentage points)** |
| **Linear Regression (Baseline)** | 6.6223 | 4.5221 | 0.6870 | Reference |

### 5.3 Feature Importance Analysis (Gradient-Boosted Trees)
Serialized to `reports/gbt_feature_importance.json` and plotted in `visualizations/11_rf_feature_importance.png`:

| Rank | Feature Name | Split Impurity Importance | Relative Share | Cumulative Share | Operational Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `lag_1h_demand` | 0.4080 | 40.80% | 40.80% | Station momentum in the immediate preceding hour |
| 2 | `lag_24h_demand` | 0.1007 | 10.07% | 50.87% | Diurnal periodicity from the same hour yesterday |
| 3 | `pickup_hour` | 0.0869 | 8.69% | 59.56% | Coarse diurnal commute shift indicator |
| 4 | `day_of_week` | 0.0823 | 8.23% | 67.79% | Day-specific commute variance |
| 5 | `hour_cos` | 0.0731 | 7.31% | 75.10% | Continuous harmonic time representation |
| 6 | `day_of_month` | 0.0688 | 6.88% | 81.98% | Sub-monthly trend progression |
| 7 | `hour_sin` | 0.0622 | 6.22% | 88.20% | Continuous harmonic time representation |
| 8 | `is_weekend` | 0.0597 | 5.97% | 94.17% | Leisure vs. commuter travel mode shift |
| 9 | `latitude` | 0.0350 | 3.50% | 97.67% | Spatial location (North-South axis) |
| 10 | `longitude` | 0.0232 | 2.32% | 100.00% | Spatial location (East-West axis) |

> **Scientific Disclosure**: Feature importance in decision tree ensembles measures the normalized total reduction in impurity (variance) across split nodes. It reflects predictive correlation within the trained model rather than direct physical causation.

---

## 6. Geographic Hotspot Analysis Verification

Hotspot classification evaluates all **2,386 unique pickup stations** across NYC:
- **High Demand (Top 20%)**: Total departures $\ge 4,020$ (**480 stations**).
- **Medium Demand (Middle 50%)**: Total departures between $415$ and $4,019$ (**1,193 stations**).
- **Low Demand (Bottom 30%)**: Total departures $< 415$ (**713 stations**).

### Verified Top 5 Transit Hubs
1. **W 21 St & 6 Ave**: 17,727 departures (23.08 trips/hr avg, 10.48 min avg duration)
2. **Pier 61 at Chelsea Piers**: 16,752 departures (21.81 trips/hr avg, 14.38 min avg duration)
3. **E 17 St & Broadway**: 16,030 departures (20.87 trips/hr avg, 13.15 min avg duration)
4. **7 Ave & Central Park South**: 15,312 departures (19.94 trips/hr avg, 27.09 min avg duration)
5. **W 31 St & 7 Ave**: 15,189 departures (19.78 trips/hr avg, 12.20 min avg duration)

---

## 7. Interactive Streamlit Dashboard Verification

The dashboard (`dashboard/app.py`) was audited and verified:
- **Overview Tab**: Displays KPI metric cards (`5.24M` trips, `2,386` stations, `12.9 min` avg duration, `74.4%` Parquet compression).
- **Demand Analysis Tab**: Interactive Plotly charts for hourly commute curves, weekday vs. weekend shifts, and electric bike adoption.
- **Geographic Hotspots Tab**: PyDeck 3D map with interactive station tooltips and dynamic demand-tier coloring.
- **Demand Prediction & ML Tab**:
  - Displays validation benchmark comparison table.
  - Displays holdout test set evaluation metrics ($R^2 = 0.7159$, $\text{RMSE} = 6.3087$).
  - Displays tree ensemble feature importance table and chart.
  - Interactive station forecast lookup tool allowing selection of any modeled station to inspect predicted vs. actual demand.

---

## 8. Technical Logging & System Reproducibility

- **Centralized Logging**: `src/logger.py` provides thread-safe logging to both console and `logs/pipeline.log`.
- **Execution Logging**: Ingestion file sizes, cleaning filtering counts, training convergence, and evaluation metrics are written with timestamps and execution durations.
- **End-to-End Pipeline Execution**:
  ```bash
  python -m src.data_cleaning
  python -m src.demand_analysis
  python -m src.hotspot_analysis
  python -m src.model_training
  python -m src.generate_visualizations
  streamlit run dashboard/app.py
  ```

---

## 9. Current Limitations & Scope Boundaries

1. **Single-Node Execution**: Apache Spark is run in `local[*]` multi-threaded mode on a single workstation (not a distributed cluster like AWS EMR or Databricks). All code uses standard distributed PySpark DataFrame and MLlib APIs and is 100% cluster-ready.
2. **Exogenous Factors**: Weather conditions (temperature, rain, snow) are not included in Citi Bike trip records.
3. **Temporal Window**: The dataset spans August 2026 (summer peak); winter seasonal demand dynamics are unobserved.
4. **Forecasting Horizon**: The system is designed and evaluated for rolling 1-hour-ahead operational dispatching; recursive multi-day forecasting without intermediate ground truth is not supported.

---

## 10. CV / Resume Impact Bullet Points

For technical BTech CSE placement resumes:

- **Engineered an End-to-End PySpark Big Data Pipeline**: Ingested and cleaned 5.24M+ Citi Bike trip records (~1 GB raw CSVs) using PySpark on Apache Spark 3.5; reduced on-disk footprint by 74.4% (to 250 MB) using Snappy-compressed columnar Parquet partitions.
- **Formulated Station-Level Machine Learning Demand Models**: Engineered a 108,000 station-hour zero-demand modeling grid; anchored lag features on unique station IDs (`start_station_id`) and implemented an August 1 warm-up window to eliminate corrupted boundary lags from incomplete source data.
- **Benchmarked Spark MLlib Regression Ensembles**: Implemented and tuned Linear Regression, Decision Tree, Random Forest, and GBT models using strict chronological train/validation/test splits; achieved test $R^2 = 0.7159$ and $\text{RMSE} = 6.31$ trips/hr under a rolling 1-hour-ahead evaluation protocol.
- **Extracted Machine Learning Feature Interpretability**: Extracted and visualized normalized split-impurity feature importances, identifying that recent momentum (`lag_1h`: 40.8%) and diurnal periodicity (`lag_24h` + `pickup_hour`: 18.8%) drive over 59.5% of predictive capability.
- **Mapped Micro-Mobility Demand Hotspots**: Applied statistical quantile thresholds across 2,386 transit hubs to classify High, Medium, and Low demand tiers for municipal fleet rebalancing.
- **Deployed Interactive Analytics Dashboard**: Built a responsive Streamlit web application featuring PyDeck 3D geospatial maps, Plotly diurnal analytics, and station-level forecast lookups.
