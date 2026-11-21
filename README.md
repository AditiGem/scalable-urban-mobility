# Scalable Urban Mobility Demand Prediction and Hotspot Analysis Using Apache Spark MLlib

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Apache Spark 3.5](https://img.shields.io/badge/Apache%20Spark-3.5.9-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Big Data analytics and machine learning system engineered with **Apache Spark (PySpark)**, **Spark SQL**, and **Spark MLlib** to analyze large-scale urban micro-mobility patterns, map geographic demand hotspots, and forecast station-level pickup demand across New York City using **5.24+ million real-world Citi Bike trips**.

---

## 1. Project Overview

Municipal bike-sharing systems represent a critical component of modern urban micro-mobility infrastructure. However, fluctuating commuter demand creates severe dock imbalances—causing empty stations during morning rush hours and overcrowded docks during evening peaks. 

This project builds a scalable, production-grade Big Data pipeline to ingest, clean, analyze, and forecast hourly station-level pickup demand. Designed with modular Python architecture, the system leverages Apache Spark's distributed DataFrame engine, columnar Parquet storage, reproducible geospatial quantile classification, and MLlib regression algorithms evaluated on strictly chronological holdout splits.

---

## 2. Problem Statement & Objectives

### Problem Statement
Bike-sharing network operators face dynamic, time-sensitive supply-demand imbalances across thousands of stations. Manual rebalancing is inefficient and costly. Reliable hourly demand forecasts and geographic hotspot classifications are needed to optimize automated van rebalancing, fleet allocation, and infrastructure planning.

### Objectives
1. **Scalable Ingestion & ETL**: Process multiple large-scale Citi Bike CSV files (~1 GB, 5.24M rows) with distributed PySpark transformations.
2. **Data Quality & Validation**: Filter out coordinate outliers, corrupted timestamps, accidental dock unlocks (<1 min), and unreturned bikes (>24 hr).
3. **Temporal & Demographic Analytics**: Uncover commute patterns across hours, days of the week, user types (`member` vs. `casual`), and bike types (`electric_bike` vs. `classic_bike`) via Spark SQL.
4. **Geographic Hotspot Mapping**: Classify 2,386 transit stations into High, Medium, and Low demand tiers using reproducible statistical quantiles.
5. **Predictive Modeling with Zero-Demand Representation**: Model station-hourly demand incorporating zero-pickup hours, engineering strictly historical lag features, and benchmarking Spark MLlib models (**Linear Regression, Decision Tree, Random Forest, GBT**).
6. **Interactive Visualization & Dashboard**: Deploy a responsive interactive Streamlit dashboard for exploratory analytics and map visualization.

---

## 3. Dataset Source & Detailed Inspection

The dataset consists of **6 extracted Citi Bike CSV files** spanning **August 2026** in New York City:

| File Name | File Size | Exact Records | Earliest Timestamp | Latest Timestamp |
| :--- | :--- | :--- | :--- | :--- |
| `202608-citibike-tripdata_1.csv` | 186.42 MB | 1,000,000 | 2026-07-31 16:40:21 | 2026-08-14 19:59:58 |
| `202608-citibike-tripdata_2.csv` | 186.11 MB | 1,000,000 | 2026-07-31 13:37:51 | 2026-08-14 19:59:58 |
| `202608-citibike-tripdata_3.csv` | 185.75 MB | 1,000,000 | 2026-07-31 02:10:56 | 2026-08-26 19:59:59 |
| `202608-citibike-tripdata_4.csv` | 186.07 MB | 1,000,000 | 2026-08-14 20:00:00 | 2026-08-26 19:59:55 |
| `202608-citibike-tripdata_5.csv` | 186.22 MB | 1,000,000 | 2026-08-14 20:00:00 | 2026-08-31 23:56:20 |
| `202608-citibike-tripdata_6.csv` | 46.02 MB | 246,236 | 2026-08-28 20:00:00 | 2026-08-31 23:58:01 |
| **Combined Raw Dataset** | **976.59 MB** | **5,246,236** | **2026-07-31 02:10:56** | **2026-08-31 23:58:01** |

### Verified Data Quality Audit
- **Unique Trip IDs**: 5,246,236 distinct `ride_id` values (**0 duplicates**).
- **Missing Data**:
  - `start_station_name` / `start_station_id`: 2,870 rows (0.05%)
  - `start_lat` / `start_lng`: 2,870 rows (0.05%)
  - `end_station_name`: 13,792 rows (0.26%)
  - `end_lat` / `end_lng`: 14,733 rows (0.28%)
- **Data Retention**: After filtering duration outliers and missing origin metadata, **5,242,103 valid records were retained (99.92% retention rate)**.

---

## 4. Technologies Used

- **Data Processing Engine**: Apache Spark 3.5.9 (PySpark DataFrame API, Spark SQL, RDDs)
- **Machine Learning**: Spark MLlib (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GBTRegressor`, `VectorAssembler`, `RegressionEvaluator`)
- **Storage Format**: Apache Parquet (Snappy-compressed columnar partitions)
- **Interactive UI & Dashboard**: Streamlit 1.64.0, Plotly 7.1.0, PyDeck 0.9.3
- **Data Science & Plotting**: Pandas 2.3.3, NumPy 2.2.6, Matplotlib 3.10.9
- **Runtime Environment**: Python 3.10.0, Java 1.8.0_481 (64-bit SE Runtime)

> **Execution Disclosure**: Apache Spark was executed in **local mode (`local[*]`)** leveraging multi-core parallel processing on a single host machine, configured with 4 GB driver memory and 8 shuffle partitions.

---

## 5. System Architecture & Folder Structure

```
bikes/
├── dataset/                               # Raw Citi Bike CSV files (Files 1 to 6)
│   ├── 202608-citibike-tripdata_1.csv
│   ├── ...
│   └── 202608-citibike-tripdata_6.csv
│
├── src/                                   # Reusable modular source code
│   ├── config.py                          # Global paths, thresholds, and dates
│   ├── logger.py                          # Centralized execution logging
│   ├── spark_session.py                   # Local SparkSession builder & tuning
│   ├── data_ingestion.py                  # PySpark multi-file ingestion & validation
│   ├── data_cleaning.py                   # Duration, coordinate validation & ETL
│   ├── demand_analysis.py                 # Spark SQL temporal & demographic queries
│   ├── hotspot_analysis.py                # Station quantile classification & hotspots
│   ├── model_training.py                  # Zero-demand grid, lag features & MLlib models
│   └── generate_visualizations.py         # Publication chart generation suite
│
├── dashboard/
│   └── app.py                             # Interactive Streamlit Web Application
│
├── notebooks/
│   └── urban_mobility_analysis.ipynb      # Step-by-step interactive workflow notebook
│
├── processed_data/                        # Columnar Parquet & CSV analytical tables
│   ├── cleaned_trips.parquet/             # 5.24M cleaned trips in 8 partitions (~250 MB)
│   ├── station_hourly_demand.parquet/     # Station-hour modeling dataset with lags
│   ├── station_hotspots.csv               # Station demand classification & coordinates
│   └── test_predictions_sample.parquet   # Out-of-sample predictions for evaluation
│
├── models/
│   └── best_demand_model/                 # Serialized best Spark MLlib model (Gradient-Boosted Trees)
│
├── visualizations/                        # 11 publication-quality charts (300 DPI)
│   ├── 01_data_cleaning_summary.png
│   ├── 02_hourly_trip_distribution.png
│   ├── 03_weekday_vs_weekend_demand.png
│   ├── 04_top10_busiest_stations.png
│   ├── 05_rider_category_comparison.png
│   ├── 06_bike_type_distribution.png
│   ├── 07_station_demand_classification.png
│   ├── 08_geographic_hotspots.png
│   ├── 09_actual_vs_predicted_demand.png
│   ├── 10_model_performance_comparison.png
│   └── 11_gbt_feature_importance.png
│
├── reports/                               # Machine-readable audit and metric JSONs
│   ├── etl_audit_report.json
│   ├── analytics_summary.json
│   ├── hotspot_summary.json
│   ├── model_evaluation_metrics.json
│   └── gbt_feature_importance.json
│
├── logs/                                  # Execution logs
│   └── pipeline.log                       # Timestamped execution run logs
│
├── backups/                               # Preserved original student notebooks
├── requirements.txt                       # Project Python dependencies
├── .gitignore                             # Clean repository ignore rules
└── README.md                              # Complete project documentation
```

---

## 6. End-to-End Data Processing Pipeline

```
[Raw CSVs (1-6)] (~1 GB)
       │
       ▼
[src/data_ingestion.py]   ── Multi-file PySpark Ingestion (5,246,236 rows)
       │
       ▼
[src/data_cleaning.py]    ── Millisecond Timestamp Casting
                          ── Duration Filter: 1 min <= duration <= 24 hr
                          ── Coordinate Bounds: NYC Box (40.5-41.05N, 74.25-73.65W)
                          ── Feature Engineering: pickup_hour, day_of_week, is_weekend
       │
       ▼
[processed_data/cleaned_trips.parquet] (5,242,103 clean trips, 99.92% retained)
       │
       ├──► [src/demand_analysis.py]   ── Spark SQL Temporal & Demographic Queries
       │
       ├──► [src/hotspot_analysis.py]  ── Station Demand Quantiles (Low/Med/High)
       │
       └──► [src/model_training.py]    ── Hourly Grid (Zero-Demand Representation)
                                       ── Strictly Past Lags (lag_1h, lag_24h)
                                       ── Chronological Train/Val/Test Split
                                       ── Spark MLlib Model Benchmarking
```

---

## 7. Key Analytical Insights (Spark SQL)

1. **Diurnal Commuter Peaks**: Hourly pickups exhibit distinct bimodal peaks on weekdays at **08:00 (Morning commute: ~300k departures)** and **17:00–18:00 (Evening rush: ~480k departures)**. In contrast, weekends display a smooth unimodal curve peaking between 13:00 and 17:00.
2. **Weekday vs. Weekend Demand**: Average daily weekday departures reach **171,448 trips/day** with an average trip duration of **12.4 minutes**. Weekend departures average **148,849 trips/day** with longer recreational durations averaging **14.9 minutes**.
3. **Rider Demographics**: **78.5% (4,118,394 trips)** of riders are annual Citi Bike subscribers (`member`), while **21.5% (1,123,709 trips)** are `casual` users. However, casual riders register significantly higher average trip durations (**17.8 minutes**) compared to commuters (**11.6 minutes**).
4. **Bike Fleet Technology**: **73.1% (3,830,529 trips)** of all departures used **electric bikes**, while **26.9% (1,411,574 trips)** used classic dock bikes—highlighting massive rider adoption of pedal-assist technology.

---

## 8. Geographic Hotspot Analysis & Classification

Analysis of **2,386 unique pickup stations** across NYC revealed severe spatial concentration of demand in Midtown Manhattan, Lower Manhattan, and the Brooklyn waterfront.

Stations were classified into 3 operational tiers using statistical quantiles:
- **High Demand (Top 20%)**: Stations with $\ge 4,020$ total monthly departures (**480 stations**).
- **Medium Demand (Middle 50%)**: Stations with between $415$ and $4,019$ departures (**1,193 stations**).
- **Low Demand (Bottom 30%)**: Peripheral/suburban docks with $< 415$ departures (**713 stations**).

### Top 5 Busiest Transit Hubs (August 2026)
1. **W 21 St & 6 Ave**: 17,727 departures (23.08 trips/hr avg)
2. **Pier 61 at Chelsea Piers**: 16,752 departures (21.81 trips/hr avg)
3. **E 17 St & Broadway**: 16,030 departures (20.87 trips/hr avg)
4. **7 Ave & Central Park South**: 15,312 departures (19.94 trips/hr avg)
5. **W 31 St & 7 Ave**: 15,189 departures (19.78 trips/hr avg)

---

## 9. Machine Learning Demand Prediction (Spark MLlib)

### Target Formulation & Zero-Demand Representation
To prevent upward-biased predictions caused by only observing positive departures, an hourly observation grid was constructed for the top commuter stations. Inactive hours were explicitly cross-joined and imputed with `demand = 0`, producing **108,000 modeling observations** across 150 stations over 30 days (720 hours per station).

### Training-Period Station Selection & Unique Station Identification
- **Selection Isolation**: To strictly prevent lookahead data leakage in candidate station selection, top commuter stations are ranked and chosen **exclusively using training-period trip counts (`2026-08-02 <= pickup_date <= 2026-08-23`)**. The broader hotspot analysis still evaluates all 2,386 stations across the entire month. The number of stations modeled is configurable via `TOP_N_STATIONS` in `src/config.py` (default: `150`).
- **Station Identification**: Grouped and partitioned by `start_station_id` (the physical dock identifier). In Citi Bike data, station names can be ambiguous (44 station names map to multiple IDs across NYC, such as `Clinton St & Grand St` which spans 2 distinct IDs). Partitioning by `start_station_id` guarantees that lag computations never mix separate physical stations.

### Boundary Data & August 01 Warm-Up Strategy
- The raw dataset begins on July 31 at 02:10 EDT (Citi Bike monthly archive boundary), containing only 1,423 trips across NYC compared to 202,163 trips on August 1.
- Because hours 0–11 on July 31 have 0 records due to collection truncation, treating them as zero demand would artificially corrupt 24-hour lag features for August 1.
- **Warm-Up Implementation**: The chronological grid is constructed from August 1 to August 31. **August 01 serves as a 24-hour warm-up window** (3,600 station-hours) used exclusively to compute valid, verified historical lags for August 02 onwards.
- August 01 is dropped from the modeling dataset, ensuring that every training, validation, and test observation has 100% complete and uncorrupted lag features.

### Rolling 1-Hour-Ahead Prediction Paradigm
The predictive task is formulated as a **rolling 1-hour-ahead demand forecast**. For any station $s$ at target hour $t$, the system uses observed features up to $t - 1$ (including `lag_1h_demand` from $t - 1$ and `lag_24h_demand` from $t - 24$). This models a realistic operational scenario where station dispatchers have access to completed pickups from the prior hour before finalizing van rebalancing for the upcoming hour. It is explicitly not a multi-day recursive forecast made in advance without intermediate ground-truth updates.

### Feature Engineering (Strictly Historical)
- **Temporal**: `pickup_hour`, `day_of_week`, `day_of_month`, `is_weekend`
- **Cyclical**: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$
- **Spatial**: Station `latitude`, `longitude`
- **Historical Lag Features**:
  - `lag_1h_demand`: Pickups at the station in the immediate prior hour ($t - 1$).
  - `lag_24h_demand`: Pickups at the station at the exact same hour yesterday ($t - 24$).

### Strict Chronological Splitting
- **Warm-Up Window**: August 01 (3,600 observations across 150 stations used for lag initialization)
- **Training Set**: August 02 – August 23 (79,200 observations, 22 days, 73.3%)
- **Validation Set**: August 24 – August 27 (14,400 observations, 4 days, 13.3%)
- **Test Set**: August 28 – August 31 (14,400 observations, 4 days, 13.3%)
- **Total Modeled Observations**: **108,000 station-hours** ($79,200 + 14,400 + 14,400 = 108,000$, 30 days × 24 hours = 720 hours per station).

### Actual Model Benchmark Results (Validation Set: August 24 – August 27)

| Algorithm | Validation RMSE | Validation MAE | Validation $R^2$ | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 7.4493 | 4.9671 | 0.6447 | Baseline linear model |
| **Decision Tree Regressor** | 7.1291 | 4.7301 | 0.6746 | Non-linear tree model |
| **Random Forest Regressor** | 6.9214 | 4.5505 | 0.6932 | Bagging ensemble |
| **Gradient-Boosted Trees (GBT)** | **6.6748** | **4.4474** | **0.7147** | **Selected Best Model** |

### Final Holdout Test Evaluation (August 28 – August 31)

| Model | Test RMSE | Test MAE | Test $R^2$ |
| :--- | :--- | :--- | :--- |
| **Gradient-Boosted Trees (GBT)** | **6.3087** | **4.3031** | **0.7159** |
| **Linear Regression (Baseline)** | **6.6223** | **4.5221** | **0.6870** |

**Interpretation**: Gradient-Boosted Trees achieved the strongest predictive accuracy, capturing sequential diurnal shifts, non-linear location dynamics, and demand momentum. GBT reduced Test RMSE by **0.3136 trips/hr**, and R² increased by 0.0289 (2.89 percentage points) compared with the Linear Regression baseline without lookahead or boundary leakage.

### Feature Importance Analysis (Gradient-Boosted Trees)

| Rank | Feature | Importance | Relative Share | Description |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `lag_1h_demand` | 0.4080 | 40.80% | Pickups at the station in immediate prior hour ($t-1$) |
| 2 | `lag_24h_demand` | 0.1007 | 10.07% | Pickups at the station at exact same hour yesterday ($t-24$) |
| 3 | `pickup_hour` | 0.0869 | 8.69% | Hour of departure (0 to 23) |
| 4 | `day_of_week` | 0.0823 | 8.23% | Day of the week index (1 = Sunday, 7 = Saturday) |
| 5 | `hour_cos` | 0.0731 | 7.31% | Cosine transformation of diurnal cycle |
| 6 | `day_of_month` | 0.0688 | 6.88% | Calendar day (1 to 31) |
| 7 | `hour_sin` | 0.0622 | 6.22% | Sine transformation of diurnal cycle |
| 8 | `is_weekend` | 0.0597 | 5.97% | Binary flag for Saturday / Sunday |
| 9 | `latitude` | 0.0350 | 3.50% | Station geographic latitude |
| 10 | `longitude` | 0.0232 | 2.32% | Station geographic longitude |

> **Methodological Note on Feature Importance**: Feature importance in Spark MLlib tree ensembles measures the normalized total reduction in impurity (variance) contributed by each feature across split nodes. It reflects **predictive association** within the trained ensemble rather than direct physical or causal mechanisms. Recent demand (`lag_1h`) and diurnal periodicity (`lag_24h` and `pickup_hour`) together account for over 59.5% of total predictive split power.

---

## 10. Interactive Streamlit Dashboard

The project includes an interactive web dashboard (`dashboard/app.py`):
1. **Dashboard Overview**: Executive KPIs, overall daily demand trends, and diurnal transit splits.
2. **Demand Analysis**: Interactive charts comparing weekday vs. weekend patterns, member vs. casual usage, and bike technology preferences.
3. **Geographic Hotspot Analysis**: Dynamic PyDeck 3D map rendering all 2,386 stations with interactive radius/color demand tier classifications and hover tooltips.
4. **Demand Prediction & ML**: Ground-truth vs. predicted demand curves, model comparison benchmark visualizer, Gradient-Boosted Trees feature importance rankings, and an interactive station demand forecast lookup tool.

---

## 11. How to Run the Project

### Prerequisites
- **Python**: 3.10 or higher
- **Java**: Java 8, 11, or 17 (verified with `java -version`)

### Dataset Placement
If cloning the repository without the raw CSV files, place the 6 monthly Citi Bike CSV files in the `dataset/` directory:
```
dataset/
├── 202608-citibike-tripdata_1.csv
├── 202608-citibike-tripdata_2.csv
├── 202608-citibike-tripdata_3.csv
├── 202608-citibike-tripdata_4.csv
├── 202608-citibike-tripdata_5.csv
└── 202608-citibike-tripdata_6.csv
```
*(The ingestion module will automatically validate schema consistency, column headers, and non-empty files before proceeding.)*

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run PySpark ETL Pipeline
Loads all 6 CSV files, cleans 5.24M records, and writes compressed Parquet:
```bash
python -m src.data_cleaning
```

### Step 3: Run Spark SQL Demand & Hotspot Analysis
Computes temporal aggregations and classifies 2,386 stations:
```bash
python -m src.demand_analysis
python -m src.hotspot_analysis
```

### Step 4: Train & Benchmark MLlib Models
Builds zero-demand hourly grid and trains Linear Regression, Decision Tree, Random Forest, and GBT:
```bash
python -m src.model_training
```

### Step 5: Generate Visualizations
Generates all 11 publication charts in `visualizations/`:
```bash
python -m src.generate_visualizations
```

### Step 6: Launch Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

> **Execution Logging**: All data ingestion, ETL cleaning, hotspot analysis, and model training events are logged with timestamps, elapsed durations, and row counts to both the console and `logs/pipeline.log`.

---

## 12. Limitations & Future Work

### Current Limitations
1. **Weather Exogenous Factors**: Extreme weather events (rain, heatwaves) impact micro-mobility demand but are not present in Citi Bike trip logs.
2. **Single-Month Observation Window**: August data reflects peak summer usage; winter seasonal dynamics are unobserved.
3. **Local Spark Execution**: While PySpark code is cluster-ready, execution was performed in `local[*]` mode on a single machine due to resource availability.

### Future Work
- Ingest real-time Open-Meteo weather APIs to join temperature and precipitation features.
- Deploy the ETL pipeline to a multi-node Databricks or AWS EMR cluster.
- Implement streaming demand ingestion using Apache Kafka and Spark Structured Streaming.

---

## 13. License & Academic Attribution
Developed as an academic BTech Computer Science & Engineering project. Open sourced under the [MIT License](LICENSE).
