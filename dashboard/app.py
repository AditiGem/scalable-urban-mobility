import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# ----------------------------------------------------
# 1. PAGE CONFIG & GLOBAL PATHS
# ----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data"
VISUALIZATIONS_DIR = ROOT_DIR / "visualizations"

st.set_page_config(
    page_title="Urban Mobility Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# 2. MODERN SAAS DARK THEME STYLING (CSS)
# ----------------------------------------------------
SAAS_CSS = """
<style>
/* Import Inter font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #CBD5E1;
}

/* App Background */
.stApp {
    background-color: #0A0E17;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Adjust Main Padding */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1440px;
}

/* Top Dashboard Header */
.dash-header {
    background: linear-gradient(180deg, #131E35 0%, #0F172A 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
}

.dash-header-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.dash-header-subtitle {
    font-size: 0.875rem;
    color: #94A3B8;
    margin: 0.25rem 0 0 0;
    font-weight: 400;
}

.dash-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.dash-badge {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 6px;
    padding: 0.35rem 0.65rem;
    font-size: 0.75rem;
    color: #E2E8F0;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
}
.badge-dot.cyan { background-color: #06B6D4; box-shadow: 0 0 8px #06B6D4; }
.badge-dot.teal { background-color: #14B8A6; box-shadow: 0 0 8px #14B8A6; }
.badge-dot.blue { background-color: #38BDF8; box-shadow: 0 0 8px #38BDF8; }

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: linear-gradient(180deg, #131C30 0%, #0F172A 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.kpi-card:hover {
    border-color: rgba(6, 182, 212, 0.35);
    transform: translateY(-2px);
}

.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.kpi-val {
    font-size: 1.85rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.kpi-sub {
    font-size: 0.775rem;
    color: #64748B;
    margin-top: 0.4rem;
}

.kpi-pill {
    font-size: 0.7rem;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    font-weight: 600;
}
.kpi-pill.cyan { background: rgba(6, 182, 212, 0.15); color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.3); }
.kpi-pill.green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }

/* Chart Containers */
.chart-card {
    background: #111C30;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.chart-card-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}

.chart-card-subtitle {
    font-size: 0.775rem;
    color: #94A3B8;
    margin-bottom: 0.75rem;
}

/* Sidebar Info Cards */
.sidebar-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.85rem;
    margin-top: 1rem;
    font-size: 0.8rem;
    line-height: 1.45;
}

.sidebar-box-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #38BDF8;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

/* Callout Info Banner */
.info-banner {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(14, 165, 233, 0.25);
    border-left: 4px solid #06B6D4;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 1.25rem;
    font-size: 0.825rem;
    color: #CBD5E1;
    line-height: 1.5;
}

/* Custom Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background-color: transparent;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 6px;
    color: #94A3B8;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 1rem;
    border: 1px solid transparent;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(6, 182, 212, 0.12) !important;
    color: #22D3EE !important;
    border: 1px solid rgba(6, 182, 212, 0.35) !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    overflow: hidden;
}
</style>
"""
st.markdown(SAAS_CSS, unsafe_allow_html=True)

# ----------------------------------------------------
# 3. ROBUST DATA LOADING
# ----------------------------------------------------
@st.cache_data
def load_data():
    errors = []
    bundle = {}

    try:
        with open(REPORTS_DIR / "etl_audit_report.json", "r", encoding="utf-8") as f:
            bundle["etl_audit"] = json.load(f)
    except Exception as e:
        errors.append(f"ETL report missing: {e}")

    try:
        with open(REPORTS_DIR / "analytics_summary.json", "r", encoding="utf-8") as f:
            bundle["analytics"] = json.load(f)
    except Exception as e:
        errors.append(f"Analytics report missing: {e}")

    try:
        with open(REPORTS_DIR / "hotspot_summary.json", "r", encoding="utf-8") as f:
            bundle["hotspots"] = json.load(f)
    except Exception as e:
        errors.append(f"Hotspot report missing: {e}")

    try:
        with open(REPORTS_DIR / "model_evaluation_metrics.json", "r", encoding="utf-8") as f:
            bundle["metrics"] = json.load(f)
    except Exception as e:
        errors.append(f"Model metrics report missing: {e}")

    try:
        bundle["stations_df"] = pd.read_csv(PROCESSED_DATA_DIR / "station_hotspots.csv")
    except Exception as e:
        errors.append(f"Station hotspots CSV missing: {e}")

    try:
        bundle["test_preds_df"] = pd.read_parquet(PROCESSED_DATA_DIR / "test_predictions_sample.parquet")
    except Exception as e:
        errors.append(f"Test predictions parquet missing: {e}")

    try:
        fi_path = REPORTS_DIR / "gbt_feature_importance.json"
        if fi_path.exists():
            with open(fi_path, "r", encoding="utf-8") as f:
                bundle["feature_importance"] = json.load(f)
        else:
            bundle["feature_importance"] = []
    except Exception:
        bundle["feature_importance"] = []

    return bundle, errors

data_bundle, loading_errors = load_data()

# ----------------------------------------------------
# 4. REUSABLE PLOTLY SAAS THEME
# ----------------------------------------------------
def apply_saas_theme(fig, height=350):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=35, b=20),
        height=height,
        font=dict(family="Inter, -apple-system, sans-serif", color="#94A3B8", size=11),
        title=dict(font=dict(size=13, color="#F8FAFC", family="Inter, -apple-system, sans-serif"), x=0.01),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#94A3B8"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(size=10, color="#94A3B8")
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=11,
            font_family="Inter, -apple-system, sans-serif",
            font_color="#F8FAFC",
            bordercolor="#06B6D4"
        )
    )
    return fig

# ----------------------------------------------------
# 5. SIDEBAR NAVIGATION & SYSTEM INFO
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; padding-top: 0.25rem;">
        <div style="background: linear-gradient(135deg, #06B6D4, #3B82F6); width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem; color: white;">⚡</div>
        <div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #F8FAFC; letter-spacing: -0.01em;">Urban Mobility</div>
            <div style="font-size: 0.7rem; color: #06B6D4; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if loading_errors:
        st.error("⚠️ Incomplete Datasets Detected")
        for err in loading_errors:
            st.caption(err)
        st.stop()

    etl_audit = data_bundle["etl_audit"]
    analytics = data_bundle["analytics"]
    hotspots = data_bundle["hotspots"]
    metrics = data_bundle["metrics"]
    stations_df = data_bundle["stations_df"]
    test_preds_df = data_bundle["test_preds_df"]
    feature_importance = data_bundle["feature_importance"]

    section = st.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Demand Analysis",
            "Geographic Hotspot Analysis",
            "Demand Prediction & ML"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # System Architecture Information Card
    st.markdown("""
    <div class="sidebar-box">
        <div class="sidebar-box-title">System Architecture</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="color: #94A3B8;">Engine:</span>
            <span style="color: #F8FAFC; font-weight: 500;">Apache Spark 3.5.9</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="color: #94A3B8;">Cluster Mode:</span>
            <span style="color: #F8FAFC; font-weight: 500;">Local (local[*])</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="color: #94A3B8;">Ingested Trips:</span>
            <span style="color: #F8FAFC; font-weight: 500;">5,246,236</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="color: #94A3B8;">Cleaned Trips:</span>
            <span style="color: #F8FAFC; font-weight: 500;">5,242,103 (99.9%)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="color: #94A3B8;">Parquet Storage:</span>
            <span style="color: #34D399; font-weight: 600;">250 MB (-74.4%)</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
            <span style="color: #94A3B8;">Hotspot Docks:</span>
            <span style="color: #F8FAFC; font-weight: 500;">2,386 stations</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #94A3B8;">ML Modeling Docks:</span>
            <span style="color: #F8FAFC; font-weight: 500;">Top 150 Hubs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 1rem; padding: 0.5rem; font-size: 0.725rem; color: #64748B; text-align: center;">
        BTech CSE Placement Project<br/>Apache Spark • MLlib • Streamlit
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 6. GLOBAL DASHBOARD HEADER
# ----------------------------------------------------
st.markdown("""
<div class="dash-header">
    <div>
        <div class="dash-header-title">
            <span style="color: #06B6D4;">Urban Mobility Intelligence</span>
        </div>
        <div class="dash-header-subtitle">
            Scalable demand forecasting and station hotspot analytics powered by Apache Spark MLlib
        </div>
    </div>
    <div class="dash-badges">
        <div class="dash-badge"><span class="badge-dot cyan"></span><b>Dataset:</b> August 2026</div>
        <div class="dash-badge"><span class="badge-dot teal"></span><b>Model:</b> Gradient-Boosted Trees</div>
        <div class="dash-badge"><span class="badge-dot blue"></span><b>Horizon:</b> Rolling 1h Ahead</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================================================
# PAGE 1: DASHBOARD OVERVIEW
# ====================================================
if section == "Dashboard Overview":
    # 4 Prominent KPI Cards
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Total Trips</span>
                <span class="kpi-pill green">99.92% Retained</span>
            </div>
            <div class="kpi-val">5.24M</div>
            <div class="kpi-sub">5,242,103 clean trips in Parquet</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Pickup Stations</span>
                <span class="kpi-pill cyan">2,386 Docks</span>
            </div>
            <div class="kpi-val">2,386</div>
            <div class="kpi-sub">NYC Metropolitan transit network</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Test R² Score</span>
                <span class="kpi-pill green">+2.89% pts</span>
            </div>
            <div class="kpi-val">0.7159</div>
            <div class="kpi-sub">+0.0289 vs Linear Regression baseline</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Test RMSE</span>
                <span class="kpi-pill cyan">Champion GBT</span>
            </div>
            <div class="kpi-val">6.3087</div>
            <div class="kpi-sub">Trips / station-hour (-0.31 vs Baseline)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Balanced Two-Column Layout for Charts
    col_row1_left, col_row1_right = st.columns(2)

    with col_row1_left:
        st.markdown('<div class="chart-card-title">Mobility Demand Trend (August 2026)</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-subtitle">Daily aggregate trip volume across the NYC Citi Bike network</div>', unsafe_allow_html=True)
        daily_trend_df = pd.DataFrame(analytics["temporal_analysis"]["daily_trend"])
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_trend_df["pickup_date"],
            y=daily_trend_df["total_trips"],
            mode="lines+markers",
            line=dict(color="#06B6D4", width=2.5),
            marker=dict(size=5, color="#22D3EE"),
            name="Daily Pickups",
            hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>"
        ))
        fig_daily = apply_saas_theme(fig_daily, height=310)
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_row1_right:
        st.markdown('<div class="chart-card-title">Diurnal Time-of-Day Transit Split</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-subtitle">Trip distribution aggregated across standardized commute windows</div>', unsafe_allow_html=True)
        diurnal_df = pd.DataFrame(analytics["temporal_analysis"]["diurnal_buckets"])
        fig_diurnal = px.bar(
            diurnal_df, x="time_period", y="total_trips",
            color="time_period",
            color_discrete_sequence=["#06B6D4", "#0EA5E9", "#3B82F6", "#6366F1"],
            text_auto=".2s"
        )
        fig_diurnal = apply_saas_theme(fig_diurnal, height=310)
        fig_diurnal.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total Pickups")
        st.plotly_chart(fig_diurnal, use_container_width=True)

    col_row2_left, col_row2_right = st.columns(2)

    with col_row2_left:
        st.markdown('<div class="chart-card-title">Rider Demographics Volume Share</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-subtitle">Annual Citi Bike subscribers vs casual recreational users</div>', unsafe_allow_html=True)
        rider_df = pd.DataFrame(analytics["rider_and_bike_analysis"]["rider_summary"])
        fig_rider = px.pie(
            rider_df, names="member_casual", values="total_trips",
            color="member_casual",
            color_discrete_map={"member": "#06B6D4", "casual": "#F59E0B"},
            hole=0.55
        )
        fig_rider.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Trips: %{value:,}<br>Share: %{percent}<extra></extra>"
        )
        fig_rider = apply_saas_theme(fig_rider, height=300)
        st.plotly_chart(fig_rider, use_container_width=True)

    with col_row2_right:
        st.markdown('<div class="chart-card-title">Bike Fleet Technology Adoption</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-subtitle">Pedal-assist electric bikes dominate mechanical classic docks</div>', unsafe_allow_html=True)
        bike_df = pd.DataFrame(analytics["rider_and_bike_analysis"]["bike_summary"])
        fig_bike = px.bar(
            bike_df, x="rideable_type", y="total_trips",
            color="rideable_type",
            color_discrete_map={"electric_bike": "#14B8A6", "classic_bike": "#3B82F6"},
            text_auto=".2s"
        )
        fig_bike = apply_saas_theme(fig_bike, height=300)
        fig_bike.update_layout(showlegend=False, xaxis_title="", yaxis_title="Departures")
        st.plotly_chart(fig_bike, use_container_width=True)

    # Top 5 Transit Hubs Quick View
    st.markdown('<div class="chart-card-title" style="margin-top: 0.5rem;">Top Transit Hubs Across Network</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-subtitle">Highest volume departure stations during August 2026</div>', unsafe_allow_html=True)
    top_5_df = pd.DataFrame(hotspots["top_10_stations"][:5])[[
        "start_station_name", "total_pickups", "avg_hourly_demand", "avg_trip_duration_min", "demand_classification"
    ]].rename(columns={
        "start_station_name": "Station Name",
        "total_pickups": "Monthly Pickups",
        "avg_hourly_demand": "Avg Pickups / Hr",
        "avg_trip_duration_min": "Avg Duration (min)",
        "demand_classification": "Tier"
    })
    st.dataframe(top_5_df, use_container_width=True, hide_index=True)

    # Pipeline Storage & Compression Banner
    st.markdown(f"""
    <div class="info-banner" style="margin-top: 1rem;">
        <b>PySpark ETL Architecture:</b> Ingested all 6 CSV raw archives ({etl_audit['ingestion']['total_size_mb']:.1f} MB, 
        {etl_audit['ingestion']['total_records']:,} raw records). Filtered 4,133 anomalous trips (<1 min accidental unlocks, 
        >24 hr unreturned bikes, coordinate outliers). Cleaned data saved in 8 Snappy-compressed Parquet partitions 
        (<b>250 MB</b>), achieving a verified <b>74.40% storage reduction</b>.
    </div>
    """, unsafe_allow_html=True)

# ====================================================
# PAGE 2: DEMAND ANALYSIS
# ====================================================
elif section == "Demand Analysis":
    st.markdown('<div class="dash-header-title" style="font-size: 1.25rem; margin-bottom: 0.25rem;">Temporal Dynamics & User Demographics</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-header-subtitle" style="margin-bottom: 1.25rem;">Comprehensive Spark SQL queries evaluating diurnal cycles, weekday-weekend variations, and vehicle technology</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Hourly & Day-of-Week Cycles",
        "Weekday vs. Weekend Dynamics",
        "Rider Profiles & Bike Technology"
    ])

    with tab1:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown('<div class="chart-card-title">24-Hour Diurnal Demand Distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card-subtitle">Distinct morning commute peak (08:00) and evening rush (17:00–18:00)</div>', unsafe_allow_html=True)
            hourly_df = pd.DataFrame(analytics["temporal_analysis"]["hourly"])
            # Highlight rush hours
            colors = ["#06B6D4" if h in [8, 17, 18] else "#1E3A5F" for h in hourly_df["pickup_hour"]]
            fig_h = go.Figure(go.Bar(
                x=hourly_df["pickup_hour"],
                y=hourly_df["total_trips"],
                marker_color=colors,
                hovertemplate="Hour %{x}:00<br>Departures: %{y:,}<extra></extra>"
            ))
            fig_h = apply_saas_theme(fig_h, height=330)
            fig_h.update_layout(xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", tick0=0, dtick=1))
            st.plotly_chart(fig_h, use_container_width=True)

        with col_t2:
            st.markdown('<div class="chart-card-title">Weekly Departure Distribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card-subtitle">Mid-week commute volume remains elevated vs weekend leisure rides</div>', unsafe_allow_html=True)
            dow_df = pd.DataFrame(analytics["temporal_analysis"]["day_of_week"])
            fig_dow = px.bar(
                dow_df, x="day_name", y="total_trips",
                color="total_trips",
                color_continuous_scale=["#1E3A5F", "#06B6D4", "#38BDF8"],
                text_auto=".2s"
            )
            fig_dow = apply_saas_theme(fig_dow, height=330)
            fig_dow.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Total Pickups")
            st.plotly_chart(fig_dow, use_container_width=True)

    with tab2:
        ww_df = pd.DataFrame(analytics["temporal_analysis"]["weekday_weekend"])
        
        # Comparative mini-KPI cards
        col_w_kpi1, col_w_kpi2 = st.columns(2)
        with col_w_kpi1:
            st.markdown("""
            <div class="kpi-card" style="margin-bottom: 1rem;">
                <div class="kpi-label"><span>Weekday Average</span><span class="kpi-pill cyan">Commuter Profile</span></div>
                <div class="kpi-val" style="color: #38BDF8;">171,448</div>
                <div class="kpi-sub">Trips/day • Avg duration: <b>12.4 mins</b> (Fast direct transit)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_w_kpi2:
            st.markdown("""
            <div class="kpi-card" style="margin-bottom: 1rem;">
                <div class="kpi-label"><span>Weekend Average</span><span class="kpi-pill green">Recreational Profile</span></div>
                <div class="kpi-val" style="color: #34D399;">148,849</div>
                <div class="kpi-sub">Trips/day • Avg duration: <b>14.9 mins</b> (+20.2% longer duration)</div>
            </div>
            """, unsafe_allow_html=True)

        col_w1, col_w2 = st.columns([1, 1.2])
        with col_w1:
            st.markdown('<div class="chart-card-title">Average Daily Volume Comparison</div>', unsafe_allow_html=True)
            fig_ww = px.bar(
                ww_df, x="day_type", y="avg_daily_trips",
                color="day_type",
                color_discrete_map={"Weekday": "#0EA5E9", "Weekend": "#10B981"},
                text_auto=".2s"
            )
            fig_ww = apply_saas_theme(fig_ww, height=300)
            fig_ww.update_layout(showlegend=False, xaxis_title="", yaxis_title="Average Pickups / Day")
            st.plotly_chart(fig_ww, use_container_width=True)

        with col_w2:
            st.markdown('<div class="chart-card-title">Commute Dynamics & Modal Contrast</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-banner" style="margin-top: 0.5rem; font-size: 0.85rem;">
                <b>Commuter vs. Leisure Behaviour:</b><br/>
                • <b>Weekday Traffic:</b> Exhibits a pronounced bimodal peak at 08:00 and 17:00–18:00. Commuters emphasize direct point-to-point transit with minimal dwell times (avg 12.4 min).<br/><br/>
                • <b>Weekend Traffic:</b> Shifts to a smooth, unimodal distribution peaking broadly from 13:00 to 17:00. Trips register 20.2% longer durations (avg 14.9 min), concentrated around waterfront greenways and parks.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown('<div class="chart-card-title">Subscriber vs. Casual Rider Profile</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card-subtitle">Members generate 78.5% of trips; casual riders ride 53.4% longer</div>', unsafe_allow_html=True)
            rider_df = pd.DataFrame(analytics["rider_and_bike_analysis"]["rider_summary"])
            fig_r_bar = px.bar(
                rider_df, x="member_casual", y="avg_duration_min",
                color="member_casual",
                color_discrete_map={"member": "#06B6D4", "casual": "#F59E0B"},
                text_auto=".1f"
            )
            fig_r_bar = apply_saas_theme(fig_r_bar, height=310)
            fig_r_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Average Duration (Minutes)")
            st.plotly_chart(fig_r_bar, use_container_width=True)

        with col_r2:
            st.markdown('<div class="chart-card-title">Bike Technology Preference</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-card-subtitle">Electric pedal-assist bikes account for 73.1% of total rides</div>', unsafe_allow_html=True)
            bike_df = pd.DataFrame(analytics["rider_and_bike_analysis"]["bike_summary"])
            fig_b_bar = px.bar(
                bike_df, x="rideable_type", y="total_trips",
                color="rideable_type",
                color_discrete_map={"electric_bike": "#14B8A6", "classic_bike": "#3B82F6"},
                text_auto=".2s"
            )
            fig_b_bar = apply_saas_theme(fig_b_bar, height=310)
            fig_b_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total Departures")
            st.plotly_chart(fig_b_bar, use_container_width=True)

# ====================================================
# PAGE 3: GEOGRAPHIC HOTSPOT ANALYSIS
# ====================================================
elif section == "Geographic Hotspot Analysis":
    st.markdown('<div class="dash-header-title" style="font-size: 1.25rem; margin-bottom: 0.25rem;">Geographic Pickup Hotspots & Station Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-header-subtitle" style="margin-bottom: 1rem;">Spatial density and reproducible quantile classification across 2,386 transit hubs in NYC</div>', unsafe_allow_html=True)

    # Filter Bar Container
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        tier_filter = st.multiselect(
            "Filter Stations by Demand Classification",
            options=["High Demand", "Medium Demand", "Low Demand"],
            default=["High Demand", "Medium Demand", "Low Demand"]
        )

    filtered_stations = stations_df[stations_df["demand_classification"].isin(tier_filter)]

    with col_f2:
        st.markdown(f"""
        <div style="display: flex; gap: 0.6rem; align-items: center; justify-content: flex-end; height: 100%; padding-top: 1.25rem;">
            <div class="dash-badge"><span class="badge-dot" style="background:#EF4444;"></span><b>High Tier:</b> ≥ 4,020 trips (480 docks)</div>
            <div class="dash-badge"><span class="badge-dot" style="background:#F59E0B;"></span><b>Med Tier:</b> 415–4,019 (1,193 docks)</div>
            <div class="dash-badge"><span class="badge-dot" style="background:#10B981;"></span><b>Low Tier:</b> < 415 (713 docks)</div>
        </div>
        """, unsafe_allow_html=True)

    # Color mapping for PyDeck map
    color_map = {
        "High Demand": [239, 68, 68, 200],    # Coral Red
        "Medium Demand": [245, 158, 11, 170], # Amber
        "Low Demand": [16, 185, 129, 130]     # Emerald
    }
    radius_map = {
        "High Demand": 110,
        "Medium Demand": 65,
        "Low Demand": 35
    }

    map_df = filtered_stations.copy()
    map_df["color"] = map_df["demand_classification"].map(color_map)
    map_df["radius"] = map_df["demand_classification"].map(radius_map)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True
    )

    view_state = pdk.ViewState(
        latitude=40.738,
        longitude=-73.985,
        zoom=11.4,
        pitch=25
    )

    tooltip = {
        "html": "<div style='font-family: Inter, sans-serif; padding: 4px;'>"
                "<b style='font-size: 13px; color: #38BDF8;'>{start_station_name}</b><br/>"
                "<span style='color: #94A3B8;'>Departures:</span> <b>{total_pickups}</b><br/>"
                "<span style='color: #94A3B8;'>Avg Hourly Demand:</span> <b>{avg_hourly_demand} trips/hr</b><br/>"
                "<span style='color: #94A3B8;'>Classification:</span> <b style='color: #F8FAFC;'>{demand_classification}</b>"
                "</div>",
        "style": {
            "backgroundColor": "#0F172A",
            "color": "#F8FAFC",
            "border": "1px solid rgba(56, 189, 248, 0.3)",
            "borderRadius": "8px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.5)"
        }
    }

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="dark"
        )
    )

    # Top 10 Busiest Stations Table
    st.markdown('<div class="chart-card-title" style="margin-top: 1rem;">Top 10 High-Traffic Transit Hubs (August 2026)</div>', unsafe_allow_html=True)
    top_10_table = pd.DataFrame(hotspots["top_10_stations"])[[
        "start_station_name", "total_pickups", "avg_hourly_demand", "avg_trip_duration_min", "demand_classification"
    ]].rename(columns={
        "start_station_name": "Station Name",
        "total_pickups": "Monthly Departures",
        "avg_hourly_demand": "Avg Departures / Hr",
        "avg_trip_duration_min": "Avg Duration (min)",
        "demand_classification": "Demand Tier"
    })
    st.dataframe(top_10_table, use_container_width=True, hide_index=True)

# ====================================================
# PAGE 4: DEMAND PREDICTION & ML
# ====================================================
elif section == "Demand Prediction & ML":
    st.markdown('<div class="dash-header-title" style="font-size: 1.25rem; margin-bottom: 0.25rem;">Station-Level Hourly Demand Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-header-subtitle" style="margin-bottom: 1.25rem;">Spark MLlib regression models benchmarked on strict chronological holdout partitions</div>', unsafe_allow_html=True)

    # Champion Model Metric Callouts
    st.markdown("""
    <div class="kpi-grid">
        <div class="kpi-card" style="border-color: rgba(6, 182, 212, 0.4);">
            <div class="kpi-label"><span>Selected Champion</span><span class="kpi-pill cyan">Lowest Val RMSE</span></div>
            <div class="kpi-val" style="color: #22D3EE;">Gradient-Boosted Trees</div>
            <div class="kpi-sub">Spark MLlib GBTRegressor</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span>Holdout Test RMSE</span><span class="kpi-pill green">-0.3136 vs Baseline</span></div>
            <div class="kpi-val">6.3087</div>
            <div class="kpi-sub">Trips per station-hour</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span>Holdout Test MAE</span><span class="kpi-pill green">-0.2190 vs Baseline</span></div>
            <div class="kpi-val">4.3031</div>
            <div class="kpi-sub">Mean absolute deviation</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label"><span>Holdout Test R² Score</span><span class="kpi-pill cyan">+2.89% pts</span></div>
            <div class="kpi-val">0.7159</div>
            <div class="kpi-sub">R² increased by 0.0289 vs Baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Verified Test Set Benchmark Table & Methodology Card
    col_test1, col_test2 = st.columns([1, 1.3])
    with col_test1:
        st.markdown('<div class="chart-card-title">Verified Holdout Test Results (Aug 28–31)</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card-subtitle">Evaluated on unseen chronological holdout partition (14,400 observations)</div>', unsafe_allow_html=True)
        test_eval_table = pd.DataFrame([
            {"Model": "Gradient-Boosted Trees (Champion)", "RMSE": 6.3087, "MAE": 4.3031, "R²": 0.7159},
            {"Model": "Linear Regression (Baseline)", "RMSE": 6.6223, "MAE": 4.5221, "R²": 0.6870}
        ])
        st.dataframe(test_eval_table, use_container_width=True, hide_index=True)

    with col_test2:
        st.markdown('<div class="chart-card-title">Predictive Formulation & Protocol</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-banner" style="margin-top: 0.25rem;">
            <b>Rolling 1-Hour-Ahead Operational Protocol:</b><br/>
            • <b>Prediction Horizon:</b> For each station at target hour <i>t</i>, actual observed pickups from the immediate prior hour (<i>t-1</i>) and previous day (<i>t-24</i>) are available to schedule van rebalancing.<br/>
            • <b>No Multi-Day Guessing:</b> This is strictly rolling 1-hour-ahead forecasting, not a 4-day forecast generated in advance without intermediate observations.<br/>
            • <b>Zero Leakage:</b> Stations selected strictly on training data (Aug 02–23). August 01 served as a 24h warm-up window to prevent boundary distortion.
        </div>
        """, unsafe_allow_html=True)

    # Validation Model Comparison Section
    st.markdown('<div class="chart-card-title" style="margin-top: 1.5rem;">Spark MLlib Model Benchmark (Validation Set: Aug 24–27)</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-subtitle">Hyperparameter tuning and algorithm selection on 14,400 validation observations</div>', unsafe_allow_html=True)

    col_bench1, col_bench2 = st.columns([1.1, 1.4])
    with col_bench1:
        comp_records = []
        for m_name, vals in metrics["model_comparison"].items():
            comp_records.append({
                "Algorithm": m_name,
                "Val RMSE": vals["validation"]["rmse"],
                "Val MAE": vals["validation"]["mae"],
                "Val R²": vals["validation"]["r2"]
            })
        comp_df = pd.DataFrame(comp_records).sort_values("Val RMSE")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.caption("R² increased by 0.0289 (2.89 percentage points) compared with the Linear Regression baseline.")

    with col_bench2:
        fig_comp = px.bar(
            comp_df, x="Algorithm", y="Val RMSE",
            color="Val R²",
            color_continuous_scale=["#1E3A5F", "#06B6D4", "#38BDF8"],
            text_auto=".2f"
        )
        fig_comp = apply_saas_theme(fig_comp, height=270)
        fig_comp.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="Validation RMSE")
        st.plotly_chart(fig_comp, use_container_width=True)

    # Feature Importance Section
    st.markdown('<div class="chart-card-title" style="margin-top: 1.5rem;">Feature Importance Profile (Gradient-Boosted Trees)</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-subtitle">Relative variance reduction across boosted decision tree splits (predictive association, not causal mechanism)</div>', unsafe_allow_html=True)

    if feature_importance:
        fi_df = pd.DataFrame(feature_importance)
        col_fi1, col_fi2 = st.columns([1.4, 1])

        with col_fi1:
            fi_sorted = fi_df.sort_values("importance", ascending=True)
            fig_fi = px.bar(
                fi_sorted, x="importance", y="feature",
                orientation="h",
                text=[f"{v*100:.1f}%" for v in fi_sorted["importance"]],
                color="importance",
                color_continuous_scale=["#1E3A5F", "#0EA5E9", "#06B6D4"]
            )
            fig_fi = apply_saas_theme(fig_fi, height=340)
            fig_fi.update_layout(coloraxis_showscale=False, xaxis_title="Split Importance", yaxis_title="")
            st.plotly_chart(fig_fi, use_container_width=True)

        with col_fi2:
            fi_display = fi_df.copy()
            fi_display["Relative Share"] = (fi_display["importance"] * 100).round(2).astype(str) + "%"
            st.dataframe(fi_display.rename(columns={"feature": "Feature", "importance": "Importance Weight"}), use_container_width=True, hide_index=True)
            st.caption("Recent demand momentum (`lag_1h`: 40.8%) and diurnal periodicity (`lag_24h` + `pickup_hour`: 18.8%) drive over 59.5% of predictive performance.")

    # Actual vs Predicted Demand Curve
    st.markdown('<div class="chart-card-title" style="margin-top: 1.5rem;">Actual vs. Predicted Hourly Demand (Chronological Holdout Set)</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-subtitle">Averaged across 150 modeled transit hubs over the 4-day holdout evaluation window (Aug 28–31)</div>', unsafe_allow_html=True)

    hourly_actual_pred = test_preds_df.groupby("pickup_hour")[["demand", "predicted_demand"]].mean().reset_index()
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=hourly_actual_pred["pickup_hour"], y=hourly_actual_pred["demand"],
        mode="lines+markers", name="Actual Demand (Ground Truth)",
        line=dict(color="#38BDF8", width=2.5),
        marker=dict(size=6, color="#0EA5E9"),
        hovertemplate="Hour %{x}:00<br>Actual: %{y:.2f} trips<extra></extra>"
    ))
    fig_pred.add_trace(go.Scatter(
        x=hourly_actual_pred["pickup_hour"], y=hourly_actual_pred["predicted_demand"],
        mode="lines+markers", name="GBT Model Predictions",
        line=dict(color="#F43F5E", width=2.5, dash="dash"),
        marker=dict(size=5, color="#FB7185"),
        hovertemplate="Hour %{x}:00<br>Predicted: %{y:.2f} trips<extra></extra>"
    ))
    fig_pred = apply_saas_theme(fig_pred, height=330)
    fig_pred.update_layout(
        xaxis=dict(title="Hour of Day (0–23)", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title="Average Pickups / Station-Hour")
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Interactive Station Demand Lookup Tool
    st.markdown('<div class="chart-card-title" style="margin-top: 1.5rem;">Interactive Station Forecast Lookup</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card-subtitle">Query test-period ground truth vs GBT forecast for any modeled station</div>', unsafe_allow_html=True)

    station_list = sorted(test_preds_df["start_station_name"].unique())
    selected_station = st.selectbox("Select Station for Sample Predictions", station_list, label_visibility="collapsed")

    sample_station_data = test_preds_df[test_preds_df["start_station_name"] == selected_station].sort_values(["pickup_date", "pickup_hour"]).tail(24)
    st.dataframe(sample_station_data.rename(columns={
        "start_station_name": "Station",
        "pickup_date": "Date",
        "pickup_hour": "Hour",
        "demand": "Actual Pickups",
        "predicted_demand": "Predicted Pickups"
    }), use_container_width=True, hide_index=True)
