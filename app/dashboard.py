import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import pytz
import hopsworks
import yaml
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PK_TZ = pytz.timezone("Asia/Karachi")


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def aqi_style(aqi):
    if aqi <= 50:
        return "🟢", "Good", "#2ecc71"
    elif aqi <= 100:
        return "🟡", "Moderate", "#f1c40f"
    elif aqi <= 150:
        return "🟠", "Unhealthy (Sensitive)", "#e67e22"
    elif aqi <= 200:
        return "🔴", "Unhealthy", "#e74c3c"
    elif aqi <= 300:
        return "🟣", "Very Unhealthy", "#8e44ad"
    else:
        return "⚫", "Hazardous", "#2c3e50"


def alert_banner(aqi):
    if aqi > 200:
        st.error("🚨 Very Unhealthy — Everyone should avoid outdoor activity")
    elif aqi > 150:
        st.warning("⚠️ Unhealthy — Sensitive groups should avoid outdoor activity")
    elif aqi > 100:
        st.warning("🟠 Moderate — Sensitive groups may be affected")
    else:
        st.success("✅ Air quality is acceptable today")


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_forecast():
    preds = pd.read_csv(
        os.path.join(DATA_DIR, "latest_predictions.csv"),
        parse_dates=["timestamp"]
    )
    daily = pd.read_csv(os.path.join(DATA_DIR, "daily_summary.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    return preds, daily


@st.cache_data(ttl=3600)
def load_current_conditions():
    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs  = project.get_feature_store()
    fv  = fs.get_feature_view("aqi_features_fv", version=1)
    df  = fv.get_batch_data().sort_values("timestamp")
    return df.iloc[-1]


@st.cache_data(ttl=3600)
def load_model_metrics():
    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )
    mr     = project.get_model_registry()
    models = mr.get_models("aqi_predictor")

    rows = []
    for m in models:
        metrics = m.training_metrics or {}
        rows.append({
            "Model":   m.name,
            "Version": m.version,
            "RMSE":    metrics.get("rmse"),
            "MAE":     metrics.get("mae"),
            "R2":      metrics.get("r2"),
        })

    df   = pd.DataFrame(rows).dropna()
    best = df.sort_values("RMSE").iloc[0]
    return df, best


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🌍 AQI Karachi")
st.sidebar.caption("Air Quality Forecast System")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 3-Day Forecast", "📊 Model Performance",
     "🔬 EDA", "🧠 SHAP", "ℹ️ About"]
)

now_pk = datetime.now(PK_TZ)
st.sidebar.markdown("---")
st.sidebar.caption(f"🕒 {now_pk:%Y-%m-%d %H:%M} PKT")
st.sidebar.caption("🔄 Data updates every hour")
st.sidebar.caption("🤖 Model retrains daily")


# ==================================================
# PAGE 1 — HOME
# ==================================================
if page == "🏠 Home":
    st.title("🌆 Karachi Air Quality Dashboard")
    st.caption(f"📍 Karachi, Pakistan  |  🕒 {now_pk:%Y-%m-%d %H:%M} PKT")

    try:
        current     = load_current_conditions()
        current_aqi = float(current["aqi"])
        emoji, category, color = aqi_style(current_aqi)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                f"""
                <div style="border:2px solid {color}; border-radius:16px;
                            padding:30px; text-align:center; background:rgba(0,0,0,0.1);">
                    <p style="font-size:48px; margin:0;">{emoji}</p>
                    <h1 style="color:{color}; font-size:64px; margin:0;">{current_aqi:.0f}</h1>
                    <h3 style="color:{color};">{category}</h3>
                    <p style="color:#aaa; font-size:12px;">
                        Last updated: {current['timestamp']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            alert_banner(current_aqi)

            st.subheader("🌡 Current Conditions")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌡 Temperature", f"{current['temperature']:.1f} °C")
            c2.metric("💧 Humidity",    f"{current['humidity']:.0f} %")
            c3.metric("💨 Wind Speed",  f"{current['wind_speed']:.1f} m/s")
            c4.metric("📈 Pressure",    f"{current['pressure']:.0f} hPa")

            st.markdown("---")
            st.subheader("📊 Quick Stats")
            c1, c2, c3 = st.columns(3)
            c1.metric("PM2.5", f"{current['pm25']:.1f} µg/m³")
            c2.metric("PM10",  f"{current['pm10']:.1f} µg/m³")
            c3.metric("O3",    f"{current['o3']:.2f} µg/m³")

    except Exception as e:
        st.error(f"Could not load current conditions: {e}")


# ==================================================
# PAGE 2 — FORECAST
# ==================================================
elif page == "🔮 3-Day Forecast":
    st.title("🔮 72-Hour AQI Forecast")

    try:
        preds, daily = load_forecast()

        st.subheader("📅 3-Day Forecast")
        daily_show = daily.head(4)
        cols = st.columns(len(daily_show))

        for col, (_, r) in zip(cols, daily_show.iterrows()):
            emoji, category, color = aqi_style(r["avg_aqi"])
            date_label = pd.to_datetime(r["date"]).strftime("%b %d")
            with col:
                st.markdown(
                    f"""
                    <div style="border:2px solid {color}; padding:20px;
                                border-radius:12px; text-align:center;
                                background:rgba(0,0,0,0.15); margin-bottom:10px;">
                        <h3 style="color:white; margin:0;">{date_label}</h3>
                        <p style="font-size:36px; margin:8px 0;">{emoji}</p>
                        <h2 style="color:{color}; margin:0;">{r['avg_aqi']}</h2>
                        <p style="color:#ccc; font-size:13px; margin:4px 0;">{category}</p>
                        <hr style="border-color:#444; margin:8px 0;">
                        <p style="color:#aaa; font-size:12px; margin:0;">
                            Min: {r['min_aqi']} | Max: {r['max_aqi']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        avg_aqi = daily["avg_aqi"].mean()
        if avg_aqi > 200:
            st.error("🚨 Very Unhealthy forecast — Avoid outdoor activity")
        elif avg_aqi > 150:
            st.warning("⚠️ Unhealthy forecast — Sensitive groups stay indoors")
        elif avg_aqi > 100:
            st.warning("🟠 May affect sensitive groups over next 3 days")
        else:
            st.success("✅ Air quality forecast acceptable for next 3 days")

        st.subheader("📈 Hourly Forecast (6-hour steps)")

        marker_colors = []
        for aqi in preds["predicted_aqi"]:
            if aqi <= 50:      marker_colors.append("#2ecc71")
            elif aqi <= 100:   marker_colors.append("#f1c40f")
            elif aqi <= 150:   marker_colors.append("#e67e22")
            elif aqi <= 200:   marker_colors.append("#e74c3c")
            elif aqi <= 300:   marker_colors.append("#8e44ad")
            else:              marker_colors.append("#2c3e50")

        fig = go.Figure()
        fig.add_hrect(y0=0,   y1=50,  fillcolor="#2ecc71", opacity=0.05, line_width=0)
        fig.add_hrect(y0=50,  y1=100, fillcolor="#f1c40f", opacity=0.07, line_width=0)
        fig.add_hrect(y0=100, y1=150, fillcolor="#e67e22", opacity=0.08, line_width=0)
        fig.add_hrect(y0=150, y1=200, fillcolor="#e74c3c", opacity=0.08, line_width=0)
        fig.add_hrect(y0=200, y1=300, fillcolor="#8e44ad", opacity=0.08, line_width=0)

        fig.add_trace(go.Scatter(
            x=preds["timestamp"],
            y=preds["predicted_aqi"],
            mode="lines+markers",
            name="Predicted AQI",
            line=dict(width=3, color="royalblue"),
            marker=dict(size=10, color=marker_colors),
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>"
        ))

        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title="AQI",
            yaxis=dict(range=[0, max(preds["predicted_aqi"].max() + 30, 200)]),
            height=420,
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Forecast Breakdown")
        cat_counts = preds["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Total Hours"] = cat_counts["Count"] * 6
        cat_counts = cat_counts.drop(columns=["Count"])
        st.dataframe(cat_counts, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load forecast: {e}")
        st.info("Run `python inference/forecast.py` first")


# ==================================================
# PAGE 3 — MODEL PERFORMANCE
# ==================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    try:
        df, best = load_model_metrics()

        st.markdown(
            f"""
            <div style="border:2px solid #f1c40f; border-radius:12px;
                        padding:20px; margin-bottom:20px;
                        background:rgba(241,196,15,0.05);">
                <h3 style="color:#f1c40f;">
                    🏆 Best Model: {best['Model']} (v{int(best['Version'])})
                </h3>
                <p style="color:white;">
                    RMSE: <b>{best['RMSE']:.4f}</b> &nbsp;|&nbsp;
                    MAE: <b>{best['MAE']:.4f}</b> &nbsp;|&nbsp;
                    R²: <b>{best['R2']:.4f}</b>
                </p>
                <p style="color:#aaa; font-size:13px;">
                    Off by ~{best['RMSE']:.0f} AQI units on average &nbsp;|&nbsp;
                    Explains {best['R2']*100:.1f}% of AQI variation
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("📋 All Models")
        st.dataframe(
            df.sort_values("RMSE").reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("📊 Metric Comparison")
        fig = go.Figure()
        for metric, color in [
            ("RMSE", "#e74c3c"),
            ("MAE",  "#e67e22"),
            ("R2",   "#2ecc71")
        ]:
            fig.add_trace(go.Bar(
                name=metric, x=df["Model"],
                y=df[metric], marker_color=color
            ))

        fig.update_layout(
            barmode="group", template="plotly_dark",
            height=400, margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📖 What Do These Metrics Mean?")
        c1, c2, c3 = st.columns(3)
        c1.info("**RMSE**\nRoot Mean Square Error.\nAverage error in AQI units. Lower is better.")
        c2.info("**MAE**\nMean Absolute Error.\nTypical prediction error. More intuitive.")
        c3.info("**R²**\nR-squared.\nHow much variance explained. Closer to 1.0 is better.")

    except Exception as e:
        st.error(f"Could not load model metrics: {e}")


# ==================================================
# PAGE 4 — EDA
# ==================================================
elif page == "🔬 EDA":
    st.title("🔬 Exploratory Data Analysis")
    st.caption("Analysis of Karachi AQI data from Dec 2025 to present")

    plot_dir = os.path.join(BASE_DIR, "eda", "plots")

    plots = [
        ("1_aqi_distribution.png",
         "AQI Distribution",
         "Histogram and category breakdown of all recorded AQI values"),
        ("2_aqi_over_time.png",
         "AQI Over Time",
         "Daily average AQI trend from December 2025 to present with 7-day rolling average"),
        ("3_hourly_patterns.png",
         "Hourly Patterns",
         "Average AQI by hour of day — identifies rush hour pollution spikes"),
        ("4_weekly_patterns.png",
         "Weekly Patterns",
         "Average AQI by day of week — weekday vs weekend comparison"),
        ("5_correlation_heatmap.png",
         "Feature Correlation Heatmap",
         "Correlation between all features and AQI target variable"),
        ("6_seasonal_patterns.png",
         "Seasonal Patterns",
         "Monthly AQI averages showing seasonal variation across Karachi"),
        ("7_weather_vs_aqi.png",
         "Weather vs AQI",
         "Scatter plots showing how temperature, humidity, wind and pressure affect AQI"),
    ]

    for filename, title, description in plots:
        path = os.path.join(plot_dir, filename)
        if os.path.exists(path):
            st.subheader(f"📈 {title}")
            st.caption(description)
            st.image(path, use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"⚠️ Plot not found: {filename}")
            st.info("Run `python eda/eda_analysis.py` to generate plots")


# ==================================================
# PAGE 5 — SHAP
# ==================================================
elif page == "🧠 SHAP":
    st.title("🧠 Model Explainability (SHAP)")
    st.caption("Understanding what drives AQI predictions")

    plot_dir = os.path.join(BASE_DIR, "explainability", "plots")

    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** explains how each feature
    contributes to the model's predictions. Higher SHAP value = more
    influence on the AQI prediction.
    """)

    plots = [
        ("shap_importance.png",
         "Feature Importance",
         "Average absolute SHAP value per feature — higher means more important"),
        ("shap_summary.png",
         "SHAP Summary (Beeswarm)",
         "Each dot is one prediction — color shows feature value, position shows impact"),
    ]

    for filename, title, description in plots:
        path = os.path.join(plot_dir, filename)
        if os.path.exists(path):
            st.subheader(f"🔍 {title}")
            st.caption(description)
            st.image(path, use_column_width=True)
            st.markdown("---")
        else:
            st.warning(f"⚠️ Plot not found: {filename}")
            st.info("Run `python explainability/shap_explain.py` to generate")

    # dependence plots
    dep_plots = [f for f in os.listdir(plot_dir)
                 if f.startswith("shap_dep_")] if os.path.exists(plot_dir) else []

    if dep_plots:
        st.subheader("📊 Top Feature Dependence Plots")
        st.caption("Shows how each top feature value affects predictions")
        for filename in sorted(dep_plots):
            path = os.path.join(plot_dir, filename)
            feat = filename.replace("shap_dep_", "").replace(".png", "")
            st.image(path, caption=f"Dependence: {feat}", use_column_width=True)


# ==================================================
# PAGE 6 — ABOUT
# ==================================================
else:
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 🌍 Karachi AQI Prediction System

    An end-to-end MLOps pipeline predicting Air Quality Index for
    **Karachi, Pakistan** up to **72 hours ahead**.

    ---

    ### 🔧 Data Pipeline
    - **Pollution source**: OpenWeather API (pm2.5, pm10, co, no2, o3, so2, nh3)
    - **Weather source**: Open-Meteo API (temperature, humidity, wind, pressure)
    - **Frequency**: Hourly data collection via GitHub Actions
    - **History**: Dec 2025 → present (3800+ rows and growing)
    - **Storage**: Hopsworks Feature Store (v1 raw, v2 engineered)

    ### 🤖 Models Trained
    | Model | RMSE | MAE | R² |
    |---|---|---|---|
    | XGBoost | 10.60 | 6.85 | 0.9558 |
    | RandomForest | 11.81 | 6.95 | 0.9451 |
    | Ridge | 22.10 | 16.31 | 0.8080 |

    ### 📊 Features Used (18 total)
    - **Time**: hour, day, month, day_of_week, is_weekend
    - **Pollution lags**: pm25_lag_1h, pm25_lag_3h, pm10_lag_1h, pm10_lag_3h
    - **Pollution rolling**: pm25_roll_3h, pm25_roll_6h, pm10_roll_3h
    - **Weather lags**: temp_lag_1h, humidity_lag_1h, wind_lag_1h
    - **Weather rolling**: temp_roll_3h, humidity_roll_3h, wind_roll_3h

     ### 🔄 Pipeline Architecture
                """)