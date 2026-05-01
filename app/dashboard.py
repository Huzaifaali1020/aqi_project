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
    page_title="AQI Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
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
        st.warning("⚠️ Unhealthy Air Quality — Sensitive groups should avoid outdoor activity")
    elif aqi <= 100:
        st.success("✅ Air quality is acceptable today")


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_forecast():
    preds = pd.read_csv(
        os.path.join(DATA_DIR, "latest_predictions.csv"),
        parse_dates=["timestamp"]
    )
    daily = pd.read_csv(
        os.path.join(DATA_DIR, "daily_summary.csv"),
        parse_dates=["date"]
    )
    return preds, daily


@st.cache_data
def load_current_conditions():
    project = hopsworks.login(api_key_value=config["hopsworks"]["api_key"])
    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_features_fv", version=1)
    df = fv.get_batch_data().sort_values("timestamp")
    return df.iloc[-1]


@st.cache_data
def load_model_metrics():
    project = hopsworks.login(api_key_value=config["hopsworks"]["api_key"])
    mr = project.get_model_registry()
    models = mr.get_models("aqi_predictor")

    rows = []
    for m in models:
        metrics = m.training_metrics or {}
        rows.append({
            "Model": m.name,
            "Version": m.version,
            "RMSE": metrics.get("rmse"),
            "MAE": metrics.get("mae"),
            "R2": metrics.get("r2"),
        })

    df = pd.DataFrame(rows).dropna()
    best = df.sort_values("RMSE").iloc[0]
    return df, best


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 3-Day Forecast", "📊 Model Performance", "ℹ️ About"]
)

# --------------------------------------------------
# PAGE 1 — HOME
# --------------------------------------------------
if page == "🏠 Home":
    st.title("🌆 Karachi Air Quality Dashboard")

    now_pk = datetime.now(PK_TZ)
    st.caption(f"📍 Karachi, Pakistan | 🕒 {now_pk:%Y-%m-%d %H:%M} PKT")

    current = load_current_conditions()
    current_aqi = float(current["aqi"])

    emoji, category, color = aqi_style(current_aqi)

    st.markdown(
        f"""
        <h1 style="color:{color};">{emoji} {current_aqi:.1f}</h1>
        <h3>{category}</h3>
        <p>Last updated: {current['timestamp']}</p>
        """,
        unsafe_allow_html=True,
    )

    alert_banner(current_aqi)

    st.subheader("🌡 Current Conditions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡 Temperature", f"{current['temperature']} °C")
    c2.metric("💧 Humidity", f"{current['humidity']} %")
    c3.metric("💨 Wind Speed", f"{current['wind_speed']} m/s")
    c4.metric("📈 Pressure", f"{current['pressure']} hPa")


# --------------------------------------------------
# PAGE 2 — FORECAST (ADDITIONS ONLY)
# --------------------------------------------------
elif page == "🔮 3-Day Forecast":
    st.title("🔮 72-Hour AQI Forecast")

    preds, daily = load_forecast()
    today = datetime.now(PK_TZ).date()

    # ---------------- TODAY BADGE ----------------
    today_row = daily[daily["date"].dt.date == today]
    if not today_row.empty:
        r = today_row.iloc[0]
        emoji, category, color = aqi_style(r["avg_aqi"])
        st.markdown(
            f"""
            <div style="border:2px solid {color}; padding:15px; border-radius:12px; margin-bottom:20px;">
                <span style="background:{color}; color:white; padding:4px 12px; border-radius:20px; font-size:12px;">
                    TODAY
                </span>
                <h3>{r['date'].strftime('%B %d')}</h3>
                <h1 style="color:{color};">{emoji} {r['avg_aqi']}</h1>
                <p>{category}</p>
                <p>Min: {r['min_aqi']} | Max: {r['max_aqi']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- NEXT 3 DAYS ONLY ----------------
    future_days = daily[daily["date"].dt.date > today].head(3)
    cols = st.columns(3)

    for col, (_, r) in zip(cols, future_days.iterrows()):
        emoji, category, color = aqi_style(r["avg_aqi"])
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #ddd; padding:15px; border-radius:10px;">
                <h4>{r['date'].strftime('%B %d')}</h4>
                <h1 style="color:{color};">{emoji} {r['avg_aqi']}</h1>
                <p>{category}</p>
                <p>Min: {r['min_aqi']} | Max: {r['max_aqi']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------- EXISTING GRAPH (UNCHANGED) ----------------
    st.subheader("📈 Hourly Forecast (6-hour steps)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds["timestamp"],
        y=preds["predicted_aqi"],
        mode="lines+markers",
        name="AQI"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- NEW STOCK STYLE GRAPH ----------------
    st.subheader("📊 AQI Trend (Today + Next 3 Days)")
    stock_fig = go.Figure()
    stock_fig.add_trace(
        go.Scatter(
            x=preds["timestamp"],
            y=preds["predicted_aqi"],
            mode="lines",
            line=dict(width=3),
            name="AQI"
        )
    )
    stock_fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="AQI"
    )
    st.plotly_chart(stock_fig, use_container_width=True)


# --------------------------------------------------
# PAGE 3 — MODEL PERFORMANCE
# --------------------------------------------------
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    df, best = load_model_metrics()

    st.success(
        f"🏆 Best Model: {best['Model']} (v{int(best['Version'])})  \n"
        f"RMSE: {best['RMSE']:.2f} | MAE: {best['MAE']:.2f} | R²: {best['R2']:.4f}"
    )

    st.subheader("📋 Model Comparison")
    st.dataframe(df, use_container_width=True)

    fig = go.Figure()
    for metric in ["RMSE", "MAE", "R2"]:
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Model"],
            y=df[metric]
        ))

    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# PAGE 4 — ABOUT
# --------------------------------------------------
else:
    st.title("ℹ️ About This Project")
    st.markdown("AQI forecasting system for Karachi using ML + Hopsworks.")