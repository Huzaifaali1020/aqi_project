import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from datetime import datetime, timedelta
import pytz
import hopsworks
import yaml
import os
import subprocess
import streamlit.components.v1 as components
from datetime import timezone

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Karachi AQI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Detect if running on Streamlit Cloud
# Never run forecast.py on cloud — only read committed CSVs
# Forecast runs via GitHub Actions daily
# --------------------------------------------------
IS_CLOUD = (
    os.path.exists("/mount/src") or
    "STREAMLIT_SHARING_MODE" in os.environ or
    os.environ.get("HOME", "") == "/home/appuser"
)

# --------------------------------------------------
# Professional CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.top-status-bar {
    height: 3px;
    background: linear-gradient(90deg, #4f7cff, #38bdf8, #34d399);
    margin: -1rem -1rem 1.5rem -1rem;
}
.page-header {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 0 0 20px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 24px;
}
.page-header-title {
    font-size: 22px; font-weight: 600;
    color: #ffffff; letter-spacing: -0.3px;
}
.page-header-meta {
    font-size: 12px; color: #6b7280;
    background: rgba(255,255,255,0.05);
    padding: 5px 14px; border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}
.aqi-main-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 28px 24px;
    text-align: center;
}
.section-title {
    font-size: 11px; font-weight: 500; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 20px 0 10px 0;
}
.alert-good {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.25);
    border-left: 3px solid #34d399; border-radius: 8px;
    padding: 10px 16px; font-size: 13px;
    color: #6ee7b7; margin-bottom: 16px;
}
.alert-moderate {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-left: 3px solid #f59e0b; border-radius: 8px;
    padding: 10px 16px; font-size: 13px;
    color: #fcd34d; margin-bottom: 16px;
}
.alert-bad {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    border-left: 3px solid #ef4444; border-radius: 8px;
    padding: 10px 16px; font-size: 13px;
    color: #fca5a5; margin-bottom: 16px;
}
.forecast-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 20px 16px;
    text-align: center;
}
[data-testid="stSidebar"] { background: #111827; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Paths and Config
# --------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

try:
    config = {
        "hopsworks": {
            "host":    st.secrets["HOPSWORKS_HOST"],
            "api_key": st.secrets["HOPSWORKS_API_KEY"],
        },
        "api": {"weather_key": st.secrets.get("WEATHER_API_KEY", "")},
        "city": {"lat": 24.8607, "lon": 67.0011}
    }
except Exception:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

PK_TZ = pytz.timezone("Asia/Karachi")


# --------------------------------------------------
# Forecast helpers
# --------------------------------------------------
def is_forecast_stale() -> bool:
    # on Streamlit Cloud NEVER auto-run forecast
    # GitHub Actions handles daily refresh
    if IS_CLOUD:
        return False

    daily_path = os.path.join(DATA_DIR, "daily_summary.csv")
    if not os.path.exists(daily_path):
        return True

    try:
        daily = pd.read_csv(daily_path)
        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        today = datetime.now(PK_TZ).date()

        first_date = daily["date"].min()
        if first_date < today:
            return True

        modified_time = datetime.fromtimestamp(
            os.path.getmtime(daily_path), tz=pytz.UTC
        )
        age_hours = (datetime.now(pytz.UTC) - modified_time).total_seconds() / 3600
        return age_hours > 6

    except Exception:
        return True


def refresh_forecast():
    # never run on Streamlit Cloud
    if IS_CLOUD:
        return

    try:
        result = subprocess.run(
            [sys.executable,
             os.path.join(BASE_DIR, "inference", "forecast.py")],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=180
        )
        if result.returncode == 0:
            st.toast("Forecast updated", icon="✓")
        else:
            st.warning(f"Forecast update failed: {result.stderr[:300]}")
    except Exception as e:
        st.warning(f"Could not refresh forecast: {e}")


def aqi_color(aqi):
    if aqi <= 50:    return "#34d399"
    elif aqi <= 100: return "#f59e0b"
    elif aqi <= 150: return "#f97316"
    elif aqi <= 200: return "#ef4444"
    elif aqi <= 300: return "#a855f7"
    else:            return "#6b7280"


def aqi_category(aqi):
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else:            return "Hazardous"


def alert_html(aqi):
    if aqi > 200:
        return '<div class="alert-bad">Air quality is very unhealthy — everyone should avoid outdoor activity</div>'
    elif aqi > 150:
        return '<div class="alert-bad">Air quality is unhealthy — sensitive groups should avoid outdoor activity</div>'
    elif aqi > 100:
        return '<div class="alert-moderate">Air quality may affect sensitive groups</div>'
    else:
        return '<div class="alert-good">Air quality is acceptable today</div>'


# --------------------------------------------------
# Data loaders
# --------------------------------------------------
@st.cache_data(ttl=1800)
def load_forecast():
    preds = pd.read_csv(
        os.path.join(DATA_DIR, "latest_predictions.csv"),
        parse_dates=["timestamp"]
    )
    daily = pd.read_csv(os.path.join(DATA_DIR, "daily_summary.csv"))
    daily["date"] = pd.to_datetime(daily["date"])
    return preds, daily


@st.cache_data(ttl=1800)
def load_current_conditions():
    path = os.path.join(DATA_DIR, "latest_features.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values("timestamp")
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
# Location map
# --------------------------------------------------
def render_location_map():
    map_html = """
    <!DOCTYPE html><html>
    <head>
    <meta charset="utf-8"/>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body{margin:0;padding:0;background:#111827;}
        #map{height:300px;width:100%;border-radius:10px;
             border:1px solid rgba(255,255,255,0.08);}
        #status{position:absolute;top:10px;right:10px;z-index:1000;
            background:rgba(17,24,39,0.9);color:#9ca3af;
            font-family:Inter,sans-serif;font-size:11px;
            padding:6px 12px;border-radius:20px;
            border:1px solid rgba(255,255,255,0.08);}
        .loc-btn{position:absolute;bottom:20px;right:10px;z-index:1000;
            background:#1e3a5f;color:#60a5fa;
            font-family:Inter,sans-serif;font-size:12px;
            padding:7px 14px;border-radius:8px;cursor:pointer;
            border:1px solid rgba(96,165,250,0.3);}
        .loc-btn:hover{background:#1e40af;}
    </style>
    </head>
    <body>
    <div style="position:relative">
        <div id="map"></div>
        <div id="status">Karachi, Pakistan</div>
        <button class="loc-btn" onclick="getMyLocation()">
            Show My Location
        </button>
    </div>
    <script>
    var map = L.map('map').setView([24.8607,67.0011],11);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        {attribution:'© OpenStreetMap © CARTO',maxZoom:19}).addTo(map);
    var karachiIcon = L.divIcon({
        html:'<div style="background:#f59e0b;width:14px;height:14px;border-radius:50%;border:2px solid #fcd34d;box-shadow:0 0 10px #f59e0b66"></div>',
        iconSize:[14,14],iconAnchor:[7,7],className:''
    });
    var karachi = L.marker([24.8607,67.0011],{icon:karachiIcon}).addTo(map)
        .bindPopup('<div style="background:#111827;color:#e5e7eb;font-family:Inter,sans-serif;font-size:12px;padding:6px"><b style="color:#f59e0b">Karachi AQI Station</b><br>24.8607°N | 67.0011°E<br>Monitoring: Active</div>');
    L.circle([24.8607,67.0011],{radius:20000,color:'#f59e0b',fillColor:'#f59e0b',fillOpacity:0.04,weight:1,dashArray:'4'}).addTo(map);
    var userM=null;
    function getMyLocation(){
        var s=document.getElementById('status');
        s.textContent='Requesting location...';
        if(!navigator.geolocation){s.textContent='Not supported';return;}
        navigator.geolocation.getCurrentPosition(function(pos){
            var lat=pos.coords.latitude,lon=pos.coords.longitude;
            if(userM)map.removeLayer(userM);
            var ui=L.divIcon({html:'<div style="background:#4f7cff;width:14px;height:14px;border-radius:50%;border:2px solid #818cf8;box-shadow:0 0 12px #4f7cff88"></div>',iconSize:[14,14],iconAnchor:[7,7],className:''});
            userM=L.marker([lat,lon],{icon:ui}).addTo(map)
                .bindPopup('<div style="background:#111827;color:#e5e7eb;font-family:Inter,sans-serif;font-size:12px;padding:6px"><b style="color:#4f7cff">Your Location</b><br>'+lat.toFixed(4)+'° | '+lon.toFixed(4)+'°</div>').openPopup();
            map.fitBounds(L.featureGroup([karachi,userM]).getBounds().pad(0.2));
            s.textContent='Showing your location';
        },function(e){
            document.getElementById('status').textContent=e.code===1?'Permission denied':'Location unavailable';
        },{timeout:10000});
    }
    </script></body></html>
    """
    components.html(map_html, height=320)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
now_pk = datetime.now(PK_TZ)

st.sidebar.markdown("""
<div style="padding:4px 0 16px 0;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:16px;">
    <div style="font-size:16px;font-weight:600;color:#ffffff;">AQI Karachi</div>
    <div style="font-size:11px;color:#6b7280;margin-top:3px;">Air Quality Forecast System</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Home", "3-Day Forecast", "Model Performance", "EDA", "SHAP", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="font-size:11px;color:#6b7280;line-height:2.0;">
    {now_pk:%Y-%m-%d %H:%M} PKT<br>
    Data updates every hour<br>
    Model retrains daily
</div>
""", unsafe_allow_html=True)

# show refresh button only when running locally
if not IS_CLOUD:
    if st.sidebar.button("Refresh Forecast", use_container_width=True):
        st.cache_data.clear()
        with st.spinner("Updating forecast..."):
            refresh_forecast()
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style="font-size:11px;color:#6b7280;margin-top:8px;
                padding:8px;background:rgba(255,255,255,0.02);
                border-radius:6px;border:1px solid rgba(255,255,255,0.06);">
        Forecast auto-updates daily via GitHub Actions
    </div>
    """, unsafe_allow_html=True)

# auto refresh — local only
if not IS_CLOUD and is_forecast_stale():
    if "refresh_done" not in st.session_state:
        st.session_state.refresh_done = True
        with st.spinner("Updating forecast data..."):
            refresh_forecast()
            st.cache_data.clear()
        st.rerun()


# ==================================================
# PAGE 1 — HOME
# ==================================================
if page == "Home":
    st.markdown('<div class="top-status-bar"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-title">Karachi Air Quality Dashboard</div>
        <div class="page-header-meta">
            Karachi, Pakistan &nbsp;|&nbsp; {now_pk:%Y-%m-%d %H:%M} PKT
        </div>
    </div>
    """, unsafe_allow_html=True)

    current = load_current_conditions()

    if current is None:
        st.warning("Current conditions data not available. "
                   "Run `python inference/forecast.py` to generate it.")
    else:
        try:
            current_aqi = float(current["aqi"])
            color       = aqi_color(current_aqi)
            category    = aqi_category(current_aqi)

            st.markdown(alert_html(current_aqi), unsafe_allow_html=True)

            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    PKT = timezone(timedelta(hours=5))
                    ts_pkt = pd.to_datetime(current['timestamp']).astimezone(PKT).strftime('%Y-%m-%d %H:%M')
                except Exception:
                    ts_pkt = str(current['timestamp'])

                st.markdown(f"""
                <div class="aqi-main-card" style="border-left:4px solid {color};">
                    <div style="font-size:72px;font-weight:700;
                                color:{color};line-height:1;margin-bottom:8px;">
                        {current_aqi:.0f}
                    </div>
                    <div style="font-size:16px;color:#d1d5db;
                                font-weight:500;margin-bottom:4px;">
                        {category}
                    </div>
                    <div style="font-size:11px;color:#6b7280;">
                        Last updated: {ts_pkt} PKT
                    </div>
                </div>
                """, unsafe_allow_html=True)


            with col2:
                st.markdown('<div class="section-title">Current Conditions</div>',
                            unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Temperature", f"{float(current['temperature']):.1f} °C")
                c2.metric("Humidity",    f"{float(current['humidity']):.0f} %")
                c3.metric("Wind Speed",  f"{float(current['wind_speed']):.1f} m/s")
                c4.metric("Pressure",    f"{float(current['pressure']):.0f} hPa")

                st.markdown('<div class="section-title" style="margin-top:20px">Pollutant Levels</div>',
                            unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("PM2.5", f"{float(current['pm25']):.1f} µg/m³")
                c2.metric("PM10",  f"{float(current['pm10']):.1f} µg/m³")
                c3.metric("O3",    f"{float(current['o3']):.2f} µg/m³")

        except Exception as e:
            st.error(f"Error displaying conditions: {e}")

    st.markdown('<div class="section-title" style="margin-top:24px">Location</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#6b7280;margin-bottom:10px;">
        Monitoring station: Karachi, Pakistan (24.8607°N, 67.0011°E).
        Click <b style="color:#60a5fa">Show My Location</b> on the map to
        see your position relative to the AQI station.
    </div>
    """, unsafe_allow_html=True)
    render_location_map()


# ==================================================
# PAGE 2 — FORECAST
# ==================================================
elif page == "3-Day Forecast":
    st.markdown('<div class="top-status-bar"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-title">72-Hour AQI Forecast</div>
        <div class="page-header-meta">Karachi, Pakistan</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        preds, daily = load_forecast()

        daily_path    = os.path.join(DATA_DIR, "daily_summary.csv")
        modified_time = datetime.fromtimestamp(
            os.path.getmtime(daily_path), tz=pytz.UTC
        ).astimezone(PK_TZ)

        st.markdown(f"""
        <div style="font-size:12px;color:#6b7280;margin-bottom:20px;">
            Forecast generated: {modified_time:%Y-%m-%d %H:%M} PKT
            &nbsp;|&nbsp; 12-step recursive prediction (6h intervals)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">3-Day Outlook</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(daily.head(4)))
        for col, (_, r) in zip(cols, daily.head(4).iterrows()):
            color    = aqi_color(r["avg_aqi"])
            category = aqi_category(r["avg_aqi"])
            date_lbl = pd.to_datetime(r["date"]).strftime("%b %d")
            with col:
                st.markdown(f"""
                <div class="forecast-card" style="border-top:3px solid {color};">
                    <div style="font-size:13px;font-weight:500;
                                color:#d1d5db;margin-bottom:12px;">{date_lbl}</div>
                    <div style="font-size:38px;font-weight:700;
                                color:{color};line-height:1;margin-bottom:6px;">
                        {r['avg_aqi']}
                    </div>
                    <div style="font-size:12px;color:#9ca3af;margin-bottom:12px;">
                        {category}
                    </div>
                    <div style="font-size:11px;color:#6b7280;padding-top:10px;
                                border-top:1px solid rgba(255,255,255,0.06);">
                        Min: {r['min_aqi']} &nbsp;|&nbsp; Max: {r['max_aqi']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        avg_aqi = daily["avg_aqi"].mean()
        if avg_aqi > 200:
            st.markdown('<div class="alert-bad">Very unhealthy forecast — avoid outdoor activity</div>', unsafe_allow_html=True)
        elif avg_aqi > 150:
            st.markdown('<div class="alert-bad">Unhealthy forecast — sensitive groups stay indoors</div>', unsafe_allow_html=True)
        elif avg_aqi > 100:
            st.markdown('<div class="alert-moderate">May affect sensitive groups over the next 3 days</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-good">Air quality forecast is acceptable for the next 3 days</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:24px">Hourly Forecast</div>',
                    unsafe_allow_html=True)

        marker_colors = [aqi_color(a) for a in preds["predicted_aqi"]]
        fig = go.Figure()
        fig.add_hrect(y0=0,   y1=50,  fillcolor="#34d399", opacity=0.04, line_width=0)
        fig.add_hrect(y0=50,  y1=100, fillcolor="#f59e0b", opacity=0.06, line_width=0)
        fig.add_hrect(y0=100, y1=150, fillcolor="#f97316", opacity=0.07, line_width=0)
        fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.07, line_width=0)
        fig.add_hrect(y0=200, y1=300, fillcolor="#a855f7", opacity=0.07, line_width=0)
        fig.add_trace(go.Scatter(
            x=preds["timestamp"], y=preds["predicted_aqi"],
            mode="lines+markers", name="Predicted AQI",
            line=dict(width=2.5, color="#4f7cff"),
            marker=dict(size=9, color=marker_colors,
                        line=dict(width=1, color="#111827")),
            hovertemplate="<b>%{x}</b><br>AQI: %{y:.1f}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            xaxis=dict(title="Time", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="AQI", gridcolor="rgba(255,255,255,0.05)",
                       range=[0, max(preds["predicted_aqi"].max()+30, 150)]),
            height=380, showlegend=False,
            margin=dict(l=40, r=40, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Forecast Breakdown</div>',
                    unsafe_allow_html=True)
        cat_counts = preds["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Total Hours"] = cat_counts["Count"] * 6
        st.dataframe(cat_counts.drop(columns=["Count"]),
                     use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load forecast: {e}")


# ==================================================
# PAGE 3 — MODEL PERFORMANCE
# ==================================================
elif page == "Model Performance":
    st.markdown('<div class="top-status-bar"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">Model Performance</div>
        <div class="page-header-meta">3 models trained daily</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        df, best = load_model_metrics()

        st.markdown(f"""
        <div style="background:rgba(79,124,255,0.06);
                    border:1px solid rgba(79,124,255,0.2);
                    border-left:4px solid #4f7cff;
                    border-radius:10px;padding:20px;margin-bottom:24px;">
            <div style="font-size:11px;color:#6b7280;text-transform:uppercase;
                        letter-spacing:0.8px;margin-bottom:8px;">Best Performing Model</div>
            <div style="font-size:20px;font-weight:600;color:#ffffff;margin-bottom:10px;">
                {best['Model']}
                <span style="font-size:13px;color:#6b7280;font-weight:400;">
                    version {int(best['Version'])}
                </span>
            </div>
            <div style="display:flex;gap:24px;">
                <div>
                    <div style="font-size:11px;color:#6b7280;margin-bottom:2px;">RMSE</div>
                    <div style="font-size:22px;font-weight:500;color:#4f7cff;">
                        {best['RMSE']:.4f}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;color:#6b7280;margin-bottom:2px;">MAE</div>
                    <div style="font-size:22px;font-weight:500;color:#38bdf8;">
                        {best['MAE']:.4f}
                    </div>
                </div>
                <div>
                    <div style="font-size:11px;color:#6b7280;margin-bottom:2px;">R²</div>
                    <div style="font-size:22px;font-weight:500;color:#34d399;">
                        {best['R2']:.4f}
                    </div>
                </div>
            </div>
            <div style="font-size:12px;color:#6b7280;margin-top:10px;">
                Off by ~{best['RMSE']:.0f} AQI units on average &nbsp;|&nbsp;
                Explains {best['R2']*100:.1f}% of AQI variation
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">All Models</div>', unsafe_allow_html=True)
        st.dataframe(df.sort_values("RMSE").reset_index(drop=True),
                     use_container_width=True)

        st.markdown('<div class="section-title" style="margin-top:20px">Metric Comparison</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        for metric, color in [("RMSE","#4f7cff"),("MAE","#38bdf8"),("R2","#34d399")]:
            fig.add_trace(go.Bar(name=metric, x=df["Model"],
                                 y=df[metric], marker_color=color,
                                 marker_line_width=0))
        fig.update_layout(barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          template="plotly_dark", height=360,
                          margin=dict(l=40,r=40,t=10,b=40),
                          legend=dict(orientation="h",y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">What These Metrics Mean</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.info("**RMSE** — Root Mean Square Error\nAverage error in AQI units. Lower is better.")
        c2.info("**MAE** — Mean Absolute Error\nTypical prediction error. More intuitive.")
        c3.info("**R²** — R-squared\nVariance explained. Closer to 1.0 is better.")

    except Exception as e:
        st.error(f"Could not load model metrics: {e}")


# ==================================================
# PAGE 4 — EDA
# ==================================================
elif page == "EDA":
    st.markdown('<div class="top-status-bar"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">Exploratory Data Analysis</div>
        <div class="page-header-meta">Dec 2025 — present</div>
    </div>
    """, unsafe_allow_html=True)

    plot_dir = os.path.join(BASE_DIR, "eda", "plots")
    plots = [
        ("1_aqi_distribution.png",   "AQI Distribution",
         "Histogram and category breakdown of all recorded AQI values"),
        ("2_aqi_over_time.png",      "AQI Over Time",
         "Daily average AQI from December 2025 to present with 7-day rolling average"),
        ("3_hourly_patterns.png",    "Hourly Patterns",
         "Average AQI by hour of day — identifies rush hour pollution spikes"),
        ("4_weekly_patterns.png",    "Weekly Patterns",
         "Average AQI by day of week — weekday vs weekend comparison"),
        ("5_correlation_heatmap.png","Feature Correlation Heatmap",
         "Correlation between all features and the AQI target variable"),
        ("6_seasonal_patterns.png",  "Seasonal Patterns",
         "Monthly AQI averages showing seasonal variation across Karachi"),
        ("7_weather_vs_aqi.png",     "Weather vs AQI",
         "How temperature, humidity, wind and pressure affect AQI"),
    ]
    for filename, title, description in plots:
        path = os.path.join(plot_dir, filename)
        if os.path.exists(path):
            st.markdown(f'<div class="section-title">{title}</div>',
                        unsafe_allow_html=True)
            st.caption(description)
            st.image(path, use_column_width=True)
            st.markdown(
                '<div style="height:1px;background:rgba(255,255,255,0.06);margin:20px 0;"></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.06);
                        border-radius:8px;padding:14px 16px;
                        font-size:13px;color:#6b7280;">
                Plot not found: {filename}
            </div>
            """, unsafe_allow_html=True)


# ==================================================
# PAGE 5 — SHAP
# ==================================================
elif page == "SHAP":
    st.markdown('<div class="top-status-bar"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="page-header">
        <div class="page-header-title">Model Explainability</div>
        <div class="page-header-meta">SHAP Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(79,124,255,0.05);
                border:1px solid rgba(79,124,255,0.15);
                border-radius:8px;padding:14px 16px;
                font-size:13px;color:#9ca3af;margin-bottom:20px;">
        <b style="color:#d1d5db;">SHAP (SHapley Additive exPlanations)</b> —
        explains how much each feature contributes to each prediction.
        Positive values push AQI higher. Negative values push AQI lower.
    </div>
    """, unsafe_allow_html=True)

    plot_dir = os.path.join(BASE_DIR, "explainability", "plots")
    for filename, title, description in [
        ("shap_importance.png", "Feature Importance",
         "Average absolute SHAP value per feature — higher means more influential"),
        ("shap_summary.png",    "SHAP Beeswarm Summary",
         "Each dot is one prediction. Color = feature value. Position = impact on AQI."),
    ]:
        path = os.path.join(plot_dir, filename)
        if os.path.exists(path):
            st.markdown(f'<div class="section-title">{title}</div>',
                        unsafe_allow_html=True)
            st.caption(description)
            st.image(path, use_column_width=True)
            st.markdown(
                '<div style="height:1px;background:rgba(255,255,255,0.06);margin:20px 0;"></div>',
                unsafe_allow_html=True)

    if os.path.exists(plot_dir):
        dep_plots = sorted([f for f in os.listdir(plot_dir)
                            if f.startswith("shap_dep_")])
        if dep_plots:
            st.markdown('<div class="section-title">Top Feature Dependence Plots</div>',
                        unsafe_allow_html=True)
            for filename in dep_plots:
                path = os.path.join(plot_dir, filename)
                feat = filename.replace("shap_dep_","").replace(".png","")
                st.caption(f"Dependence: {feat}")
                st.image(path, use_column_width=True)


# ==================================================
# PAGE 6 — ABOUT
# ==================================================
else:
    st.markdown('<div class="top-status-bar"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(79,124,255,0.08),rgba(56,189,248,0.05));
                border:1px solid rgba(79,124,255,0.15);
                border-radius:14px;padding:32px;margin-bottom:28px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
                <div style="font-size:24px;font-weight:700;color:#ffffff;
                            letter-spacing:-0.3px;margin-bottom:4px;">
                    Karachi AQI Predictor
                </div>
                <div style="font-size:13px;color:#6b7280;margin-bottom:16px;">
                    End-to-End MLOps Pipeline for Air Quality Forecasting
                </div>
                <div style="font-size:13px;color:#9ca3af;line-height:1.8;">
                    Predicts Air Quality Index for
                    <b style="color:#d1d5db">Karachi, Pakistan</b>
                    up to <b style="color:#d1d5db">72 hours ahead</b>
                    using machine learning, automated data pipelines,
                    and a live interactive dashboard.
                </div>
            </div>
            <div style="text-align:right;min-width:180px;">
                <div style="font-size:11px;color:#6b7280;text-transform:uppercase;
                            letter-spacing:0.8px;margin-bottom:8px;">Developed by</div>
                <div style="font-size:15px;font-weight:600;color:#ffffff;">
                    Syed Huzaifa Ali
                </div>
                <div style="font-size:12px;color:#6b7280;margin-top:4px;">
                    10Pearls Data Science Internship
                </div>
                <div style="font-size:12px;color:#6b7280;">2026</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="section-title">Data Pipeline</div>
        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;padding:16px;
                    font-size:13px;color:#9ca3af;line-height:2.0;">
            <b style="color:#d1d5db;">Pollution Source</b><br>
            OpenWeather Air Pollution API<br>
            PM2.5, PM10, CO, NO, NO2, O3, SO2, NH3<br><br>
            <b style="color:#d1d5db;">Weather Source</b><br>
            Open-Meteo API (free, no key required)<br>
            Temperature, Humidity, Wind Speed, Pressure<br><br>
            <b style="color:#d1d5db;">Collection Frequency</b><br>
            Hourly via GitHub Actions<br><br>
            <b style="color:#d1d5db;">Historical Data</b><br>
            December 2025 → present (3800+ rows)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:20px">Model Results</div>',
                    unsafe_allow_html=True)
        try:
            df_metrics, best_metric = load_model_metrics()
            st.dataframe(
                df_metrics.sort_values("RMSE").reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )
        except Exception:
            # fallback to current best known values
            results_data = {
                "Model": ["XGBoost", "RandomForest", "Ridge"],
                "RMSE": [3.75, 11.81, 22.10],
                "MAE": [2.34, 6.95, 16.31],
                "R²": [0.9942, 0.9451, 0.8080],
            }
            st.dataframe(
                pd.DataFrame(results_data),
                use_container_width=True,
                hide_index=True
            )

    with col2:
        st.markdown("""
        <div class="section-title">Features Used (18 total)</div>
        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;padding:16px;
                    font-size:13px;color:#9ca3af;line-height:2.0;">
            <b style="color:#4f7cff;">Time</b><br>
            hour, day, month, day_of_week, is_weekend<br><br>
            <b style="color:#38bdf8;">Pollution Lags</b><br>
            pm25_lag_1h, pm25_lag_3h, pm10_lag_1h, pm10_lag_3h<br><br>
            <b style="color:#34d399;">Pollution Rolling</b><br>
            pm25_roll_3h, pm25_roll_6h, pm10_roll_3h<br><br>
            <b style="color:#f59e0b;">Weather Lags</b><br>
            temp_lag_1h, humidity_lag_1h, wind_lag_1h<br><br>
            <b style="color:#f97316;">Weather Rolling</b><br>
            temp_roll_3h, humidity_roll_3h, wind_roll_3h
        </div>

        <div class="section-title" style="margin-top:20px">AQI Scale Reference</div>
        <div style="background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px;padding:16px;
                    font-size:13px;line-height:2.2;">
            <span style="color:#34d399;">Good (0–50)</span> — No health impact<br>
            <span style="color:#f59e0b;">Moderate (51–100)</span> — Very sensitive people<br>
            <span style="color:#f97316;">Sensitive (101–150)</span> — Sensitive groups affected<br>
            <span style="color:#ef4444;">Unhealthy (151–200)</span> — Everyone affected<br>
            <span style="color:#a855f7;">Very Unhealthy (201–300)</span> — Health alert<br>
            <span style="color:#6b7280;">Hazardous (301–500)</span> — Emergency conditions
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:24px;padding:16px 20px;
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;display:flex;
                justify-content:space-between;
                font-size:12px;color:#6b7280;">
        <span>Karachi, Pakistan (24.8607°N, 67.0011°E)</span>
        <span>Hourly data collection</span>
        <span>Daily model retraining</span>
        <span>72-hour recursive forecast</span>
        <span>Streamlit Cloud</span>
    </div>
    """, unsafe_allow_html=True)