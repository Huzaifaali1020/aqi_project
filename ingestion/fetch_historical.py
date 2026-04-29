import requests
import pandas as pd
import hopsworks
from datetime import datetime, timezone
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


# --------------------------------------------------
# EPA PM2.5 → AQI formula
# --------------------------------------------------
def pm25_to_aqi(pm25: float) -> int:
    breakpoints = [
        (0.0,   12.0,    0,  50),
        (12.1,  35.4,   51, 100),
        (35.5,  55.4,  101, 150),
        (55.5,  150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round(
                (aqi_hi - aqi_lo) / (bp_hi - bp_lo) * (pm25 - bp_lo) + aqi_lo
            )
    return 500


# --------------------------------------------------
# Free historical weather from Open-Meteo
# --------------------------------------------------
def fetch_historical_weather_openmeteo(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float
) -> dict:
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure"
        f"&timezone=UTC"
    )

    print(f"📡 Fetching weather from Open-Meteo {start_date} → {end_date} ...")
    response = requests.get(url).json()

    if "hourly" not in response:
        print(f"❌ Open-Meteo error: {response}")
        return {}

    hourly = response["hourly"]

    weather_by_ts = {}
    for i, time_str in enumerate(hourly["time"]):
        ts = pd.to_datetime(time_str, utc=True)
        weather_by_ts[ts] = {
            "temperature": float(hourly["temperature_2m"][i])
                           if hourly["temperature_2m"][i] is not None else None,
            "humidity":    float(hourly["relativehumidity_2m"][i])
                           if hourly["relativehumidity_2m"][i] is not None else None,
            "wind_speed":  float(hourly["windspeed_10m"][i])
                           if hourly["windspeed_10m"][i] is not None else None,
            "pressure":    float(hourly["surface_pressure"][i])
                           if hourly["surface_pressure"][i] is not None else None,
        }

    print(f"✅ Got weather for {len(weather_by_ts)} hours from Open-Meteo")
    return weather_by_ts


# --------------------------------------------------
# Fetch historical pollution + weather
# --------------------------------------------------
def fetch_historical(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    lat     = config["city"]["lat"]
    lon     = config["city"]["lon"]
    api_key = config["api"]["weather_key"]

    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())

    # ── 1. Pollution from OpenWeather ────────────
    pollution_url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={api_key}"
    )
    print(f"📡 Fetching pollution {start_dt.date()} → {end_dt.date()} ...")
    pollution_resp = requests.get(pollution_url).json()

    if "list" not in pollution_resp:
        print(f"❌ Pollution API error: {pollution_resp}")
        return pd.DataFrame()

    print(f"✅ Got {len(pollution_resp['list'])} pollution records")

    # ── 2. Weather from Open-Meteo (free) ────────
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date   = end_dt.strftime("%Y-%m-%d")

    weather_by_ts = fetch_historical_weather_openmeteo(
        start_date, end_date, lat, lon
    )

    # ── 3. Combine pollution + weather ───────────
    records = []
    missing_weather = 0

    for record in pollution_resp["list"]:
        components = record["components"]
        pm25 = float(components.get("pm2_5", components.get("pm25", 0)))

        ts = datetime.fromtimestamp(
            record["dt"], tz=timezone.utc
        ).replace(minute=0, second=0, microsecond=0)

        weather = weather_by_ts.get(ts)
        if weather is None:
            missing_weather += 1
            weather = {
                "temperature": None,
                "humidity":    None,
                "wind_speed":  None,
                "pressure":    None,
            }

        records.append({
            "timestamp":   ts,
            "aqi":         pm25_to_aqi(pm25),
            "co":          float(components.get("co",   0)),
            "no":          float(components.get("no",   0)),
            "no2":         float(components.get("no2",  0)),
            "o3":          float(components.get("o3",   0)),
            "pm10":        float(components.get("pm10", 0)),
            "pm25":        pm25,
            "so2":         float(components.get("so2",  0)),
            "nh3":         float(components.get("nh3",  0)),
            "temperature": weather["temperature"],
            "humidity":    weather["humidity"],
            "wind_speed":  weather["wind_speed"],
            "pressure":    weather["pressure"],
        })

    if missing_weather > 0:
        print(f"⚠️ {missing_weather} rows had no weather match — will be dropped")

    df = pd.DataFrame(records)

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[(df["aqi"] > 5) & (df["aqi"] < 499)].reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)

    print(f"✅ Got {len(df)} clean hourly rows after filtering")
    print(f"🌡️  Temp range    : {df['temperature'].min():.1f} – {df['temperature'].max():.1f} °C")
    print(f"💧 Humidity range : {df['humidity'].min():.1f} – {df['humidity'].max():.1f} %")
    print(f"💨 Wind range     : {df['wind_speed'].min():.1f} – {df['wind_speed'].max():.1f} m/s")

    return df


# --------------------------------------------------
# Upload to Hopsworks v1
# --------------------------------------------------
def upload_to_hopsworks(df: pd.DataFrame):
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    fg_v1 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Raw hourly air quality + weather data",
        online_enabled=False
    )

    print(f"📤 Uploading {len(df)} rows to Hopsworks v1 ...")
    fg_v1.insert(df, write_options={"wait_for_job": True})
    print(f"✅ Successfully uploaded {len(df)} rows to aqi_features v1")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    start = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime.now(tz=timezone.utc)

    df_hist = fetch_historical(start, end)

    if df_hist.empty:
        print("❌ No data returned — check API keys and network")
    else:
        print(f"\n📊 Total rows  : {len(df_hist)}")
        print(f"📅 Date range  : {df_hist['timestamp'].min()} → {df_hist['timestamp'].max()}")
        print(f"📈 AQI range   : {df_hist['aqi'].min()} – {df_hist['aqi'].max()}")

        out_path = os.path.join(BASE_DIR, "data", "historical_raw.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_hist.to_csv(out_path, index=False)
        print(f"💾 Saved to {out_path}")

        upload_to_hopsworks(df_hist)