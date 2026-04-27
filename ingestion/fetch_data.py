import requests
import pandas as pd
from datetime import datetime, timezone
import pytz
import os, yaml

# --------------------------------------------------
# Load config.yaml
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def fetch_data():
    # --------------------------------------------------
    # Fetch AQI data
    # --------------------------------------------------
    aqi_resp = requests.get(
        f"{config['api']['aqi_url']}?token={config['api']['aqi_key']}"
    ).json()

    # --------------------------------------------------
    # Fetch Weather data
    # --------------------------------------------------
    weather_resp = requests.get(
        config["api"]["weather_url"],
        params={
            "q": config["city"],
            "appid": config["api"]["weather_key"],
            "units": "metric"
        }
    ).json()

    # --------------------------------------------------
    # ✅ HOURLY TIME BUCKET (CRITICAL FIX)
    # --------------------------------------------------
    now_utc = datetime.now(timezone.utc)

    # 🔥 round DOWN to hour
    hour_utc = now_utc.replace(minute=0, second=0, microsecond=0)

    # epoch milliseconds (PRIMARY KEY)
    timestamp = int(hour_utc.timestamp() * 1000)
    print("UTC Hour:", hour_utc)
    print("Epoch timestamp (ms):", timestamp)
    # human-readable times
    pk_tz = pytz.timezone("Asia/Karachi")
    event_time_pk = hour_utc.astimezone(pk_tz)

    # --------------------------------------------------
    # Return dataframe
    # --------------------------------------------------
    return pd.DataFrame([{
        "timestamp": timestamp,           # ✅ PRIMARY KEY (hourly)
        "aqi": int(aqi_resp["data"]["aqi"]),
        "pm25": int(aqi_resp["data"]["iaqi"].get("pm25", {}).get("v", 0)),
        "pm10": int(aqi_resp["data"]["iaqi"].get("pm10", {}).get("v", 0)),
        "no2": int(aqi_resp["data"]["iaqi"].get("no2", {}).get("v", 0)),
        "temp": float(weather_resp["main"]["temp"]),
        "humidity": int(weather_resp["main"]["humidity"]),
        "wind": float(weather_resp["wind"]["speed"]),
    }])

if __name__ == "__main__":
    df = fetch_data()
    print(df)