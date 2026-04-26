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
    # Time handling (BEST PRACTICE)
    # --------------------------------------------------
    # Epoch timestamp in milliseconds (PRIMARY KEY)
    timestamp = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    # UTC datetime (for debugging)
    event_time_utc = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

    # Pakistan local time (human readable)
    pk_tz = pytz.timezone("Asia/Karachi")
    event_time_pk = event_time_utc.astimezone(pk_tz)

    # --------------------------------------------------
    # Return dataframe
    # --------------------------------------------------
    return pd.DataFrame([{
        "timestamp": timestamp,                 # ✅ primary key (epoch ms)
        "event_time_utc": event_time_utc,       # UTC datetime
        "event_time_pk": event_time_pk,         # Pakistan local time
        "aqi": int(aqi_resp["data"]["aqi"]),
        "pm25": int(aqi_resp["data"]["iaqi"].get("pm25", {}).get("v", 0)),
        "pm10": int(aqi_resp["data"]["iaqi"].get("pm10", {}).get("v", 0)),
        "no2": int(aqi_resp["data"]["iaqi"].get("no2", {}).get("v", 0)),
        "temp": float(weather_resp["main"]["temp"]),
        "humidity": int(weather_resp["main"]["humidity"]),
        "wind": float(weather_resp["wind"]["speed"]),
    }])