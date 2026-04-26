import requests
import pandas as pd
from datetime import datetime
import os
import yaml

# ----------------------------
# Load config.yaml
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------
# Fetch AQI + Weather data
# ----------------------------
def fetch_data():
    # ---- AQICN (Air Quality) ----
    aqi_resp = requests.get(
        f"{config['api']['aqi_url']}?token={config['api']['aqi_key']}"
    )
    aqi_resp.raise_for_status()
    aqi = aqi_resp.json()

    iaqi = aqi["data"].get("iaqi", {})

    # ---- OpenWeather (Weather) ----
    weather_resp = requests.get(
        config["api"]["weather_url"],
        params={
            "q": config["city"]["name"],
            "appid": config["api"]["weather_key"],
            "units": "metric"
        }
    )
    weather_resp.raise_for_status()
    weather = weather_resp.json()

    # ----------------------------
    # Build DataFrame
    # ----------------------------
    return pd.DataFrame([{
        "timestamp": datetime.now(),  # local time (same as yesterday)
        "aqi": aqi["data"]["aqi"],
        "pm25": iaqi.get("pm25", {}).get("v", 0),
        "pm10": iaqi.get("pm10", {}).get("v", 0),
        "no2": iaqi.get("no2", {}).get("v", 0),
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "wind": weather["wind"]["speed"]
    }])

# ----------------------------
# Run manually
# ----------------------------
if __name__ == "__main__":
    df = fetch_data()
    print(df)