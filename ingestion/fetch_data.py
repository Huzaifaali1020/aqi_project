import requests
import pandas as pd
from datetime import datetime
import os, yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def fetch_data():
    aqi_resp = requests.get(
        f"{config['api']['aqi_url']}?token={config['api']['aqi_key']}"
    ).json()

    weather_resp = requests.get(
        config["api"]["weather_url"],
        params={
            "q": config["city"],
            "appid": config["api"]["weather_key"],
            "units": "metric"
        }
    ).json()

    return pd.DataFrame([{
        "timestamp": datetime.now(),
        "aqi": int(aqi_resp["data"]["aqi"]),
        "pm25": int(aqi_resp["data"]["iaqi"].get("pm25", {}).get("v", 0)),
        "pm10": int(aqi_resp["data"]["iaqi"].get("pm10", {}).get("v", 0)),
        "no2": int(aqi_resp["data"]["iaqi"].get("no2", {}).get("v", 0)),
        "temp": float(weather_resp["main"]["temp"]),
        "humidity": int(weather_resp["main"]["humidity"]),
        "wind": float(weather_resp["wind"]["speed"]),
    }])