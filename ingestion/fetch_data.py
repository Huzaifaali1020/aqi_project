import requests
import pandas as pd
from datetime import datetime
import os, yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def fetch_data():
    aqi = requests.get(
        f"{config['api']['aqi_url']}?token={config['api']['aqi_key']}"
    ).json()

    weather = requests.get(
        config["api"]["weather_url"],
        params={
            "q": config["city"],
            "appid": config["api"]["weather_key"],
            "units": "metric"
        }
    ).json()

    return pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "aqi": aqi["data"]["aqi"],
        "pm25": aqi["data"]["iaqi"].get("pm25", {}).get("v", 0),
        "pm10": aqi["data"]["iaqi"].get("pm10", {}).get("v", 0),
        "no2": aqi["data"]["iaqi"].get("no2", {}).get("v", 0),
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "wind": weather["wind"]["speed"]
    }])