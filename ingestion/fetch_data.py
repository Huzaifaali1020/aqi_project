import requests
import pandas as pd
from datetime import datetime
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def fetch_data():
    # --- Weather API ---
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

    # --- Air Pollution API ---
    pollution_resp = requests.get(
        config["api"]["air_pollution_url"],
        params={
            "lat": config["city"]["lat"],
            "lon": config["city"]["lon"],
            "appid": config["api"]["weather_key"]
        }
    )
    pollution_resp.raise_for_status()
    pollution = pollution_resp.json()

    components = pollution["list"][0]["components"]

    return pd.DataFrame([{
        "timestamp": datetime.now(),
        "temp": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "wind": weather["wind"]["speed"],
        "pm25": components.get("pm2_5", 0),
        "pm10": components.get("pm10", 0),
        "no2": components.get("no2", 0),
        "co": components.get("co", 0),
        "o3": components.get("o3", 0)
    }])

if __name__ == "__main__":
    df = fetch_data()
    print(df.head())