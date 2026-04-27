import requests
import pandas as pd
from datetime import datetime, timezone
import os, yaml


# --------------------------------------------------
# Load config.yaml
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def fetch_data():
    lat = config["location"]["lat"]
    lon = config["location"]["lon"]
    api_key = config["api"]["weather_key"]

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )

    response = requests.get(url).json()
    record = response["list"][0]

    # Hourly UTC timestamp (rounded)
    timestamp = datetime.fromtimestamp(
        record["dt"], tz=timezone.utc
    ).replace(minute=0, second=0, microsecond=0)

    components = record["components"]

    df = pd.DataFrame([{
        "timestamp": timestamp,
        "aqi": record["main"]["aqi"],   # AQI (1–5)
        "co": float(components["co"]),
        "no": float(components["no"]),
        "no2": float(components["no2"]),
        "o3": float(components["o3"]),
        "pm10": float(components["pm10"]),
        "pm25": float(components["pm2_5"]),
        "so2": float(components["so2"]),
        "nh3": int(components["nh3"])
    }])

    return df


if __name__ == "__main__":
    print(fetch_data())