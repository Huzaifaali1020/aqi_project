import requests
import pandas as pd
from datetime import datetime, timezone
import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


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


def fetch_current_weather(lat: float, lon: float) -> dict:
    """Fetch current weather from Open-Meteo — completely free, no API key"""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relativehumidity_2m,windspeed_10m,surface_pressure"
        f"&timezone=UTC"
    )
    response = requests.get(url).json()

    if "current" not in response:
        print(f"⚠️ Open-Meteo error: {response}")
        return {
            "temperature": 0.0,
            "humidity":    0.0,
            "wind_speed":  0.0,
            "pressure":    0.0,
        }

    current = response["current"]
    return {
        "temperature": float(current["temperature_2m"]),
        "humidity":    float(current["relativehumidity_2m"]),
        "wind_speed":  float(current["windspeed_10m"]),
        "pressure":    float(current["surface_pressure"]),
    }


def fetch_data():
    lat     = config["city"]["lat"]
    lon     = config["city"]["lon"]
    api_key = config["api"]["weather_key"]

    # ── 1. Air pollution from OpenWeather ────────
    pollution_url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )
    pollution_resp = requests.get(pollution_url).json()
    record         = pollution_resp["list"][0]
    components     = record["components"]

    # wall-clock timestamp — never repeats
    timestamp = datetime.now(tz=timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )

    pm25     = float(components["pm2_5"])
    real_aqi = pm25_to_aqi(pm25)

    # ── 2. Weather from Open-Meteo (free) ────────
    weather = fetch_current_weather(lat, lon)

    print(f"📍 timestamp  : {timestamp}")
    print(f"📍 pm25={pm25} → AQI={real_aqi}")
    print(f"🌡️  temp={weather['temperature']}°C  "
          f"humidity={weather['humidity']}%  "
          f"wind={weather['wind_speed']}m/s  "
          f"pressure={weather['pressure']}hPa")

    df = pd.DataFrame([{
        "timestamp":   timestamp,
        "aqi":         real_aqi,
        "co":          float(components["co"]),
        "no":          float(components["no"]),
        "no2":         float(components["no2"]),
        "o3":          float(components["o3"]),
        "pm10":        float(components["pm10"]),
        "pm25":        pm25,
        "so2":         float(components["so2"]),
        "nh3":         float(components["nh3"]),
        "temperature": weather["temperature"],
        "humidity":    weather["humidity"],
        "wind_speed":  weather["wind_speed"],
        "pressure":    weather["pressure"],
    }])

    return df


if __name__ == "__main__":
    print(fetch_data())