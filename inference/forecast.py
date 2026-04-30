import hopsworks
import pandas as pd
import yaml
import os
import numpy as np

# --------------------------------------------------
# Load config.yaml
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# Feature engineering function
# --------------------------------------------------
def transform_features(df: pd.DataFrame) -> pd.DataFrame:

    # ---- Basic prep ----
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Normalize column names (VERY IMPORTANT) ----
    df = df.rename(columns={
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "wind_speed": "wind_speed_10m"
    })

    # ---- Time features ----
    df["hour"]        = df["timestamp"].dt.hour
    df["day"]         = df["timestamp"].dt.day
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)

    # ---- Rush hour (Karachi traffic proxy) ----
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # ---- Season encoding ----
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,      # winter
        3: 1, 4: 1, 5: 1,       # spring
        6: 2, 7: 2, 8: 2, 9: 2, # monsoon
        10: 3, 11: 3           # autumn
    })

    # ---- AQI dynamics (CRITICAL) ----
    df["aqi_change_1h"] = df["aqi"].diff(1)
    df["aqi_change_3h"] = df["aqi"].diff(3)

    # ---- Pollution lag features ----
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    # ---- Pollution rolling features ----
    df["pm25_roll_3h"] = df["pm25"].rolling(3).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(6).mean()
    df["pm10_roll_3h"] = df["pm10"].rolling(3).mean()

    # ---- Weather lag features ----
    df["temp_lag_1h"]     = df["temperature_2m"].shift(1)
    df["humidity_lag_1h"] = df["relative_humidity_2m"].shift(1)
    df["wind_lag_1h"]     = df["wind_speed_10m"].shift(1)

    # ---- Weather rolling features ----
    df["temp_roll_3h"]     = df["temperature_2m"].rolling(3).mean()
    df["humidity_roll_3h"] = df["relative_humidity_2m"].rolling(3).mean()
    df["wind_roll_3h"]     = df["wind_speed_10m"].rolling(3).mean()

    # ---- Interaction & stagnation ----
    df["temp_humidity"] = (
        df["temperature_2m"] * df["relative_humidity_2m"] / 100
    )

    df["is_stagnant"] = (df["wind_speed_10m"] < 1.5).astype(int)

    # ---- Target: 6-hour ahead AQI ----
    df["aqi_t_plus_6h"] = df["aqi"].shift(-6)

    # ---- Final cleanup ----
    df = df.dropna().reset_index(drop=True)

    return df

# --------------------------------------------------
# Main feature pipeline
# --------------------------------------------------
def run_feature_pipeline():
    print("🚀 Starting feature engineering pipeline")

    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    # ---- Read raw features (v1) ----
    fg_v1 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Raw hourly air quality + weather data",
        online_enabled=False
    )

    df_raw = fg_v1.read().sort_values("timestamp").reset_index(drop=True)
    print(f"📥 Read {len(df_raw)} rows from v1")

    if len(df_raw) < 24:
        print("⏳ Not enough data yet (need >= 24 rows)")
        return

    # ---- Basic AQI sanity cleaning ----
    df_raw = df_raw[(df_raw["aqi"] > 5) & (df_raw["aqi"] < 499)]
    df_raw = df_raw.reset_index(drop=True)

    # ---- Feature engineering ----
    df_fe = transform_features(df_raw)

    if df_fe.empty:
        print("⚠️ Engineered DataFrame is empty — skipping insert")
        return

    # ---- Drop metadata ----
    df_fe = df_fe.drop(
        columns=["event_time", "ingestion_time"],
        errors="ignore"
    )

    print("📊 Engineered features preview:")
    print(df_fe.tail(3))
    print(f"\n📊 Total features: {len(df_fe.columns)}")

    # ---- Insert engineered features (v2) ----
    fg_v2 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["timestamp"],
        description="Engineered hourly AQI + weather features",
        online_enabled=False
    )

    fg_v2.insert(
        df_fe,
        write_options={"wait_for_job": True}
    )

    print("✅ Feature engineering pipeline completed successfully")

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    run_feature_pipeline()