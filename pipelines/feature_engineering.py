import hopsworks
import pandas as pd
import yaml
import os
import time

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

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.ffill()

    df["hour"]        = df["timestamp"].dt.hour
    df["day"]         = df["timestamp"].dt.day
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)

    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    df["pm25_roll_3h"] = df["pm25"].rolling(3, min_periods=1).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(6, min_periods=1).mean()
    df["pm10_roll_3h"] = df["pm10"].rolling(3, min_periods=1).mean()

    df["temp_lag_1h"]     = df["temperature"].shift(1)
    df["humidity_lag_1h"] = df["humidity"].shift(1)
    df["wind_lag_1h"]     = df["wind_speed"].shift(1)

    df["temp_roll_3h"]     = df["temperature"].rolling(3, min_periods=1).mean()
    df["humidity_roll_3h"] = df["humidity"].rolling(3, min_periods=1).mean()
    df["wind_roll_3h"]     = df["wind_speed"].rolling(3, min_periods=1).mean()

    df["aqi_next_hour"] = df["aqi"].shift(-1)

    df = df.dropna(subset=[
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm10_lag_1h",
        "pm10_lag_3h",

    ]).reset_index(drop=True)

    df["aqi_next_hour"] = df["aqi_next_hour"].fillna(df["aqi"])

    return df


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def run_feature_pipeline():
    print(" Starting feature engineering pipeline")

    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    # ── Read from v1 ─────────────────────────────
    fg_v1 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        online_enabled=False
    )

    df_raw = fg_v1.read().sort_values("timestamp").reset_index(drop=True)
    print(f" Read {len(df_raw)} rows from v1")

    if len(df_raw) < 10:
        print(" Not enough data")
        return

    # ── Clean bad rows ───────────────────────────
    df_raw = df_raw[
        (df_raw["aqi"] > 5) & (df_raw["aqi"] < 499)
    ].reset_index(drop=True)

    if len(df_raw) < 10:
        print(" Not enough clean data")
        return

    # ── Feature engineering ──────────────────────
    df_fe = transform_features(df_raw)

    if df_fe.empty:
        print(" Empty feature dataframe — skipping")
        return

    df_fe = df_fe.drop(
        columns=["event_time", "ingestion_time"],
        errors="ignore"
    )

    print(f" Engineered {len(df_fe)} rows")
    print(f" Latest timestamp: {df_fe['timestamp'].max()}")

    # ── Insert into v2 with wait ──────────────────
    fg_v2 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["timestamp"],
        online_enabled=False
    )

    print(" Inserting into v2 (waiting for job to finish)...")

    fg_v2.insert(
        df_fe,
        write_options={"wait_for_job": False}   # ← KEY FIX
    )

    print(" Feature engineering pipeline completed successfully")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    run_feature_pipeline()