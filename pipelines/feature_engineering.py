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

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    df["pm25_roll_3h"] = df["pm25"].rolling(3, min_periods=1).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(6, min_periods=1).mean()
    df["pm10_roll_3h"] = df["pm10"].rolling(3, min_periods=1).mean()

    df["temp_lag_1h"] = df["temperature"].shift(1)
    df["humidity_lag_1h"] = df["humidity"].shift(1)
    df["wind_lag_1h"] = df["wind_speed"].shift(1)

    df["temp_roll_3h"] = df["temperature"].rolling(3, min_periods=1).mean()
    df["humidity_roll_3h"] = df["humidity"].rolling(3, min_periods=1).mean()
    df["wind_roll_3h"] = df["wind_speed"].rolling(3, min_periods=1).mean()

    df["aqi_next_hour"] = df["aqi"].shift(-1)

    df = df.dropna(subset=[
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm10_lag_1h",
        "pm10_lag_3h"
    ]).reset_index(drop=True)

    return df


# --------------------------------------------------
# SAFE MATERIALIZATION FUNCTION (IMPORTANT FIX)
# --------------------------------------------------
def safe_materialize(fg):
    state = fg.materialization_job.get_state()

    print(f"📡 Materialization state: {state}")

    if state in ["RUNNING", "SUBMITTED"]:
        print("⛔ Materialization already running — skipping to avoid queue")
        return

    print("🚀 Starting materialization job...")
    fg.materialization_job.run()


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def run_feature_pipeline():
    print("🚀 Starting feature engineering pipeline")

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    # ---------------- RAW FG ----------------
    fg_v1 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False
    )

    df_raw = fg_v1.read().sort_values("timestamp").reset_index(drop=True)
    print(f"📥 Read {len(df_raw)} rows from v1")

    if len(df_raw) < 5:
        print("⏳ Not enough data")
        return

    df_raw = df_raw[(df_raw["aqi"] > 5) & (df_raw["aqi"] < 499)]
    df_raw = df_raw.reset_index(drop=True)

    if len(df_raw) < 5:
        print("⏳ Not enough clean data")
        return

    # ---------------- FEATURE ENGINEERING ----------------
    df_fe = transform_features(df_raw)

    if df_fe.empty:
        print("⚠️ Empty feature dataframe")
        return

    df_fe = df_fe.drop(columns=["event_time", "ingestion_time"], errors="ignore")

    print("📊 Latest engineered row:")
    print(df_fe.tail(1))

    # ---------------- FEATURE GROUP V2 ----------------
    fg_v2 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["timestamp"],
        online_enabled=False
    )

    latest_row = df_fe.dropna(subset=["aqi_next_hour"]).tail(1)

    print("📤 Inserting latest row...")
    fg_v2.insert(latest_row)

    # ---------------- SAFE MATERIALIZATION (FIX) ----------------
    #safe_materialize(fg_v2)

    #while True:
        #state = fg_v2.materialization_job.get_state()
        #print("📡 Materialization state:", state)

        #if state == "FINISHED":
         #   print("✅ Materialization completed")
          #  break

        #if state in ["FAILED", "KILLED"]:
         #   raise Exception(f"❌ Materialization failed: {state}")

        #time.sleep(10)

    # ---------------- VERIFY ----------------
    df_check = fg_v2.read()
    print(df_check.tail())
    print(df_check.shape)

    print("✅ Pipeline completed successfully")


# --------------------------------------------------
# ENTRY
# --------------------------------------------------
if __name__ == "__main__":
    run_feature_pipeline()