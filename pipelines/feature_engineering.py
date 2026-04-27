import hopsworks
import pandas as pd
import yaml
import os
from datetime import datetime

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
    # Ensure correct datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fill gaps
    df = df.ffill().bfill()

    # -------------------------
    # Time-based features
    # -------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # -------------------------
    # Lag features (SAFE)
    # -------------------------
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    # -------------------------
    # Rolling features
    # -------------------------
    df["pm25_roll_3h"] = df["pm25"].rolling(3).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(6).mean()
    df["pm10_roll_3h"] = df["pm10"].rolling(3).mean()

    # -------------------------
    # Target (NEXT HOUR AQI)
    # -------------------------
    df["aqi_next_hour"] = df["aqi"].shift(-1)

    # -------------------------
    # DROP rows with NaNs
    # (this REQUIRES >= 2 rows)
    # -------------------------
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

    # -------- v1: RAW FEATURES --------
    fg_v1 = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df_raw = fg_v1.read().sort_values("timestamp")

    print(f"📥 Read {len(df_raw)} rows from v1")

    # IMPORTANT: need history
    if len(df_raw) < 2:
        print("⏳ Not enough data yet for feature engineering (need >= 2 rows)")
        return

    # -------- FEATURE ENGINEERING --------
    df_fe = transform_features(df_raw)

    if df_fe.empty:
        print("⚠️ Engineered DataFrame is empty — skipping insert")
        return

    # 🚨 CRITICAL FIX
    # v2 schema DOES NOT have these
    df_fe = df_fe.drop(
        columns=["event_time", "ingestion_time"],
        errors="ignore"
    )

    print("📊 Inserting engineered features:")
    print(df_fe.tail())

    # -------- v2: ENGINEERED FEATURES --------
    fg_v2 = fs.get_feature_group(
        name="aqi_features",
        version=2
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