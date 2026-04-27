import hopsworks
import yaml
import os
import pandas as pd

from features.transform_features import transform_features

# ----------------------------
# Load config
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ----------------------------
# Feature Pipeline
# ----------------------------
def run_feature_pipeline():

    # Login
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    # ----------------------------
    # Feature Group v1 (RAW)
    # ----------------------------
    fg_v1 = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df_raw = fg_v1.read()

    # ----------------------------
    # Transform features
    # ----------------------------
    df_transformed = transform_features(df_raw)

    # ----------------------------
    # Feature Group v2 (ENGINEERED)
    # ----------------------------
    fg_v2 = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        primary_key=["timestamp"],
        event_time="event_time",
        description="Engineered AQI features",
        online_enabled=False
    )

    # ============================
    # ✅ STEP 4 — INSERT INTO v2
    # ============================
    print("📊 Inserting engineered features:")
    print(df_transformed.head())

    fg_v2.insert(
        df_transformed,
        write_options={"wait_for_job": True}
    )

    print("✅ Feature Group v2 populated successfully")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    run_feature_pipeline()