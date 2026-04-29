import hopsworks
import pandas as pd
import yaml
import os

from ingestion.fetch_data import fetch_data
from pipelines.feature_engineering import run_feature_pipeline

# --------------------------------------------------
# Load config
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def run_hourly_ingestion():
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    fg_raw = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Raw hourly air quality + weather data",
        online_enabled=False
    )

    df = fetch_data()
    if df is None:
        print("⚠️ No data fetched")
        return

    new_timestamp = df["timestamp"].iloc[0]

    # ── Skip if already exists ───────────────────
    try:
        existing = fg_raw.read()
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        if new_timestamp in existing["timestamp"].values:
            print(f"⚠️ {new_timestamp} already exists — skipping")
            return
    except Exception:
        pass

    print(f"📥 Inserting new row for {new_timestamp}")
    fg_raw.insert(df, write_options={"wait_for_job": True})
    print("✅ Raw data inserted into v1")

    run_feature_pipeline()
    print("⚙️ Feature engineering completed")


if __name__ == "__main__":
    run_hourly_ingestion()