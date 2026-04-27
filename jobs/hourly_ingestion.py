import hopsworks
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
    # --------------------------------------------------
    # Login to Hopsworks
    # --------------------------------------------------
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    # --------------------------------------------------
    # RAW FEATURE GROUP (v1) — CREATE IF NOT EXISTS
    # --------------------------------------------------
    fg_raw = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Raw hourly air quality data from OpenWeather",
        online_enabled=False
    )

    # --------------------------------------------------
    # Fetch & insert raw data
    # --------------------------------------------------
    df = fetch_data()
    print("📥 Inserting raw data:")
    print(df)

    fg_raw.insert(df, write_options={"wait_for_job": True})
    print("✅ Raw data inserted into FG v1")

    # --------------------------------------------------
    # Run feature engineering → v2
    # --------------------------------------------------
    run_feature_pipeline()
    print("⚙️ Feature engineering pipeline completed")


if __name__ == "__main__":
    run_hourly_ingestion()