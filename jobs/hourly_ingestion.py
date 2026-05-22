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

print("CONFIG:", config)

# --------------------------------------------------
# Hourly ingestion pipeline
# --------------------------------------------------
def run_hourly_ingestion():
    hops_config = config.get("hopsworks", {})
    host = hops_config.get("host")
    api_key = hops_config.get("api_key")

    if not host or not api_key:
        raise ValueError("Missing hopsworks config in YAML")

    project = hopsworks.login(
        host=host,
        api_key_value=api_key
    )
    fs = project.get_feature_store()

    # Raw feature group (SOURCE OF TRUTH)
    fg_raw = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Raw hourly air quality + weather data",
        online_enabled=False
    )

    df = fetch_data()

    if df is None or df.empty:
        print("⚠️ No data fetched")
        return

    print(f"📥 Inserting raw data for {df['timestamp'].iloc[0]}")
    fg_raw.insert(df, write_options={"wait_for_job": True})
    print("✅ Raw data inserted into aqi_features v1")

    # Run feature engineering AFTER raw ingestion
    run_feature_pipeline()
    print("⚙️ Feature engineering completed successfully")


if __name__ == "__main__":
    run_hourly_ingestion()