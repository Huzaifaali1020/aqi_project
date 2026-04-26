# ingestion/backfill.py
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
import yaml
import os
from ingestion.fetch_data import fetch_data

# Load config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

def backfill(days=30):
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Historical AQI + weather data"
    )

    rows = []

    for i in range(days):
        print(f"⏳ Fetching data for {i+1} day(s) ago")
        df = fetch_data()

        # Simulate historical timestamps
        df["timestamp"] = datetime.now() - timedelta(days=i)
        rows.append(df)

    full_df = pd.concat(rows, ignore_index=True)

    fg.insert(full_df)
    print(f"✅ Backfilled {len(full_df)} rows into Feature Store")

if __name__ == "__main__":
    backfill(days=30)