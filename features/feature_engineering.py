import os
import hopsworks
import pandas as pd
from ingestion.fetch_data import fetch_data

def run_feature_pipeline():
    # 1. Fetch data
    df = fetch_data()
    print("Data fetched:")
    print(df.head())

    # 2. Login to Hopsworks using ENV key
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"]
    )
    fs = project.get_feature_store()

    # 3. Get or create feature group
    fg = fs.get_or_create_feature_group(
        name="air_quality_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly air quality and weather features"
    )
    # 4. Insert data
    fg.insert(df)

    print("✅ Data successfully inserted into Hopsworks")

if __name__ == "__main__":
    run_feature_pipeline()