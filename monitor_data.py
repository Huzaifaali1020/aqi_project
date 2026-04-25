import hopsworks
import pandas as pd
import os

def monitor_feature_group():
    # 1. Login to Hopsworks (uses API key from env variable)
    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY")
    )

    fs = project.get_feature_store()

    # 2. Get the feature group
    fg = fs.get_feature_group(
        name="air_quality_features",
        version=1
    )

    # 3. Read all data
    df = fg.read()

    # 4. Show stats
    print("\n==============================")
    print("📊 FEATURE GROUP MONITOR")
    print("==============================")
    print(f"Total rows: {len(df)}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData preview:")
    print(df)

if __name__ == "__main__":
    monitor_feature_group()