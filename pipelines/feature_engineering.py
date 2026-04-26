import hopsworks
import yaml
import os
from ingestion.fetch_data import fetch_data
from features.transform_features import transform_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def run_feature_pipeline():
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )

    fs = project.get_feature_store()


    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly air quality and weather features"
    )

    df = fetch_data()
    df = transform_features(df)
    print("Data fetched:")
    print(df)

    fg.insert(df)
    print("Data successfully inserted into Hopsworks")

if __name__ == "__main__":
    run_feature_pipeline()