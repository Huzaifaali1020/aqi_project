import hopsworks
import yaml
import os

# ✅ IMPORT fetch_data
from ingestion.fetch_data import fetch_data

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def run_feature_pipeline():
    # ✅ SAME AS YESTERDAY (NO ENV VAR)
    api_key_value = config["hopsworks"]["api_key"]

    project = hopsworks.login(
        api_key_value=api_key_value
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    df = fetch_data()
    fg.insert(df)

if __name__ == "__main__":
    run_feature_pipeline()