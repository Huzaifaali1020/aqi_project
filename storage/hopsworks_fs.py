import hopsworks
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def read_features():
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )

    fs = project.get_feature_store()

    fv = fs.get_feature_view(
        name="air_quality_fv",   # must match feature view
        version=1
    )

    df = fv.get_batch_data()

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"✅ Loaded {len(df)} rows from Feature View")
    print("Columns:", list(df.columns))

    return df