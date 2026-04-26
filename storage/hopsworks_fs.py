import hopsworks
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def read_features():
    project = hopsworks.login(api_key_value=config["hopsworks"]["api_key"])
    fs = project.get_feature_store()

    fv = fs.get_feature_view(
        name="aqi_features_fv",
        version=1
    )

    # ✅ THIS reads the offline training dataset
    df = fv.get_batch_data()

    if df.empty:
        raise ValueError("❌ No training data found")

    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"✅ Loaded {len(df)} rows from Feature View")
    print("Columns:", df.columns.tolist())

    return df