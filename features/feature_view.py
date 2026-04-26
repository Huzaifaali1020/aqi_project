import hopsworks
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def create_feature_view():
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name="aqi_features",
        version=1
    )

    fv = fs.get_or_create_feature_view(
        name="aqi_features_fv",
        version=1,
        query=fg.select_all(),
        description="AQI feature view for training"
    )

    print("✅ Feature View created (or already exists)")

if __name__ == "__main__":
    create_feature_view()