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
        name="aqi_features",   # ✅ FIXED
        version=1
    )

    fv = fs.get_or_create_feature_view(
        name="air_quality_fv",
        version=1,
        query=fg.select_all(),
        description="Feature view for AQI prediction"
    )

    # 🔥 THIS IS CRITICAL
    fv.create_batch_scoring_dataset()

    print("✅ Feature View created & materialized")


if __name__ == "__main__":
    create_feature_view()