import os
os.environ["TMP"] = r"C:\tmp"
os.environ["TEMP"] = r"C:\tmp"

import hopsworks
import yaml
from pathlib import Path

# Load config
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Login
project = hopsworks.login(
    api_key_value=config["hopsworks"]["api_key"]
)

fs = project.get_feature_store()

# Create / get Feature Group
fg = fs.get_or_create_feature_group(
    name="karachi_aqi_features",
    version=1,
    primary_key=["timestamp"],
    description="AQI and weather features for Karachi"
)

print("✅ Feature Group ready:", fg.name)