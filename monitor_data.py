import hopsworks
import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

project = hopsworks.login(api_key_value=config["hopsworks"]["api_key"])
fs = project.get_feature_store()

# ── Check v1 (raw) ──────────────────────────────
fg_v1 = fs.get_feature_group(name="aqi_features", version=1)
df_v1 = fg_v1.read()
print(f"📊 v1 raw rows:        {len(df_v1)}")
print(f"📅 v1 latest timestamp: {df_v1['timestamp'].max()}")
print(f"📅 v1 oldest timestamp: {df_v1['timestamp'].min()}")

# ── Check v2 (engineered) ───────────────────────
fg_v2 = fs.get_feature_group(name="aqi_features", version=2)
df_v2 = fg_v2.read()
print(f"\n📊 v2 engineered rows:  {len(df_v2)}")
print(f"📅 v2 latest timestamp: {df_v2['timestamp'].max()}")
print(f"📅 v2 oldest timestamp: {df_v2['timestamp'].min()}")

# ── AQI stats ───────────────────────────────────
print(f"\n📈 AQI min:  {df_v1['aqi'].min()}")
print(f"📈 AQI max:  {df_v1['aqi'].max()}")
print(f"📈 AQI mean: {df_v1['aqi'].mean():.1f}")