import hopsworks
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def create_training_data():
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()

    fv = fs.get_feature_view(
        name="aqi_features_fv",
        version=1
    )

    print("📦 Creating versioned training dataset ...")

    X_train, X_test, y_train, y_test = fv.train_test_split(
        test_size=0.2,
        description="AQI training dataset with weather features"
    )

    print(f"✅ Training dataset created")
    print(f"🔹 X_train: {X_train.shape}")
    print(f"🔹 X_test:  {X_test.shape}")
    print(f"🔹 Features: {X_train.columns.tolist()}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    create_training_data()