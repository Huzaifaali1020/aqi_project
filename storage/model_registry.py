# storage/model_registry.py
import hopsworks
import yaml
import os
import joblib
import tempfile


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def save_model(model, model_name="aqi_predictor"):
    """
    Saves trained model to Hopsworks Model Registry
    """

    api_key_value = config["hopsworks"]["api_key"]

    project = hopsworks.login(
        api_key_value=api_key_value
    )

    mr = project.get_model_registry()

    # Save model temporarily
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model.pkl")
        joblib.dump(model, model_path)

        model_obj = mr.python.create_model(
            name=model_name,
            description="AQI prediction model (weather + pollution features)"
        )

        model_obj.save(model_path)

    print("Model successfully saved to Hopsworks Model Registry")