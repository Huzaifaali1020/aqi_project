import hopsworks
import yaml
import os
import joblib
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def save_model(model, model_name="aqi_predictor", rmse=None, y_test=None, y_pred=None):
    """
    Saves trained model to Hopsworks Model Registry with metrics and evaluation plot
    """

    MAX_RETRIES = 3

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"💾 Connecting to Hopsworks (attempt {attempt}/{MAX_RETRIES}) ...")

            project = hopsworks.login(
                api_key_value=config["hopsworks"]["api_key"]
            )

            mr = project.get_model_registry()

            with tempfile.TemporaryDirectory() as tmp_dir:

                # ── Save model file ──────────────────────────
                model_path = os.path.join(tmp_dir, "model.pkl")
                joblib.dump(model, model_path)

                # ── Save evaluation plot ─────────────────────
                if y_test is not None and y_pred is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(y_test.values, label="Actual AQI",    color="steelblue")
                    ax.plot(y_pred,        label="Predicted AQI", color="coral", linestyle="--")
                    ax.set_title("Actual vs Predicted AQI (Test Set)")
                    ax.set_xlabel("Hour")
                    ax.set_ylabel("AQI")
                    ax.legend()
                    plt.tight_layout()
                    plot_path = os.path.join(tmp_dir, "evaluation.png")
                    plt.savefig(plot_path)
                    plt.close()
                    print("📊 Evaluation plot saved")

                # ── Create model object with metrics ─────────
                metrics = {"rmse": round(float(rmse), 4)} if rmse is not None else {}

                model_obj = mr.python.create_model(
                    name=model_name,
                    metrics=metrics,
                    description="AQI prediction model (weather + pollution features)"
                )

                model_obj.save(tmp_dir)  # saves entire folder (model + plot)

            print("✅ Model successfully saved to Hopsworks Model Registry")
            return

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"🔄 Retrying in 15 seconds ...")
                time.sleep(15)
            else:
                print("❌ All retries failed.")
                raise