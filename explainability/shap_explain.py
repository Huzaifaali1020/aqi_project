import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hopsworks
import yaml
import os
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PLOT_DIR = os.path.join(BASE_DIR, "explainability", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

FEATURE_COLS = [
    "hour", "day", "month", "day_of_week", "is_weekend",
    "pm25_lag_1h", "pm25_lag_3h",
    "pm10_lag_1h", "pm10_lag_3h",
    "pm25_roll_3h", "pm25_roll_6h", "pm10_roll_3h",
    "temp_lag_1h", "humidity_lag_1h", "wind_lag_1h",
    "temp_roll_3h", "humidity_roll_3h", "wind_roll_3h",
]

LEAKAGE_COLS = [
    "timestamp", "aqi",
    "co", "no", "no2", "o3", "so2", "nh3",
    "pm10", "pm25",
    "temperature", "humidity", "wind_speed", "pressure",
    "aqi_next_hour"
]


def load_model_and_data():
    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )

    mr         = project.get_model_registry()
    models     = mr.get_models(name="aqi_predictor")
    model_meta = sorted(models, key=lambda m: m.version)[-1]
    print(f" Loading model v{model_meta.version}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_meta.download(tmp_dir)
        model = joblib.load(os.path.join(tmp_dir, "model.pkl"))

    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_features_fv", version=1)
    df = fv.get_batch_data()
    df = df.sort_values("timestamp").reset_index(drop=True)

    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    X = X[FEATURE_COLS].dropna().reset_index(drop=True)

    print(f" Data loaded: {X.shape}")
    return model, X


def run_shap_analysis(model, X):
    print(" Computing SHAP values ...")
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    # ── Bar chart ────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(" Saved: shap_importance.png")

    # ── Beeswarm ─────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary — Feature Impact on AQI", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(" Saved: shap_summary.png")

    # ── Dependence plots for top 3 features ──────
    top_features = X.columns[
        np.argsort(np.abs(shap_values.values).mean(0))[::-1]
    ][:3]

    for feat in top_features:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values.values, X, show=False)
        plt.title(f"SHAP Dependence — {feat}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"shap_dep_{feat}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Saved: shap_dep_{feat}.png")

    # ── Print importance table ───────────────────
    importance = pd.DataFrame({
        "Feature":    X.columns,
        "SHAP_Value": np.abs(shap_values.values).mean(0)
    }).sort_values("SHAP_Value", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 45)
    print(" FEATURE IMPORTANCE (SHAP)")
    print("=" * 45)
    for _, row in importance.iterrows():
        bar = "█" * int(row["SHAP_Value"] * 10)
        print(f"{row['Feature']:<20} {bar} {row['SHAP_Value']:.4f}")
    print("=" * 45)

    return shap_values


if __name__ == "__main__":
    print(" Starting SHAP analysis\n")
    model, X = load_model_and_data()
    run_shap_analysis(model, X)
    print(f"\n Plots saved to: {PLOT_DIR}")
    print(" SHAP analysis complete")