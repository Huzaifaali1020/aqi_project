import hopsworks
import numpy as np
import yaml
import os
import joblib
import tempfile
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def main():
    # --------------------------------------------------
    # Connect to Hopsworks
    # --------------------------------------------------
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # --------------------------------------------------
    # Load training data from Feature View (Option C)
    # --------------------------------------------------
    fv = fs.get_feature_view(
        name="aqi_features_fv",
        version=1
    )

    print("📦 Loading training data from Feature View ...")
    X_train, X_test, y_train, y_test = fv.get_train_test_split(
        training_dataset_version=1
    )

    print(f"✅ Loaded training data")
    print(f"🔹 X_train: {X_train.shape}")
    print(f"🔹 X_test:  {X_test.shape}")

    # --------------------------------------------------
    # Flatten y to Series (Feature View returns DataFrame)
    # --------------------------------------------------
    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    # --------------------------------------------------
    # Clean bad rows
    # --------------------------------------------------
    train_mask = (y_train > 5) & (y_train < 499)
    test_mask  = (y_test  > 5) & (y_test  < 499)

    X_train = X_train[train_mask].reset_index(drop=True)
    y_train = y_train[train_mask].reset_index(drop=True)
    X_test  = X_test[test_mask].reset_index(drop=True)
    y_test  = y_test[test_mask].reset_index(drop=True)

    print(f"📊 After cleaning: train={len(X_train)}, test={len(X_test)}")

    # --------------------------------------------------
    # Leakage check
    # --------------------------------------------------
    print("\n🔍 LEAKAGE CHECK:")
    print(f"Correlation of pm25_lag_1h  with aqi_next_hour: "
          f"{X_train['pm25_lag_1h'].corr(y_train):.4f}")
    print(f"Correlation of temp_lag_1h  with aqi_next_hour: "
          f"{X_train['temp_lag_1h'].corr(y_train):.4f}")
    print(f"Correlation of wind_lag_1h  with aqi_next_hour: "
          f"{X_train['wind_lag_1h'].corr(y_train):.4f}")
    print("──────────────────────────────────────────────\n")

    # --------------------------------------------------
    # Drop leakage columns
    # --------------------------------------------------
    LEAKAGE_COLS = [
        "timestamp",
        "aqi",
        "co", "no", "no2", "o3", "so2", "nh3",
        "pm10", "pm25",
        "temperature", "humidity", "wind_speed", "pressure",
    ]
    cols_to_drop = [c for c in LEAKAGE_COLS if c in X_train.columns]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test  = X_test.drop(columns=cols_to_drop)

    print(f"✅ Features used: {X_train.columns.tolist()}")
    print(f"🔹 Train size: {X_train.shape}")
    print(f"🔹 Test size:  {X_test.shape}")

    # --------------------------------------------------
    # 3 Models
    # --------------------------------------------------
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
    }

    # --------------------------------------------------
    # Train & Evaluate
    # --------------------------------------------------
    results = {}

    for name, model in models.items():
        print(f"\n🤖 Training {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        results[name] = {
            "rmse":  rmse,
            "mae":   mae,
            "r2":    r2,
            "model": model,
            "preds": preds,
        }

        print(f"📈 {name} — RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")

    # --------------------------------------------------
    # Print comparison table
    # --------------------------------------------------
    print("\n" + "-" * 62)
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 62)
    for name, r in sorted(results.items(), key=lambda x: x[1]["rmse"]):
        marker = " ← best" if name == min(
            results, key=lambda x: results[x]["rmse"]) else ""
        print(f"{name:<20} {r['rmse']:>8.4f} "
              f"{r['mae']:>8.4f} {r['r2']:>8.4f}{marker}")
    print("-" * 62)

    # --------------------------------------------------
    # Select best model
    # --------------------------------------------------
    best_name  = min(results, key=lambda x: results[x]["rmse"])
    best       = results[best_name]
    best_model = best["model"]
    best_preds = best["preds"]
    best_rmse  = best["rmse"]
    best_mae   = best["mae"]
    best_r2    = best["r2"]

    print(f"\n🏆 Best model : {best_name}")
    print(f"   RMSE       : {best_rmse:.4f}")
    print(f"   MAE        : {best_mae:.4f}")
    print(f"   R²         : {best_r2:.4f}")

    # --------------------------------------------------
    # Save model + plot + register with lineage
    # --------------------------------------------------
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n💾 Saving to Hopsworks (attempt {attempt}/{MAX_RETRIES}) ...")

            with tempfile.TemporaryDirectory() as tmp_dir:

                # ── Save model file ──────────────────────
                model_path = os.path.join(tmp_dir, "model.pkl")
                joblib.dump(best_model, model_path)

                # ── Save evaluation plot ─────────────────
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(y_test.values, label="Actual AQI",
                        color="steelblue", linewidth=1)
                ax.plot(best_preds,    label="Predicted AQI",
                        color="coral",  linewidth=1, linestyle="--")
                ax.set_title(
                    f"Actual vs Predicted AQI — {best_name} "
                    f"(RMSE={best_rmse:.2f}, MAE={best_mae:.2f}, "
                    f"R²={best_r2:.3f})"
                )
                ax.set_xlabel("Hour index (test set)")
                ax.set_ylabel("AQI")
                ax.legend()
                plt.tight_layout()
                plot_path = os.path.join(tmp_dir, "evaluation.png")
                plt.savefig(plot_path)
                plt.close()
                print("📊 Evaluation plot saved")

                # ── Model schema (correct Hopsworks format) ──
                input_schema  = Schema(X_train)
                output_schema = Schema(y_test)
                model_schema  = ModelSchema(
                    input_schema=input_schema,
                    output_schema=output_schema
                )

                # ── Register with Feature View lineage ───
                model_obj = mr.python.create_model(
                    name="aqi_predictor",
                    metrics={
                        "rmse": round(float(best_rmse), 4),
                        "mae":  round(float(best_mae),  4),
                        "r2":   round(float(best_r2),   4),
                    },
                    model_schema=model_schema,
                    feature_view=fv,
                    training_dataset_version=1,
                    description=(
                        f"AQI 6h forecast — {best_name} "
                        f"(RMSE={best_rmse:.2f})"
                    )
                )

                model_obj.save(tmp_dir)

            print("✅ Model successfully saved to Hopsworks Model Registry")
            print("-" * 62)
            print(f"✅ Best model : {best_name}")
            print(f"✅ RMSE       : {best_rmse:.4f}")
            print(f"✅ MAE        : {best_mae:.4f}")
            print(f"✅ R²         : {best_r2:.4f}")
            print("🎉 Model registered with full Feature View lineage")
            return

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print("🔄 Retrying in 15 seconds ...")
                time.sleep(15)
            else:
                print("❌ All retries failed.")
                raise


if __name__ == "__main__":
    main()