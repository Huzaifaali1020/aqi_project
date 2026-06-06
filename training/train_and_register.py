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

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


def main():
    # --------------------------------------------------
    # Connect to Hopsworks
    # --------------------------------------------------
    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # --------------------------------------------------
    # Load FRESH data from Feature Group v2
    # --------------------------------------------------
    fg_v2 = fs.get_feature_group("aqi_features", version=2)

    print("Loading fresh data from Feature Group v2 ...")
    df = fg_v2.read()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(df)} fresh rows")
    print(f"Latest data: {df['timestamp'].max()}")

    TARGET = "aqi_next_hour"

    # check column exists
    if TARGET not in df.columns:
        raise ValueError(
            f"Column {TARGET} not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # clean bad rows
    df = df[(df["aqi"] > 5) & (df["aqi"] < 499)]
    df = df[(df[TARGET] > 5) & (df[TARGET] < 499)]
    df = df.dropna(subset=[TARGET])
    df = df.reset_index(drop=True)

    print(f"After cleaning: {len(df)} rows")

    # time based 80/20 split
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    y_train = train_df.pop(TARGET)
    y_test  = test_df.pop(TARGET)
    X_train = train_df
    X_test  = test_df

    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    # additional clean
    train_mask = (y_train > 5) & (y_train < 499)
    test_mask  = (y_test  > 5) & (y_test  < 499)

    X_train = X_train[train_mask].reset_index(drop=True)
    y_train = y_train[train_mask].reset_index(drop=True)
    X_test  = X_test[test_mask].reset_index(drop=True)
    y_test  = y_test[test_mask].reset_index(drop=True)

    print(f"After cleaning: train={len(X_train)}, test={len(X_test)}")

    # --------------------------------------------------
    # Leakage check
    # --------------------------------------------------
    print("\nLEAKAGE CHECK:")
    print(f"Correlation pm25_lag_1h with target: "
          f"{X_train['pm25_lag_1h'].corr(y_train):.4f}")
    print(f"Correlation temp_lag_1h with target: "
          f"{X_train['temp_lag_1h'].corr(y_train):.4f}")
    print(f"Correlation wind_lag_1h with target: "
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

    print(f"Features used: {X_train.columns.tolist()}")
    print(f"Train size: {X_train.shape}")
    print(f"Test size:  {X_test.shape}")

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
    # Train and Evaluate
    # --------------------------------------------------
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
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

        print(f"{name} — RMSE: {rmse:.4f}  MAE: {mae:.4f}  R²: {r2:.4f}")

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

    print(f"\nBest model : {best_name}")
    print(f"RMSE       : {best_rmse:.4f}")
    print(f"MAE        : {best_mae:.4f}")
    print(f"R²         : {best_r2:.4f}")

    # --------------------------------------------------
    # Save model + plot + register
    # --------------------------------------------------
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\nSaving to Hopsworks (attempt {attempt}/{MAX_RETRIES}) ...")

            with tempfile.TemporaryDirectory() as tmp_dir:

                # save model
                model_path = os.path.join(tmp_dir, "model.pkl")
                joblib.dump(best_model, model_path)

                # save evaluation plot
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
                plt.savefig(os.path.join(tmp_dir, "evaluation.png"))
                plt.close()
                print("Evaluation plot saved")

                # model schema
                input_schema  = Schema(X_train)
                output_schema = Schema(y_test)
                model_schema  = ModelSchema(
                    input_schema=input_schema,
                    output_schema=output_schema
                )


                model_obj = mr.python.create_model(
                    name="aqi_predictor",
                    metrics={
                        "rmse": round(float(best_rmse), 4),
                        "mae":  round(float(best_mae),  4),
                        "r2":   round(float(best_r2),   4),
                    },
                    model_schema=model_schema,
                    description=(
                        f"AQI 6h forecast — {best_name} "
                        f"(RMSE={best_rmse:.2f})"
                    )
                )

                model_obj.save(tmp_dir)

            print("Model successfully saved to Hopsworks Model Registry")
            print("-" * 62)
            print(f"Best model : {best_name}")
            print(f"RMSE       : {best_rmse:.4f}")
            print(f"MAE        : {best_mae:.4f}")
            print(f"R²         : {best_r2:.4f}")
            return

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print("Retrying in 15 seconds ...")
                time.sleep(15)
            else:
                print("All retries failed.")
                raise


if __name__ == "__main__":
    main()