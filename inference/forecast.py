import hopsworks
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import tempfile
from datetime import datetime, timezone, timedelta

# --------------------------------------------------
# Paths & config
# --------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


# --------------------------------------------------
# AQI category helper
# --------------------------------------------------
def aqi_category(aqi: float):
    if aqi <= 50:
        return {"category": "Good",                      "emoji": "🟢"}
    elif aqi <= 100:
        return {"category": "Moderate",                  "emoji": "🟡"}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive",   "emoji": "🟠"}
    elif aqi <= 200:
        return {"category": "Unhealthy",                 "emoji": "🔴"}
    elif aqi <= 300:
        return {"category": "Very Unhealthy",            "emoji": "🟣"}
    else:
        return {"category": "Hazardous",                 "emoji": "⚫"}


# --------------------------------------------------
# Load best model from registry
# --------------------------------------------------
def load_model(project):
    mr     = project.get_model_registry()
    models = mr.get_models("aqi_predictor")

    best_model = None
    best_rmse  = float("inf")

    for m in models:
        metrics = m.training_metrics or {}
        rmse    = float(metrics.get("rmse", 9999))
        if rmse < best_rmse:
            best_rmse  = rmse
            best_model = m

    if best_model is None:
        raise RuntimeError("No model found in registry")

    print(f"Loading model v{best_model.version} — RMSE: {best_rmse:.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        best_model.download(tmp)
        model = joblib.load(os.path.join(tmp, "model.pkl"))

    return model


# --------------------------------------------------
# Load feature history from Feature View
# --------------------------------------------------
def load_latest_features(project):
    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_features_fv", version=1)

    df = fv.get_batch_data()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # forward fill only — no bfill to avoid future leakage
    df = df.ffill()

    print(f"Loaded {len(df)} rows from Feature View")
    print(f"Latest timestamp: {df['timestamp'].max()}")

    # validate data is not empty or all zeros
    if df["pm25"].mean() < 0.1:
        raise ValueError(
            f"Feature data looks corrupt — pm25 mean is {df['pm25'].mean():.4f}. "
            "Check Hopsworks Feature Store."
        )

    return df


# --------------------------------------------------
# Build feature row for one prediction step
# --------------------------------------------------
def build_features(history: pd.DataFrame, ts: datetime) -> pd.Series:

    def col_mean(col, n):
        vals = history[col].iloc[-n:] if len(history) >= n else history[col]
        mean = vals.mean()
        return float(mean) if not pd.isna(mean) else 0.0

    def col_last(col):
        val = history[col].iloc[-1] if len(history) > 0 else 0.0
        return float(val) if not pd.isna(val) else 0.0

    row = {
        # time features
        "hour":        ts.hour,
        "day":         ts.day,
        "month":       ts.month,
        "day_of_week": ts.weekday(),
        "is_weekend":  int(ts.weekday() >= 5),

        # PM2.5 lags
        "pm25_lag_1h": col_last("pm25"),
        "pm25_lag_3h": col_mean("pm25", 3),

        # PM10 lags
        "pm10_lag_1h": col_last("pm10"),
        "pm10_lag_3h": col_mean("pm10", 3),

        # PM rolling
        "pm25_roll_3h": col_mean("pm25", 3),
        "pm25_roll_6h": col_mean("pm25", 6),
        "pm10_roll_3h": col_mean("pm10", 3),

        # weather lags
        "temp_lag_1h":     col_last("temperature"),
        "humidity_lag_1h": col_last("humidity"),
        "wind_lag_1h":     col_last("wind_speed"),

        # weather rolling
        "temp_roll_3h":     col_mean("temperature", 3),
        "humidity_roll_3h": col_mean("humidity",    3),
        "wind_roll_3h":     col_mean("wind_speed",  3),
    }

    # safety — replace any NaN or inf
    for k, v in row.items():
        if pd.isna(v) or np.isinf(v):
            row[k] = 0.0

    return pd.Series(row)


# --------------------------------------------------
# 72-hour recursive forecast
# --------------------------------------------------
def forecast_72h(model, df: pd.DataFrame) -> pd.DataFrame:

    FEATURE_COLS = [
        "hour", "day", "month", "day_of_week", "is_weekend",
        "pm25_lag_1h", "pm25_lag_3h",
        "pm10_lag_1h", "pm10_lag_3h",
        "pm25_roll_3h", "pm25_roll_6h", "pm10_roll_3h",
        "temp_lag_1h", "humidity_lag_1h", "wind_lag_1h",
        "temp_roll_3h", "humidity_roll_3h", "wind_roll_3h",
    ]

    history  = df.copy()
    now      = datetime.now(timezone.utc)
    # align to next 6-hour boundary
    start_ts = now.replace(minute=0, second=0, microsecond=0)
    remainder = start_ts.hour % 6
    if remainder != 0:
        start_ts += timedelta(hours=(6 - remainder))

    predictions = []
    print(f"\nStarting 72h forecast from {start_ts}")

    for step in range(12):
        ts = start_ts + timedelta(hours=step * 6)

        feature_row = build_features(history, ts)
        X           = pd.DataFrame([feature_row])[FEATURE_COLS]

        # validate features before predicting
        if X.isnull().any().any():
            print(f"Warning: NaN in features at step {step}, filling with 0")
            X = X.fillna(0.0)

        pred     = float(model.predict(X)[0])
        pred     = max(10.0, min(500.0, pred))   # clamp — minimum 10, not 1

        cat = aqi_category(pred)

        predictions.append({
            "timestamp":     ts,
            "predicted_aqi": round(pred, 1),
            "category":      cat["category"],
            "emoji":         cat["emoji"],
            "hours_ahead":   (step + 1) * 6,
        })

        print(f"  +{(step+1)*6:2d}h  {ts.strftime('%Y-%m-%d %H:%M')}  "
              f"AQI: {pred:6.1f}  {cat['emoji']} {cat['category']}")

        # update history with predicted AQI so next step uses it
        new_row              = history.iloc[-1].copy()
        new_row["timestamp"] = ts
        new_row["aqi"]       = pred
        new_row["pm25"]      = history["pm25"].iloc[-1]  # keep last real PM2.5
        history = pd.concat(
            [history, new_row.to_frame().T], ignore_index=True
        )

    return pd.DataFrame(predictions)


# --------------------------------------------------
# Aggregate to 3-day daily summary
# --------------------------------------------------
def aggregate_to_3_days(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = df.groupby("date").agg(
        avg_aqi  = ("predicted_aqi", "mean"),
        min_aqi  = ("predicted_aqi", "min"),
        max_aqi  = ("predicted_aqi", "max"),
        category = ("category", lambda x: x.mode()[0]),
        emoji    = ("emoji",    lambda x: x.mode()[0]),
    ).reset_index()

    daily["avg_aqi"] = daily["avg_aqi"].round(1)
    daily["min_aqi"] = daily["min_aqi"].round(1)
    daily["max_aqi"] = daily["max_aqi"].round(1)

    return daily


# --------------------------------------------------
# Main
# --------------------------------------------------
def run_forecast():
    print("Starting AQI forecast pipeline\n")

    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )

    model = load_model(project)
    df    = load_latest_features(project)

    # save latest features locally for dashboard home page
    out_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "latest_features.csv"), index=False)
    print("Saved latest_features.csv")

    preds = forecast_72h(model, df)
    daily = aggregate_to_3_days(preds)

    # print summary
    print(f"\n{'='*55}")
    print("3-DAY FORECAST SUMMARY")
    print(f"{'='*55}")
    for _, row in daily.iterrows():
        print(f"  {row['emoji']}  {row['date']}  "
              f"Avg: {row['avg_aqi']:5.1f}  "
              f"Range: {row['min_aqi']} - {row['max_aqi']}  "
              f"{row['category']}")
    print(f"{'='*55}\n")

    # save CSVs
    preds.to_csv(os.path.join(out_dir, "latest_predictions.csv"), index=False)
    daily.to_csv(os.path.join(out_dir, "daily_summary.csv"),       index=False)

    print("Saved latest_predictions.csv")
    print("Saved daily_summary.csv")
    print("\nForecast pipeline completed successfully")


if __name__ == "__main__":
    run_forecast()