import hopsworks
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import tempfile
from datetime import datetime, timezone, timedelta, date

# --------------------------------------------------
# Paths & config
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

# --------------------------------------------------
# AQI category helper
# --------------------------------------------------
def aqi_category(aqi: float):
    if aqi <= 50:
        return {"category": "Good", "emoji": "🟢"}
    elif aqi <= 100:
        return {"category": "Moderate", "emoji": "🟡"}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive", "emoji": "🟠"}
    elif aqi <= 200:
        return {"category": "Unhealthy", "emoji": "🔴"}
    elif aqi <= 300:
        return {"category": "Very Unhealthy", "emoji": "🟣"}
    else:
        return {"category": "Hazardous", "emoji": "⚫"}


# --------------------------------------------------
# Load model (best RMSE)
# --------------------------------------------------
def load_model(project):
    mr = project.get_model_registry()
    models = mr.get_models("aqi_predictor")

    best_model = None
    best_score = float("inf")

    for m in models:
        metrics = m.training_metrics or {}
        rmse = metrics.get("rmse", 999)
        mae = metrics.get("mae", 999)
        r2 = metrics.get("r2", 0)

        score = rmse + mae - (r2 * 10)

        if score < best_score:
            best_score = score
            best_model = m

    if best_model is None:
        raise RuntimeError("No model found")

    with tempfile.TemporaryDirectory() as tmp:
        best_model.download(tmp)
        model = joblib.load(os.path.join(tmp, "model.pkl"))

    return model


# --------------------------------------------------
# Load feature history
# --------------------------------------------------
def load_latest_features(project):
    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_features_fv", version=1)

    df = fv.get_batch_data()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # IMPORTANT CLEANING
    df = df.fillna(method="ffill").fillna(method="bfill")

    return df


# --------------------------------------------------
# SAFE FEATURE BUILDER (NO HISTORY CORRUPTION)
# --------------------------------------------------
def build_features(history, ts):
    last = history.iloc[-1]

    def safe(col, default=0):
        return float(last[col]) if col in history.columns else default

    row = {
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        "day_of_week": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),

        "pm25_lag_1h": safe("pm25"),
        "pm25_lag_3h": history["pm25"].iloc[-3:].mean(),
        "pm10_lag_1h": safe("pm10"),
        "pm10_lag_3h": history["pm10"].iloc[-3:].mean(),

        "pm25_roll_3h": history["pm25"].iloc[-3:].mean(),
        "pm25_roll_6h": history["pm25"].iloc[-6:].mean(),
        "pm10_roll_3h": history["pm10"].iloc[-3:].mean(),

        "temp_lag_1h": safe("temperature"),
        "humidity_lag_1h": safe("humidity"),
        "wind_lag_1h": safe("wind_speed"),

        "temp_roll_3h": history["temperature"].iloc[-3:].mean(),
        "humidity_roll_3h": history["humidity"].iloc[-3:].mean(),
        "wind_roll_3h": history["wind_speed"].iloc[-3:].mean(),
    }

    # FINAL SAFETY (NO NaN EVER)
    for k in row:
        if pd.isna(row[k]) or np.isinf(row[k]):
            row[k] = 0.0

    return pd.Series(row)


# --------------------------------------------------
# 72-hour forecast (FIXED)
# --------------------------------------------------
def forecast_72h(model, df):

    FEATURE_COLS = [
        "hour", "day", "month", "day_of_week", "is_weekend",
        "pm25_lag_1h", "pm25_lag_3h",
        "pm10_lag_1h", "pm10_lag_3h",
        "pm25_roll_3h", "pm25_roll_6h", "pm10_roll_3h",
        "temp_lag_1h", "humidity_lag_1h", "wind_lag_1h",
        "temp_roll_3h", "humidity_roll_3h", "wind_roll_3h",
    ]

    history = df.copy()

    now = datetime.now(timezone.utc)
    start_ts = now.replace(minute=0, second=0, microsecond=0)
    start_ts += timedelta(hours=(6 - start_ts.hour % 6) % 6)

    predictions = []

    print(f"\nStarting forecast from {start_ts}")

    for step in range(12):
        ts = start_ts + timedelta(hours=step * 6)

        X_row = build_features(history, ts)
        X = pd.DataFrame([X_row])[FEATURE_COLS]

        pred = float(model.predict(X)[0])

        # CLAMP SAFETY
        pred = max(1.0, min(500.0, pred))

        cat = aqi_category(pred)

        predictions.append({
            "timestamp": ts,
            "predicted_aqi": round(pred, 1),
            "category": cat["category"],
            "emoji": cat["emoji"],
            "hours_ahead": (step + 1) * 6
        })

        print(f"+{(step+1)*6}h → {pred:.1f} {cat['emoji']}")

    return pd.DataFrame(predictions)


# --------------------------------------------------
# DAILY AGGREGATION
# --------------------------------------------------
def aggregate_to_3_days(df):
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby("date").agg(
        avg_aqi=("predicted_aqi", "mean"),
        min_aqi=("predicted_aqi", "min"),
        max_aqi=("predicted_aqi", "max"),
        category=("category", lambda x: x.mode()[0]),
        emoji=("emoji", lambda x: x.mode()[0]),
    ).reset_index()

    return daily.round(1)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def run_forecast():

    print("Starting AQI forecast pipeline")

    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )

    model = load_model(project)
    df = load_latest_features(project)

    preds = forecast_72h(model, df)
    daily = aggregate_to_3_days(preds)

    out = os.path.join(BASE_DIR, "data")
    os.makedirs(out, exist_ok=True)

    preds.to_csv(os.path.join(out, "latest_predictions.csv"), index=False)
    daily.to_csv(os.path.join(out, "daily_summary.csv"), index=False)

    print("Forecast completed successfully")


if __name__ == "__main__":
    run_forecast()