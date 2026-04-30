import hopsworks
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import tempfile
from datetime import datetime, timezone, timedelta
from datetime import date
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
def aqi_category(aqi: float) -> dict:
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
# Load BEST model automatically (lowest RMSE)
# --------------------------------------------------
def load_model(project):
    mr = project.get_model_registry()
    models = mr.get_models("aqi_predictor")

    best_model = None
    best_score = float("inf")
    best_metrics = None

    print("🔍 Selecting best model using RMSE + MAE + R²")

    for m in models:
        metrics = m.training_metrics or {}

        rmse = metrics.get("rmse", float("inf"))
        mae = metrics.get("mae", float("inf"))
        r2 = metrics.get("r2", -1.0)

        # Composite score
        score = rmse + mae - (r2 * 10)

        print(
            f"▶ v{m.version} | "
            f"RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f} → SCORE={score:.3f}"
        )

        if score < best_score:
            best_score = score
            best_model = m
            best_metrics = (rmse, mae, r2)

    if best_model is None:
        raise RuntimeError("❌ No model found")

    rmse, mae, r2 = best_metrics

    print(
        f"\n🏆 BEST MODEL SELECTED\n"
        f"   Version : {best_model.version}\n"
        f"   RMSE    : {rmse:.3f}\n"
        f"   MAE     : {mae:.3f}\n"
        f"   R²      : {r2:.3f}\n"
        f"   SCORE   : {best_score:.3f}"
    )

    with tempfile.TemporaryDirectory() as tmp:
        best_model.download(tmp)
        model = joblib.load(os.path.join(tmp, "model.pkl"))

    return model
# --------------------------------------------------
# Load feature history (context only)
# --------------------------------------------------
def load_latest_features(project):
    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_features_fv", version=1)

    df = fv.get_batch_data()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"📥 Loaded {len(df)} rows from Feature View")
    print(f"📅 Latest feature timestamp: {df['timestamp'].max()}")

    return df


# --------------------------------------------------
# Update lag + rolling features
# --------------------------------------------------
def update_features(history, new_aqi, new_ts):
    last = history.iloc[-1].copy()
    last["timestamp"] = new_ts
    last["aqi"] = new_aqi

    history = pd.concat([history, last.to_frame().T], ignore_index=True)
    n = len(history)

    row = {
        "hour": new_ts.hour,
        "day": new_ts.day,
        "month": new_ts.month,
        "day_of_week": new_ts.weekday(),
        "is_weekend": int(new_ts.weekday() >= 5),

        "pm25_lag_1h": history["pm25"].iloc[-2] if n >= 2 else history["pm25"].mean(),
        "pm25_lag_3h": history["pm25"].iloc[-4] if n >= 4 else history["pm25"].mean(),
        "pm10_lag_1h": history["pm10"].iloc[-2] if n >= 2 else history["pm10"].mean(),
        "pm10_lag_3h": history["pm10"].iloc[-4] if n >= 4 else history["pm10"].mean(),

        "pm25_roll_3h": history["pm25"].iloc[-3:].mean(),
        "pm25_roll_6h": history["pm25"].iloc[-6:].mean(),
        "pm10_roll_3h": history["pm10"].iloc[-3:].mean(),

        "temp_lag_1h": history["temperature"].iloc[-2],
        "humidity_lag_1h": history["humidity"].iloc[-2],
        "wind_lag_1h": history["wind_speed"].iloc[-2],

        "temp_roll_3h": history["temperature"].iloc[-3:].mean(),
        "humidity_roll_3h": history["humidity"].iloc[-3:].mean(),
        "wind_roll_3h": history["wind_speed"].iloc[-3:].mean(),
    }

    return pd.Series(row), history


# --------------------------------------------------
# CORRECT 72-hour forecast (FROM TODAY)
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

    # 🔑 START FROM CURRENT TIME (rounded UP to next 6h)
    now = datetime.now(timezone.utc)
    start_ts = now.replace(minute=0, second=0, microsecond=0)
    start_ts += timedelta(hours=(6 - start_ts.hour % 6) % 6)

    print(f"\n🔮 Starting TRUE 72h forecast from {start_ts}")
    print("   Predicting every 6 hours = 12 steps\n")

    predictions = []
    last_aqi = float(df["aqi"].iloc[-1])

    for step in range(1, 13):
        ts = start_ts + timedelta(hours=(step - 1) * 6)

        feature_row, history = update_features(history, last_aqi, ts)
        X = pd.DataFrame([feature_row])[FEATURE_COLS]

        pred = float(model.predict(X)[0])
        pred = max(0, min(500, pred))

        cat = aqi_category(pred)

        predictions.append({
            "timestamp": ts,
            "predicted_aqi": round(pred, 1),
            "category": cat["category"],
            "emoji": cat["emoji"],
            "hours_ahead": step * 6
        })

        last_aqi = pred

        print(f"   +{step*6:2d}h  {ts:%Y-%m-%d %H:%M}  AQI: {pred:6.1f}  {cat['emoji']}")

    return pd.DataFrame(predictions)


# --------------------------------------------------
# Daily aggregation
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

    daily[["avg_aqi", "min_aqi", "max_aqi"]] = daily[
        ["avg_aqi", "min_aqi", "max_aqi"]
    ].round(1)

    return daily


# --------------------------------------------------
# Main runner
# --------------------------------------------------
def run_forecast():
    print("🚀 Starting AQI forecast pipeline\n")

    project = hopsworks.login(api_key_value=config["hopsworks"]["api_key"])
    model = load_model(project)
    df = load_latest_features(project)

    preds = forecast_72h(model, df)
    daily = aggregate_to_3_days(preds)

    print("\n📅 3-DAY AQI FORECAST — KARACHI")
    print("=" * 55)

    today = date.today()

    for _, r in daily.iterrows():
        circle = "🔴" if r["date"] == today else r["emoji"]
        print(f"{circle} {r['date']} | Avg {r['avg_aqi']} | {r['category']}")
    print("=" * 55)

    out = os.path.join(BASE_DIR, "data")
    os.makedirs(out, exist_ok=True)

    preds.to_csv(os.path.join(out, "latest_predictions.csv"), index=False)
    daily.to_csv(os.path.join(out, "daily_summary.csv"), index=False)

    print("\n✅ Forecast completed successfully")


if __name__ == "__main__":
    run_forecast()