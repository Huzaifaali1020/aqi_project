import hopsworks
import pandas as pd
import numpy as np
import joblib
import yaml
import os
import tempfile
from datetime import datetime, timezone, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


# --------------------------------------------------
# AQI category helper
# --------------------------------------------------
def aqi_category(aqi: float) -> dict:
    if aqi <= 50:
        return {"category": "Good",                    "color": "green",  "emoji": "🟢"}
    elif aqi <= 100:
        return {"category": "Moderate",                "color": "yellow", "emoji": "🟡"}
    elif aqi <= 150:
        return {"category": "Unhealthy for Sensitive", "color": "orange", "emoji": "🟠"}
    elif aqi <= 200:
        return {"category": "Unhealthy",               "color": "red",    "emoji": "🔴"}
    elif aqi <= 300:
        return {"category": "Very Unhealthy",          "color": "purple", "emoji": "🟣"}
    else:
        return {"category": "Hazardous",               "color": "maroon", "emoji": "⚫"}


# --------------------------------------------------
# Load model from Hopsworks registry
# --------------------------------------------------
def load_model(project):
    mr = project.get_model_registry()

    model_meta = mr.get_model(
        name="aqi_predictor",
        version=31
    )

    print(f"📦 Loading model: {model_meta.name} v{model_meta.version}")
    print(f"   RMSE: {model_meta.training_metrics.get('rmse', 'N/A')}")
    print(f"   MAE:  {model_meta.training_metrics.get('mae',  'N/A')}")
    print(f"   R²:   {model_meta.training_metrics.get('r2',   'N/A')}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_meta.download(tmp_dir)
        model_path = os.path.join(tmp_dir, "model.pkl")
        model = joblib.load(model_path)

    print("✅ Model loaded successfully")
    return model


# --------------------------------------------------
# Load latest features from Feature Store
# --------------------------------------------------
def load_latest_features(project) -> pd.DataFrame:
    fs = project.get_feature_store()

    fv = fs.get_feature_view(
        name="aqi_features_fv",
        version=1
    )

    df = fv.get_batch_data()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"📥 Loaded {len(df)} rows from Feature View")
    print(f"📅 Latest timestamp: {df['timestamp'].max()}")

    return df


# --------------------------------------------------
# Update lag and rolling features for next step
# --------------------------------------------------
def update_features(history: pd.DataFrame, new_aqi: float,
                    new_timestamp: datetime) -> tuple:
    """
    Given history DataFrame and new predicted AQI,
    build the feature row for the NEXT prediction step.
    """
    last      = history.iloc[-1].copy()
    last["timestamp"]   = new_timestamp
    last["aqi"]         = new_aqi

    history = pd.concat(
        [history, last.to_frame().T], ignore_index=True
    )

    n   = len(history)
    row = {}

    # ── Time features ────────────────────────────
    row["hour"]        = new_timestamp.hour
    row["day"]         = new_timestamp.day
    row["month"]       = new_timestamp.month
    row["day_of_week"] = new_timestamp.weekday()
    row["is_weekend"]  = int(new_timestamp.weekday() >= 5)

    # ── PM2.5 lags ───────────────────────────────
    row["pm25_lag_1h"] = float(history["pm25"].iloc[-2]) if n >= 2 else np.nan
    row["pm25_lag_3h"] = float(history["pm25"].iloc[-4]) if n >= 4 else np.nan

    # ── PM10 lags ────────────────────────────────
    row["pm10_lag_1h"] = float(history["pm10"].iloc[-2]) if n >= 2 else np.nan
    row["pm10_lag_3h"] = float(history["pm10"].iloc[-4]) if n >= 4 else np.nan

    # ── PM2.5 rolling ────────────────────────────
    row["pm25_roll_3h"] = float(history["pm25"].iloc[-3:].mean()) if n >= 3 else float(history["pm25"].mean())
    row["pm25_roll_6h"] = float(history["pm25"].iloc[-6:].mean()) if n >= 6 else float(history["pm25"].mean())

    # ── PM10 rolling ─────────────────────────────
    row["pm10_roll_3h"] = float(history["pm10"].iloc[-3:].mean()) if n >= 3 else float(history["pm10"].mean())

    # ── Weather lags ─────────────────────────────
    row["temp_lag_1h"]     = float(history["temperature"].iloc[-2]) if n >= 2 else np.nan
    row["humidity_lag_1h"] = float(history["humidity"].iloc[-2])    if n >= 2 else np.nan
    row["wind_lag_1h"]     = float(history["wind_speed"].iloc[-2])  if n >= 2 else np.nan

    # ── Weather rolling ──────────────────────────
    row["temp_roll_3h"]     = float(history["temperature"].iloc[-3:].mean()) if n >= 3 else float(history["temperature"].mean())
    row["humidity_roll_3h"] = float(history["humidity"].iloc[-3:].mean())    if n >= 3 else float(history["humidity"].mean())
    row["wind_roll_3h"]     = float(history["wind_speed"].iloc[-3:].mean())  if n >= 3 else float(history["wind_speed"].mean())

    return pd.Series(row), history


# --------------------------------------------------
# 72-hour recursive forecast
# --------------------------------------------------
def forecast_72h(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Recursively predicts AQI for next 72 hours.
    Predicts every 6 hours = 12 steps total.
    Each prediction feeds into next step as lag features.
    """
    FEATURE_COLS = [
        "hour", "day", "month", "day_of_week", "is_weekend",
        "pm25_lag_1h", "pm25_lag_3h",
        "pm10_lag_1h", "pm10_lag_3h",
        "pm25_roll_3h", "pm25_roll_6h", "pm10_roll_3h",
        "temp_lag_1h", "humidity_lag_1h", "wind_lag_1h",
        "temp_roll_3h", "humidity_roll_3h", "wind_roll_3h",
    ]

    history     = df.copy()
    last_ts     = pd.to_datetime(df["timestamp"].max())
    predictions = []

    print(f"\n🔮 Starting 72h forecast from {last_ts} ...")
    print(f"   Predicting every 6 hours = 12 steps total\n")

    for step in range(1, 13):
        next_ts = last_ts + timedelta(hours=step * 6)

        # use last predicted AQI or last known AQI
        last_aqi = (
            predictions[-1]["predicted_aqi"]
            if predictions
            else float(df["aqi"].iloc[-1])
        )

        feature_row, history = update_features(
            history, last_aqi, next_ts
        )

        X        = pd.DataFrame([feature_row])[FEATURE_COLS]
        pred_aqi = float(model.predict(X)[0])
        pred_aqi = max(0, min(500, pred_aqi))  # clamp to valid AQI range

        cat = aqi_category(pred_aqi)

        predictions.append({
            "timestamp":     next_ts,
            "predicted_aqi": round(pred_aqi, 1),
            "category":      cat["category"],
            "color":         cat["color"],
            "emoji":         cat["emoji"],
            "hours_ahead":   step * 6,
        })

        print(f"   +{step*6:2d}h  {next_ts.strftime('%Y-%m-%d %H:%M')}  "
              f"AQI: {pred_aqi:6.1f}  {cat['emoji']} {cat['category']}")

    return pd.DataFrame(predictions)


# --------------------------------------------------
# Aggregate to 3-day daily summary
# --------------------------------------------------
def aggregate_to_3_days(df_pred: pd.DataFrame) -> pd.DataFrame:
    df_pred["date"] = pd.to_datetime(df_pred["timestamp"]).dt.date

    daily = df_pred.groupby("date").agg(
        avg_aqi  = ("predicted_aqi", "mean"),
        min_aqi  = ("predicted_aqi", "min"),
        max_aqi  = ("predicted_aqi", "max"),
        category = ("category",      lambda x: x.mode()[0]),
        emoji    = ("emoji",         lambda x: x.mode()[0]),
    ).reset_index()

    daily["avg_aqi"] = daily["avg_aqi"].round(1)
    daily["min_aqi"] = daily["min_aqi"].round(1)
    daily["max_aqi"] = daily["max_aqi"].round(1)

    return daily


# --------------------------------------------------
# Main
# --------------------------------------------------
def run_forecast():
    print("🚀 Starting AQI forecast pipeline\n")

    # ── Connect to Hopsworks ─────────────────────
    project = hopsworks.login(
        api_key_value=config["hopsworks"]["api_key"]
    )

    # ── Load model ───────────────────────────────
    model = load_model(project)

    # ── Load latest features ─────────────────────
    df = load_latest_features(project)

    # ── Run 72h forecast ─────────────────────────
    df_predictions = forecast_72h(model, df)

    # ── Aggregate to 3 days ──────────────────────
    df_daily = aggregate_to_3_days(df_predictions)

    # ── Print 3-day summary ──────────────────────
    print("\n" + "=" * 58)
    print("📅  3-DAY AQI FORECAST SUMMARY — KARACHI")
    print("=" * 58)
    for _, row in df_daily.iterrows():
        print(f"  {row['emoji']}  {row['date']}  |  "
              f"Avg: {row['avg_aqi']:5.1f}  |  "
              f"Range: {row['min_aqi']} – {row['max_aqi']}  |  "
              f"{row['category']}")
    print("=" * 58)

    # ── Save predictions locally ─────────────────
    out_dir     = os.path.join(BASE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)

    hourly_path = os.path.join(out_dir, "latest_predictions.csv")
    daily_path  = os.path.join(out_dir, "daily_summary.csv")

    df_predictions.to_csv(hourly_path, index=False)
    df_daily.to_csv(daily_path,        index=False)

    print(f"\n💾 Hourly predictions → {hourly_path}")
    print(f"💾 Daily summary      → {daily_path}")
    print("\n✅ Forecast pipeline completed successfully")

    return df_predictions, df_daily


if __name__ == "__main__":
    run_forecast()