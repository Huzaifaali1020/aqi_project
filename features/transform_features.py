import pandas as pd

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    # ----------------------------
    # Timestamp handling
    # ----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ----------------------------
    # Forward fill ONLY (no leakage)
    # ----------------------------
    df.ffill(inplace=True)

    # ----------------------------
    # Time-based features
    # ----------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ----------------------------
    # Lag features
    # ----------------------------
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)

    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    # ----------------------------
    # Rolling features
    # ----------------------------
    df["pm25_roll_3h"] = df["pm25"].rolling(window=3, min_periods=1).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(window=6, min_periods=1).mean()
    df["pm10_roll_3h"] = df["pm10"].rolling(window=3, min_periods=1).mean()

    # ----------------------------
    # Target variable
    # ----------------------------
    df["aqi_next_hour"] = df["aqi"].shift(-1)

    # ----------------------------
    # REQUIRED COLUMNS GUARANTEE
    # ----------------------------
    required_cols = [
        "pm10_lag_1h",
        "pm10_lag_3h",
        "pm10_roll_3h"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # ----------------------------
    # Event / ingestion metadata
    # ----------------------------
    df["event_time"] = df["timestamp"]
    df["ingestion_time"] = pd.Timestamp.utcnow()

    # ----------------------------
    # Final cleanup
    # ----------------------------
    model_required = [
        "pm25_lag_1h",
        "pm25_lag_3h",
        "pm10_lag_1h",
        "pm10_lag_3h",
        "aqi_next_hour"
    ]

    df.dropna(subset=model_required, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df