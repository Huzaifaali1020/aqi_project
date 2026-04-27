import pandas as pd

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    # Rolling
    df["pm25_roll_3h"] = df["pm25"].rolling(3).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(6).mean()
    df["pm10_roll_3h"] = df["pm10"].rolling(3).mean()

    # Target
    df["aqi_next_hour"] = df["aqi"].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df