import pandas as pd

def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Change features
    df["pm25_change"] = df["pm25"].diff().fillna(0)
    df["pm10_change"] = df["pm10"].diff().fillna(0)

    # Lag features
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)

    # Rolling features
    df["pm25_roll_3h"] = df["pm25"].rolling(3).mean()
    df["pm25_roll_6h"] = df["pm25"].rolling(6).mean()

    # Target
    df["aqi_next_hour"] = df["aqi"].shift(-1)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

