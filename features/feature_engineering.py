def create_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    df = df.sort_values("timestamp")

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_roll_mean"] = df["aqi"].rolling(3).mean()
    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1"]

    return df.dropna()