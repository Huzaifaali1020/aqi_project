import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from storage.hopsworks_fs import read_features
from storage.model_registry import save_model

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = read_features()

print(f"📊 Training data shape: {df.shape}")
print(f"📊 Columns: {df.columns.tolist()}")

MIN_ROWS = 10
if len(df) < MIN_ROWS:
    print(f"❌ Not enough data. Found {len(df)} rows, need {MIN_ROWS}.")
    exit(0)

# --------------------------------------------------
# Clean bad rows
# --------------------------------------------------
df = df[(df["aqi"] > 5) & (df["aqi"] < 499)]
df = df[(df["aqi_next_hour"] > 5) & (df["aqi_next_hour"] < 499)]
df = df.reset_index(drop=True)
print(f"📊 After cleaning: {len(df)} rows")

# ── ADD THIS BLOCK HERE ──────────────────────
print("\n🔍 LEAKAGE CHECK:")
print(df[["timestamp", "aqi", "aqi_next_hour", "pm25_lag_1h", "pm25_roll_3h"]].head(10).to_string())
print(f"\nCorrelation of pm25_lag_1h  with aqi_next_hour: {df['pm25_lag_1h'].corr(df['aqi_next_hour']):.4f}")
print(f"Correlation of pm25_roll_3h with aqi_next_hour: {df['pm25_roll_3h'].corr(df['aqi_next_hour']):.4f}")
print(f"Correlation of temp_lag_1h  with aqi_next_hour: {df['temp_lag_1h'].corr(df['aqi_next_hour']):.4f}")
print(f"Correlation of wind_lag_1h  with aqi_next_hour: {df['wind_lag_1h'].corr(df['aqi_next_hour']):.4f}")
print("──────────────────────────────────────────────\n")
# --------------------------------------------------
# Target & leakage columns
# --------------------------------------------------
TARGET = "aqi_next_hour"

LEAKAGE_COLS = [
    "timestamp",
    "aqi",
    "co", "no", "no2", "o3", "so2", "nh3",
    "pm10", "pm25",
    "temperature", "humidity", "wind_speed", "pressure",
    TARGET
]

cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]

# --------------------------------------------------
# Split FIRST then drop leakage columns
# --------------------------------------------------
split_idx = int(len(df) * 0.8)
train_df  = df.iloc[:split_idx].copy()
test_df   = df.iloc[split_idx:].copy()

X_train = train_df.drop(columns=cols_to_drop)
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=cols_to_drop)
y_test  = test_df[TARGET]

print(f"✅ Features used for training: {X_train.columns.tolist()}")
print(f"✅ Target: {TARGET}")
print(f"🔹 Train size: {X_train.shape}")
print(f"🔹 Test size:  {X_test.shape}")

# --------------------------------------------------
# 3 Models
# --------------------------------------------------
models = {
    "Ridge": Ridge(
        alpha=1.0
    ),
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
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = (rmse, model, preds)
    print(f"📈 {name} RMSE: {rmse:.4f}")

# --------------------------------------------------
# Print comparison table
# --------------------------------------------------
print("\n--------------------------------------------------")
print(f"{'Model':<20} {'RMSE':>10}")
print("--------------------------------------------------")
for name, (rmse, _, __) in sorted(results.items(), key=lambda x: x[1][0]):
    marker = " ← best" if name == min(results, key=lambda x: results[x][0]) else ""
    print(f"{name:<20} {rmse:>10.4f}{marker}")
print("--------------------------------------------------")

# --------------------------------------------------
# Select best model
# --------------------------------------------------
best_model_name             = min(results, key=lambda x: results[x][0])
best_rmse, best_model, best_preds = results[best_model_name]

print(f"\n🏆 Best model: {best_model_name} (RMSE: {best_rmse:.4f})")

# --------------------------------------------------
# Save best model to Hopsworks
# --------------------------------------------------
save_model(
    best_model,
    rmse=best_rmse,
    y_test=y_test,
    y_pred=best_preds
)

print("--------------------------------------------------")
print(f"✅ Best model: {best_model_name}")
print(f"✅ Final RMSE: {best_rmse:.4f}")
print("🎉 Model successfully registered without leakage")