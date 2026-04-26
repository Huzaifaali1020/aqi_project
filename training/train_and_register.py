import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from storage.hopsworks_fs import read_features
from storage.model_registry import save_model

df = read_features()

print(f"Training data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

MIN_ROWS = 3   # temporarily low for testing
if len(df) < MIN_ROWS:
    print(f"❌ Not enough data to train. Found {len(df)} rows, need {MIN_ROWS}.")
    exit(0)

X = df.drop(columns=["aqi_next_hour", "timestamp"])
y = df["aqi_next_hour"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # ✅ Compatible RMSE calculation
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = (rmse, model)

best_model_name = min(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]

save_model(best_model)

print(f"✅ Best model: {best_model_name}")
print(f"✅ RMSE: {results[best_model_name][0]}")