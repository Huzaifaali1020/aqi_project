import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from storage.hopsworks_fs import read_features
from storage.model_registry import save_model

# 1. Load historical features from Feature Store
df = read_features()

# 2. Separate features and target
X = df.drop(columns=["aqi", "timestamp"])
y = df["aqi"]

# 3. Time-based train-test split
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Define models
models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ),
}

results = {}

# 5. Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    results[name] = (rmse, model)

# 6. Select best model
best_model_name = min(results, key=lambda x: results[x][0])
best_model = results[best_model_name][1]

# 7. Save best model to Model Registry
save_model(best_model)

print(f"Best model: {best_model_name}")
print(f"RMSE: {results[best_model_name][0]}")