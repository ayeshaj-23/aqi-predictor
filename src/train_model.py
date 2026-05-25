import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load engineered dataset
df = pd.read_csv("features/feature_store.csv")

print("Dataset loaded successfully!")
print(df.head())

# Select input features
X = df[
    [
        "pm25",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "dust",
        "uv_index",
        "hour",
        "day",
        "month",
        "aqi_change"
    ]
]

# Target
y = df["future_aqi"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# =========================
# RANDOM FOREST MODEL
# =========================

print("\nTraining Random Forest model...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\n=== Random Forest Results ===")
print(f"MAE: {rf_mae:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R2 Score: {rf_r2:.2f}")

# =========================
# RIDGE MODEL
# =========================

print("\nTraining Ridge Regression model...")

ridge_model = Ridge()

ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\n=== Ridge Regression Results ===")
print(f"MAE: {ridge_mae:.2f}")
print(f"RMSE: {ridge_rmse:.2f}")
print(f"R2 Score: {ridge_r2:.2f}")

# =========================
# SAVE MODELS (NEW PART)
# =========================

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(ridge_model, "models/ridge.pkl")

print("\nModels saved successfully ✔")

# =========================
# MODEL SELECTION (NEW)
# =========================

if ridge_r2 > rf_r2:
    best_model = ridge_model
    best_name = "Ridge Regression"
    best_r2 = ridge_r2
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_r2 = rf_r2

print("\n=========================")
print("BEST MODEL SELECTED")
print("=========================")
print(f"Model: {best_name}")
print(f"R2 Score: {best_r2:.2f}")

joblib.dump(best_model, "models/best_model.pkl")

print("Best model saved as best_model.pkl ✔")