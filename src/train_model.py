import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("features/feature_store.csv")

print("Dataset loaded successfully!")
print(df.head())


# =========================
# FEATURES & TARGET
# =========================

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

y = df["future_aqi"]


# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


# =========================
# RANDOM FOREST
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
# RIDGE REGRESSION
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
# XGBOOST MODEL
# =========================

print("\nTraining XGBoost model...")

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\n=== XGBoost Results ===")
print(f"MAE: {xgb_mae:.2f}")
print(f"RMSE: {xgb_rmse:.2f}")
print(f"R2 Score: {xgb_r2:.2f}")


# =========================
# MODEL SELECTION
# =========================

print("\n=========================")
print("BEST MODEL SELECTION")
print("=========================")

best_model = rf_model
best_score = rf_r2

best_name = "Random Forest"

if ridge_r2 > best_score:
    best_model = ridge_model
    best_score = ridge_r2
    best_name = "Ridge Regression"

if xgb_r2 > best_score:
    best_model = xgb_model
    best_score = xgb_r2
    best_name = "XGBoost"


print(f"Model: {best_name}")
print(f"R2 Score: {best_score:.2f}")


# =========================
# SAVE MODEL
# =========================

joblib.dump(best_model, "models/best_model.pkl")

print("Best model saved successfully ✔")