import pandas as pd
import numpy as np
import os
import json
import joblib
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Loading Feature Store...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
store = FeatureStore(repo_path=os.path.join(BASE_DIR, "feature_store"))

# ----------------------------
# STEP 1: ENTITY DATAFRAME
# ----------------------------
df = pd.read_csv(os.path.join(BASE_DIR, "feature_store/data/karachi_features.csv"))
df["time"] = pd.to_datetime(df["time"])
entity_df = df[["aqi_id", "time"]].copy()

# ----------------------------
# STEP 2: GET FEATURES FROM FEAST
# ----------------------------
print("Fetching features from Feast...")
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "aqi_features:temperature_2m",
        "aqi_features:relative_humidity_2m",
        "aqi_features:wind_speed_10m",
        "aqi_features:aqi",
        "aqi_features:aqi_change",
        "aqi_features:aqi_lag_1",
        "aqi_features:aqi_lag_2",
        "aqi_features:aqi_lag_3",
        "aqi_features:aqi_rolling_3h",
        "aqi_features:aqi_rolling_6h",
        "aqi_features:hour",
        "aqi_features:day",
        "aqi_features:month",
        "aqi_features:weekday",
        "aqi_features:future_aqi",
    ],
).to_df()

print(f"✅ Loaded {len(training_df)} rows from Feast")
training_df = training_df.dropna()

# ----------------------------
# STEP 3: FEATURES & TARGET
# ----------------------------
FEATURES = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "aqi", "aqi_change", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_rolling_3h", "aqi_rolling_6h", "hour", "day", "month", "weekday"
]
TARGET = "future_aqi"

X = training_df[FEATURES]
y = training_df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# STEP 4: TRAIN MODELS
# ----------------------------
models = {
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "ridge": Ridge(),
    "xgboost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
}

os.makedirs(os.path.join(BASE_DIR, "model_registry"), exist_ok=True)

results = {}
best_model = None
best_score = -np.inf
best_name = ""

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2   = r2_score(y_test, pred)

    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2  : {r2:.4f}")

    results[name] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Save each model
    joblib.dump(model, os.path.join(BASE_DIR, f"model_registry/{name}.pkl"))

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# ----------------------------
# STEP 5: SAVE BEST MODEL
# ----------------------------
joblib.dump(best_model, os.path.join(BASE_DIR, "model_registry/best_model.pkl"))

metadata = {
    "best_model": best_name,
    "r2_score": float(best_score),
    "features": FEATURES,
    "all_models": results
}

with open(os.path.join(BASE_DIR, "model_registry/metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n=============================")
print(f"✅ Best Model : {best_name}")
print(f"✅ Best R2    : {best_score:.4f}")
print("✅ All models saved to model_registry/")
print("=============================")