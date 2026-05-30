import pandas as pd
import numpy as np
import os

# =========================
# LOAD RAW DATA
# =========================
df = pd.read_csv("feature_store/data/karachi_3months.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# =========================
# SYNTHETIC AQI
# =========================
df["aqi"] = (
    df["temperature_2m"] * 1.1 +
    df["relative_humidity_2m"] * 0.6 +
    df["wind_speed_10m"] * 2.0
).round(2)

# =========================
# TIME FEATURES
# =========================
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday

# =========================
# ENGINEERED FEATURES
# =========================
df["aqi_change"] = df["aqi"].diff().fillna(0)
df["aqi_lag_1"] = df["aqi"].shift(1).fillna(method="bfill")
df["aqi_lag_2"] = df["aqi"].shift(2).fillna(method="bfill")
df["aqi_lag_3"] = df["aqi"].shift(3).fillna(method="bfill")
df["aqi_rolling_3h"] = df["aqi"].rolling(3, min_periods=1).mean()
df["aqi_rolling_6h"] = df["aqi"].rolling(6, min_periods=1).mean()

# =========================
# TARGET
# =========================
df["future_aqi"] = df["aqi"].shift(-1)
df = df.dropna()

# =========================
# ADD ENTITY ID FOR FEAST
# =========================
df["aqi_id"] = range(len(df))

# =========================
# SAVE TO FEATURE STORE
# =========================
os.makedirs("feature_store/data", exist_ok=True)
df.to_csv("feature_store/data/karachi_features.csv", index=False)
df.to_parquet("feature_store/data/karachi_features.parquet", index=False)

print("✅ Feature engineering completed")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(2))