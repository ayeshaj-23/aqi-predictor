import pandas as pd
import os

# =========================
# LOAD RAW DATA
# =========================

df = pd.read_csv("data/aqi_dataset.csv")

print("Raw dataset loaded!")

# =========================
# CONVERT TIMESTAMP
# =========================

df["timestamp"] = pd.to_datetime(df["timestamp"])

# =========================
# TIME FEATURES
# =========================

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# =========================
# AQI CHANGE FEATURE
# =========================

df["aqi_change"] = df["aqi"].diff()

# fill first null value
df["aqi_change"] = df["aqi_change"].fillna(0)

# =========================
# TARGET COLUMN
# =========================

df["future_aqi"] = df["aqi"].shift(-1)

# remove last null row
df = df.dropna()

# =========================
# CREATE FEATURE STORE FOLDER
# =========================

os.makedirs("features", exist_ok=True)

# =========================
# SAVE TO FEATURE STORE
# =========================

FEATURE_STORE_PATH = "features/feature_store.csv"

df.to_csv(FEATURE_STORE_PATH, index=False)

print("\nFeature engineering complete!")
print(f"Saved to: {FEATURE_STORE_PATH}")

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)