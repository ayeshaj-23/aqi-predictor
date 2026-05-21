import pandas as pd

# File paths
INPUT_FILE = "data/aqi_dataset.csv"
OUTPUT_FILE = "data/aqi_features.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Convert timestamp column to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create time-based features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# AQI change compared to previous row
df["aqi_change"] = df["aqi"].diff()

# Create future AQI target
# Shift AQI upward by 1 row
df["future_aqi"] = df["aqi"].shift(-1)

# Remove rows with missing values
df = df.dropna()

# Save engineered dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Feature engineering complete!")
print(f"Saved engineered dataset to: {OUTPUT_FILE}")

# Show first few rows
print(df.head())