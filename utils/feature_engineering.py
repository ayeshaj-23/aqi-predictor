import pandas as pd

df = pd.read_csv("data/aqi_dataset.csv")

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Time features
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month

# Derived features
df["aqi_change"] = df["aqi"].diff()
df["future_aqi"] = df["aqi"].shift(-1)

# Clean data
df.dropna(inplace=True)

# Save
df.to_csv("data/aqi_features.csv", index=False)

print("Feature engineering complete ✔")
print(df.head())
print("Rows:", len(df))