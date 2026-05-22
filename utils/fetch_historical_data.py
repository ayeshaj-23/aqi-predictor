import requests
import pandas as pd

# Karachi coordinates
LAT = 24.8607
LON = 67.0011

# Open-Meteo Air Quality API (historical)
url = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,uv_index"
    f"&start_date=2026-02-20"
    f"&end_date=2026-05-22"
    f"&timezone=auto"
)

print("Fetching historical data...")

response = requests.get(url)

# Safety check
if response.status_code != 200:
    print("API failed:", response.status_code)
    exit()

data = response.json()["hourly"]

# Build dataset
df = pd.DataFrame({
    "timestamp": data["time"],
    "pm25": data["pm2_5"],
    "pm10": data["pm10"],
    "carbon_monoxide": data["carbon_monoxide"],
    "nitrogen_dioxide": data["nitrogen_dioxide"],
    "sulphur_dioxide": data["sulphur_dioxide"],
    "ozone": data["ozone"],
    "dust": data["dust"],
    "uv_index": data["uv_index"]
})

# Simple AQI approximation
df["aqi"] = (
    df["pm25"] * 0.5 +
    df["pm10"] * 0.3 +
    df["nitrogen_dioxide"] * 0.2
)

# Save dataset
df.to_csv("data/aqi_dataset.csv", index=False)

print("DONE ✔ Dataset created")
print("Rows:", len(df))
print(df.head())