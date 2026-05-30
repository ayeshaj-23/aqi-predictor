import requests
import pandas as pd
import os

LAT = 24.8607
LON = 67.0011

START_DATE = "2025-12-01"
END_DATE = "2026-03-01"

URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT}&longitude={LON}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
)

response = requests.get(URL)
data = response.json()

# safety check
if "hourly" not in data:
    print("API ERROR RESPONSE:", data)
    raise Exception("No hourly data returned from API")

# build dataframe
df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"])

# create folder if not exists
os.makedirs("data", exist_ok=True)

# save dataset
output_path = "data/karachi_3months.csv"
df.to_csv(output_path, index=False)

print("✅ Data saved successfully:", output_path)
print(df.head())
print("Rows:", len(df))