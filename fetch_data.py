import requests
import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("AQICN_API_KEY")

def get_clean_data():
    url = f"https://api.waqi.info/feed/karachi/?token={TOKEN}"
    response = requests.get(url).json()

    # Handle API errors
    if response.get("status") != "ok":
        print(f"API error: {response}")
        return None

    data = response["data"]

    cleaned = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": data.get("aqi"),
        "pm25": data["iaqi"].get("pm25", {}).get("v"),
        "pm10": data["iaqi"].get("pm10", {}).get("v"),
        "temp": data["iaqi"].get("t", {}).get("v"),
        "humidity": data["iaqi"].get("h", {}).get("v"),
        "wind": data["iaqi"].get("w", {}).get("v"),
    }

    return cleaned

if __name__ == "__main__":
    data = get_clean_data()

    if data is None:
        print("No data fetched, skipping.")
        exit(0)

    print(data)

    os.makedirs("data", exist_ok=True)

    # Save to JSON
    file_path = "data/aqi_dataset.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            dataset = json.load(f)
    else:
        dataset = []
    dataset.append(data)
    with open(file_path, "w") as f:
        json.dump(dataset, f, indent=4)

    # Also save/append to CSV
    csv_path = "data/aqi_dataset.csv"
    df_new = pd.DataFrame([data])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)

    print("✅ Data fetched and saved successfully")