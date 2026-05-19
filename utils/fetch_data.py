import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import csv

CSV_FILE = "data/aqi_dataset.csv"

load_dotenv()

TOKEN = os.getenv("AQICN_API_KEY")


def get_clean_data():
    url = f"https://api.waqi.info/feed/here/?token={TOKEN}"
    data = requests.get(url).json()["data"]

    cleaned = {
        "aqi": data["aqi"],
        "pm25": data["iaqi"].get("pm25", {}).get("v"),
        "pm10": data["iaqi"].get("pm10", {}).get("v"),
        "temp": data["iaqi"].get("t", {}).get("v"),
        "humidity": data["iaqi"].get("h", {}).get("v"),
        "wind": data["iaqi"].get("w", {}).get("v"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return cleaned

if __name__ == "__main__":
    data = get_clean_data()
    print(data)

    os.makedirs("data", exist_ok=True)

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        # write header ONLY once
        if not file_exists:
            writer.writeheader()

        writer.writerow(data)