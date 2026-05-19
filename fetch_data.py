import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
#loading

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
    }

    return cleaned

if __name__ == "__main__":
    data = get_clean_data()
    print(data)

    os.makedirs("data", exist_ok=True)

    file_path = "data/aqi_dataset.json"

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            dataset = json.load(f)
    else:
        dataset = []

    dataset.append(data)

    with open(file_path, "w") as f:
        json.dump(dataset, f, indent=4)