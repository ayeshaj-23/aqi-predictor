import time
from fetch_data import get_clean_data
import os
import csv

CSV_FILE = "data/aqi_dataset.csv"

os.makedirs("data", exist_ok=True)

# create file if not exists
file_exists = os.path.exists(CSV_FILE)

with open(CSV_FILE, "a", newline="") as f:
    writer = None

    for i in range(20):  # 👈 THIS replaces manual runs
        data = get_clean_data()

        if writer is None:
            writer = csv.DictWriter(f, fieldnames=data.keys())

            if not file_exists:
                writer.writeheader()

        writer.writerow(data)
        print(f"Saved row {i+1}")

        time.sleep(60)  # wait 1 minute between API calls