import time
from fetch_data import get_clean_data
import os
import csv

CSV_FILE = "data/aqi_dataset.csv"

os.makedirs("data", exist_ok=True)

file_exists = os.path.exists(CSV_FILE)

i = 0

while True:

    data = get_clean_data()

    # If API fails, skip safely
    if data is None:
        print("API failed, skipping this cycle...")
        time.sleep(60)
        continue

    # Write to CSV safely
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        # Write header only once
        if not file_exists:
            writer.writeheader()
            file_exists = True

        writer.writerow(data)

    i += 1
    print(f"Saved row {i}")

    # wait before next API call
    time.sleep(60)