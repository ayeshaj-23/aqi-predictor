if __name__ == "__main__":
    data = get_clean_data()
    print(data)

    os.makedirs("data", exist_ok=True)

    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        # write header once
        if not file_exists:
            writer.writerow(["timestamp", "aqi", "pm25", "pm10", "temp", "humidity", "wind"])

        writer.writerow([
            data["timestamp"],
            data["aqi"],
            data["pm25"],
            data["pm10"],
            data["temp"],
            data["humidity"],
            data["wind"]
        ])