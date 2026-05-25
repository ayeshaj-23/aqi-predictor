import joblib
import pandas as pd
import numpy as np

model = joblib.load("models/best_model.pkl")


# =========================
# BASE INPUT (you can adjust)
# =========================

base_input = {
    "pm25": 32.5,
    "pm10": 37.4,
    "carbon_monoxide": 495.0,
    "nitrogen_dioxide": 40.0,
    "sulphur_dioxide": 12.0,
    "ozone": 20.0,
    "dust": 25.0,
    "uv_index": 6,
    "hour": 12,
    "day": 25,
    "month": 5,
    "aqi_change": 0.0
}


# =========================
# FORECAST GENERATION
# =========================

def generate_day(input_data, drift):
    """Simulate future pollution changes"""
    
    new_data = input_data.copy()

    # small realistic changes (not random noise only)
    new_data["pm25"] += drift * 1.5
    new_data["pm10"] += drift * 1.2
    new_data["carbon_monoxide"] += drift * 2

    new_data["aqi_change"] = drift

    df = pd.DataFrame([new_data])

    return model.predict(df)[0]


# =========================
# 3-DAY FORECAST
# =========================

print("\n 3-DAY AQI FORECAST")
print("========================")

day1 = generate_day(base_input, drift=0)
day2 = generate_day(base_input, drift=2)
day3 = generate_day(base_input, drift=4)

print(f"Day 1: AQI = {day1:.2f}")
print(f"Day 2: AQI = {day2:.2f}")
print(f"Day 3: AQI = {day3:.2f}")