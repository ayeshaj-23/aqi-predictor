import joblib
import pandas as pd

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/best_model.pkl")

# =========================
# LOAD FEATURE STORE
# =========================

df = pd.read_csv("features/feature_store.csv")

features = [
    "pm25",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "uv_index",
    "hour",
    "day",
    "month",
    "aqi_change"
]

# Take latest row
current_input = df[features].iloc[[-1]].copy()

# =========================
# FORECAST LOOP
# =========================

predictions = []

for i in range(3):

    pred = model.predict(current_input)[0]
    predictions.append(pred)

    # Simulate future progression
    current_input["hour"] += 1
    current_input["aqi_change"] = pred * 0.01

# =========================
# RESULTS
# =========================

print("\n🌫️ 3-DAY AQI FORECAST")
print("========================")

for i, p in enumerate(predictions, 1):
    print(f"Day {i}: AQI = {p:.2f}")