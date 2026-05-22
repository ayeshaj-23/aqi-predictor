import joblib
import pandas as pd

model = joblib.load("models/best_model.pkl")

feature_names = [
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

def predict_aqi(values):
    df = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(df)
    return prediction[0]


# Example test
result = predict_aqi([
    120, 80, 0.5,
    20, 10,
    30, 5, 6,
    14, 22, 5, 0
])

print("Predicted AQI:", result)