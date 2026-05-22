from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AQI Predictor</title>
    <style>
        body {
            font-family: Arial;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            width: 400px;
        }
        input {
            width: 100%;
            padding: 5px;
            margin-bottom: 10px;
        }
        button {
            padding: 10px;
            background: #2e86de;
            color: white;
            border: none;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            background: #eaf2f8;
            border-radius: 5px;
        }
    </style>
</head>

<body>

<div class="container">
<h1>🌍 AQI Predictor</h1>

<form method="POST">

PM2.5: <input name="pm25" required>
PM10: <input name="pm10" required>
CO: <input name="co" required>
NO2: <input name="no2" required>
SO2: <input name="so2" required>
Ozone: <input name="ozone" required>
Dust: <input name="dust" required>
UV Index: <input name="uv" required>
Hour: <input name="hour" required>
Day: <input name="day" required>
Month: <input name="month" required>
AQI Change: <input name="change" required>

<button type="submit">Predict AQI</button>
</form>

{% if prediction is not none %}
<div class="result">
    <h2>Predicted AQI: {{ prediction }}</h2>
    <h3>Air Quality: {{ category }}</h3>
</div>
{% endif %}

</div>

</body>
</html>
"""

def get_category(aqi):
    if aqi <= 50:
        return "Good 😊"
    elif aqi <= 100:
        return "Moderate 🙂"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 😷"
    else:
        return "Unhealthy ☠️"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    category = None

    if request.method == "POST":
        data = [
            float(request.form["pm25"]),
            float(request.form["pm10"]),
            float(request.form["co"]),
            float(request.form["no2"]),
            float(request.form["so2"]),
            float(request.form["ozone"]),
            float(request.form["dust"]),
            float(request.form["uv"]),
            float(request.form["hour"]),
            float(request.form["day"]),
            float(request.form["month"]),
            float(request.form["change"])
        ]

        df = pd.DataFrame([data], columns=[
            "pm25","pm10","carbon_monoxide","nitrogen_dioxide",
            "sulphur_dioxide","ozone","dust","uv_index",
            "hour","day","month","aqi_change"
        ])

        prediction = model.predict(df)[0]
        category = get_category(prediction)

    return render_template_string(HTML, prediction=prediction, category=category)


if __name__ == "__main__":
    app.run(debug=True)