from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # ✅ FIX ADDED HERE
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")


# =========================
# AQI CATEGORY + ALERTS
# =========================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good 🟢", "Air quality is safe."

    elif aqi <= 100:
        return "Moderate 🟡", "Air quality is acceptable."

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups 🟠", \
               "People with breathing issues should reduce outdoor activity."

    elif aqi <= 200:
        return "Unhealthy 🔴", \
               "Health alert: avoid prolonged outdoor exposure."

    else:
        return "Very Unhealthy ⚠️", \
               "Hazardous air quality detected. Stay indoors if possible."


# =========================
# CREATE CHART FUNCTION
# =========================

def create_chart(values, title):

    plt.figure()

    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("AQI")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    chart = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return chart


# =========================
# HTML UI
# =========================

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
            padding: 25px;
            border-radius: 12px;
            width: 450px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            color: #2e86de;
        }

        input {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }

        button {
            padding: 12px;
            background: #2e86de;
            color: white;
            border: none;
            cursor: pointer;
            width: 100%;
            border-radius: 5px;
        }

        .result {
            margin-top: 20px;
            padding: 15px;
            background: #eaf2f8;
            border-radius: 8px;
        }

        .warning {
            margin-top: 10px;
            padding: 10px;
            background: #fdebd0;
            border-left: 5px solid orange;
            border-radius: 5px;
        }

        img {
            width: 100%;
            margin-top: 10px;
            border-radius: 8px;
        }

    </style>
</head>

<body>

<div class="container">

<h1> AQI Predictor</h1>

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

    <div class="warning">
        <b>Alert:</b> {{ warning }}
    </div>

</div>

{% if chart %}
<div class="result">
    <h3> 3-Day Forecast</h3>
    <img src="data:image/png;base64,{{ chart }}">
</div>
{% endif %}

{% endif %}

</div>

</body>
</html>
"""


# =========================
# ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    category = None
    warning = None
    chart = None

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

        ])

        prediction = model.predict(df)[0]
        prediction = round(prediction, 2)

        category, warning = get_aqi_category(prediction)

        # simple 3-day forecast (visual demo using model output)
        forecast = [prediction, prediction + 2, prediction + 4]

        chart = create_chart(forecast, "3-Day AQI Forecast")

    return render_template_string(
        HTML,
        prediction=prediction,
        category=category,
        warning=warning,
        chart=chart
    )


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)