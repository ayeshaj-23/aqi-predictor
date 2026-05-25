import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL + DATA
# =========================

model = joblib.load("models/best_model.pkl")

df = pd.read_csv("features/feature_store.csv")

# Features used in training (same order is IMPORTANT)
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

X = df[features]

# =========================
# SHAP EXPLAINER
# =========================

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# =========================
# SUMMARY PLOT
# =========================

plt.title("SHAP Feature Importance - AQI Model")
shap.summary_plot(shap_values, X, show=False)

plt.savefig("shap_summary.png", bbox_inches='tight')

print("SHAP analysis completed!")
print("Saved plot as shap_summary.png")