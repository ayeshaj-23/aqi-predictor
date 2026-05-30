import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt
from feast import FeatureStore

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

# =========================
# LOAD MODEL + METADATA
# =========================
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model_registry/best_model.pkl"))
    with open(os.path.join(BASE_DIR, "model_registry/metadata.json")) as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_features():
    store = FeatureStore(repo_path=os.path.join(BASE_DIR, "feature_store"))
    df = pd.read_csv(os.path.join(BASE_DIR, "feature_store/data/karachi_features.csv"))
    df["time"] = pd.to_datetime(df["time"])
    entity_df = df[["aqi_id", "time"]].copy()
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "aqi_features:temperature_2m",
            "aqi_features:relative_humidity_2m",
            "aqi_features:wind_speed_10m",
            "aqi_features:aqi",
            "aqi_features:aqi_change",
            "aqi_features:aqi_lag_1",
            "aqi_features:aqi_lag_2",
            "aqi_features:aqi_lag_3",
            "aqi_features:aqi_rolling_3h",
            "aqi_features:aqi_rolling_6h",
            "aqi_features:hour",
            "aqi_features:day",
            "aqi_features:month",
            "aqi_features:weekday",
            "aqi_features:future_aqi",
        ],
    ).to_df().dropna()
    return training_df

def get_aqi_category(aqi):
    if aqi <= 50:   return "Good", "🟢", "#00e400"
    elif aqi <= 100: return "Moderate", "🟡", "#ffff00"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups", "🟠", "#ff7e00"
    elif aqi <= 200: return "Unhealthy", "🔴", "#ff0000"
    else:            return "Very Unhealthy ⚠️", "🟣", "#8f3f97"

# =========================
# LOAD DATA
# =========================
model, meta = load_model()

st.title("🌫️ Karachi AQI Predictor")
st.markdown("Real-time Air Quality Index prediction using Machine Learning")

with st.spinner("Loading features from Feast..."):
    df = load_features()

FEATURES = meta["features"]

# =========================
# CURRENT AQI
# =========================
latest = df.iloc[-1]
current_aqi = round(float(latest["aqi"]), 2)
category, emoji, color = get_aqi_category(current_aqi)

col1, col2, col3 = st.columns(3)
col1.metric("🌡️ Temperature", f"{latest['temperature_2m']:.1f} °C")
col2.metric("💧 Humidity", f"{latest['relative_humidity_2m']:.1f} %")
col3.metric("💨 Wind Speed", f"{latest['wind_speed_10m']:.1f} km/h")

st.markdown("---")
st.markdown(f"### Current AQI: <span style='color:{color}; font-size:2em'>{current_aqi} {emoji} {category}</span>", unsafe_allow_html=True)

if current_aqi > 150:
    st.error("⚠️ Hazardous AQI Alert! Stay indoors and avoid outdoor activities.")
elif current_aqi > 100:
    st.warning("⚠️ Moderate Alert: Sensitive groups should limit outdoor exposure.")
else:
    st.success("✅ Air quality is acceptable.")

# =========================
# 3-DAY FORECAST
# =========================
st.markdown("---")
st.subheader("📅 3-Day AQI Forecast")

last_rows = df[FEATURES].tail(72)
forecast_preds = model.predict(last_rows)

hours = list(range(1, 73))
forecast_df = pd.DataFrame({"Hour": hours, "Predicted AQI": forecast_preds})
daily = forecast_df.groupby(forecast_df["Hour"].apply(lambda x: f"Day {(x-1)//24 + 1}"))["Predicted AQI"].mean().reset_index()
daily.columns = ["Day", "Average AQI"]

col1, col2, col3 = st.columns(3)
for i, (_, row) in enumerate(daily.iterrows()):
    aqi_val = round(row["Average AQI"], 1)
    cat, emj, _ = get_aqi_category(aqi_val)
    [col1, col2, col3][i].metric(f"{row['Day']}", f"{aqi_val} {emj}", cat)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(forecast_df["Hour"], forecast_df["Predicted AQI"], color="#2e86de", linewidth=2)
ax.fill_between(forecast_df["Hour"], forecast_df["Predicted AQI"], alpha=0.2, color="#2e86de")
ax.set_xlabel("Hour")
ax.set_ylabel("AQI")
ax.set_title("72-Hour AQI Forecast")
ax.axhline(y=100, color="orange", linestyle="--", alpha=0.5, label="Moderate threshold")
ax.axhline(y=150, color="red", linestyle="--", alpha=0.5, label="Unhealthy threshold")
ax.legend()
st.pyplot(fig)

# =========================
# HISTORICAL AQI CHART
# =========================
st.markdown("---")
st.subheader("📈 Historical AQI (Last 7 Days)")

hist = df.tail(168)[["time", "aqi"]] if "time" in df.columns else df.tail(168)[["aqi"]]
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(range(len(hist)), hist["aqi"].values, color="#27ae60", linewidth=1.5)
ax2.set_xlabel("Hour")
ax2.set_ylabel("AQI")
ax2.set_title("Historical AQI - Last 7 Days")
st.pyplot(fig2)

# =========================
# MODEL PERFORMANCE
# =========================
st.markdown("---")
st.subheader("🏆 Model Registry")

all_models = meta.get("all_models", {})
perf_df = pd.DataFrame(all_models).T.reset_index()
perf_df.columns = ["Model", "MAE", "RMSE", "R²"]
perf_df = perf_df.round(4)
st.dataframe(perf_df, use_container_width=True)
st.success(f"✅ Best Model: **{meta['best_model']}** with R² = {meta['r2_score']:.4f}")

# =========================
# SHAP FEATURE IMPORTANCE
# =========================
st.markdown("---")
st.subheader("🧠 SHAP Feature Importance")

with st.spinner("Computing SHAP values..."):
    X_sample = df[FEATURES].tail(100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig3)

st.markdown("---")
st.caption("Data: Open-Meteo | Feature Store: Feast | Model: XGBoost")