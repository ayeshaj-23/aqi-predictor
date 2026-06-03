# Karachi AQI Predictor — End-to-End MLOps Project

An automated, end-to-end Air Quality Index (AQI) prediction system for Karachi, Pakistan.
Built as part of the 10 Pearls SHINE Internship 2026 at Habib University.

---

## System Architecture

Open-Meteo API + AQICN API → Feature Pipeline (hourly) → Feast Feature Store → Training Pipeline (daily) → Model Registry → Streamlit Dashboard

---

## Project Structure

aqi-predictor/
├── fetch_data.py                  # Hourly data collection from AQICN API
├── utils/feature_engineering.py   # Feature engineering pipeline
├── feature_store/                 # Feast feature store setup
│   ├── feature_repo.py
│   ├── feature_store.yaml
│   └── data/
├── src/train_model.py             # Training pipeline (reads from Feast)
├── model_registry/                # Saved models + metadata
│   ├── random_forest.pkl
│   ├── ridge.pkl
│   ├── xgboost.pkl
│   ├── best_model.pkl
│   └── metadata.json
├── app/streamlit_app.py           # Streamlit dashboard
├── notebooks/
│   ├── eda.py                     # EDA script
│   └── eda_plots/                 # 8 EDA charts
├── .github/workflows/
│   └── aqi_pipeline.yml           # CI/CD pipeline
└── requirements.txt

---

## Features

- Fetches hourly weather and AQI data from Open-Meteo and AQICN APIs
- Engineers 14 features including lag features, rolling averages, and time features
- Stores features in Feast feature store (Parquet + SQLite)
- Trains 3 ML models daily and selects the best automatically
- Displays 3-day AQI forecast on an interactive Streamlit dashboard
- SHAP feature importance explanations
- Hazard alerts for unhealthy AQI levels
- Fully automated via GitHub Actions (hourly + daily jobs)

---

## Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 2.15 | 3.07 | 0.9327 |
| Ridge Regression | 2.18 | 3.08 | 0.9322 |
| XGBoost | 2.13 | 2.97 | 0.9370 |

**Best Model: XGBoost (R² = 0.9370)**

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run feature engineering
python utils/feature_engineering.py

# Apply Feast feature store
cd feature_store && python run_feast.py

# Train models
python src/train_model.py

# Run dashboard
streamlit run app/streamlit_app.py
```

---

## CI/CD Pipeline

Two automated GitHub Actions jobs:
- **Feature pipeline** — runs every hour, fetches data and updates features
- **Training pipeline** — runs every day, retrains models and updates model registry

---

## Tech Stack

Python, Scikit-learn, XGBoost, Feast, GitHub Actions, Streamlit, SHAP, Open-Meteo API, AQICN API

---

## Author

Ayesha Jawed
10 Pearls SHINE Internship 2026 | Habib University