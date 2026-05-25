# AQI Predictor (End-to-End ML + MLOps Project)

This project predicts Air Quality Index (AQI) using real-world pollution and weather data.

It demonstrates a full machine learning pipeline — from data collection to deployment — following MLOps principles.

---

## What this project does

- Collects AQI + weather data from an external API  
- Processes raw data into ML-ready features  
- Stores features in a simulated feature store (CSV-based, Hopsworks-compatible design)  
- Trains multiple machine learning models  
- Automatically selects the best performing model  
- Deploys predictions using a Flask web application  
- Automates training using GitHub Actions (CI/CD pipeline)

---

## System Architecture
API → Raw Data → Feature Engineering → Feature Store → Model Training → Model Selection → Model Registry → Flask Web App


---

## Pipeline Workflow

### Data Collection
- Fetches AQI and weather data from external API
- Stores raw dataset in `data/aqi_dataset.csv`

### Feature Engineering
- Time-based features:
  - hour, day, month
- Derived features:
  - AQI change rate
- Output saved in feature store

---

### Feature Store
- CSV-based simulated feature store
- Designed to be compatible with Hopsworks-style architecture

---

### Model Training

Models used:
- Random Forest Regressor  
- Ridge Regression  
- XGBoost (optional comparison)

Evaluation metrics:
- MAE  
- RMSE  
- R² Score  

Best model is selected automatically based on R² score.

---

### Model Deployment

- Flask web application
- Real-time AQI prediction
- Health category + alerts
- 3-day forecast visualization

---

### CI/CD Pipeline

Implemented using GitHub Actions:

- Automated feature engineering (scheduled runs)
- Automated model training pipeline
- Continuous model improvement workflow

---

## Model Performance

| Model            | MAE  | RMSE | R² Score |
|------------------|------|------|----------|
| Random Forest    | ~2.2 | ~4.2 | 0.94     |
| Ridge Regression | ~1.7 | ~3.0 | 0.97     |
| XGBoost          | ~2.4 | ~5.1 | 0.91     |

Best Model: Ridge Regression

---

## Web Application

Users can:
- Input pollutant values
- Get real-time AQI prediction
- View air quality category
- See health alerts
- View 3-day forecast graph

## Author

Ayesha Jawed  
Social Development & Policy | Habib University  

---

## Note

This project demonstrates an end-to-end MLOps pipeline including feature engineering, model training, automated selection, and deployment.