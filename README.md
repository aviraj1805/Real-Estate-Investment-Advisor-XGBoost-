# **Real Estate Investment Advisor**
### Predicting Property Profitability & Future Value using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-FF6600?style=flat-square&logo=xgboost&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)

---

## Overview

A production-grade machine learning application that empowers real estate investors with data-backed decisions. The system runs **two parallel ML pipelines** — a classification engine to determine investment quality and a regression engine to forecast 5-year property value — all accessible through an interactive Streamlit dashboard.

> Built as part of an ML internship project under the domain of **Real Estate / Investment / Financial Analytics**.

---

## Problem Statement

Real estate investment decisions are often driven by intuition rather than data. This project addresses that gap by building an intelligent system that:

- **Classifies** whether a property is a *Good Investment* or not, based on multi-factor domain rules
- **Predicts** the estimated property price after 5 years using city-based compound growth modeling
- **Deploys** results through an interactive web UI with visual market insights

---

## System Architecture

```
india_housing_prices.csv
         │
         ▼
┌─────────────────────┐
│   Data Cleaning &   │
│   Preprocessing     │  ← Imputation, Encoding, Outlier Handling
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Engineering│  ← Good_Investment Label, Future_Price_5Y
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│        EDA          │  ← Price Trends, Correlations, Location Analysis
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│  CLF   │ │  REG   │  ← XGBoost / Random Forest
│ Model  │ │ Model  │
└────┬───┘ └───┬────┘
     │         │
     └────┬────┘
          ▼
┌─────────────────────┐
│    MLflow Registry  │  ← Experiment Tracking + Model Versioning
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit App     │  ← Prediction UI + Visual Insights Dashboard
└─────────────────────┘
```

---

## Key Features

| Feature | Description |
|---|---|
| Investment Classifier | Binary classification with confidence score |
| Price Forecaster | 5-year future value with city-based growth rates |
| Location Intelligence | City & state-wise price trend visualizations |
| Model Transparency | Feature importance charts for every prediction |
| Experiment Tracking | Full MLflow integration with model registry |
| Interactive Dashboard | Streamlit UI with 3-tab analytics panel |

---

## Project Structure

```
Real-Estate-Investment-Advisor/
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_mlflow_tracking.ipynb
│
├── data/
│   ├── india_housing_prices.csv          # Raw dataset
│   └── engineered_data.csv               # Processed + feature engineered
│
├── models/
│   ├── rf_clf.pkl                        # Random Forest Classifier
│   ├── xgb_clf.pkl                       # XGBoost Classifier
│   ├── rf_reg.pkl                        # Random Forest Regressor
│   ├── xgb_reg.pkl                       # XGBoost Regressor
│   └── scaler.pkl                        # StandardScaler
│
├── mlruns/                            # MLflow experiment logs
│
├── app.py                             # Streamlit application
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** `india_housing_prices.csv` — India residential property listings

| Feature | Description |
|---|---|
| State / City / Locality | Geographic identifiers |
| Property_Type | Apartment, Villa, House |
| BHK | Number of bedrooms |
| Size_in_SqFt | Property area |
| Price_in_Lakhs | Listed price |
| Price_per_SqFt | Normalized price metric |
| Year_Built / Age_of_Property | Property age |
| Furnished_Status | Unfurnished / Semi / Fully |
| Nearby_Schools / Hospitals | Infrastructure proximity |
| Public_Transport_Accessibility | Connectivity score |
| Parking_Space / Security / Amenities | Lifestyle features |

---

## Feature Engineering

### Target 1 — `Good_Investment` (Classification)
A property is labeled **Good Investment = 1** if it satisfies ≥ 3 of 4 conditions:

```python
conditions = [
    Price_in_Lakhs  <= city_median_price,
    Price_per_SqFt  <= dataset_median_ppsf,
    Nearby_Schools  >= 3,
    Public_Transport_Accessibility >= 3
]
Good_Investment = 1 if sum(conditions) >= 3 else 0
```

### Target 2 — `Future_Price_5Y` (Regression)
City-based compound growth model:

```python
Future_Price_5Y = Price_in_Lakhs × (1 + growth_rate) ^ 5

# Growth rates:
# Metro cities (Mumbai, Delhi, Bangalore): 10%
# Tier-2 cities: 8%
# All others: 7%
```

---

## Models & Results

### Classification — Good Investment Prediction

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | ~91% | ~0.90 | ~0.91 | ~0.90 | ~0.96 |
| **XGBoost**   | **~93%** | **~0.92** | **~0.93** | **~0.92** | **~0.97** |

### Regression — Future Price Prediction

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Random Forest | ~8.2 | ~5.9 | ~0.96 |
| **XGBoost**   | **~7.1** | **~5.1** | **~0.97** |

> XGBoost selected as best model for both tasks and registered in MLflow Model Registry.

---

## MLflow Experiment Tracking

All training runs are tracked with MLflow:

```
Experiment: real_estate_investment_advisor
│
├── Run: RandomForest_Classifier
│   ├── Params: n_estimators=100, max_depth=None
│   └── Metrics: accuracy, f1, roc_auc
│
├── Run: XGBoost_Classifier          ← Registered as "best_classifier"
│   ├── Params: n_estimators=100, learning_rate=0.1
│   └── Metrics: accuracy, f1, roc_auc
│
├── Run: RandomForest_Regressor
│   └── Metrics: rmse, mae, r2
│
└── Run: XGBoost_Regressor           ← Registered as "best_regressor"
    └── Metrics: rmse, mae, r2
```

To launch MLflow UI:
```bash
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000
```

---

## Streamlit Application

The app provides a full investor-facing interface:

**Sidebar Input Form**
- Property details: City, Type, BHK, Size, Price, Furnishing
- Infrastructure: Schools, Hospitals, Transport, Parking
- Building: Floor, Total Floors, Age

**Prediction Output**
```
┌─────────────────┬──────────────────┬─────────────────┬──────────────────┐
│ Investment      │ Confidence Score │ Price (5Y)      │ Appreciation     │
│ Good ✅ / Avoid ❌│      89%      │ ₹125.4 Lakhs    │     +47.5%       │
└─────────────────┴──────────────────┴─────────────────┴──────────────────┘
```

**3-Tab Visual Dashboard**
- Market Overview — Price distributions, property type comparisons
- Location Insights — City-wise trends, price vs size scatter
- Feature Importance — Top 15 drivers of investment decision

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/aviraj1805/Real-Estate-Investment-Advisor-XGBoost-.git
cd Real-Estate-Investment-Advisor-XGBoost-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```
01 → 02 → 03 → 04 → 05 → 06
```

### 4. Launch Streamlit app
```bash
streamlit run app.py
```

### Running on Google Colab
```python
!pip install streamlit pyngrok
from pyngrok import ngrok
!streamlit run app.py &
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8501)
print(public_url)
```

---

## Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=1.7.0
mlflow>=2.8.0
streamlit>=1.28.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
pyngrok>=6.0.0
```

---

## Project Evaluation Metrics

| Criteria | Implementation |
|---|---|
| Data Handling | Median/mode imputation, label encoding, outlier capping |
| Classification Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Regression Metrics | RMSE, MAE, R² Score, Actual vs Predicted plots |
| Deployment | Fully functional Streamlit UI with live predictions |
| Experimentation | MLflow runs, metric logging, model registry versioning |

---

## Business Use Cases

- **Individual Investors** — Get instant data-backed buy/avoid decisions with confidence scores
- **Real Estate Platforms** — Automate investment analysis for every new property listing
- **Property Buyers** — Identify high-return properties in developing areas before prices rise
- **Financial Advisors** — Provide clients with 5-year property value forecasts

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Models | XGBoost, Scikit-learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Experiment Tracking | MLflow |
| Deployment | Streamlit |
| Environment | Google Colab / Local |

---

## Author

**AVIRAJ VIRAPE**
- GitHub: [@aviraj1805](https://github.com/aviraj1805)
- LinkedIn : [@avirajvirape](https://www.linkedin.com/in/avirajvirape/)
---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ | Real Estate Investment Advisor v1.0 | ML Internship Project
</p>
