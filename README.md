# 🌿 Auckland Air Quality Intelligence Dashboard

This dashboard is part of the **GDDA713 Capstone Project**, developed in collaboration with **Auckland Council**. All datasets used in this analysis were provided directly by Auckland Council.

The project focuses on understanding pollution behaviour in Auckland’s city centre and developing AI-powered forecasting models to support evidence-based urban planning and air-quality management.

---

# 📌 Project Overview

This project investigates how:

- 🚗 Traffic activity
- 🚶 Pedestrian movement
- 🌦 Weather conditions
- ⏳ Time-series patterns

influence:

- AQI (Air Quality Index)
- PM2.5
- PM10
- NO₂

The dashboard combines:

- Data Analytics
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- Time-Series Forecasting
- Interactive Visualisation

into one professional Streamlit application.

---

# 🎯 Project Objectives

## Goal 1 — Pollution Analysis

Analyse:

- pollution behaviour
- environmental relationships
- seasonal patterns
- traffic impact
- weather influence

---

## Goal 2 — AI Forecasting

Predict:

- future AQI
- future PM2.5
- future NO₂

using:

- historical pollution
- weather
- traffic
- pedestrian activity
- time-series modelling

---

# 🛠 Technologies Used

| Area | Technology |
|---|---|
| Dashboard | Python + Streamlit |
| Data Processing | Pandas + NumPy |
| Visualisation | Plotly + Matplotlib + Seaborn |
| Machine Learning | Scikit-learn |
| Forecasting | XGBoost + Random Forest + GRU |
| Deep Learning | TensorFlow / Keras |
| Styling | Custom CSS |
| Deployment | GitHub + Streamlit Cloud |

---

# 📂 Project Structure

```text
AQI_Project/
│
├── Home.py
├── ui_components.py
├── requirements.txt
│
├── pages/
│   ├── 1_📊_Data_Visualisation.py
│   ├── 2_🧹_Data_Cleaning.py
│   ├── 3_📈_EDA.py
│   ├── 4_⚙️_Feature_Engineering.py
│   ├── 5_🤖_Model_Development.py
│
├── assets/
├── screenshots/
│
└── README.md
```

---

# 📊 Dashboard Workflow

```text
UPLOAD DATA
     ↓
DATA VISUALISATION
     ↓
DATA CLEANING
     ↓
EDA ANALYSIS
     ↓
FEATURE ENGINEERING
     ↓
MODEL DEVELOPMENT
     ↓
AQI FORECASTING
```

---

# 📥 Supported File Types

The dashboard supports:

- CSV
- XLS
- XLSX

datasets for upload and processing.

---

# 📈 Dashboard Modules

## 🏠 Home Page

- Dataset upload
- Dataset merging
- Filtering by year/month
- Dashboard KPIs
- Project overview
- Workflow guidance

---

## 📊 Data Visualisation

Interactive visualisations including:

- Dataset Overview
- Summary Statistics
- Missing Value Analysis
- Basic Visualisation
- correlation heatmaps
- Missing Value Timeline including months and columns

---

## 🧹 Data Cleaning

Cleaning operations include:

- missing value handling
- infinite value replacement
- datetime processing
- negative value correction
- high-missing column removal
- high-missing month filtering
- median imputation
- time interpolation

---

## 📈 Exploratory Data Analysis (EDA)

Analyses:

- temporal pollution patterns
- traffic vs pollution relationships
- weather impact
- seasonal behaviour
- correlation analysis

---

## ⚙️ Feature Engineering

Created advanced features such as:

- cyclical time encoding
- lag features
- rolling statistics
- traffic dispersion index
- thermal humidity index
- seasonal features
- redundancy filtering

---

## 🤖 Model Development & Forecasting

Implemented models:

- Random Forest
- XGBoost
- GRU

Includes:

- model evaluation
- AQI prediction
- 24-hour forecasting
- performance metrics
- interactive forecast graphs

---

# 🔬 Machine Learning Metrics

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² Score | Model Performance Quality |

---

# 🌫 Key Research Insights

The analysis identified strong relationships between:

- traffic and NO₂
- NOX and AQI
- PM2.5 and PM10
- pedestrian activity and traffic density

Time-series analysis also revealed:

- daily pollution cycles
- weekly traffic patterns
- seasonal AQI variation

---

# 🚀 Deployment

The application is deployed using:

- GitHub
- Streamlit Cloud

---

# 👥 Team Members

- Seema Devi
- Deshika Jayatilaka
- Yaqing Zhang (Sarah)

---

# 🎓 Supervisors

- Dr Louis Boamponsem — External Supervisor
- Dr Sara Zandi — Internal Supervisor

---

# 🔮 Future Improvements

- Real-time AQI monitoring
- Live API integration
- Advanced GRU forecasting
- Explainable AI integration
- Smart-city environmental monitoring support

---

# 🌱 Final Project Statement

This project demonstrates how AI and environmental analytics can be combined to better understand urban air pollution and support smarter, data-driven environmental decision-making for future smart cities.