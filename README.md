# Sales Forecasting System

## Objective
This project builds an end-to-end time series forecasting system for predicting the next 8 weeks of sales for different states using historical sales data.



# Features

- Multiple forecasting models
- Automatic model comparison
- Feature engineering
- REST API using FastAPI
- Production-style project structure
- Time-series validation
- Forecasting pipeline



# Models Implemented

1. ARIMA
2. Prophet
3. XGBoost
4. LSTM



# Feature Engineering

Implemented:
- Lag Features (t-1, t-7, t-30)
- Rolling Mean
- Rolling Standard Deviation
- Day of Week
- Month
- Year



# Evaluation Metrics

- MAE
- RMSE



# Best Performing Model

LSTM achieved the best performance among all implemented models.



# API Endpoints

## Home
GET /

## Prediction
POST /predict



# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- XGBoost
- Prophet
- FastAPI



# Run Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run API

```bash
uvicorn api.app:app --reload
```



# Project Structure

```text
sales-forecasting-system/
```



# Author

Sathwik Sharma Deshapathi