from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()


model = joblib.load("models/xgboost_model.pkl")


@app.get("/")
def home():
    return {"message": "Sales Forecasting API Running Successfully"}


@app.post("/predict")
def predict(data: dict):

    
    df = pd.DataFrame([data])

    
    prediction = model.predict(df)

    return {
        "forecasted_sales": float(prediction[0])
    }