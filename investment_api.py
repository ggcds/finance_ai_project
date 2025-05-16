
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Investment Strategy API")

# Tente carregar os modelos
try:
    model = load_model("models/investment_model.h5")
    scaler = joblib.load("models/scaler_investment.pkl")
    print("✅ Modelo e scaler de investimento carregados com sucesso.")
except Exception as e:
    print("❌ Erro ao carregar modelo ou scaler:", e)
    model = None
    scaler = None

class InvestmentData(BaseModel):
    stock_volatility: float
    interest_rate: float
    inflation_rate: float
    past_return: float

@app.post("/predict")
def predict_investment(data: InvestmentData):
    try:
        if model is None or scaler is None:
            raise ValueError("Modelo ou scaler não carregado.")

        input_array = np.array([[data.stock_volatility, data.interest_rate, data.inflation_rate, data.past_return]])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        expected_return = float(prediction[0][0])

        return {
            "expected_return": round(expected_return, 4)
        }
    except Exception as e:
        print("❌ Erro durante a previsão de investimento:", e)
        return {"error": str(e)}
