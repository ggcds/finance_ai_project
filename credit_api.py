from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Credit Risk Prediction API")

# Tente carregar os modelos e registre qualquer falha
try:
    model = load_model("models/credit_model.h5")
    scaler = joblib.load("models/scaler_credit.pkl")
    print("✅ Modelo e scaler carregados com sucesso.")
except Exception as e:
    print("❌ Erro ao carregar modelo ou scaler:", e)
    model = None
    scaler = None

class CreditData(BaseModel):
    income: float
    age: int
    loan_amount: float
    score: int

@app.post("/predict")
def predict_credit_risk(data: CreditData):
    try:
        if model is None or scaler is None:
            raise ValueError("Modelo ou scaler não carregado.")

        input_array = np.array([[data.income, data.age, data.loan_amount, data.score]])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        default_prob = float(prediction[0][0])

        return {
            "default_probability": round(default_prob, 4),
            "risk_level": "High" if default_prob > 0.5 else "Low"
        }
    except Exception as e:
        print("❌ Erro durante a previsão:", e)
        return {"error": str(e)}
