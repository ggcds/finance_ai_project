
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = FastAPI(title="Finance AI Unified API")

# ======== Carregamento de modelos ==========
def load_models():
    base = "models"
    models = {}
    try:
        models["credit_model"] = load_model(f"{base}/credit_model.h5")
        models["credit_scaler"] = joblib.load(f"{base}/scaler_credit.pkl")
        print("✅ Modelos de crédito carregados.")
    except Exception as e:
        print("❌ Erro carregando crédito:", e)

    try:
        models["fraud_model"] = load_model(f"{base}/fraud_model.h5")
        models["fraud_scaler"] = joblib.load(f"{base}/scaler_fraud.pkl")
        print("✅ Modelos de fraude carregados.")
    except Exception as e:
        print("❌ Erro carregando fraude:", e)

    try:
        models["investment_model"] = load_model(f"{base}/investment_model.h5")
        models["investment_scaler"] = joblib.load(f"{base}/scaler_investment.pkl")
        print("✅ Modelos de investimento carregados.")
    except Exception as e:
        print("❌ Erro carregando investimento:", e)

    return models

models = load_models()

# ========== CREDIT ==========
class CreditData(BaseModel):
    income: float
    age: int
    loan_amount: float
    score: int

@app.post("/predict/credit")
def predict_credit(data: CreditData):
    try:
        scaler = models["credit_scaler"]
        model = models["credit_model"]
        X = np.array([[data.income, data.age, data.loan_amount, data.score]])
        X_scaled = scaler.transform(X)
        prob = float(model.predict(X_scaled)[0][0])
        return {
            "default_probability": round(prob, 4),
            "risk_level": "High" if prob > 0.5 else "Low"
        }
    except Exception as e:
        return {"error": str(e)}

# ========== FRAUD ==========
class FraudData(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float


@app.post("/predict/fraud")
def predict_fraud(data: FraudData):
    try:
        scaler = models["fraud_scaler"]
        model = models["fraud_model"]
        X = np.array([[data.feature_0, data.feature_1, data.feature_2, data.feature_3,
                       data.feature_4, data.feature_5, data.feature_6, data.feature_7,
                       data.feature_8, data.feature_9]])
        X_scaled = scaler.transform(X).reshape((1, 10, 1))
        prob = float(model.predict(X_scaled)[0][0])
        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": prob > 0.5
        }
    except Exception as e:
        return {"error": str(e)}

# ========== INVESTMENT ==========
class InvestmentData(BaseModel):
    stock_volatility: float
    interest_rate: float
    inflation_rate: float
    past_return: float

@app.post("/predict/investment")
def predict_investment(data: InvestmentData):
    try:
        scaler = models["investment_scaler"]
        model = models["investment_model"]
        X = np.array([[data.stock_volatility, data.interest_rate, data.inflation_rate, data.past_return]])
        X_scaled = scaler.transform(X)
        expected = float(model.predict(X_scaled)[0][0])
        return {
            "expected_return": round(expected, 4)
        }
    except Exception as e:
        return {"error": str(e)}
