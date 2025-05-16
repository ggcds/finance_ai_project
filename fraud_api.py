
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Fraud Detection API")

# Carregar modelo e scaler
try:
    model = load_model("models/fraud_model.h5")
    scaler = joblib.load("models/scaler_fraud.pkl")
    print("✅ Modelo e scaler de fraude carregados com sucesso.")
except Exception as e:
    print("❌ Erro ao carregar modelo ou scaler:", e)
    model = None
    scaler = None

# Definir os campos de entrada corretos
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
        if model is None or scaler is None:
            raise ValueError("Modelo ou scaler não carregado.")

        X = np.array([[data.feature_0, data.feature_1, data.feature_2, data.feature_3,
                       data.feature_4, data.feature_5, data.feature_6, data.feature_7,
                       data.feature_8, data.feature_9]])
        X_scaled = scaler.transform(X).reshape((1, 10, 1))
        prediction = model.predict(X_scaled)
        prob = float(prediction[0][0])

        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": prob > 0.5
        }
    except Exception as e:
        return {"error": str(e)}
