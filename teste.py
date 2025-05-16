import joblib
from tensorflow.keras.models import load_model

scaler = joblib.load("models/scaler_investment.pkl")
model = load_model("models/investment_model.h5")
print("✅ Ambos os arquivos estão corretos.")