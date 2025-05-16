import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Carregar o CSV
df = pd.read_csv("data/credit_data.csv")

# Verifique os nomes das colunas
print("Colunas disponíveis:", df.columns.tolist())

# Usar apenas colunas de entrada relevantes
X = df[['income', 'age', 'loan_amount', 'score']]
y = df['default']

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Modelo
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Salvar modelo e scaler
os.makedirs("models", exist_ok=True)
model.save("models/credit_model.h5")
joblib.dump(scaler, "models/scaler_credit.pkl")

print("✅ Modelo e scaler salvos com sucesso!")
