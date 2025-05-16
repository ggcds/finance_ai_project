# Finance AI Project

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)

---

## 📖 Overview

O **Finance AI Project** demonstra a aplicação de técnicas de Deep Learning e Machine Learning em três casos de uso financeiros:

1. **Análise de Risco de Crédito**: Modelagem e predição de probabilidade de inadimplência.
2. **Detecção de Fraudes**: Identificação de transações suspeitas em tempo real.
3. **Otimização de Investimentos**: Geração de recomendações de carteira com base em modelos preditivos.

Cada módulo inclui pipelines de preparação de dados, treinamento de modelos e APIs para integração.

---

## ✨ Funcionalidades

- 📊 **Pré-processamento de Dados**: Normalização, feature engineering e validação cruzada.
- 🤖 **Modelos Treinados**:
  - `credit_model.h5` com seu script de treino (`train_credit_model.py`).
  - `fraud_model.h5` e pipeline em `fraud_detection_model.py`.
  - `investment_model.h5` e otimizador em `investment_optimizer.py`.
- 🚀 **APIs REST** (`FastAPI`):
  - Endpoints de predição unificados em `unified_api.py`.
- 📓 **Notebooks Interativos**:
  - `01_credit_risk_analysis.ipynb` – diagnóstico e métricas.
  - `02_fraud_detection.ipynb` – exploração e testes.
  - `03_investment_strategy.ipynb` – backtesting e visualizações.

---

## 📂 Estrutura do Projeto

```text
├── data
│   ├── credit_data.csv
│   ├── market_data.csv
│   └── transactions.csv
├── models
│   ├── credit_model.h5
│   ├── fraud_model.h5
│   ├── investment_model.h5
│   ├── scaler_credit.pkl
│   ├── scaler_fraud.pkl
│   └── scaler_investment.pkl
├── notebooks
│   ├── 01_credit_risk_analysis.ipynb
│   ├── 02_fraud_detection.ipynb
│   └── 03_investment_strategy.ipynb
├── utils
│   ├── credit_api.py
│   ├── fraud_api.py
│   ├── investment_api.py
│   └── unified_api.py
├── train_credit_model.py
├── train_fraud_model.py
├── train_investment_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalação

> Requisitos: Python 3.8+, `conda` ou `venv`.

1. Clone este repositório:
   ```bash
   git clone https://github.com/ggcds/finance_ai_project.git
   cd finance_ai_project
   ```
2. Crie e ative o ambiente:
   ```bash
   python -m venv .env
   source .env/bin/activate        # macOS/Linux
   .env\Scripts\activate         # Windows PowerShell
   ```
3. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Uso

Antes de iniciar o servidor, execute os notebooks em ordem para gerar os modelos (`*.pkl` e `*.h5`) na pasta `models/`:

1. Abra o Jupyter na pasta de notebooks:
   ```bash
   jupyter notebook notebooks/
   ```
2. Execute os notebooks:
   - `01_credit_risk_analysis.ipynb`
   - `02_fraud_detection.ipynb`
   - `03_investment_strategy.ipynb`

3. Inicie o servidor FastAPI:
   ```bash
   uvicorn unified_api:app --reload
   ```
4. Acesse `http://localhost:8000/docs` para explorar os endpoints de cada módulo.
