# AIML Fraud Detection Pipeline (Hands-on)

This is a complete, **interview-ready** hands-on project you can run locally:
- Synthetic data generation
- EDA + feature engineering
- LightGBM training with PR-AUC
- SHAP explainability artifacts
- FastAPI serving
- Drift monitor (PSI/KS)
- Minimal tests

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# (optional) regenerate data
python src/generate_data.py --rows 20000 --out data/transactions.csv

# Build features
python src/features.py

# EDA (exports PNGs to reports/eda/)
python src/eda.py

# Train model (saves model.pkl + metrics.json)
python src/train.py

# Explainability (saves SHAP plots to reports/shap/)
python src/explain.py

# Serve API
uvicorn src.serve:app --host 0.0.0.0 --port 8080
# In another terminal:
curl -X POST http://localhost:8080/score -H "Content-Type: application/json" -d '{"amount": 123.4, "merchant_cat": 5411, "new_device": 0, "dist_from_home_km": 2.3, "hour": 14}'
```

## Structure

```
aiml-fraud-pipeline/
├─ data/                     # synthetic transactions (csv)
├─ reports/
│  ├─ eda/                   # PNGs from eda.py
│  └─ shap/                  # PNGs from explain.py
├─ src/
│  ├─ generate_data.py       # synthetic data generator
│  ├─ eda.py                 # quick EDA charts
│  ├─ features.py            # feature engineering
│  ├─ train.py               # LightGBM training & metrics
│  ├─ explain.py             # SHAP plots
│  └─ serve.py               # FastAPI scoring service
├─ tests/
│  └─ data_tests.py          # basic data sanity checks
├─ requirements.txt
└─ README.md
```

## Notes
- LightGBM & SHAP are used; install build tools if needed.
- Plots avoid seaborn and style settings.
- For a full MLOps pipeline (shadow/canary, registry), see Chapters 8–9 notes.
