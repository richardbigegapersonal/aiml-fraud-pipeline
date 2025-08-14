from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import joblib

app = FastAPI()
model = joblib.load("models/model.pkl")
threshold = 0.75

class Txn(BaseModel):
    amount: float
    merchant_cat: int
    new_device: int
    dist_from_home_km: float
    hour: int

    @field_validator("amount")
    @classmethod
    def non_negative(cls, v):
        if v < 0: raise ValueError("amount must be >= 0")
        return v

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/score")
def score(txn: Txn):
    x = [[txn.amount, txn.merchant_cat, txn.new_device, txn.dist_from_home_km, txn.hour]]
    p = float(model.predict_proba(x)[0][1])
    return {"fraud_prob": p, "flag": bool(p >= threshold)}
