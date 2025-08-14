import pandas as pd
from pathlib import Path

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "amount": df["amount"],
        "merchant_cat": df["merchant_cat"],
        "new_device": df["new_device"],
        "dist_from_home_km": df["dist_from_home_km"],
        "hour": df["hour"]
    })
    return out

if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
    X = build_features(df)
    y = df["is_fraud"].astype(int)
    Path("data").mkdir(exist_ok=True, parents=True)
    X.to_csv("data/features.csv", index=False)
    y.to_frame("is_fraud").to_csv("data/labels.csv", index=False)
    print({"rows": len(X), "positives": int(y.sum())})
