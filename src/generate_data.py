import argparse, math, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def main(rows, out):
    rng = np.random.default_rng(42)
    n_customers = 2500
    mcc_choices = np.array([5411, 5812, 5999, 4821, 4899, 4111, 5311, 6011, 5814, 5732])
    risk_map = {5411:0.2, 5812:0.3, 5999:0.4, 4821:0.35, 4899:0.25, 4111:0.15, 5311:0.18, 6011:0.5, 5814:0.45, 5732:0.28}
    customers = np.arange(1, n_customers+1)
    start_ts = datetime(2025, 6, 1)
    def random_ts():
        return start_ts + timedelta(seconds=int(rng.integers(0, 60*60*24*60)))
    rows_out = []
    for i in range(rows):
        cid = int(rng.choice(customers))
        amt_base = float(abs(rng.normal(60, 45)))
        if rng.random() < 0.08:
            amt_base *= rng.uniform(5, 20)
        amount = round(max(0.5, amt_base), 2)
        mcc = int(rng.choice(mcc_choices))
        ts = random_ts()
        hour = ts.hour
        new_device = int(rng.random() < 0.03)
        dist_km = float(abs(rng.normal(10, 200)))
        base_rate = 0.006
        risk = base_rate
        risk += 0.006 if amount > 500 else 0.0
        risk += 0.004 if new_device else 0.0
        risk += 0.005 if (hour >= 0 and hour <= 5) else 0.0
        risk += 0.006 if dist_km > 800 else 0.0
        risk += 0.004 * risk_map[mcc]
        is_fraud = int(rng.random() < min(0.95, risk))
        rows_out.append({
            "transaction_id": i+1,
            "customer_id": cid,
            "timestamp": ts.isoformat(),
            "amount": amount,
            "merchant_cat": mcc,
            "new_device": new_device,
            "dist_from_home_km": round(dist_km, 3),
            "hour": hour,
            "is_fraud": is_fraud
        })
    import pandas as pd
    df = pd.DataFrame(rows_out).sort_values("timestamp").reset_index(drop=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print({"rows": len(df), "fraud_rate": float(df["is_fraud"].mean())})

if __name__ == "__main__":
    import numpy as np, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", type=str, default="data/transactions.csv")
    args = ap.parse_args()
    main(args.rows, args.out)
