import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/transactions.csv")
EDA_DIR = Path("reports/eda"); EDA_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["timestamp"])

plt.figure()
df["amount"].hist(bins=50)
plt.title("Transaction Amount Distribution"); plt.xlabel("Amount"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(EDA_DIR / "amount_distribution.png"); plt.close()

plt.figure()
df["timestamp"].dt.hour.value_counts().sort_index().plot(kind="bar")
plt.title("Transactions by Hour"); plt.xlabel("Hour of Day"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(EDA_DIR / "transactions_by_hour.png"); plt.close()

plt.figure()
bins = pd.cut(df["amount"], bins=[0,25,50,100,200,500,1000,5000,100000], include_lowest=True)
df.groupby(bins)["is_fraud"].mean().plot(kind="bar")
plt.title("Fraud Rate by Amount Bucket"); plt.xlabel("Amount Bucket"); plt.ylabel("Fraud Rate")
plt.tight_layout(); plt.savefig(EDA_DIR / "fraud_rate_by_amount_bucket.png"); plt.close()

print("EDA charts saved to reports/eda/")
