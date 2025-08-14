import json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier
from pathlib import Path

X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/labels.csv")["is_fraud"].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LGBMClassifier(n_estimators=800, num_leaves=63, class_weight="balanced", learning_rate=0.05)
clf.fit(X_train, y_train)

p = clf.predict_proba(X_val)[:,1]
metrics = {
    "val_roc_auc": float(roc_auc_score(y_val, p)),
    "val_pr_auc": float(average_precision_score(y_val, p)),
    "pos_rate_val": float(y_val.mean())
}
Path("models").mkdir(exist_ok=True)
joblib.dump(clf, "models/model.pkl")
with open("models/metrics.json","w") as f: json.dump(metrics, f, indent=2)

print(metrics)
