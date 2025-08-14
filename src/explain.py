import joblib, shap, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

model = joblib.load("models/model.pkl")
X = pd.read_csv("data/features.csv").sample(1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

out_dir = Path("reports/shap"); out_dir.mkdir(parents=True, exist_ok=True)

plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig(out_dir / "summary_bar.png")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout(); plt.savefig(out_dir / "summary_beeswarm.png")
plt.close()

print("SHAP plots saved to reports/shap/")
