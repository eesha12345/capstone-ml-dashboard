"""Reproducibly generate the capstone dataset and train the deployed model.

Run from the project root:
    python train_model.py
"""
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
n = 1500
age = rng.integers(18, 76, n)
annual_income = np.clip(rng.normal(70000, 30000, n), 20000, 200000).round().astype(int)
credit_score = np.clip(rng.normal(680, 75, n), 300, 850).round().astype(int)
loan_amount = np.clip(rng.normal(45000, 22000, n), 5000, 150000).round().astype(int)
employment_years = np.clip(rng.normal(8, 7, n), 0, 35).round(1)
debt_to_income = np.clip(rng.normal(0.30, 0.12, n), 0.05, 0.70).round(3)

z = (
    1.35
    - 0.012 * (credit_score - 650)
    + 0.95 * (debt_to_income - 0.30)
    + 0.000010 * (loan_amount - 45000)
    - 0.000003 * (annual_income - 70000)
    - 0.018 * (employment_years - 8)
    + 0.004 * (age - 40)
)
prob_default = 1 / (1 + np.exp(-z))
default = rng.binomial(1, prob_default, n)

df = pd.DataFrame({
    "age": age,
    "annual_income": annual_income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "employment_years": employment_years,
    "debt_to_income": debt_to_income,
    "default": default,
})
df.to_csv(DATA_DIR / "customer_accounts.csv", index=False)

features = ["age", "annual_income", "credit_score", "loan_amount", "employment_years", "debt_to_income"]
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["default"], test_size=0.20, random_state=42, stratify=df["default"]
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
    "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
    "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
    "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "test_size": int(len(y_test)),
    "features": features,
    "target": "default",
    "dataset_note": "Synthetic customer-account dataset generated reproducibly with random seed 42 for this internship capstone.",
}

with open(MODEL_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"Saved {len(df):,} records to {DATA_DIR / 'customer_accounts.csv'}")
print(f"Saved trained model to {MODEL_DIR / 'model.pkl'}")
print(f"Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
