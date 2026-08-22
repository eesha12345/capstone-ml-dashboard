import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.title("📊 Exploratory Data Analysis Portal")
st.subheader("Platform Operational Performance Metrics")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "customer_accounts.csv"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

df = load_data()
metrics = load_metrics()

c1, c2 = st.columns(2)
with c1:
    st.metric(label="Total Processed Records", value=f"{len(df):,} Accounts")
with c2:
    st.metric(label="Model Prediction Accuracy", value=f"{metrics['accuracy'] * 100:.1f}%", delta=f"ROC-AUC {metrics['roc_auc']:.3f}")

st.markdown("---")
st.subheader("Data Trend Distribution Vector")

# Keep the original line-chart style, but use genuine dataset-derived values.
age_bins = pd.cut(df["age"], bins=10)
trend = df.groupby(age_bins, observed=False)["default"].mean().reset_index()
trend["age"] = trend["age"].apply(lambda x: f"{int(x.left)}-{int(x.right)}")
trend = trend.set_index("age")
st.line_chart(trend["default"].rename("Default Rate"))

left, right = st.columns(2)
with left:
    st.subheader("Account Data Sample")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
with right:
    st.subheader("Target Distribution")
    counts = df["default"].value_counts().rename(index={0: "No Default", 1: "Default"})
    st.bar_chart(counts)

st.markdown("---")
st.subheader("Model Evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
m2.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
m3.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")
m4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.subheader("Confusion Matrix")
cm = pd.DataFrame(
    metrics["confusion_matrix"],
    index=["Actual: No Default", "Actual: Default"],
    columns=["Predicted: No Default", "Predicted: Default"],
)
st.dataframe(cm, use_container_width=True)

st.caption("Dataset note: synthetic customer-account data generated reproducibly with seed 42 for this capstone project.")
