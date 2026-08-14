"""Interactive Dry Bean classification dashboard."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
META = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))

st.set_page_config(page_title="BeanScope ML Lab", page_icon="🫘", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#f8faf7 0%,#eef5e9 100%);}
[data-testid="stMetric"] {background:#ffffff;border:1px solid #d9e7d1;border-radius:12px;padding:12px;}
.hero {padding:1.5rem;border-radius:18px;background:linear-gradient(120deg,#24452f,#52734d);color:white;margin-bottom:1rem;}
.hero h1 {margin:0;color:white}.hero p {margin:.4rem 0 0;color:#e8f2e4}
</style>
<div class="hero"><h1>🫘 BeanScope ML Lab</h1><p>Live multiclass prediction and model evaluation for seven Dry Bean varieties</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def calculate_metrics(y_true, y_pred, probabilities):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


with st.sidebar:
    st.header("Experiment controls")
    model_name = st.selectbox(
        "Select a classification model",
        list(META["model_files"]),
    )
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"], help="Use test_data.csv. The Class column is optional for prediction, but required for live evaluation.")
    st.caption("Expected: 16 numeric feature columns; optional `Class` ground-truth column.")

comparison = pd.DataFrame(META["metrics"]).T
comparison.index.name = "Model"
tabs = st.tabs(["Live predictions", "Model comparison", "Dataset guide"])

with tabs[1]:
    st.subheader("Held-out test-set comparison")
    st.caption("Stratified 80/20 split, random_state=42. Multiclass AUC uses macro one-vs-rest; Precision, Recall and F1 use macro averaging.")
    st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, color="#cde8c5"), width="stretch")
    st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])

with tabs[2]:
    st.subheader("Input schema")
    st.write("The UCI Dry Bean dataset contains 13,611 observations, 16 numeric shape features, and seven bean classes.")
    st.code(", ".join(META["feature_names"]), language=None)
    st.write("Classes: " + ", ".join(META["class_names"]))
    st.download_button("Download bundled test_data.csv", (ROOT / "test_data.csv").read_bytes(), "test_data.csv", "text/csv")

with tabs[0]:
    if uploaded is None:
        st.info("Upload `test_data.csv` from the sidebar to generate live predictions. You can download a copy from Dataset guide.")
    else:
        try:
            data = pd.read_csv(uploaded)
            missing = [c for c in META["feature_names"] if c not in data.columns]
            if missing:
                st.error("Missing required feature columns: " + ", ".join(missing))
                st.stop()
            X = data[META["feature_names"]].apply(pd.to_numeric, errors="raise")
            model = load_model(META["model_files"][model_name])
            encoder = load_model("label_encoder.joblib")
            predicted_codes = model.predict(X)
            probabilities = model.predict_proba(X)
            predicted_labels = encoder.inverse_transform(predicted_codes)
            output = data.copy()
            output["Predicted_Class"] = predicted_labels
            output["Prediction_Confidence"] = probabilities.max(axis=1)
            st.success(f"Generated {len(output):,} predictions with {model_name}.")

            if "Class" in data.columns:
                known = data["Class"].astype(str)
                unknown = sorted(set(known) - set(META["class_names"]))
                if unknown:
                    st.error("Unknown labels in Class: " + ", ".join(unknown))
                    st.stop()
                y_true = encoder.transform(known)
                live = calculate_metrics(y_true, predicted_codes, probabilities)
                cols = st.columns(6)
                for col, (metric, value) in zip(cols, live.items()):
                    col.metric(metric, f"{value:.4f}")

                left, right = st.columns([1, 1.15])
                with left:
                    st.subheader("Confusion matrix")
                    matrix = confusion_matrix(y_true, predicted_codes, labels=range(len(META["class_names"])))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGn", xticklabels=META["class_names"], yticklabels=META["class_names"], ax=ax)
                    ax.set(xlabel="Predicted class", ylabel="Actual class")
                    plt.xticks(rotation=45, ha="right")
                    st.pyplot(fig)
                    plt.close(fig)
                with right:
                    st.subheader("Classification report")
                    report = classification_report(y_true, predicted_codes, target_names=META["class_names"], output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch")
            else:
                st.warning("Predictions are available, but evaluation metrics require a `Class` ground-truth column.")

            st.subheader("Prediction preview")
            st.dataframe(output[[*(['Class'] if 'Class' in output else []), "Predicted_Class", "Prediction_Confidence"]].head(100), width="stretch")
            st.download_button("Download all predictions", output.to_csv(index=False).encode("utf-8"), f"{model_name.lower().replace(' ', '_')}_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")
