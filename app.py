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

st.set_page_config(page_title="BeanScope ML Lab", page_icon="🫘", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp {background:radial-gradient(circle at 8% 4%,rgba(222,238,202,.9),transparent 24%),radial-gradient(circle at 95% 38%,rgba(244,207,178,.42),transparent 22%),linear-gradient(145deg,#f9fbf7 0%,#edf5e9 52%,#fff8ef 100%);}
.block-container {max-width:none;width:100%;padding:1.5rem 2rem 3rem;}
[data-testid="stSidebar"] {display:none;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#fff,#f7fbf4);border:1px solid #cfe0c8;border-top:4px solid #d87855;border-radius:14px;padding:14px;box-shadow:0 7px 22px rgba(35,69,47,.09);}
.hero {position:relative;overflow:hidden;padding:2.25rem 2.4rem;border-radius:26px;background:radial-gradient(circle at 90% 12%,rgba(235,188,111,.48) 0,transparent 25%),radial-gradient(circle at 75% 120%,rgba(126,164,104,.7) 0,transparent 34%),linear-gradient(120deg,#123923,#315c3f 62%,#5e7d4f);color:white;margin-bottom:1.5rem;box-shadow:0 18px 42px rgba(27,65,42,.22);border-bottom:5px solid #d8835d;}
.hero h1 {margin:0;color:white;font-size:2.55rem;text-shadow:0 2px 12px rgba(0,0,0,.16)}.hero p {margin:.55rem 0 0;color:#f2f8ee;font-size:1.08rem}.hero-tag {display:inline-block;background:#e7f5b9;color:#23452f;padding:.34rem .75rem;border-radius:999px;font-weight:800;font-size:.76rem;margin-bottom:.8rem;letter-spacing:.055em;box-shadow:0 4px 12px rgba(0,0,0,.12)}
.control-title {font-size:1.35rem;font-weight:750;color:#173c29;margin-bottom:.15rem}.control-subtitle {color:#5c6d60;margin-bottom:.8rem}
.model-card {height:100%;background:linear-gradient(145deg,#fff,#f8fbf5);border:1px solid #d6e4d0;border-left:7px solid #d87855;border-radius:17px;padding:1.15rem 1.25rem;box-shadow:0 9px 26px rgba(35,69,47,.1);}
.model-card .eyebrow {color:#6c806d;font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em}.model-card h3 {color:#1e462e;margin:.25rem 0 .4rem}.model-card p {color:#58675b;margin:0}.score-pill {display:inline-block;margin-top:.65rem;margin-right:.35rem;background:#edf5e9;color:#28523a;padding:.25rem .55rem;border-radius:8px;font-size:.78rem;font-weight:700}
div[role="radiogroup"] {background:#fff;border:1px solid #d5e2cf;border-radius:14px;padding:.55rem .8rem;box-shadow:0 5px 18px rgba(35,69,47,.05);}
div[role="radiogroup"] label {padding:.4rem .55rem;border-radius:9px;}
[data-testid="stFileUploader"] {background:linear-gradient(135deg,#fff,#f8fbf5);border:2px dashed #76906d;border-radius:18px;padding:.7rem 1rem;box-shadow:0 7px 22px rgba(35,69,47,.07);}
[data-testid="stFileUploader"] section {background:#f1f7ed;border-radius:12px;}
.workspace {background:rgba(255,255,255,.72);border:1px solid #d6e4d0;border-radius:22px;padding:1.15rem 1.3rem .4rem;margin:.4rem 0 1.25rem;box-shadow:0 10px 30px rgba(35,69,47,.07);}
.section-kicker {display:inline-block;color:#8a442f;background:#f8e3d7;padding:.3rem .62rem;border-radius:8px;font-size:.73rem;font-weight:850;letter-spacing:.11em;text-transform:uppercase;margin-bottom:.45rem;}
.model-grid-card {min-height:165px;background:linear-gradient(145deg,#fff,#fbfdf9);border:1px solid #d7e4d0;border-top:5px solid var(--accent);border-radius:18px;padding:1rem 1.05rem;box-shadow:0 7px 20px rgba(35,69,47,.07);margin-bottom:.8rem;transition:transform .2s ease,box-shadow .2s ease;}
.model-grid-card:hover {transform:translateY(-3px);box-shadow:0 12px 28px rgba(35,69,47,.13);}
.model-icon {width:44px;height:44px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:color-mix(in srgb,var(--accent) 16%,white);font-size:1.4rem;margin-bottom:.65rem;box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 28%,white);}
.model-grid-card h4 {color:#173c29;margin:0 0 .3rem;font-size:1rem;}.model-grid-card p {color:#627064;font-size:.82rem;line-height:1.35;margin:0 0 .65rem;}
.model-type {font-size:.68rem;color:#52734d;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
.mini-score {display:inline-block;background:#f1f6ee;color:#31523a;border-radius:7px;padding:.2rem .42rem;margin-right:.25rem;font-size:.7rem;font-weight:700;}
.selector-shell {width:100%;background:linear-gradient(120deg,#fff,#eef7e8 65%,#fff0e6);border:3px solid #78966b;border-radius:24px;padding:1.6rem 1.8rem 1.4rem;margin:.7rem 0 1.55rem;box-shadow:0 18px 42px rgba(35,69,47,.2);}
.step-badge {display:inline-flex;align-items:center;justify-content:center;width:27px;height:27px;border-radius:50%;background:#315c3f;color:white;font-weight:800;margin-right:.45rem;}
[data-testid="stSelectbox"] {width:100% !important;max-width:none !important;margin:.25rem 0 .4rem;}
[data-testid="stSelectbox"] [data-baseweb="select"] {width:100% !important;max-width:none !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {background:linear-gradient(110deg,#244f35,#52734d 62%,#78905b) !important;border:4px solid #dcecaf !important;border-radius:20px !important;min-height:118px;padding:0 2rem;box-shadow:0 14px 32px rgba(23,60,41,.34) !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {background:linear-gradient(110deg,#52734d,#78966b) !important;border-color:#dcecaf !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] div,[data-testid="stSelectbox"] [data-baseweb="select"] span,[data-testid="stSelectbox"] [data-baseweb="select"] input {color:#fff !important;-webkit-text-fill-color:#fff !important;font-size:2.1rem !important;line-height:1.25 !important;font-weight:850 !important;letter-spacing:.015em;}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {fill:#fff !important;color:#fff !important;width:40px;height:40px;}
[data-baseweb="popover"] [role="listbox"] {background:#f3f8f0 !important;border:1px solid #9db694 !important;}
[data-baseweb="popover"] [role="option"] {min-height:84px;padding:1.1rem 1.55rem !important;}
[data-baseweb="popover"] [role="option"],[data-baseweb="popover"] [role="option"] * {color:#174a2d !important;-webkit-text-fill-color:#174a2d !important;font-size:1.55rem !important;line-height:1.3 !important;font-weight:750;}
[data-baseweb="popover"] [role="option"]:hover {background:#d7ead0 !important;}
[data-baseweb="popover"] [role="option"] {color:#174a2d !important;-webkit-text-fill-color:#174a2d !important;background:#fff !important;}
[data-baseweb="popover"] [role="option"]:hover,[data-baseweb="popover"] [aria-selected="true"] {background:#dcebd5 !important;color:#0d4225 !important;-webkit-text-fill-color:#0d4225 !important;}
div[data-testid="stPills"] button {min-height:45px;background:#fff;color:#244a32 !important;border:1px solid #b9cdb1;}
div[data-testid="stPills"] button p,div[data-testid="stPills"] button span {color:#244a32 !important;font-weight:750;}
div[data-testid="stPills"] button:hover {background:#dfeeda !important;border-color:#52734d;color:#173c29 !important;box-shadow:0 4px 12px rgba(35,69,47,.13);}
div[data-testid="stPills"] button:hover p,div[data-testid="stPills"] button:hover span {color:#173c29 !important;}
div[data-testid="stPills"] button[aria-pressed="true"] {background:#315c3f !important;border-color:#315c3f;color:#fff !important;}
div[data-testid="stPills"] button[aria-pressed="true"] p,div[data-testid="stPills"] button[aria-pressed="true"] span {color:#fff !important;}
[data-testid="stTabs"] [role="tablist"] {gap:.5rem;background:#e7efe2;padding:.4rem;border-radius:14px;}
[data-testid="stTabs"] button[role="tab"] {border-radius:10px;padding:.5rem 1.1rem;font-weight:750;color:#31523a !important;background:#f6faf3;}
[data-testid="stTabs"] button[role="tab"] p {color:#31523a !important;font-weight:750;}
[data-testid="stTabs"] button[role="tab"]:hover {background:#cfdfc8;color:#173c29 !important;}
[data-testid="stTabs"] button[role="tab"]:hover p {color:#173c29 !important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {background:#315c3f;color:#fff !important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {color:#fff !important;}
/* Streamlit 1.61 renders st.pills as stButtonGroup. */
[data-testid="stButtonGroup"] button {background:#eef5e9 !important;border:1px solid #9db694 !important;color:#174a2d !important;box-shadow:none !important;}
[data-testid="stButtonGroup"] button * {color:#174a2d !important;-webkit-text-fill-color:#174a2d !important;opacity:1 !important;}
[data-testid="stButtonGroup"] button:hover {background:#d7ead0 !important;border-color:#315c3f !important;}
[data-testid="stButtonGroup"] button:hover * {color:#0d4225 !important;-webkit-text-fill-color:#0d4225 !important;opacity:1 !important;}
[data-testid="stButtonGroup"] button[aria-pressed="true"] {background:#315c3f !important;border-color:#315c3f !important;}
[data-testid="stButtonGroup"] button[aria-pressed="true"] * {color:#fff !important;-webkit-text-fill-color:#fff !important;}
/* BaseWeb tab rules keep inactive and hovered labels readable. */
[data-baseweb="tab-list"] {background:#e2ecdd !important;}
button[data-baseweb="tab"] {background:#f3f8f0 !important;color:#176238 !important;opacity:1 !important;}
button[data-baseweb="tab"] * {color:#176238 !important;-webkit-text-fill-color:#176238 !important;opacity:1 !important;}
button[data-baseweb="tab"]:hover {background:#cfe2c8 !important;}
button[data-baseweb="tab"]:hover * {color:#0d4225 !important;-webkit-text-fill-color:#0d4225 !important;opacity:1 !important;}
button[data-baseweb="tab"][aria-selected="true"] {background:#315c3f !important;}
button[data-baseweb="tab"][aria-selected="true"] * {color:#fff !important;-webkit-text-fill-color:#fff !important;opacity:1 !important;}
button[data-testid="stBaseButton-pills"],button[data-testid="stBaseButton-pills"] * {color:#174a2d !important;-webkit-text-fill-color:#174a2d !important;}
button[data-testid="stBaseButton-pills"]:hover,button[data-testid="stBaseButton-pills"]:hover * {color:#0d4225 !important;-webkit-text-fill-color:#0d4225 !important;background:#d7ead0 !important;}
[data-testid="stTabs"] button[role="tab"]:not([aria-selected="true"]),[data-testid="stTabs"] button[role="tab"]:not([aria-selected="true"]) * {color:#176238 !important;-webkit-text-fill-color:#176238 !important;opacity:1 !important;}
.stDownloadButton button,.stButton button {background:linear-gradient(110deg,#315c3f,#52734d);color:#fff;border:0;border-radius:11px;font-weight:750;box-shadow:0 6px 16px rgba(35,69,47,.18);}
.stDownloadButton button:hover,.stButton button:hover {background:linear-gradient(110deg,#244a32,#3f663f);color:#fff;transform:translateY(-1px);}
[data-testid="stAlert"] {border-radius:14px;box-shadow:0 6px 18px rgba(35,69,47,.06);}
/* Final tab override: every navigation label stays green in every state. */
.stTabs [data-baseweb="tab-list"] button[role="tab"],
.stTabs [data-baseweb="tab-list"] button[role="tab"] *,
[data-testid="stTabs"] button[role="tab"],
[data-testid="stTabs"] button[role="tab"] * {
    color:#064b2a !important;
    -webkit-text-fill-color:#064b2a !important;
    opacity:1 !important;
    text-shadow:none !important;
}
.stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"],
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background:#d9ead3 !important;
    color:#043d21 !important;
    border-bottom:3px solid #064b2a !important;
}
.stTabs [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] *,
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
    color:#043d21 !important;
    -webkit-text-fill-color:#043d21 !important;
}
.stTabs [data-baseweb="tab-list"] button[role="tab"]:hover,
[data-testid="stTabs"] button[role="tab"]:hover {background:#cbe2c4 !important;}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] span,
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p {
    color:#07552f !important;
    -webkit-text-fill-color:#07552f !important;
    font-weight:800 !important;
}
@media (max-width:800px) {.hero h1 {font-size:1.9rem}.block-container {padding-left:1rem;padding-right:1rem}.model-grid-card {min-height:auto}.selector-shell {padding:1.15rem 1rem 1rem}[data-testid="stSelectbox"] [data-baseweb="select"] > div {min-height:88px;padding:0 1.2rem}[data-testid="stSelectbox"] [data-baseweb="select"] div,[data-testid="stSelectbox"] [data-baseweb="select"] span,[data-testid="stSelectbox"] [data-baseweb="select"] input {font-size:1.5rem !important;}}
</style>
<div class="hero"><span class="hero-tag">MACHINE LEARNING • MULTICLASS CLASSIFICATION</span><h1>🫘 BeanScope ML Lab</h1><p>Explore six classifiers, upload unseen measurements, and compare predictions across seven dry bean varieties.</p></div>
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


MODEL_DETAILS = {
    "Logistic Regression": "A strong, interpretable linear baseline trained on standardized measurements.",
    "Decision Tree": "A rule-based nonlinear classifier that is simple to interpret and visualize.",
    "k-Nearest Neighbors": "Classifies each bean from the labels of its closest standardized neighbors.",
    "Gaussian Naive Bayes": "A fast probabilistic classifier based on class-conditional feature distributions.",
    "Random Forest (Ensemble)": "An ensemble of decision trees and the overall winner on this test split.",
    "Support Vector Machine": "An RBF-kernel classifier with the highest multiclass AUC in this experiment.",
}

MODEL_PRESENTATION = {
    "Logistic Regression": ("Linear model", "📈", "#4f7cac"),
    "Decision Tree": ("Rule-based model", "🌳", "#619b66"),
    "k-Nearest Neighbors": ("Instance-based model", "🧭", "#9b72cf"),
    "Gaussian Naive Bayes": ("Probabilistic model", "🎲", "#df9b45"),
    "Random Forest (Ensemble)": ("Ensemble model", "🌲", "#317454"),
    "Support Vector Machine": ("Kernel model", "🎯", "#d56d55"),
}

st.markdown('<div class="section-kicker">Model laboratory</div><div class="control-title">Meet the six trained classifiers</div><div class="control-subtitle">Each model approaches the same seven-class problem differently. Held-out performance is visible on every card.</div>', unsafe_allow_html=True)
model_names = list(META["model_files"])
for row_start in range(0, len(model_names), 3):
    cards = st.columns(3, gap="medium")
    for card, name in zip(cards, model_names[row_start:row_start + 3]):
        score = META["metrics"][name]
        model_type, icon, accent = MODEL_PRESENTATION[name]
        with card:
            st.markdown(
                f'''<div class="model-grid-card" style="--accent:{accent}"><div class="model-icon">{icon}</div><div class="model-type">{model_type}</div>
                <h4>{name}</h4><p>{MODEL_DETAILS[name]}</p><span class="mini-score">ACC {score["Accuracy"]:.3f}</span>
                <span class="mini-score">F1 {score["F1"]:.3f}</span><span class="mini-score">AUC {score["AUC"]:.3f}</span></div>''',
                unsafe_allow_html=True,
            )

st.markdown('<div class="selector-shell"><div class="control-title"><span class="step-badge">1</span>Select the model to run</div><div class="control-subtitle">This selection controls live predictions, evaluation metrics, and diagnostic charts.</div>', unsafe_allow_html=True)
model_name = st.selectbox(
    "Select model",
    model_names,
    index=0,
    label_visibility="collapsed",
    placeholder="Choose a classification model",
    width="stretch",
)
st.markdown('</div>', unsafe_allow_html=True)

model_score = META["metrics"][model_name]
model_col, upload_col = st.columns([1, 1.45], gap="large", vertical_alignment="center")
with model_col:
    st.markdown(
        f'''<div class="model-card"><div class="eyebrow">Selected model</div><h3>{model_name}</h3>
        <p>{MODEL_DETAILS[model_name]}</p><span class="score-pill">Accuracy {model_score["Accuracy"]:.2%}</span>
        <span class="score-pill">F1 {model_score["F1"]:.3f}</span><span class="score-pill">AUC {model_score["AUC"]:.3f}</span></div>''',
        unsafe_allow_html=True,
    )
with upload_col:
    uploaded = st.file_uploader(
        "STEP 2 — UPLOAD TEST DATA (CSV)",
        type=["csv"],
        help="Use test_data.csv. The Class column is optional for prediction, but required for live evaluation.",
    )

st.caption("Evaluation uses a stratified 20% test split with random_state=42. Metrics update for the selected model and uploaded labels.")

comparison = pd.DataFrame(META["metrics"]).T
comparison.index.name = "Model"
tabs = st.tabs([
    ":green[Live predictions]",
    ":green[Model comparison]",
    ":green[Dataset guide]",
])

with tabs[1]:
    st.subheader("Held-out test-set comparison")
    st.caption("Stratified 80/20 split, random_state=42. Multiclass AUC uses macro one-vs-rest; Precision, Recall and F1 use macro averaging.")
    comparison_style = comparison.style.format("{:.4f}").highlight_max(
        axis=0,
        props="background-color:#cde8c5;color:#12351f;font-weight:800;",
    )
    st.dataframe(comparison_style, width="stretch")
    st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])

with tabs[2]:
    st.subheader("Input schema")
    st.write("The UCI Dry Bean dataset contains 13,611 observations, 16 numeric shape features, and seven bean classes.")
    st.code(", ".join(META["feature_names"]), language=None)
    st.write("Classes: " + ", ".join(META["class_names"]))
    st.download_button("Download bundled test_data.csv", (ROOT / "test_data.csv").read_bytes(), "test_data.csv", "text/csv")

with tabs[0]:
    if uploaded is None:
        st.info("Choose a model above and upload `test_data.csv` to generate live predictions. You can download a copy from Dataset guide.")
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
