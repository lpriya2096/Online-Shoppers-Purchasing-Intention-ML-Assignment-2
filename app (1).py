import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Online Shoppers' Purchase Predictor", page_icon="🛒", layout="centered")

NAVY, TEAL, GOLD = "#1f3a5f", "#1b7f8c", "#f2a900"

st.markdown("""
<style>
.block-container {max-width: 980px; padding-top: 1.2rem; padding-bottom: 2.5rem;}
footer {visibility: hidden;}
.hero {background: linear-gradient(120deg, #1f3a5f 0%, #1b7f8c 100%); border-radius: 20px;
       padding: 30px 34px 26px; margin-bottom: 20px; box-shadow: 0 10px 26px rgba(31,58,95,.28);
       border-bottom: 5px solid #f2a900;}
.hero .kicker {color: #cfe3ee; font-size: .78rem; font-weight: 700; letter-spacing: .16em;}
.hero h1 {color: #ffffff; font-size: 2.15rem; line-height: 1.15; margin: 6px 0 8px; padding: 0;}
.hero p {color: #e3eef6; font-size: 1.02rem; margin: 0;}
.hero .chip {display: inline-block; background: rgba(255,255,255,.16); color: #fff; font-size: .8rem;
             font-weight: 600; padding: 4px 12px; border-radius: 999px; margin: 14px 8px 0 0;}
[data-testid="stForm"] {border: 1px solid #d9e2ec; border-radius: 18px; padding: 18px 22px 14px;
                        background: rgba(27,127,140,.05);}
[data-testid="stFormSubmitButton"] button {background: #1b7f8c; color: #fff; border: 0; border-radius: 12px;
                        font-weight: 700; padding: .55rem 1.6rem;}
[data-testid="stFormSubmitButton"] button:hover {background: #1f3a5f; color: #fff;}
.sect {font-size: 1.25rem; font-weight: 800; margin: 26px 0 6px; padding-left: 12px; border-left: 6px solid #f2a900;}
.pill {display: inline-block; background: #e6f1f6; color: #1f3a5f; font-size: .82rem; font-weight: 700;
       padding: 5px 14px; border-radius: 999px; margin: 0 8px 6px 0;}
.mt {width: 100%; border-collapse: separate; border-spacing: 12px; table-layout: fixed; margin: 0 -12px;}
.mc {background: #ffffff; border: 1px solid #e1e7ef; border-radius: 16px; padding: 16px 18px 14px;
     box-shadow: 0 4px 14px rgba(31,58,95,.08); vertical-align: top;}
.mc .nm {font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: #5b6577; font-weight: 800;}
.mc .vl {font-size: 2.05rem; font-weight: 800; color: #1d2330; line-height: 1.15; margin: 4px 0 9px;}
.mc .bar {height: 7px; border-radius: 7px; background: #e8edf4; overflow: hidden;}
.mc .bar i {display: block; height: 100%; border-radius: 7px;}
.mc .ht {font-size: .78rem; color: #6b7686; margin-top: 9px; line-height: 1.3;}
.cm {width: 100%; border-collapse: separate; border-spacing: 10px; table-layout: fixed;}
.cm th {background: #1f3a5f; color: #ffffff; font-size: .8rem; font-weight: 700; padding: 10px 6px; border-radius: 10px;}
.cm th.e {background: transparent;}
.cm td {border-radius: 16px; text-align: center; padding: 20px 8px; color: #1d2330;}
.cm .n {font-size: 2rem; font-weight: 800; line-height: 1.1;}
.cm .l {font-size: .78rem; font-weight: 700; margin-top: 4px; letter-spacing: .03em;}
.cm .p {font-size: .74rem; opacity: .75; margin-top: 2px;}
.steps {width: 100%; border-collapse: separate; border-spacing: 12px; table-layout: fixed; margin: 0 -12px;}
.steps td {background: #ffffff; border: 1px dashed #b8c6d8; border-radius: 16px; padding: 18px; text-align: center;
           color: #1d2330; vertical-align: top;}
.steps b {display: inline-block; width: 30px; height: 30px; line-height: 30px; border-radius: 50%;
          background: #f2a900; color: #1f3a5f; font-size: .95rem;}
.steps div {margin-top: 8px; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "model/knn.joblib",
    "Gaussian Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine (SVM)": "model/svm.joblib",
}


@st.cache_resource(show_spinner=False, max_entries=2)
def load_model(model_path: str):
    return joblib.load(model_path)


def blend(hex_color: str, strength: float) -> str:
    """Mix a colour with white (strength 0 = white, 1 = full colour)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(255 - (255 - c) * strength)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def metric_cards(scores):
    meta = {
        "Accuracy": ("#1b7f8c", "Overall share of correct predictions"),
        "AUC": ("#6c5ce7", "How well purchases are ranked above non-purchases"),
        "Precision": ("#2f80ed", "Predicted purchases that were real"),
        "Recall": ("#f2994a", "Real purchases the model found"),
        "F1 Score": ("#27ae60", "Balance between precision and recall"),
        "MCC": ("#e0556b", "Quality across all four outcomes"),
    }
    cells = []
    for label, score in scores:
        color, hint = meta[label]
        width = max(0.0, min(1.0, score)) * 100
        cells.append(
            f'<td class="mc" style="border-top:5px solid {color};">'
            f'<div class="nm">{label}</div><div class="vl">{score:.4f}</div>'
            f'<div class="bar"><i style="width:{width:.1f}%;background:{color};"></i></div>'
            f'<div class="ht">{hint}</div></td>')
    rows = "".join(f"<tr>{''.join(cells[i:i + 3])}</tr>" for i in (0, 3))
    return f'<table class="mt">{rows}</table>'


def confusion_html(tn, fp, fn, tp):
    total = tn + fp + fn + tp

    def cell(value, label, good):
        share = value / total
        bg = blend("#27ae60" if good else "#e0556b", 0.18 + 0.75 * min(1.0, share * (1.15 if good else 6)))
        return (f'<td style="background:{bg};"><div class="n">{value:,}</div>'
                f'<div class="l">{label}</div><div class="p">{share:.1%} of all sessions</div></td>')

    return (
        '<table class="cm"><colgroup><col style="width:20%"><col style="width:40%"><col style="width:40%"></colgroup>'
        '<tr><th class="e"></th><th>Predicted: No Purchase</th><th>Predicted: Purchase</th></tr>'
        f'<tr><th>Actual:<br>No Purchase</th>{cell(tn, "True Negative", True)}{cell(fp, "False Positive", False)}</tr>'
        f'<tr><th>Actual:<br>Purchase</th>{cell(fn, "False Negative", False)}{cell(tp, "True Positive", True)}</tr>'
        '</table>')


st.markdown(
    '<div class="hero"><div class="kicker">MACHINE LEARNING &middot; CLASSIFICATION</div>'
    "<h1>Online Shoppers' Purchasing Intention</h1>"
    "<p>Pick a trained model, upload your test data, and see how well it predicts which visits end in a purchase.</p>"
    '<span class="chip">5 trained models</span><span class="chip">6 evaluation metrics</span>'
    '<span class="chip">No retraining</span></div>',
    unsafe_allow_html=True)

with st.form("evaluation_form"):
    left, right = st.columns([1, 1.3], gap="medium")
    with left:
        selected_model = st.selectbox("Classification model", list(MODEL_FILES))
    with right:
        uploaded_file = st.file_uploader("Test CSV (including Revenue)", type=["csv"])
    run_evaluation = st.form_submit_button("Run Evaluation", type="primary")

if not run_evaluation:
    st.markdown(
        '<table class="steps"><tr>'
        '<td><b>1</b><div>Choose a classification model</div></td>'
        '<td><b>2</b><div>Upload <code>test_data.csv</code></div></td>'
        '<td><b>3</b><div>Click <i>Run Evaluation</i> to see the results</div></td>'
        '</tr></table>', unsafe_allow_html=True)
    st.stop()

if uploaded_file is None:
    st.error("Please upload a test CSV before running the evaluation.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Cannot read CSV: {exc}")
    st.stop()

if test_df.empty or "Revenue" not in test_df.columns:
    st.error("CSV must contain rows and a Revenue target column.")
    st.stop()

raw_target = test_df["Revenue"].astype(str).str.strip().str.lower()
y = raw_target.map({"true": 1, "false": 0, "1": 1, "0": 0})
if y.isna().any():
    st.error("Revenue must contain only True/False or 1/0, without missing values.")
    st.stop()
y = y.astype(int)
if y.nunique() != 2:
    st.error("Test CSV must include both Revenue classes (0 and 1) to calculate AUC.")
    st.stop()

X = test_df.drop(columns=["Revenue"])
if len(test_df) != 2466:
    st.warning(f"Uploaded {len(test_df):,} rows. Use the held-out test_data.csv (2,466 rows) for assignment results.")

model_path = BASE_DIR / MODEL_FILES[selected_model]
if not model_path.is_file():
    st.error(f"Model file missing: {MODEL_FILES[selected_model]}")
    st.stop()

try:
    with st.spinner("Evaluating saved model..."):
        model = load_model(str(model_path))
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
        elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("preprocessing"), "feature_names_in_"):
            expected = list(model.named_steps["preprocessing"].feature_names_in_)
        else:
            expected = None
        if expected is not None:
            missing = [name for name in expected if name not in X.columns]
            if missing:
                st.error("Missing input columns: " + ", ".join(missing))
                st.stop()
            X = X[expected]
        predictions = model.predict(X)
        class_labels = list(model.classes_)
        if 1 not in class_labels:
            st.error("Saved model does not contain positive class 1.")
            st.stop()
        probabilities = model.predict_proba(X)[:, class_labels.index(1)]
except Exception as exc:
    st.error(f"Model loading or prediction failed: {exc}")
    st.stop()

scores = [
    ("Accuracy", accuracy_score(y, predictions)),
    ("AUC", roc_auc_score(y, probabilities)),
    ("Precision", precision_score(y, predictions, zero_division=0)),
    ("Recall", recall_score(y, predictions, zero_division=0)),
    ("F1 Score", f1_score(y, predictions, zero_division=0)),
    ("MCC", matthews_corrcoef(y, predictions)),
]

st.markdown('<div class="sect">Model Evaluation</div>', unsafe_allow_html=True)
st.markdown(
    f'<span class="pill">{selected_model}</span><span class="pill">{len(test_df):,} rows</span>'
    f'<span class="pill">{X.shape[1]} features</span>', unsafe_allow_html=True)
st.markdown(metric_cards(scores), unsafe_allow_html=True)

tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
st.markdown('<div class="sect">Confusion Matrix</div>', unsafe_allow_html=True)
st.markdown(confusion_html(int(tn), int(fp), int(fn), int(tp)), unsafe_allow_html=True)
st.caption(f"TN = {tn:,} · FP = {fp:,} · FN = {fn:,} · TP = {tp:,}")

with st.expander("Evaluation metric formulas"):
    st.latex(r"Accuracy = \frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"Precision = \frac{TP}{TP+FP}")
    st.latex(r"Recall = \frac{TP}{TP+FN}")
    st.latex(r"F1 = \frac{2\cdot Precision\cdot Recall}{Precision+Recall}")
    st.latex(r"MCC = \frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}")
    st.markdown("**AUC:** Area under the ROC curve across decision thresholds; calculated from predicted probabilities.")
