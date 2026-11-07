import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛒", layout="centered")

# Palette: plum + violet (brand), coral (mistakes), mint (correct), sun (accent), cream (background)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&display=swap');
[data-testid="stAppViewContainer"] {background: #fffaf3; color: #231942;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 980px; padding-top: 1.2rem; padding-bottom: 2.5rem;}
footer {visibility: hidden;}
[data-testid="stWidgetLabel"] p, label p {color: #231942 !important; font-weight: 600;}
h1, h2, h3, .pop {font-family: 'Poppins', 'Segoe UI', sans-serif;}

.hero {background: radial-gradient(circle at 92% 12%, rgba(255,200,69,.40) 0, rgba(255,200,69,0) 38%),
                    linear-gradient(135deg, #2d1b4e 0%, #5b2fa0 100%);
       border-radius: 24px; padding: 30px 34px; margin-bottom: 22px;
       box-shadow: 0 12px 30px rgba(45,27,78,.30); border-bottom: 6px solid #ffc845;}
.hero table {width: 100%; border-collapse: collapse;}
.hero td {vertical-align: middle; padding: 0;}
.hero .kick {color: #ffd98a; font-size: .76rem; font-weight: 700; letter-spacing: .22em;}
.hero h1 {color: #ffffff; font-size: 2.1rem; font-weight: 800; line-height: 1.18; margin: 8px 0 10px; padding: 0;}
.hero p {color: #e9defa; font-size: 1rem; margin: 0; line-height: 1.5;}
.hero .tag {display: inline-block; background: rgba(255,255,255,.14); color: #ffffff; font-size: .78rem;
            font-weight: 600; padding: 5px 13px; border-radius: 999px; margin: 16px 8px 0 0;}
.hero .cart {width: 96px; height: 96px; line-height: 96px; text-align: center; font-size: 3rem;
             border-radius: 50%; background: rgba(255,200,69,.22); border: 2px dashed #ffc845;}

.step {font-family: 'Poppins', 'Segoe UI', sans-serif; font-size: 1.15rem; font-weight: 700; color: #231942;
       line-height: 1.2; margin: 26px 0 10px;}
.step.up {color: #1e40af;}
.step.up b {background: #2563eb; color: #ffffff;}
.step b {display: inline-block; width: 32px; height: 32px; line-height: 32px; text-align: center; border-radius: 50%;
         background: #ffc845; color: #2d1b4e; margin-right: 10px; font-size: .95rem;}

[data-testid="stFileUploader"] section {background: #eaf2ff; border: 2px dashed #2563eb; border-radius: 18px;}
[data-testid="stFileUploader"] * {color: #1e3a8a !important;}
[data-testid="stFileUploader"] svg {fill: #2563eb !important;}
[data-testid="stFileUploader"] section button {background: #ffffff; border: 1px solid #2563eb; border-radius: 10px;}
[data-testid="stForm"] {border: 1px solid #e3d9f5; border-radius: 20px; padding: 18px 22px 14px; background: #ffffff;
                        box-shadow: 0 4px 16px rgba(91,47,160,.07);}
[data-testid="stFormSubmitButton"] button {background: linear-gradient(135deg, #ff6f61, #ff8a5c); color: #ffffff; border: 0;
        border-radius: 14px; font-weight: 700; padding: .6rem 1.8rem; box-shadow: 0 6px 14px rgba(255,111,97,.35);}
[data-testid="stFormSubmitButton"] button:hover {background: #2d1b4e; color: #ffffff;}

.chips {margin: 4px 0 10px;}
.chip {display: inline-block; background: #efe7fb; color: #3b1f5e; font-size: .82rem; font-weight: 700;
       padding: 6px 15px; border-radius: 999px; margin: 0 8px 8px 0;}
.chip.sun {background: #fff0c2; color: #6b4a00;}
.bal {width: 100%; border-collapse: collapse; margin: 6px 0 4px; table-layout: fixed;}
.bal td {padding: 9px 12px; font-size: .84rem; font-weight: 700; color: #ffffff; white-space: nowrap; overflow: hidden;}

.sect {font-family: 'Poppins', 'Segoe UI', sans-serif; font-size: 1.25rem; font-weight: 800; color: #231942;
       line-height: 1.15; margin: 26px 0 10px; padding: 3px 0 3px 12px; border-left: 6px solid #ff6f61;}
.sect.blue {border-left-color: #2563eb;}

.verdict {background: #2d1b4e; color: #f4ecff; border-radius: 18px; padding: 16px 22px; margin: 6px 0 14px;
          font-size: .98rem; line-height: 1.6;}
.verdict b {color: #ffc845;}

.mt {width: 100%; border-collapse: separate; border-spacing: 12px; table-layout: fixed; margin: 0 -12px;}
.mc {background: #ffffff; border: 1px solid #ece4f8; border-radius: 18px; padding: 16px 18px 14px;
     box-shadow: 0 6px 16px rgba(45,27,78,.08); vertical-align: top;}
.mc .nm {font-size: .74rem; letter-spacing: .1em; text-transform: uppercase; color: #6b5f82; font-weight: 800;}
.mc .vl {font-family: 'Poppins', 'Segoe UI', sans-serif; font-size: 2.1rem; font-weight: 800; color: #231942;
         line-height: 1.2; margin: 4px 0 9px;}
.mc .bar {height: 8px; border-radius: 8px; background: #efe9f8; overflow: hidden;}
.mc .bar i {display: block; height: 100%; border-radius: 8px;}
.mc .ht {font-size: .78rem; color: #7a6f90; margin-top: 9px; line-height: 1.35;}

.cmw {max-width: 600px; margin: 0 auto;}
.cm {width: 100%; border-collapse: separate; border-spacing: 8px; table-layout: fixed;}
.cm th {background: #ece5f8; color: #3b1f5e; font-size: .74rem; font-weight: 700; padding: 8px 5px; border-radius: 10px;}
.cm th.e {background: transparent;}
.cm td {border-radius: 14px; text-align: center; padding: 11px 6px; color: #231942;}
.cm .ic {font-size: 1rem; line-height: 1.1;}
.cm .n {font-family: 'Poppins', 'Segoe UI', sans-serif; font-size: 1.55rem; font-weight: 800; line-height: 1.15;}
.cm .l {font-size: .76rem; font-weight: 700; margin-top: 2px;}
.cm .t {font-size: .62rem; letter-spacing: .06em; opacity: .7; margin-top: 1px;}
.cm .p {font-size: .66rem; opacity: .75; margin-top: 1px;}
.hint {background: #ffffff; border: 2px dashed #cdbbf0; border-radius: 18px; padding: 18px 22px; color: #3b1f5e; font-size: .98rem;}
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
    """Mix a colour with white (0 = white, 1 = full colour)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    mix = lambda c: int(255 - (255 - c) * strength)
    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


def balance_bar(pos: int, total: int) -> str:
    pos_pct = pos / total * 100
    neg_pct = 100 - pos_pct
    pos_w = max(pos_pct, 12)          # keep the small segment readable
    neg_w = 100 - pos_w
    return (f'<table class="bal"><tr>'
            f'<td style="width:{neg_w:.1f}%;background:#5b2fa0;border-radius:12px 0 0 12px;">No purchase &middot; {neg_pct:.1f}%</td>'
            f'<td style="width:{pos_w:.1f}%;background:#ff6f61;border-radius:0 12px 12px 0;">Purchase &middot; {pos_pct:.1f}%</td>'
            f'</tr></table>')


def metric_cards(scores):
    meta = {
        "Accuracy": ("#5b2fa0", "🎯", "Overall share of correct predictions"),
        "AUC": ("#2f80ed", "📈", "How well buyers rank above non-buyers"),
        "Precision": ("#12b886", "🔍", "Predicted purchases that were real"),
        "Recall": ("#ff8a3d", "🧲", "Real purchases the model caught"),
        "F1 Score": ("#e64980", "⚖️", "Balance between precision and recall"),
        "MCC": ("#f0a500", "🧩", "Quality across all four outcomes"),
    }
    cells = []
    for label, score in scores:
        color, icon, hint = meta[label]
        width = max(0.0, min(1.0, score)) * 100
        cells.append(
            f'<td class="mc" style="border-top:5px solid {color};">'
            f'<div class="nm">{icon} {label}</div><div class="vl">{score:.4f}</div>'
            f'<div class="bar"><i style="width:{width:.1f}%;background:{color};"></i></div>'
            f'<div class="ht">{hint}</div></td>')
    rows = "".join(f"<tr>{''.join(cells[i:i + 3])}</tr>" for i in (0, 3))
    return f'<table class="mt">{rows}</table>'


def confusion_html(tn, fp, fn, tp):
    total = tn + fp + fn + tp

    def cell(value, icon, label, tag, good):
        share = value / total
        base = "#12b886" if good else "#ff6f61"
        bg = blend(base, 0.14 + 0.24 * min(1.0, share * (1.15 if good else 6)))   # mild pastel shades
        return (f'<td style="background:{bg};"><div class="ic">{icon}</div><div class="n">{value:,}</div>'
                f'<div class="l">{label}</div><div class="t">{tag}</div>'
                f'<div class="p">{share:.1%} of all sessions</div></td>')

    return (
        '<div class="cmw"><table class="cm"><colgroup><col style="width:22%"><col style="width:39%"><col style="width:39%"></colgroup>'
        '<tr><th class="e"></th><th>Predicted: No Purchase</th><th>Predicted: Purchase</th></tr>'
        f'<tr><th>Actual:<br>No Purchase</th>{cell(tn, "✅", "Browsers spotted", "TRUE NEGATIVE", True)}'
        f'{cell(fp, "🚨", "False alarms", "FALSE POSITIVE", False)}</tr>'
        f'<tr><th>Actual:<br>Purchase</th>{cell(fn, "😕", "Missed buyers", "FALSE NEGATIVE", False)}'
        f'{cell(tp, "🛍️", "Buyers caught", "TRUE POSITIVE", True)}</tr>'
        '</table></div>')


# ------------------------------ Header ------------------------------
st.markdown(
    '<div class="hero"><table><tr><td>'
    '<div class="kick">ONLINE SHOPPERS PURCHASING INTENTION</div>'
    "<h1>Will this visit end in a purchase?</h1>"
    "<p>Pick a trained classification model, upload your test data and see how well it "
    "tells buyers from browsers.</p>"
    '<span class="tag">5 trained models</span><span class="tag">6 evaluation metrics</span>'
    '<span class="tag">No retraining</span>'
    '</td><td style="width:120px;text-align:right;"><div class="cart" style="display:inline-block;">🛒</div></td>'
    '</tr></table></div>',
    unsafe_allow_html=True)

# ------------------------------ Step 1: upload + preview ------------------------------
st.markdown('<div class="step up"><b>1</b>Upload your test data</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Test CSV (including Revenue)", type=["csv"])

test_df = None
if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Cannot read CSV: {exc}")
        st.stop()

    if test_df.empty:
        st.error("The uploaded CSV has no rows.")
        st.stop()

    st.markdown('<div class="sect blue">Dataset Preview</div>', unsafe_allow_html=True)
    chips = (f'<span class="chip">{len(test_df):,} rows</span>'
             f'<span class="chip">{test_df.shape[1]} columns</span>'
             f'<span class="chip sun">{int(test_df.isna().sum().sum())} missing values</span>')
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    if "Revenue" in test_df.columns:
        flags = test_df["Revenue"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        if flags.notna().all():
            st.markdown(balance_bar(int(flags.sum()), len(flags)), unsafe_allow_html=True)

    n_rows = st.number_input("Rows to show", min_value=1, max_value=len(test_df),
                             value=min(10, len(test_df)), step=5)
    st.dataframe(test_df.head(int(n_rows)), hide_index=True)

# ------------------------------ Step 2: model + run ------------------------------
st.markdown('<div class="step"><b>2</b>Choose a model and run</div>', unsafe_allow_html=True)
with st.form("evaluation_form"):
    selected_model = st.selectbox("Classification model", list(MODEL_FILES))
    run_evaluation = st.form_submit_button("Run Evaluation", type="primary")

if not run_evaluation:
    st.markdown('<div class="hint">Upload <b>test_data.csv</b>, choose a model, then press '
                '<b>Run Evaluation</b> to see the metrics and the confusion matrix.</div>',
                unsafe_allow_html=True)
    st.stop()

if test_df is None:
    st.error("Please upload a test CSV before running the evaluation.")
    st.stop()

if "Revenue" not in test_df.columns:
    st.error("CSV must contain a Revenue target column.")
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
tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, predictions, labels=[0, 1]).ravel())

# ------------------------------ Results ------------------------------
st.markdown('<div class="sect">Model Evaluation</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="chips"><span class="chip">{selected_model}</span><span class="chip">{len(test_df):,} rows</span>'
    f'<span class="chip">{X.shape[1]} features</span></div>', unsafe_allow_html=True)

summary = []
if tp + fn > 0:
    summary.append(f"Out of <b>{tp + fn:,}</b> real purchases, it caught <b>{tp:,}</b> ({tp / (tp + fn):.1%}).")
if tp + fp > 0:
    summary.append(f"Of the <b>{tp + fp:,}</b> sessions it flagged as purchases, <b>{tp:,}</b> were real ({tp / (tp + fp):.1%}).")
if summary:
    st.markdown(f'<div class="verdict">{" ".join(summary)}</div>', unsafe_allow_html=True)

st.markdown(metric_cards(scores), unsafe_allow_html=True)

st.markdown('<div class="sect">Confusion Matrix</div>', unsafe_allow_html=True)
st.markdown(confusion_html(tn, fp, fn, tp), unsafe_allow_html=True)
st.caption(f"TN = {tn:,} · FP = {fp:,} · FN = {fn:,} · TP = {tp:,}")

with st.expander("Evaluation metric formulas"):
    st.latex(r"Accuracy = \frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"Precision = \frac{TP}{TP+FP}")
    st.latex(r"Recall = \frac{TP}{TP+FN}")
    st.latex(r"F1 = \frac{2\cdot Precision\cdot Recall}{Precision+Recall}")
    st.latex(r"MCC = \frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}")
    st.markdown("**AUC:** Area under the ROC curve across decision thresholds; calculated from predicted probabilities.")
