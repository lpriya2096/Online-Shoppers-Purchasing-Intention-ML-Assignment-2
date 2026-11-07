import html as _html
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Purchase Intent Predictor", page_icon="🛒", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {--navy:#14243d;--blue:#2563eb;--muted:#64748b;--line:#e2e8f0;--surface:#ffffff;}
html, body, [class*="css"], .stApp {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:#f5f7fb;color:var(--navy);}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1150px;padding-top:1.6rem;padding-bottom:4rem;}
footer {visibility:hidden;}
h1,h2,h3,p {font-family:'DM Sans',sans-serif;}
[data-testid="stWidgetLabel"] p,label p {color:#243650!important;font-weight:650;}
[data-testid="stCaptionContainer"] {color:#64748b!important;}
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#12243e 0%,#1c3b68 62%,#2563eb 135%);border-radius:22px;padding:36px 40px;margin-bottom:26px;box-shadow:0 16px 35px rgba(20,36,61,.12);}
.hero:after {content:'';position:absolute;width:310px;height:310px;right:-120px;top:-170px;border-radius:50%;border:60px solid rgba(255,255,255,.045);pointer-events:none;}
.hero table,.hero tr,.hero td {border:0!important;background:transparent!important;}
.hero table {width:100%;border-collapse:collapse;}
.hero td {vertical-align:middle;padding:0!important;}
.hero .kick {color:#a8d3ff;font-size:.76rem;font-weight:800;letter-spacing:.16em;}
.hero h1 {color:#fff;font-size:clamp(1.65rem,3.3vw,2.65rem);font-weight:800;letter-spacing:-.035em;line-height:1.2;margin:10px 0 12px;padding:0;}
.hero p {color:#d5e5fa;font-size:1rem;line-height:1.65;max-width:670px;margin:0;}
.hero .tag {display:inline-block;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.17);color:#f0f7ff;font-size:.76rem;font-weight:650;padding:6px 13px;border-radius:100px;margin:20px 7px 0 0;}
.hero .cart {width:90px;height:90px;line-height:90px;text-align:center;font-size:2.6rem;border-radius:22px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);}
.step {font-size:1.16rem;font-weight:800;color:var(--navy);margin:28px 0 13px;display:flex;align-items:center;letter-spacing:-.02em;}
.step b {display:inline-block;width:32px;height:32px;line-height:32px;text-align:center;border-radius:10px;background:#e8f0ff;color:#2563eb;margin-right:12px;font-size:.9rem;}
.step.up,.step.up b {color:#2563eb;}
.sect {font-size:1.27rem;font-weight:800;color:var(--navy);margin:28px 0 7px;letter-spacing:-.025em;}
.sect.blue {color:var(--navy);}
.sub {color:var(--muted);font-size:.91rem;line-height:1.5;margin:0 0 16px;}
[data-testid="stFileUploader"] section {background:#e9f3ff;border:1.5px dashed #3b82f6;border-radius:16px;padding:12px;}
[data-testid="stFileUploader"] section button {background:#fff;border:1px solid #cbdaf5;border-radius:9px;}
/* Uploaded CSV: readable filename and file details on a light blue surface */
[data-testid="stFileUploaderFile"] {background:#dbeafe!important;border:1px solid #60a5fa!important;border-radius:12px;padding:7px 10px;}
[data-testid="stFileUploaderFile"] span,[data-testid="stFileUploaderFile"] small,[data-testid="stFileUploaderFile"] p {color:#17365d!important;opacity:1!important;font-weight:650;}
[data-testid="stFileUploaderFile"] svg {color:#1d4ed8!important;}
[data-testid="stFileUploaderFile"] button {color:#1e40af!important;}
[data-testid="stFileUploader"] section {background:#eff6ff!important;}
[data-testid="stForm"] {background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px 24px 18px;box-shadow:0 5px 18px rgba(20,36,61,.035);}
[data-testid="stFormSubmitButton"] button {background:#2563eb;color:#fff;border:0;border-radius:10px;font-weight:800;padding:.65rem 1.5rem;min-height:44px;}
[data-testid="stFormSubmitButton"] button:hover {background:#1d4ed8;color:#fff;}
[data-baseweb="select"] > div {background:#eee9ff!important;border:1.5px solid #7c3aed!important;border-radius:10px!important;}
[data-baseweb="select"] [data-testid="stMarkdownContainer"], [data-baseweb="select"] span, [data-baseweb="select"] input {color:#281451!important;font-weight:700!important;}
[data-baseweb="select"] svg {fill:#5b21b6!important;color:#5b21b6!important;}
[data-baseweb="popover"] [role="option"] {color:#281451!important;background:#fff!important;}
[data-baseweb="popover"] [role="option"]:hover {background:#eee9ff!important;}
[data-testid="stExpander"] {background:#fff;border:1px solid var(--line);border-radius:12px;}
.chips {margin:8px 0 14px;}
.chip {display:inline-block;background:#eaf1ff;color:#254b84;font-size:.79rem;font-weight:750;padding:7px 13px;border-radius:9px;margin:0 7px 7px 0;}
.chip.sun {background:#dcfce7;color:#166534;}
.bal {width:100%;border-collapse:collapse;margin:10px 0 16px;table-layout:fixed;}
.bal td {padding:10px 12px!important;font-size:.81rem;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden;border:0!important;}
.tw {max-height:340px;overflow:auto;border:1px solid #93b9fa;border-radius:13px;background:#fff;}
.dt {border-collapse:separate;border-spacing:0;width:100%;font-size:.82rem;margin:0;}
.dt th {position:sticky;top:0;background:#315f99;color:#ffffff;font-weight:750;padding:11px 12px!important;text-align:left;white-space:nowrap;border:0!important;border-bottom:1px solid var(--line)!important;}
.dt th.tg {background:#315f99;color:#ffffff;}
.dt td {padding:9px 12px!important;white-space:nowrap;color:#27364a;background:#fff;border:0!important;border-bottom:1px solid #edf1f6!important;}
.dt td.num {text-align:right;}
.dt tr:nth-child(even) td {background:#eaf2ff;}
.dt td.tg {background:#e2edff!important;}
.bd {display:inline-block;padding:3px 10px;border-radius:8px;font-weight:750;font-size:.74rem;color:#fff;}
.bd.p {background:#2563eb;}.bd.n {background:#64748b;}
.note {color:var(--muted);font-size:.84rem;margin:10px 0 0;}
.verdict {background:#eaf2ff;border:1px solid #cfe0ff;color:#26476d;border-radius:14px;padding:17px 20px;margin:10px 0 20px;font-size:.95rem;line-height:1.65;}
.verdict b {color:#164bb3;}
.mt {width:100%;border-collapse:separate;border-spacing:12px;table-layout:fixed;margin:0 -12px;}
.mc {background:#fff;border:1px solid var(--line)!important;border-radius:15px;padding:20px 18px!important;box-shadow:0 5px 18px rgba(20,36,61,.045);vertical-align:top;}
.mc .nm {font-size:.78rem;letter-spacing:.025em;color:#61728a;font-weight:800;}
.mc .vl {font-size:clamp(1.4rem,2.6vw,2rem);font-weight:800;color:var(--navy);line-height:1.3;margin:7px 0 12px;}
.mc .bar {height:7px;border-radius:8px;background:#edf2f7;overflow:hidden;}
.mc .bar i {display:block;height:100%;border-radius:8px;}
.mc .ht {font-size:.79rem;color:#718096;margin-top:11px;line-height:1.4;}
.cmw {max-width:760px;margin:12px auto 0;padding:16px;background:#fff;border:1px solid #dbe4ef;border-radius:18px;}
.cm {width:100%;border-collapse:separate;border-spacing:7px;table-layout:fixed;}
.cm th {background:#eff4fb;color:#334967;font-size:.78rem;font-weight:750;padding:12px 6px!important;border:0!important;border-radius:9px;}
.cm th.e {background:transparent;}
.cm td {border:1px solid transparent!important;border-radius:12px;text-align:center;padding:19px 7px!important;color:#1e344e;}
.cm td.correct {background:#e0f5ed;border-color:#a8dfc9!important;}
.cm td.incorrect {background:#fff0ed;border-color:#fac7bc!important;}
.cm .n {font-size:clamp(1.5rem,3vw,2.15rem);font-weight:800;line-height:1.2;}
.cm .l {font-size:.84rem;font-weight:750;margin-top:6px;}
.cm .t {font-size:.68rem;letter-spacing:.045em;color:#53667d;margin-top:4px;}
.cm .p {font-size:.73rem;color:#53667d;margin-top:5px;}
.key {text-align:center;font-size:.82rem;color:#64748b;margin:12px 0 0;}
.key i {display:inline-block;width:12px;height:12px;border-radius:4px;margin:0 5px -1px 12px;}
.hint {background:#fff;border:1px solid #d8e4f6;border-radius:14px;padding:18px 20px;color:#4a6080;font-size:.94rem;}
@media(max-width:700px){.block-container{padding:1rem .85rem 3rem;}.hero{padding:25px 22px;border-radius:16px;}.hero .cart{display:none!important;}.hero td:last-child{display:none;}.mt{border-spacing:6px;margin:0 -6px;}.mc{padding:12px 8px!important;}.mc .nm{font-size:.66rem;}.mc .ht{font-size:.7rem;}.cm{border-spacing:4px;}.cm th{font-size:.65rem;}.cm .n{font-size:1.25rem;}.cmw{padding:7px;}}
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


def fmt_num(v) -> str:
    if pd.isna(v):
        return ""
    if float(v).is_integer():
        return f"{int(v):,}" if abs(v) >= 10000 else str(int(v))
    return f"{v:,.3f}".rstrip("0").rstrip(".")


def preview_table(df: pd.DataFrame, n: int) -> str:
    """Light, readable HTML table; the Revenue (target) column is highlighted with badges."""
    view = df.head(n)
    if "Revenue" in view.columns:          # show the target first so it is never hidden by scrolling
        view = view[["Revenue"] + [c for c in view.columns if c != "Revenue"]]
    head = "".join(
        f'<th class="{"tg" if c == "Revenue" else ""}">{_html.escape(str(c))}</th>' for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for c in view.columns:
            v = row[c]
            if c == "Revenue":
                key = str(v).strip().lower()
                if key in ("true", "1"):
                    cell = '<span class="bd p">Purchase</span>'
                elif key in ("false", "0"):
                    cell = '<span class="bd n">No purchase</span>'
                else:
                    cell = _html.escape(str(v))
                cells.append(f'<td class="tg">{cell}</td>')
            elif pd.api.types.is_number(v) and not pd.api.types.is_bool(v):
                cells.append(f'<td class="num">{fmt_num(v)}</td>')
            else:
                cells.append(f'<td>{_html.escape(str(v))}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return f'<div class="tw"><table class="dt"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def balance_bar(pos: int, total: int) -> str:
    pos_pct = pos / total * 100
    neg_pct = 100 - pos_pct
    pos_w = max(pos_pct, 12)          # keep the small segment readable
    neg_w = 100 - pos_w
    return (f'<table class="bal"><tr>'
            f'<td style="width:{neg_w:.1f}%;background:#64748b;border-radius:12px 0 0 12px;">No purchase &middot; {neg_pct:.1f}%</td>'
            f'<td style="width:{pos_w:.1f}%;background:#2563eb;border-radius:0 12px 12px 0;">Purchase &middot; {pos_pct:.1f}%</td>'
            f'</tr></table>')


def metric_cards(scores):
    meta = {
        "Accuracy": ("#2563eb", "🎯", "Overall share of correct predictions"),
        "AUC": ("#0e7490", "📈", "How well buyers rank above non-buyers"),
        "Precision": ("#059669", "🔍", "Predicted purchases that were real"),
        "Recall": ("#d97706", "🧲", "Real purchases the model caught"),
        "F1 Score": ("#7c3aed", "⚖️", "Balance between precision and recall"),
        "MCC": ("#475569", "🧩", "Quality across all four outcomes"),
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
    """Compact 2×2 matrix with consistent colors and clear axis labels."""
    total = tn + fp + fn + tp

    def cell(value, label, tag, correct):
        status = "correct" if correct else "incorrect"
        share = value / total if total else 0
        return (
            f'<td class="{status}"><div class="n">{value:,}</div>'
            f'<div class="l">{label}</div><div class="t">{tag}</div>'
            f'<div class="p">{share:.1%} of sessions</div></td>'
        )

    return (
        '<div class="cmw"><table class="cm">'
        '<colgroup><col style="width:23%"><col style="width:38.5%"><col style="width:38.5%"></colgroup>'
        '<tr><th class="e">Actual ↓ / Predicted →</th>'
        '<th>No purchase (0)</th><th>Purchase (1)</th></tr>'
        f'<tr><th>No purchase (0)</th>{cell(tn, "Correctly rejected", "TRUE NEGATIVE · TN", True)}'
        f'{cell(fp, "False alarm", "FALSE POSITIVE · FP", False)}</tr>'
        f'<tr><th>Purchase (1)</th>{cell(fn, "Missed purchase", "FALSE NEGATIVE · FN", False)}'
        f'{cell(tp, "Correctly detected", "TRUE POSITIVE · TP", True)}</tr>'
        '</table><div class="key"><i style="background:#a8dfc9;"></i>Correct prediction'
        '<i style="background:#fac7bc;"></i>Incorrect prediction</div></div>'
    )


# ------------------------------ Header ------------------------------
st.markdown(
    '<div class="hero"><table><tr><td>'
    '<div class="kick">ONLINE SHOPPERS PURCHASING INTENTION</div>'
    "<h1>Online Shopping Purchase Prediction</h1>"
    "<p>Upload labeled test data, evaluate a pretrained classifier, and explore how accurately it "
    "identifies purchasing sessions.</p>"
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
    st.markdown('<div class="sub">A quick look at your file. The bar shows how many sessions ended in a purchase.</div>',
                unsafe_allow_html=True)
    chips = (f'<span class="chip">{len(test_df):,} rows</span>'
             f'<span class="chip">{test_df.shape[1]} columns</span>'
             f'<span class="chip sun">{int(test_df.isna().sum().sum())} missing values</span>')
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    if "Revenue" in test_df.columns:
        flags = test_df["Revenue"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        if flags.notna().all():
            st.markdown(balance_bar(int(flags.sum()), len(flags)), unsafe_allow_html=True)

    st.caption("Preview: first 5 rows")
    st.markdown(preview_table(test_df, 5), unsafe_allow_html=True)
    if "Revenue" in test_df.columns:
        st.markdown(f'<div class="note">The highlighted <b>Revenue</b> column (shown first) is what the models predict: Purchase or No purchase. '
                    f'The other {test_df.shape[1] - 1} columns are the inputs they use.</div>', unsafe_allow_html=True)

# ------------------------------ Step 2: model + run ------------------------------
st.markdown('<div class="step"><b>2</b>Choose a model and run</div>', unsafe_allow_html=True)
with st.form("evaluation_form"):
    selected_model = st.selectbox("Classification model", list(MODEL_FILES))
    run_evaluation = st.form_submit_button("Run Evaluation", type="primary")

if not run_evaluation:
    st.markdown('<div class="hint">Upload <b>test_data.csv</b>, choose a model, then press '
                '<b>Run Evaluation</b> to see the scores and the confusion matrix.</div>',
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
st.markdown('<div class="sub">Six scores for the selected model on your uploaded test data. Most scores range from 0 to 1 '
            '(MCC from -1 to 1), and higher is better.</div>', unsafe_allow_html=True)
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
st.markdown('<div class="sub">Each row shows what really happened and each column shows what the model predicted, '
            'so the tiles count the four possible outcomes.</div>', unsafe_allow_html=True)
st.markdown(confusion_html(tn, fp, fn, tp), unsafe_allow_html=True)
st.caption(f"TN = {tn:,} · FP = {fp:,} · FN = {fn:,} · TP = {tp:,}")

with st.expander("Evaluation metric formulas"):
    st.latex(r"Accuracy = \frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"Precision = \frac{TP}{TP+FP}")
    st.latex(r"Recall = \frac{TP}{TP+FN}")
    st.latex(r"F1 = \frac{2\cdot Precision\cdot Recall}{Precision+Recall}")
    st.latex(r"MCC = \frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}")
    st.markdown("**AUC:** Area under the ROC curve across decision thresholds; calculated from predicted probabilities.")
