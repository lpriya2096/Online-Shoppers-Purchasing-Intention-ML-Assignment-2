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
[data-testid="stForm"] {background:#fff;border:1px solid var(--line);border-radius:16px;padding:22px 24px 18px;box-shadow:0 5px 18px rgba(20,36,61,.035);}
[data-testid="stFormSubmitButton"] button {background:#2563eb;color:#fff;border:0;border-radius:10px;font-weight:800;padding:.65rem 1.5rem;min-height:44px;}
[data-testid="stFormSubmitButton"] button:hover {background:#1d4ed8;color:#fff;}
[data-testid="stExpander"] {background:#fff;border:1px solid var(--line);border-radius:12px;}
.chips {margin:8px 0 14px;}
.chip {display:inline-block;background:#eaf1ff;color:#254b84;font-size:.79rem;font-weight:750;padding:7px 13px;border-radius:9px;margin:0 7px 7px 0;}
.chip.sun {background:#dcfce7;color:#166534;}
.bal {width:100%;border-collapse:collapse;margin:10px 0 16px;table-layout:fixed;}
.bal td {padding:10px 12px!important;font-size:.81rem;font-weight:700;color:#fff;white-space:nowrap;overflow:hidden;border:0!important;}
.tw {max-height:340px;overflow:auto;border:1px solid #93b9fa;border-radius:13px;background:#fff;}
.dt {border-collapse:separate;border-spacing:0;width:100%;font-size:.82rem;margin:0;}
.dt th {position:static;background:#315f99;color:#ffffff;font-weight:750;padding:11px 12px!important;text-align:left;white-space:nowrap;border:0!important;border-bottom:1px solid var(--line)!important;}
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
@media(max-width:700px){.block-container{padding:1rem .85rem 3rem;}.hero{padding:25px 22px;border-radius:16px;}.hero .cart{display:none!important;}.hero td:last-child{display:none;}.mt{border-spacing:6px;margin:0 -6px;}.mc{padding:12px 8px!important;}.mc .nm{font-size:.66rem;}.mc .ht{font-size:.7rem;}.cm{border-spacing:4px;}.cm th{font-size:.65rem;}.cm .n{font-size:1.25rem;}.cmw{padding:7px;}}


/* Recolour the real uploaded-file widget. No duplicate filename card. */
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] > div,
[data-testid="stFileUploaderFile"] > div > div,
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileData"],
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploaderFile"] [data-testid="stFileUploaderFileSize"],
[data-testid="stFileUploaderFile"] [data-baseweb="tag"] {
  background:#e3f1ff!important;background-color:#e3f1ff!important;
  color:#203957!important;box-shadow:none!important;
}
[data-testid="stFileUploaderFile"] {display:flex!important;border:1px solid #a9cbea!important;border-radius:11px!important;}
[data-testid="stFileUploaderFile"] :is(p,span,small,div,svg) {
  color:#203957!important;-webkit-text-fill-color:#203957!important;
}
[data-testid="stFileUploaderFile"] button {background:#d0e6fc!important;border-radius:8px!important;}
/* Closed model selector */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="combobox"] {
 background:#e4f2ff!important;color:#213f60!important;
 border:1px solid #b5d4ef!important;border-radius:12px!important;
}
[data-testid="stSelectbox"] [role="combobox"] *,
[data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stSelectbox"] svg {
 color:#213f60!important;-webkit-text-fill-color:#213f60!important;
}
/* OPEN dropdown: BaseWeb renders it in a portal outside the selectbox. */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [data-baseweb="menu"],
div[data-baseweb="popover"] [role="listbox"],
ul[role="listbox"],ul[data-baseweb="menu"] {
 background:#e8f3ff!important;background-color:#e8f3ff!important;
 color:#213f60!important;border-color:#b5d4ef!important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"],
ul[role="listbox"] li,
ul[role="listbox"] [role="option"] {
 background:#e8f3ff!important;color:#213f60!important;
 -webkit-text-fill-color:#213f60!important;
 font-family:'DM Sans',sans-serif!important;
}
div[data-baseweb="popover"] li *,
div[data-baseweb="popover"] [role="option"] *,
ul[role="listbox"] li * {
 color:#213f60!important;-webkit-text-fill-color:#213f60!important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] [aria-selected="true"],
ul[role="listbox"] li:hover {
 background:#cfe6ff!important;
}
/* One consistent heading style */
.step.up,.step.model-heading {
 font-family:'DM Sans',sans-serif;font-size:1.27rem;font-weight:800;
 letter-spacing:-.025em;color:#274d8b!important;margin:28px 0 13px;
}

/* Streamlit virtual dropdown is rendered OUTSIDE stSelectbox.
   Target the portal and menu layers, including Streamlit's own test IDs. */
[data-testid="stSelectboxVirtualDropdown"],
[data-testid="stSelectboxVirtualDropdown"] > div,
[data-testid="stSelectboxVirtualDropdown"] [data-baseweb="popover"],
[data-testid="stSelectboxVirtualDropdown"] [data-baseweb="menu"],
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
[data-testid="stVirtualDropdown"],
[data-testid="stVirtualDropdown"] > div,
[data-baseweb="popover"] [data-baseweb="menu"] > div,
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[data-baseweb="menu"] > div {
 background:#e8f3ff!important;
 background-color:#e8f3ff!important;
 color:#213f60!important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-testid="stSelectboxVirtualDropdown"] li,
[data-testid="stVirtualDropdown"] [role="option"],
[data-testid="stVirtualDropdown"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="menu"] li {
 background:#e8f3ff!important;
 color:#213f60!important;
 -webkit-text-fill-color:#213f60!important;
}
[data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
[data-testid="stSelectboxVirtualDropdown"] li *,
[data-testid="stVirtualDropdown"] [role="option"] *,
[data-baseweb="menu"] li * {
 color:#213f60!important;
 -webkit-text-fill-color:#213f60!important;
}
[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
[data-testid="stVirtualDropdown"] [aria-selected="true"],
[data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="menu"] li:hover {
 background:#cfe6ff!important;
}

/* Uploaded CSV: recolor Streamlit's nested file-pill surfaces. */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] > div > div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [data-baseweb="tag"] {
 background:#e4f2ff!important;background-color:#e4f2ff!important;
 color:#203957!important;box-shadow:none!important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
 border:1px solid #abccec!important;border-radius:12px!important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
 color:#203957!important;-webkit-text-fill-color:#203957!important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
 background:#cce5fc!important;border-radius:9px!important;
}
/* Unified typography for every section heading. */
.section-heading {font-family:'DM Sans',sans-serif!important;font-size:1.27rem!important;
 font-weight:800!important;letter-spacing:-.025em!important;line-height:1.35!important;
 color:#274d8b!important;margin:28px 0 13px!important;}
/* Confusion matrix: CSS grid gives real, consistent gaps between every cell. */
.cmw {max-width:620px;margin:18px auto 0;padding:22px;background:#fff;
 border:1px solid #dbe6f2;border-radius:18px;box-shadow:0 7px 22px rgba(23,53,89,.06);}
.matrix-axis {text-align:center;font-size:.86rem;font-weight:750;color:#496682;margin-bottom:15px;}
.matrix-grid {display:grid;grid-template-columns: minmax(105px,1fr) minmax(0,1.3fr) minmax(0,1.3fr);
 gap:12px;align-items:stretch;}
.matrix-label {display:flex;align-items:center;justify-content:center;text-align:center;
 background:#edf3fb;color:#294967;font-size:.82rem;font-weight:750;
 border-radius:11px;padding:12px 8px;min-width:0;}
.matrix-label.row-label {justify-content:flex-start;text-align:left;padding-left:14px;}
.matrix-cell {min-width:0;min-height:125px;display:flex;flex-direction:column;
 justify-content:center;align-items:center;text-align:center;border-radius:13px;
 padding:18px 8px;border:1px solid transparent;}
.matrix-cell.tn {background:#dff4ee;border-color:#b6e2d3;}
.matrix-cell.tp {background:#e0ecff;border-color:#b6cdf4;}
.matrix-cell.fp {background:#fff0dc;border-color:#f0d4aa;}
.matrix-cell.fn {background:#fae8ee;border-color:#eec6d4;}
.matrix-cell .n {font-size:1.9rem;font-weight:800;line-height:1.2;color:#193955;}
.matrix-cell .l {font-size:.79rem;font-weight:700;color:#294967;margin-top:9px;}
.matrix-foot {text-align:center;color:#64748b;font-size:.75rem;margin-top:15px;}
/* Dataset preview: balanced typography and distinct column headings. */
.tw {border:1px solid #c6d9f1;border-radius:13px;overflow:auto;max-height:520px;}
.dt {font-family:'DM Sans',sans-serif!important;font-size:.86rem!important;}
.dt th {background:#315f99!important;color:#fff!important;font-size:.78rem!important;
 letter-spacing:.01em;font-weight:750!important;padding:12px 14px!important;}
.dt td {font-size:.83rem!important;padding:11px 14px!important;color:#233c58!important;}
.dt tr:nth-child(even) td {background:#eef5ff!important;}
.dt td.tg {background:#e4efff!important;}
.dt tr:hover td {background:#e0edff!important;}
.preview-caption {font-family:'DM Sans',sans-serif;color:#4c6686;
 font-size:.85rem;font-weight:650;margin:15px 0 10px;}
@media(max-width:600px) {
 .cmw{padding:12px}.matrix-grid{gap:6px;grid-template-columns:minmax(75px,1fr) repeat(2,minmax(0,1.3fr))}
 .matrix-label{font-size:.68rem;padding:8px 4px}
 .matrix-label.row-label{padding-left:7px}
 .matrix-cell{min-height:100px;padding:10px 4px}
 .matrix-cell .n{font-size:1.35rem}.matrix-cell .l{font-size:.65rem}
}

/* Preview row selection: mild blue pills, independent of the model dropdown. */
[data-testid="stRadio"] [role="radiogroup"] {gap:9px;flex-wrap:wrap;}
[data-testid="stRadio"] label {font-family:'DM Sans',sans-serif;color:#234363;
 background:#eaf3ff;border:1px solid #c9dbf2;border-radius:10px;padding:7px 11px;}
[data-testid="stRadio"] label:has(input:checked) {background:#d8eaff;border-color:#78a9e5;}
/* Keep table headers above, not overlapping, the body rows. */
.tw .dt th {position:static!important;top:auto!important;z-index:auto!important;}
.tw .dt {border-collapse:separate!important;border-spacing:0!important;}
.tw .dt tbody tr {height:42px;}

/* Uploaded file chip: override Streamlit's dark theme at every nested layer. */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] :is(div,span,p,small),
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [data-baseweb="tag"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [class*="file"],
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] [class*="File"] {
 background-color:#e3f1ff!important;background-image:none!important;
 color:#203957!important;-webkit-text-fill-color:#203957!important;
 box-shadow:none!important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
 border:1px solid #aacbed!important;border-radius:12px!important;
}
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] :is(svg,path) {
 color:#203957!important;fill:currentColor!important;
}
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
    """Medium-sized confusion matrix with clearly separated grid cells."""
    def cell(value, title, kind):
        return (
            f'<div class="matrix-cell {kind}"><div class="n">{value:,}</div>'
            f'<div class="l">{title}</div></div>'
        )

    return (
        '<div class="cmw"><div class="matrix-axis">Predicted outcome →</div>'
        '<div class="matrix-grid">'
        '<div class="matrix-label">Actual outcome ↓</div>'
        '<div class="matrix-label">No purchase</div>'
        '<div class="matrix-label">Purchase</div>'
        '<div class="matrix-label row-label">No purchase</div>'
        + cell(tn, "True negative", "tn")
        + cell(fp, "False positive", "fp")
        + '<div class="matrix-label row-label">Purchase</div>'
        + cell(fn, "False negative", "fn")
        + cell(tp, "True positive", "tp")
        + '</div><div class="matrix-foot">Green / blue = correct predictions'
        ' · Amber / rose = incorrect predictions</div></div>'
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
st.markdown('<div class="section-heading">Upload your test data</div>', unsafe_allow_html=True)
st.caption("Upload a labeled CSV containing the Revenue column to evaluate the saved models.")
uploaded_file = st.file_uploader(
    "Upload CSV", type=["csv"], label_visibility="collapsed"
)

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

    st.markdown('<div class="section-heading">Dataset Preview</div>', unsafe_allow_html=True)
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

    # A horizontal radio avoids the model dropdown CSS and makes the selected count explicit.
    preview_count = st.radio(
        "Rows to preview",
        options=[5, 10, 20, 50, 100],
        index=0,
        horizontal=True,
        key="dataset_preview_row_count",
    )
    preview_rows = test_df.head(preview_count)
    st.markdown(
        f'<div class="preview-caption">Showing {len(preview_rows):,} of '
        f'{len(test_df):,} rows</div>',
        unsafe_allow_html=True,
    )
    st.markdown(preview_table(preview_rows, len(preview_rows)), unsafe_allow_html=True)
    if "Revenue" in test_df.columns:
        st.markdown(f'<div class="note">The highlighted <b>Revenue</b> column (shown first) is what the models predict: Purchase or No purchase. '
                    f'The other {test_df.shape[1] - 1} columns are the inputs they use.</div>', unsafe_allow_html=True)

# ------------------------------ Step 2: model + run ------------------------------
st.markdown('<div class="section-heading">Choose a model and run</div>', unsafe_allow_html=True)
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
st.markdown('<div class="section-heading">Model Evaluation</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
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
