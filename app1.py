import html as _html
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report, roc_curve
)

st.set_page_config(page_title="ShopSense | Purchase Intelligence", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root{--ink:#162a43;--blue:#2854a1;--sky:#dceafb;--line:#dce5ef;--muted:#64758c;--paper:#f6f8fc}
html,body,.stApp,[data-testid="stAppViewContainer"]{font-family:'DM Sans',sans-serif!important;background:var(--paper)!important;color:var(--ink);color-scheme:light}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1170px;padding-top:1.4rem;padding-bottom:4rem}
footer{visibility:hidden}
h1,h2,h3,p{font-family:'DM Sans',sans-serif}
[data-testid="stWidgetLabel"] p{color:#29476a!important;font-weight:700}
[data-testid="stCaptionContainer"]{color:#77879d!important}
/* ShopSense shopping analytics design */
[data-testid="stSidebar"]{background:#101f35!important;border-right:1px solid #29425f}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{background:#101f35!important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] a,[data-testid="stSidebar"] span{color:#dce9f8}
[data-testid="stSidebar"] a{text-decoration:none!important}
[data-testid="stSidebar"] a:hover{color:#8fd4ff!important}
.sidebrand{padding:12px 4px 20px;border-bottom:1px solid #29425f;margin-bottom:18px}
.sidebrand .mark{display:inline-flex;align-items:center;justify-content:center;width:43px;height:43px;background:#1b4774;border:1px solid #386d9d;border-radius:13px;font-size:24px;margin-bottom:12px}
.sidebrand strong{display:block;color:#fff;font-size:1.2rem;font-weight:800;letter-spacing:-.035em}
.sidebrand small{display:block;color:#a9c4e3;font-size:.76rem;margin-top:5px}
.sidelabel{font-size:.7rem;font-weight:800;letter-spacing:.15em;color:#8daed0;margin:17px 0 10px}
.sidelink{display:block;padding:10px 13px;margin:4px 0;background:#172c47;border:1px solid #294563;border-radius:10px;color:#e5f1ff!important;font-size:.85rem;font-weight:650}
.sidefoot{margin-top:25px;padding:15px;background:#19324d;border:1px solid #2d506e;border-radius:12px;color:#c9e2ff;font-size:.8rem;line-height:1.65}
.hero{position:relative;overflow:hidden;background:linear-gradient(112deg,#10233c 0%,#183f68 58%,#176d98 100%);border-radius:20px;padding:34px 38px;margin:3px 0 20px;box-shadow:0 12px 28px rgba(18,39,67,.12);border-left:5px solid #5ec3db}
.hero:after{content:'';position:absolute;width:270px;height:270px;border:45px solid rgba(255,255,255,.045);border-radius:50%;right:-125px;top:-150px;pointer-events:none}
.hero table,.hero tr,.hero td{border:0!important;background:transparent!important}
.hero table{width:100%;border-collapse:collapse}.hero td{vertical-align:middle;padding:0!important}
.hero .kick{color:#9cddf6;font-size:.74rem;font-weight:800;letter-spacing:.15em}
.hero h1{color:#fff!important;font-size:clamp(1.8rem,3vw,2.55rem);font-weight:800;letter-spacing:-.035em;line-height:1.2;margin:10px 0 12px}
.hero p{color:#d7e9f7;font-size:.96rem;line-height:1.6;max-width:670px;margin:0}
.hero .tag{display:inline-block;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.18);color:#f3faff;font-size:.72rem;font-weight:700;padding:6px 12px;border-radius:100px;margin:18px 7px 0 0}
.hero .cart{width:85px;height:85px;line-height:85px;text-align:center;font-size:2.5rem;border-radius:20px;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2)}
.overview{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px;margin:18px 0 28px}
.overview .item{background:#fff;border:1px solid var(--line);border-top:3px solid #48a5c8;border-radius:13px;padding:17px 20px;box-shadow:0 4px 13px rgba(30,59,91,.035)}
.overview .label{font-size:.7rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#6d829a}
.overview .value{font-size:1.65rem;font-weight:800;color:#1e4267;margin-top:6px}
.overview .detail{font-size:.75rem;color:#73879d;margin-top:4px}
.section-heading{font-family:'DM Sans',sans-serif!important;font-size:1.26rem!important;font-weight:800!important;letter-spacing:-.025em!important;color:#244b79!important;margin:30px 0 9px!important}
.sub{color:var(--muted);font-size:.9rem;line-height:1.55;margin:0 0 15px}
.workflow{background:#edf6fc;border:1px solid #cce3f1;border-radius:12px;padding:12px 16px;color:#355d7c;font-size:.83rem;margin:6px 0 18px}
.workflow b{color:#1d547b}
/* Keep Streamlit's working uploader, but do not show the unstyleable native black chip. */
[data-testid="stFileUploader"] section{background:#f1f7ff!important;border:1px solid #bad1e8!important;border-radius:13px!important;box-shadow:none!important}
[data-testid="stFileUploader"] section button{background:#fff!important;border:1px solid #b8cde4!important;color:#254a73!important;border-radius:9px!important}
[data-testid="stFileUploaderFile"],[data-testid="stFileUploaderFileData"]{display:none!important}
.chips{margin:9px 0 14px}.chip{display:inline-block;background:#eaf3fc;border:1px solid #cfe0ef;color:#315c84;font-size:.77rem;font-weight:750;padding:8px 13px;border-radius:9px;margin:0 7px 7px 0}
.bal{width:100%;border-collapse:collapse;margin:11px 0 15px;table-layout:fixed}.bal td{padding:11px 12px!important;font-size:.79rem;font-weight:750;color:#fff;white-space:nowrap;overflow:hidden;border:0!important}
.preview-inline-label{font-family:'DM Sans',sans-serif;color:#285487;font-size:.91rem;font-weight:750;white-space:nowrap}
[data-testid="stNumberInput"] [data-baseweb="input"],[data-testid="stNumberInput"] [data-baseweb="base-input"]{background:#dceafb!important;border:1px solid #b9d2ee!important;border-radius:9px!important;box-shadow:none!important;outline:none!important}
[data-testid="stNumberInput"] input{background:#dceafb!important;color:#285487!important;-webkit-text-fill-color:#285487!important;font-family:'DM Sans',sans-serif!important;font-weight:700!important}
[data-testid="stNumberInput"] button{background:#dceafb!important;color:#285487!important;border-color:#b9d2ee!important;box-shadow:none!important}
[data-testid="stNumberInput"] button:hover{background:#c5def9!important}
[data-testid="stNumberInput"] button svg{color:#285487!important}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within{border-color:#78a8d9!important;box-shadow:0 0 0 1px #78a8d9!important}
.preview-caption{color:#4c6686;font-size:.84rem;font-weight:700;margin:15px 0 10px}
/* Raw dataset: plain text only. Never relabel True/False or reorder columns. */
.tw{width:100%;max-height:440px;overflow:auto;background:#fff;border:1px solid #d5e0ec;border-radius:12px;box-shadow:0 4px 14px rgba(30,62,95,.05)}
.tw table.dt{width:max-content;min-width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:12.5px}
.tw table.dt thead th{background:#f5f8fc!important;color:#285487!important;font-size:12px!important;font-weight:700!important;padding:12px 15px!important;text-align:left;white-space:nowrap;border:0!important;border-bottom:1px solid #dce5ee!important}
.tw table.dt tbody td{background:#fff!important;color:#334155!important;font-family:'DM Sans',sans-serif!important;font-size:12.5px!important;font-weight:400!important;padding:11px 15px!important;white-space:nowrap;border:0!important;border-bottom:1px solid #edf1f5!important}
.note{color:#6b7e95;font-size:.82rem;margin:11px 0 0}
[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px 23px 18px;box-shadow:0 5px 16px rgba(20,36,61,.04)}
[data-testid="stSelectbox"] [data-baseweb="select"],[data-testid="stSelectbox"] [data-baseweb="select"]>div,[data-testid="stSelectbox"] [role="combobox"],[data-testid="stSelectbox"] [data-baseweb="select"]>div>div{background:#e7f2ff!important;color:#25496c!important;border-color:#bdd4eb!important}
[data-testid="stSelectbox"] [data-baseweb="select"] svg{color:#25496c!important}
[data-testid="stSelectbox"] [role="combobox"]{border:1px solid #bdd4eb!important;border-radius:10px!important;box-shadow:none!important}
[data-baseweb="popover"],[data-baseweb="menu"],[data-baseweb="popover"] [role="listbox"],[data-testid="stSelectboxVirtualDropdown"],[data-baseweb="menu"] li,[data-baseweb="menu"] [role="option"]{background:#eaf4ff!important;color:#25496c!important}
[data-baseweb="menu"] li:hover,[data-baseweb="menu"] [aria-selected="true"]{background:#cfe6ff!important}
[data-testid="stFormSubmitButton"] button{background:#2563eb!important;color:#fff!important;border:0!important;border-radius:10px!important;font-weight:800!important;padding:.65rem 1.5rem!important;min-height:44px}
[data-testid="stFormSubmitButton"] button:hover{background:#1d4ed8!important}
.verdict{background:#eaf2ff;border:1px solid #cfe0ff;color:#26476d;border-radius:14px;padding:17px 20px;margin:10px 0 20px;font-size:.94rem;line-height:1.65}.verdict b{color:#164bb3}
.mt{width:100%;border-collapse:separate;border-spacing:12px;table-layout:fixed;margin:0 -12px}
.mc{background:#fff;border:1px solid var(--line)!important;border-radius:15px;padding:19px 17px!important;box-shadow:0 5px 18px rgba(20,36,61,.045);vertical-align:top}
.mc .nm{font-size:.78rem;color:#61728a;font-weight:800}.mc .vl{font-size:clamp(1.4rem,2.6vw,2rem);font-weight:800;color:var(--ink);line-height:1.3;margin:7px 0 12px}.mc .bar{height:7px;border-radius:8px;background:#edf2f7;overflow:hidden}.mc .bar i{display:block;height:100%;border-radius:8px}.mc .ht{font-size:.77rem;color:#718096;margin-top:11px;line-height:1.4}
.cmw{max-width:620px;margin:18px auto 0;padding:22px;background:#fff;border:1px solid #dbe6f2;border-radius:18px;box-shadow:0 7px 22px rgba(23,53,89,.06)}
.matrix-axis{text-align:center;font-size:.86rem;font-weight:750;color:#496682;margin-bottom:15px}.matrix-grid{display:grid;grid-template-columns:minmax(105px,1fr) minmax(0,1.3fr) minmax(0,1.3fr);gap:12px;align-items:stretch}.matrix-label{display:flex;align-items:center;justify-content:center;text-align:center;background:#edf3fb;color:#294967;font-size:.82rem;font-weight:750;border-radius:11px;padding:12px 8px;min-width:0}.matrix-label.row-label{justify-content:flex-start;text-align:left;padding-left:14px}.matrix-cell{min-width:0;min-height:125px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border-radius:13px;padding:18px 8px;border:1px solid transparent}.matrix-cell.tn{background:#dff4ee;border-color:#b6e2d3}.matrix-cell.tp{background:#e0ecff;border-color:#b6cdf4}.matrix-cell.fp{background:#fff0dc;border-color:#f0d4aa}.matrix-cell.fn{background:#fae8ee;border-color:#eec6d4}.matrix-cell .n{font-size:1.9rem;font-weight:800;line-height:1.2;color:#193955}.matrix-cell .l{font-size:.79rem;font-weight:700;color:#294967;margin-top:9px}.matrix-foot{text-align:center;color:#64748b;font-size:.75rem;margin-top:15px}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--line);border-radius:12px}[data-testid="stExpander"] details>summary{background:#e9f2ff!important;color:#28518a!important}[data-testid="stExpander"] details>summary *{color:#28518a!important}[data-testid="stExpander"] .katex,[data-testid="stExpander"] .katex *{color:#28518a!important}[data-testid="stExpander"] .katex-display{margin:.5em 0!important}
@media(max-width:700px){.block-container{padding:1rem .8rem 3rem}.hero{padding:25px 20px}.hero .cart{display:none!important}.hero td:last-child{display:none}.overview{grid-template-columns:1fr}.mt{border-spacing:5px;margin:0 -5px}.mc{padding:12px 8px!important}.mc .nm{font-size:.67rem}.mc .ht{font-size:.7rem}.cmw{padding:12px}.matrix-grid{gap:6px;grid-template-columns:minmax(75px,1fr) repeat(2,minmax(0,1.3fr))}.matrix-cell{min-height:100px;padding:10px 4px}.matrix-cell .n{font-size:1.35rem}}


/* Model comparison: restrained typography and an easy-to-scan table. */
.compare-scroll{overflow-x:auto;border:1px solid #dce5ef;border-radius:12px;background:#fff;margin:12px 0 8px}
.compare-table{width:100%;min-width:760px;border-collapse:collapse;font-family:'DM Sans',sans-serif;font-size:13px;color:#334155;font-variant-numeric:tabular-nums}
.compare-table th{background:#f2f6fb;color:#285487;font-weight:700;text-align:right;padding:14px 12px;border-bottom:1px solid #dce5ef;white-space:nowrap}
.compare-table th:first-child,.compare-table td:first-child{text-align:left;padding-left:18px;min-width:175px}
.compare-table td{background:#fff;text-align:right;padding:13px 12px;border-bottom:1px solid #edf1f5;white-space:nowrap;font-weight:500}
.compare-table tbody tr:last-child td{border-bottom:0}
.compare-table tbody tr:hover td{background:#f8fbff}
.compare-table td.model-name{color:#253d5a;font-weight:650}
.compare-table td.best-score{color:#205f91;font-weight:800;background:#eef6fd}
.comparison-chart-title{font-size:1.12rem!important;margin-top:26px!important}

/* Deliberate, restrained table and chart styling */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:8px;flex-wrap:wrap}
[data-testid="stTabs"] [data-baseweb="tab"]{background:#edf4fb!important;border-radius:10px 10px 0 0;color:#345875!important;font-weight:700!important}
[data-testid="stTabs"] [aria-selected="true"]{background:#dbeeff!important;color:#173e6a!important}
[data-testid="stDataFrame"]{border:1px solid #d8e4f0;border-radius:12px;overflow:hidden}
[data-testid="stDownloadButton"] button,[data-testid="stButton"] button{border-radius:10px!important;font-weight:700!important}
[data-testid="stButton"] button[kind="primary"]{background:#2563eb!important;color:white!important}
.insight-panel{background:#fff;border:1px solid #d8e5f0;border-left:4px solid #19a3aa;border-radius:12px;padding:17px 20px;margin:14px 0 24px;color:#34516e;line-height:1.6}
.insight-panel strong{color:#1e486c}
.data-note{background:#eef6fc;border:1px solid #d3e6f4;border-radius:12px;padding:14px 18px;margin:12px 0;color:#355b7a;font-size:.88rem}
/* Hide default uploader chip only when our own accessible file summary is shown. */
[data-testid="stFileUploaderFile"]{display:none!important}

/* Tabs: make labels and active indicator visible in light and dark browser themes. */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  display:flex!important; flex-wrap:wrap!important; gap:8px!important;
  background:#ffffff!important; border-bottom:1px solid #d4e2f0!important;
  padding:7px 4px!important; min-height:52px!important;
}
[data-testid="stTabs"] button[data-baseweb="tab"],
[data-testid="stTabs"] [role="tab"] {
  display:inline-flex!important; visibility:visible!important; opacity:1!important;
  min-height:39px!important; padding:9px 14px!important;
  background:#eaf2fc!important; border:1px solid #c6d9ed!important;
  border-radius:9px!important; color:#24496e!important;
  font-family:'DM Sans',sans-serif!important; font-size:.88rem!important;
  font-weight:700!important; -webkit-text-fill-color:#24496e!important;
}
[data-testid="stTabs"] button[data-baseweb="tab"] *,
[data-testid="stTabs"] [role="tab"] * {
  color:#24496e!important; -webkit-text-fill-color:#24496e!important;
  opacity:1!important; visibility:visible!important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  background:#d4e9ff!important; border-color:#6b9bc9!important;
  box-shadow:inset 0 -3px 0 #2865a0!important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:#2865a0!important;}
[data-testid="stTabs"] [data-baseweb="tab-border"] {background:#d4e2f0!important;}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {background:#fff!important;color:#203b59!important;padding-top:15px!important;}


/* Dataset & methodology tables: light surfaces and consistent dashboard typography. */
.schema-table-wrap{background:#fff;border:1px solid #d5e2ef;border-radius:12px;overflow:hidden;margin:12px 0 18px;box-shadow:0 3px 12px rgba(24,57,89,.035)}
.schema-scroll{max-height:460px;overflow:auto}
.schema-table{width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif!important;font-size:13px;color:#334155;background:#fff}
.schema-table th{background:#edf4fc!important;color:#285487!important;text-align:left;font-size:13px;font-weight:750;padding:13px 16px;border-bottom:1px solid #d5e2ef;white-space:nowrap}
.schema-table td{background:#fff!important;color:#334155!important;padding:12px 16px;border-bottom:1px solid #e8eef5;line-height:1.4;overflow-wrap:anywhere}
.schema-table tr:last-child td{border-bottom:0}
.schema-table tbody tr:nth-child(even) td{background:#fafcff!important}
.schema-table td:last-child{text-align:left}
.schema-scroll .schema-table th{position:sticky;top:0;z-index:1}
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
    """Display CSV records as uploaded, without translating Revenue values."""
    view = df.head(n)
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for value in row:
            shown = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{_html.escape(shown)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
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


# Navigation is a simple set of section links; model evaluation stays on one page.
with st.sidebar:
    st.markdown('''<div class="sidebrand"><div class="mark">🛍️</div>
        <strong>ShopSense AI</strong><small>Purchase Intent Analytics</small></div>
        <div class="sidelabel">WORKSPACE</div>
        <a class="sidelink" href="#overview">⌂ &nbsp; Dashboard overview</a>
        <a class="sidelink" href="#dataset">▤ &nbsp; Dataset preview</a>
        <a class="sidelink" href="#evaluate">◈ &nbsp; Evaluate a model</a>
        <a class="sidelink" href="#results">▥ &nbsp; Model results</a>
        <a class="sidelink" href="#comparison">◫ &nbsp; Compare models</a>
        <a class="sidelink" href="#methodology">◇ &nbsp; Data &amp; methodology</a>
        <div class="sidefoot">✦ <b>Inference only</b><br>Five saved classifiers · Six metrics<br>No model retraining</div>''', unsafe_allow_html=True)

# ------------------------------ Header ------------------------------
st.markdown(
    '<div id="overview"></div><div class="hero"><table><tr><td>'
    '<div class="kick">BITS PILANI · AIML · ASSIGNMENT 2</div>'
    "<h1>Online Shopping Purchase Prediction</h1>"
    "<p>Upload labeled test data, evaluate a pretrained classifier, and explore how accurately it "
    "identifies purchasing sessions.</p>"
    '<span class="tag">5 trained models</span><span class="tag">6 evaluation metrics</span>'
    '<span class="tag">No retraining</span>'
    '</td><td style="width:120px;text-align:right;"><div class="cart" style="display:inline-block;">🛒</div></td>'
    '</tr></table></div>',
    unsafe_allow_html=True)

st.markdown(
    '<div class="overview">'
    '<div class="item"><div class="label">Saved classifiers</div><div class="value">05</div><div class="detail">Ready for evaluation</div></div>'
    '<div class="item"><div class="label">Evaluation metrics</div><div class="value">06</div><div class="detail">Accuracy · AUC · Precision · Recall · F1 · MCC</div></div>'
    '<div class="item"><div class="label">Training required</div><div class="value">None</div><div class="detail">Use existing model artifacts</div></div>'
    '</div>', unsafe_allow_html=True)
st.markdown('<div class="workflow"><b>Workflow</b> &nbsp; 01 Upload labeled CSV &nbsp; → &nbsp; 02 Inspect original records &nbsp; → &nbsp; 03 Select classifier &nbsp; → &nbsp; 04 Evaluate</div>', unsafe_allow_html=True)

# ------------------------------ Step 1: upload + preview ------------------------------
st.markdown('<div id="dataset"></div><div class="section-heading">Upload your test data</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="sub">View the original CSV records, column values, and class distribution.</div>',
                unsafe_allow_html=True)
    chips = (f'<span class="chip">{len(test_df):,} rows</span>'
             f'<span class="chip">{test_df.shape[1]} columns</span>'
             f'<span class="chip sun">{int(test_df.isna().sum().sum())} missing values</span>')
    st.markdown(f'<div class="chips dataset-records">{chips}</div>', unsafe_allow_html=True)

    if "Revenue" in test_df.columns:
        flags = test_df["Revenue"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
        if flags.notna().all():
            st.markdown(balance_bar(int(flags.sum()), len(flags)), unsafe_allow_html=True)

    # Keep the original row-count input, aligned beside its label.
    label_col, count_col, _ = st.columns([1.45, 1.25, 4.3], vertical_alignment="center")
    with label_col:
        st.markdown('<div class="preview-inline-label">Rows to preview</div>', unsafe_allow_html=True)
    with count_col:
        preview_count = st.number_input(
            "Rows to preview", min_value=1, max_value=100, value=5,
            step=1, label_visibility="collapsed", key="dataset_preview_count_input"
        )
    preview_rows = test_df.head(preview_count)
    st.markdown(
        f'<div class="preview-caption">Showing {len(preview_rows):,} of '
        f'{len(test_df):,} rows</div>',
        unsafe_allow_html=True,
    )
    st.markdown(preview_table(preview_rows, len(preview_rows)), unsafe_allow_html=True)
    if "Revenue" in test_df.columns:
        st.markdown(f'<div class="note">The <b>Revenue</b> column is the target used for model evaluation. '
                    f'The other {test_df.shape[1] - 1} columns are the inputs they use.</div>', unsafe_allow_html=True)

# ------------------------------ Step 2: evaluate saved model ------------------------------
st.markdown('<div id="evaluate"></div><div class="section-heading">Evaluate purchase intent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Select one of the five pretrained classifiers. Evaluation uses the uploaded CSV without fitting or retraining.</div>', unsafe_allow_html=True)
with st.form("evaluation_form"):
    selected_model = st.selectbox("Active Classification model", list(MODEL_FILES))
    run_evaluation = st.form_submit_button("Run Evaluation", type="primary")


def prepare_data(df):
    if "Revenue" not in df.columns:
        raise ValueError("CSV must contain a Revenue target column.")
    target = df["Revenue"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    if target.isna().any():
        raise ValueError("Revenue must contain only True/False or 1/0, without missing values.")
    if target.nunique() != 2:
        raise ValueError("Both Revenue classes must be present to calculate AUC.")
    return df.drop(columns=["Revenue"]), target.astype(int)


def evaluate_saved_model(name, features, actual):
    path = BASE_DIR / MODEL_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved model: {MODEL_FILES[name]}")
    model = load_model(str(path))
    expected = getattr(model, "feature_names_in_", None)
    if expected is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            expected = getattr(step, "feature_names_in_", None)
            if expected is not None:
                break
    aligned = features
    if expected is not None:
        missing = [col for col in expected if col not in features.columns]
        if missing:
            raise ValueError(f"Missing columns for {name}: {', '.join(missing)}")
        aligned = features[list(expected)]
    predicted = model.predict(aligned)
    if not hasattr(model, "predict_proba"):
        raise ValueError(f"{name} has no predict_proba method; AUC cannot be calculated.")
    classes = list(model.classes_)
    positive_index = next((i for i, label in enumerate(classes) if str(label).lower() in ("1", "true")), None)
    if positive_index is None:
        raise ValueError(f"{name} has no positive class 1.")
    probabilities = model.predict_proba(aligned)[:, positive_index]
    scores = {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, probabilities),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1 Score": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(actual, predicted, labels=[0, 1]).ravel())
    return {"scores": scores, "predicted": predicted, "probabilities": probabilities,
            "counts": (tn, fp, fn, tp), "features": aligned.shape[1]}


# Store the evaluation request, not stale predictions; recompute against the current CSV.
if run_evaluation:
    st.session_state["shop_evaluated_model"] = selected_model
    st.session_state["shop_evaluated_file"] = (uploaded_file.name, uploaded_file.size) if uploaded_file else None

active_name = st.session_state.get("shop_evaluated_model")
active_file = st.session_state.get("shop_evaluated_file")
current_file = (uploaded_file.name, uploaded_file.size) if uploaded_file else None
results = None
X = y = None
if test_df is not None:
    try:
        X, y = prepare_data(test_df)
    except ValueError as exc:
        st.info(str(exc))

if active_name and active_file == current_file and X is not None:
    try:
        with st.spinner("Evaluating saved model..."):
            results = evaluate_saved_model(active_name, X, y)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
elif not active_name:
    st.info("Upload a labelled CSV, choose a classifier, and select Run Evaluation to view its results.")

if test_df is not None and len(test_df) != 2466:
    st.warning(f"Uploaded {len(test_df):,} rows. For the assignment, use the held-out test_data.csv (2,466 rows).")

if results is not None:
    st.markdown('<div id="results"></div><div class="section-heading">Model evaluation results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chips"><span class="chip">{_html.escape(active_name)}</span>'
                f'<span class="chip">{len(test_df):,} test rows</span>'
                f'<span class="chip">{results["features"]} input features</span></div>', unsafe_allow_html=True)
    scores = results["scores"]
    st.markdown(metric_cards(list(scores.items())), unsafe_allow_html=True)
    tab_matrix, tab_report, tab_roc, tab_predictions = st.tabs(
        ["Confusion matrix", "Classification report", "ROC curve", "Test predictions"]
    )
    tn, fp, fn, tp = results["counts"]
    with tab_matrix:
        st.markdown('<div class="sub">Rows are actual outcomes; columns are model predictions.</div>', unsafe_allow_html=True)
        st.markdown(confusion_html(tn, fp, fn, tp), unsafe_allow_html=True)
        st.caption(f"TN = {tn:,} · FP = {fp:,} · FN = {fn:,} · TP = {tp:,}")
    with tab_report:
        report = classification_report(y, results["predicted"], labels=[0, 1],
                                       target_names=["No purchase", "Purchase"],
                                       output_dict=True, zero_division=0)
        rows = []
        for label in ["No purchase", "Purchase", "macro avg", "weighted avg"]:
            d = report[label]
            rows.append({"Class": label, "Precision": d["precision"], "Recall": d["recall"],
                         "F1": d["f1-score"], "Support": int(d["support"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Precision, recall and F1 are calculated separately for each actual class.")
    with tab_roc:
        fpr, tpr, _ = roc_curve(y, results["probabilities"])
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.plot(fpr, tpr, color="#176b91", lw=2.6, label=f"{active_name} · AUC {scores['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], color="#9ca9b7", ls="--", lw=1.3, label="Random baseline")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        ax.grid(alpha=.17)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    with tab_predictions:
        prediction_table = test_df.copy()
        prediction_table["Predicted_Revenue"] = results["predicted"].astype(bool)
        prediction_table["Purchase_Probability"] = np.round(results["probabilities"], 4)
        st.caption("Original CSV values are preserved. The two prediction columns are appended at the end.")
        st.dataframe(prediction_table.head(100), hide_index=True, use_container_width=True)
        st.download_button("Download test predictions (CSV)",
                           prediction_table.to_csv(index=False).encode("utf-8"),
                           file_name="shopsense_test_predictions.csv", mime="text/csv")
    with st.expander("Evaluation metric formulas", expanded=False):
        st.latex(r"Accuracy = \frac{TP+TN}{TP+TN+FP+FN}")
        st.latex(r"Precision = \frac{TP}{TP+FP}")
        st.latex(r"Recall = \frac{TP}{TP+FN}")
        st.latex(r"F1 = \frac{2\cdot Precision\cdot Recall}{Precision+Recall}")
        st.latex(r"MCC = \frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}")
        st.markdown("**AUC:** Area under the ROC curve, calculated using predicted probabilities.")

# ------------------------------ Five-model comparison ------------------------------
st.markdown('<div id="comparison"></div><div class="section-heading">Model comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Compare the six evaluation metrics across all five classifiers. Scores are shown to four decimal places for readability.</div>', unsafe_allow_html=True)

# A supplied comparison file is used only as a clearly labelled reference.
# Evaluations of uploaded data are calculated separately using the saved models.
REFERENCE_COMPARISON = Path(__file__).resolve().parent / "model_comparison.csv"
metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
reference_comparison = None
if REFERENCE_COMPARISON.is_file():
    try:
        candidate = pd.read_csv(REFERENCE_COMPARISON)
        if all(col in candidate.columns for col in ["ML Model Name", *metric_columns]):
            for col in metric_columns:
                candidate[col] = pd.to_numeric(candidate[col], errors="raise")
            reference_comparison = candidate[["ML Model Name", *metric_columns]].copy()
    except (ValueError, OSError, pd.errors.ParserError):
        st.warning("The model_comparison.csv reference could not be read.")

compare_clicked = st.button("Evaluate all five models", disabled=X is None, type="secondary")
if compare_clicked and X is not None:
    comparison_rows = []
    comparison_errors = []
    with st.spinner("Evaluating saved classifiers..."):
        for name in MODEL_FILES:
            try:
                result = evaluate_saved_model(name, X, y)
                comparison_rows.append({"ML Model Name": name, **{("F1" if key == "F1 Score" else key): value for key, value in result["scores"].items()}})
            except Exception as exc:
                comparison_errors.append(f"{name}: {exc}")
    st.session_state["shop_comparison"] = pd.DataFrame(comparison_rows)
    st.session_state["shop_comparison_errors"] = comparison_errors
    st.session_state["shop_comparison_file"] = current_file

live_comparison = st.session_state.get("shop_comparison")
if (isinstance(live_comparison, pd.DataFrame) and not live_comparison.empty
        and st.session_state.get("shop_comparison_file") == current_file):
    comparison = live_comparison.copy()
    st.caption("Results calculated from your currently uploaded test data.")
elif reference_comparison is not None:
    comparison = reference_comparison.copy()
    st.caption("Reference results from model_comparison.csv. Upload test data and select Evaluate all five models to recalculate.")
else:
    comparison = None
    st.info("Upload a labelled test CSV and select Evaluate all five models to see the comparison.")

if comparison is not None:
    # Keep column names, model names, and numeric data unchanged; format only the display.
    metrics = [col for col in metric_columns if col in comparison.columns]
    display_cols = ["ML Model Name", *metrics]
    best_per_metric = {col: comparison[col].max() for col in metrics}
    header = ''.join(f'<th>{_html.escape(col)}</th>' for col in display_cols)
    body_rows = []
    for _, row_data in comparison.iterrows():
        cells = [f'<td class="model-name">{_html.escape(str(row_data["ML Model Name"]))}</td>']
        for metric in metrics:
            val = float(row_data[metric])
            highlight = ' class="best-score"' if np.isclose(val, best_per_metric[metric]) else ''
            cells.append(f'<td{highlight}>{val:.4f}</td>')
        body_rows.append('<tr>' + ''.join(cells) + '</tr>')
    st.markdown('<div class="compare-scroll"><table class="compare-table"><thead><tr>'
                + header + '</tr></thead><tbody>' + ''.join(body_rows)
                + '</tbody></table></div>', unsafe_allow_html=True)
    st.caption("Light-blue numbers indicate the highest value for that metric. All metrics are higher-is-better.")


    st.markdown('<div class="section-heading comparison-chart-title">Metric comparison charts</div>', unsafe_allow_html=True)
    chosen_metrics = st.multiselect("Metrics to display", metrics, default=["Accuracy", "AUC", "F1", "MCC"])
    if chosen_metrics:
        # Two small, separate charts per row. Keep the labels horizontal and legible.
        for start in range(0, len(chosen_metrics), 2):
            group = chosen_metrics[start:start + 2]
            fig, axes = plt.subplots(1, len(group), figsize=(12, 3.5), squeeze=False)
            for ax, metric in zip(axes[0], group):
                values = comparison[metric].astype(float).to_numpy()
                best = int(np.argmax(values))
                colors = ["#2874A6" if i == best else "#AED6F1" for i in range(len(values))]
                positions = np.arange(len(values))
                ax.bar(positions, values, color=colors, width=0.6)
                ax.set_xticks(positions, comparison["ML Model Name"], rotation=25, ha="right", fontsize=8)
                ax.set_title(metric, fontsize=11, fontweight="bold", color="#285487", pad=10)
                ax.tick_params(axis="y", labelsize=8)
                ax.grid(axis="y", alpha=0.17)
                ax.set_axisbelow(True)
                lower = min(0, float(np.min(values)))
                upper = max(0.05, float(np.max(values)))
                margin = max(0.04, (upper - lower) * 0.13)
                ax.set_ylim(lower - (margin if lower < 0 else 0), upper + margin)
                for x, v in zip(positions, values):
                    ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 4 if v >= 0 else -12),
                                textcoords="offset points", ha="center", fontsize=8, color="#34465d")
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
            fig.tight_layout(w_pad=2.8)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            plt.close(fig)

    for error in (st.session_state.get("shop_comparison_errors", []) if st.session_state.get("shop_comparison_file") == current_file else []):
        st.warning(error)

# ------------------------------ Dataset & methodology ------------------------------
st.markdown('<div id="methodology"></div><div class="section-heading">Dataset & methodology</div>', unsafe_allow_html=True)
st.markdown('<div class="insight-panel"><strong>Online Shoppers Purchasing Intention</strong><br>'
            'This binary classification task predicts whether an online browsing session ends in a purchase. '
            'The target column is <b>Revenue</b>; the remaining columns describe the session. '
            'All evaluation is performed using saved model artifacts—no training takes place in this app.</div>',
            unsafe_allow_html=True)
if test_df is not None:
    summary_rows = [{"Property": "Uploaded records", "Value": f"{len(test_df):,}"},
                    {"Property": "CSV columns", "Value": str(test_df.shape[1])},
                    {"Property": "Input columns", "Value": str(test_df.shape[1] - int("Revenue" in test_df.columns))},
                    {"Property": "Missing values", "Value": f"{int(test_df.isna().sum().sum()):,}"},
                    {"Property": "Target column", "Value": "Revenue" if "Revenue" in test_df.columns else "Not found"}]
    # Render as a light, readable HTML table instead of Streamlit's dark dataframe grid.
    summary_body = "".join(
        f'<tr><td>{_html.escape(str(item["Property"]))}</td>'
        f'<td>{_html.escape(str(item["Value"]))}</td></tr>'
        for item in summary_rows
    )
    st.markdown(
        '<div class="schema-table-wrap"><table class="schema-table">'
        '<thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>'
        + summary_body + '</tbody></table></div>', unsafe_allow_html=True
    )
    with st.expander("View column types and missing values"):
        schema = pd.DataFrame({"Column": test_df.columns, "Data type": test_df.dtypes.astype(str).values,
                               "Missing": test_df.isna().sum().values})
        schema_body = "".join(
            '<tr>' + ''.join(f'<td>{_html.escape(str(value))}</td>' for value in record) + '</tr>'
            for record in schema.itertuples(index=False, name=None)
        )
        st.markdown(
            '<div class="schema-table-wrap schema-scroll"><table class="schema-table">'
            '<thead><tr><th>Column</th><th>Data type</th><th>Missing</th></tr></thead><tbody>'
            + schema_body + '</tbody></table></div>', unsafe_allow_html=True
        )
else:
    st.caption("Upload your labelled test CSV to see its actual structure and data-quality summary.")
st.markdown('<div class="data-note">Evaluation workflow: read the labelled CSV → validate Revenue → '
            'align input features to the saved pipeline → predict classes and probabilities → calculate '
            'Accuracy, AUC, Precision, Recall, F1 and MCC. The uploaded CSV is never modified.</div>',
            unsafe_allow_html=True)
