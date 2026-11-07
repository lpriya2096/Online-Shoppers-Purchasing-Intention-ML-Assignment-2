import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Online Shopping Purchase Prediction", page_icon="🛒", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1050px; padding-top: 1rem; padding-bottom: 1rem;}
h1 {font-size: 1.65rem !important;}
[data-testid="stMetricValue"] {font-size: 1.25rem;}
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
    
st.write("")
st.title("Online Shoppers' Purchasing Intention")

st.caption(
    "Compare the performance of five trained machine "
    "learning models using test data."
)

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

    # ---- Dataset viewer ----
    with st.expander("View dataset", expanded=True):
        tab_preview, tab_balance, tab_stats = st.tabs(
            ["Data preview", "Class balance", "Summary statistics"])

        with tab_preview:
            n_rows = st.number_input("Rows to display", min_value=1,
                                     max_value=len(test_df),
                                     value=min(10, len(test_df)), step=5)
            st.dataframe(test_df.head(int(n_rows)), hide_index=True)
            st.caption(f"{test_df.shape[0]:,} rows · {test_df.shape[1]} columns · "
                       f"{int(test_df.isna().sum().sum())} missing values")

        with tab_balance:
            if "Revenue" in test_df.columns:
                balance = (test_df["Revenue"].astype(str)
                           .value_counts().rename_axis("Revenue")
                           .reset_index(name="Sessions"))
                balance["Share (%)"] = (balance["Sessions"] / len(test_df) * 100).round(1)
                st.dataframe(balance, hide_index=True)
                st.bar_chart(balance.set_index("Revenue")["Sessions"])
            else:
                st.info("No Revenue column found in this file.")

        with tab_stats:
            st.dataframe(test_df.describe().T.round(3))

with st.form("evaluation_form"):
    selected_model = st.selectbox("Classification model", list(MODEL_FILES))
    run_evaluation = st.form_submit_button("Run Evaluation", type="primary")

if not run_evaluation:
    st.info("Upload **test_data.csv**, select a model, and click **Run Evaluation**.")
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
st.subheader("Model Evaluation")
for col, (label, score) in zip(st.columns(6), scores):
    col.metric(label, f"{score:.4f}")
st.caption(f"{selected_model} · {len(test_df):,} rows · {X.shape[1]} features")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
cm_table = pd.DataFrame(
    [[int(tn), int(fp)], [int(fn), int(tp)]],
    index=["Actual: No Purchase", "Actual: Purchase"],
    columns=["Predicted: No Purchase", "Predicted: Purchase"],
)
st.table(cm_table)
st.caption(f"TN = {tn:,} · FP = {fp:,} · FN = {fn:,} · TP = {tp:,}")

with st.expander("Evaluation metric formulas"):
    st.latex(r"Accuracy = \frac{TP+TN}{TP+TN+FP+FN}")
    st.latex(r"Precision = \frac{TP}{TP+FP}")
    st.latex(r"Recall = \frac{TP}{TP+FN}")
    st.latex(r"F1 = \frac{2\cdot Precision\cdot Recall}{Precision+Recall}")
    st.latex(r"MCC = \frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}")
    st.markdown("**AUC:** Area under the ROC curve across decision thresholds; calculated from predicted probabilities.")
