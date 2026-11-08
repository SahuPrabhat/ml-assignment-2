# Streamlit Web Application - ML Assignment 2
# Student: Prabhat Sahu
# M.Tech (AIML) - BITS Pilani

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Page configuration
st.set_page_config(
    page_title="ML Classification App - BITS Pilani",
    page_icon="🔬",
    layout="wide"
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, "test_data.csv")

# Model mapping
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

# Load scaler helper
@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

# Load model helper
@st.cache_resource
def load_saved_model(model_name):
    filename = MODEL_FILES.get(model_name)
    if not filename:
        return None
    model_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Compute all 6 metrics
def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc
    }

# App header
st.title("🔬 Breast Cancer Classification App")
st.markdown("""
**BITS Pilani - M.Tech (AIML) | Machine Learning Assignment - 2**  
*Demonstration of 5 Machine Learning Classifiers evaluated on the Breast Cancer Wisconsin Diagnostic Dataset.*
""")

st.divider()

# Sidebar controls
st.sidebar.header("Controls & Settings")

# Model selection dropdown
selected_model_name = st.sidebar.selectbox(
    "Choose ML Model:",
    options=list(MODEL_FILES.keys()),
    index=0
)

st.sidebar.markdown("---")

# Dataset upload option
st.sidebar.subheader("Dataset Upload (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must have features + 'target'):",
    type=["csv"]
)

# Load test data
test_df = None
data_source_label = ""

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        data_source_label = f"Uploaded File: {uploaded_file.name} ({len(test_df)} rows)"
        st.sidebar.success("Custom test dataset loaded!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
else:
    if os.path.exists(DEFAULT_TEST_PATH):
        test_df = pd.read_csv(DEFAULT_TEST_PATH)
        data_source_label = f"Default Test Set (test_data.csv - {len(test_df)} rows)"
        st.sidebar.info("Using pre-split test_data.csv")
    else:
        st.sidebar.warning("Default test_data.csv not found.")

# Quick download button for sample test data
if os.path.exists(DEFAULT_TEST_PATH):
    with open(DEFAULT_TEST_PATH, "rb") as f:
        st.sidebar.download_button(
            label="Download sample test_data.csv",
            data=f,
            file_name="sample_test_data.csv",
            mime="text/csv"
        )

# Validate dataset
if test_df is None:
    st.error("No test dataset available. Please upload a test CSV file.")
    st.stop()

if "target" not in test_df.columns:
    st.error("Uploaded test CSV must contain a 'target' column for evaluation.")
    st.stop()

# Prepare features and target
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"].astype(int)

# Load preprocessing artifacts and model
scaler = load_scaler()
model = load_saved_model(selected_model_name)

if model is None:
    st.error(f"Could not load saved model for {selected_model_name}. Ensure model file exists in model/ folder.")
    st.stop()

# Preprocess features according to model requirements
if selected_model_name == "Decision Tree" or scaler is None:
    X_input = X_test
else:
    X_input = scaler.transform(X_test)

# Predict on test data
y_pred = model.predict(X_input)

# Probabilities for AUC
y_prob = None
if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_input)[:, 1]
elif hasattr(model, "decision_function"):
    df_scores = model.decision_function(X_input)
    y_prob = 1 / (1 + np.exp(-df_scores))

# Compute metrics
metrics = compute_metrics(y_test, y_pred, y_prob)

# Main layout
st.subheader(f"Active Model: {selected_model_name}")
st.caption(f"Evaluated on: {data_source_label}")

# Display evaluation metrics in cards
m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
m_col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
m_col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
m_col3.metric("Precision", f"{metrics['Precision']:.4f}")
m_col4.metric("Recall", f"{metrics['Recall']:.4f}")
m_col5.metric("F1 Score", f"{metrics['F1']:.4f}")
m_col6.metric("MCC Score", f"{metrics['MCC']:.4f}")

st.write("")

# Confusion Matrix and Classification Report layout
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "ROC Curve"])

with tab1:
    col_cm, col_cm_info = st.columns([1.5, 1])
    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Malignant (0)", "Benign (1)"],
            yticklabels=["Malignant (0)", "Benign (1)"],
            ax=ax
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"Confusion Matrix - {selected_model_name}")
        st.pyplot(fig)
        plt.close(fig)

    with col_cm_info:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        **Matrix Breakdown:**
        - **True Negatives (TN):** {tn} (Malignant correctly identified)
        - **False Positives (FP):** {fp} (Malignant misclassified as Benign)
        - **False Negatives (FN):** {fn} (Benign misclassified as Malignant)
        - **True Positives (TP):** {tp} (Benign correctly identified)
        
        *Total test instances:* {len(y_test)}
        """)

with tab2:
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=4))

with tab3:
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(5.5, 4))
        ax_roc.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {metrics['AUC']:.4f})")
        ax_roc.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"ROC Curve - {selected_model_name}")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        plt.close(fig_roc)
    else:
        st.info("Probability estimates not available for ROC curve.")

st.divider()

# Comparison Table across all 5 models
st.subheader("All Models Comparison on this Dataset")
st.markdown("Below is the dynamic comparison of all five trained classifiers evaluated on the active test set:")

all_results = []
for name in MODEL_FILES.keys():
    m = load_saved_model(name)
    if m is not None:
        if name == "Decision Tree" or scaler is None:
            m_input = X_test
        else:
            m_input = scaler.transform(X_test)
        
        pred = m.predict(m_input)
        prob = None
        if hasattr(m, "predict_proba"):
            prob = m.predict_proba(m_input)[:, 1]
        elif hasattr(m, "decision_function"):
            sc = m.decision_function(m_input)
            prob = 1 / (1 + np.exp(-sc))
            
        m_eval = compute_metrics(y_test, pred, prob)
        all_results.append({
            "ML Model Name": name,
            "Accuracy": round(m_eval["Accuracy"], 4),
            "AUC": round(m_eval["AUC"], 4),
            "Precision": round(m_eval["Precision"], 4),
            "Recall": round(m_eval["Recall"], 4),
            "F1": round(m_eval["F1"], 4),
            "MCC": round(m_eval["MCC"], 4)
        })

comparison_df = pd.DataFrame(all_results)
st.dataframe(comparison_df)

# Highlight winning model
best_model_row = comparison_df.sort_values(by=["Accuracy", "AUC"], ascending=False).iloc[0]
st.success(f"🏆 **Top Performing Model on active test set:** {best_model_row['ML Model Name']} (Accuracy: {best_model_row['Accuracy']:.4f}, AUC: {best_model_row['AUC']:.4f}, MCC: {best_model_row['MCC']:.4f})")

# Preview raw test data
with st.expander("View Uploaded / Active Test Data Table"):
    st.dataframe(test_df.head(20))
