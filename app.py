import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="ML Classification Model Evaluation",
    page_icon="🤖",
    layout="wide",
)

st.title("Machine Learning Classification Model Evaluation")
st.write(
    "Upload the test dataset, select a trained classifier, "
    "and evaluate its performance."
)


# -----------------------------
# Model configuration
# -----------------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "K-Nearest Neighbor": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

SCALER_PATH = "model/scaler.joblib"

TARGET_COLUMN = "Diagnosis"


# -----------------------------
# Load test data
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload test CSV file",
    type=["csv"],
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.write(f"Rows: {test_data.shape[0]}")
    st.write(f"Columns: {test_data.shape[1]}")
    st.dataframe(test_data.head())

    # Check target column
    if TARGET_COLUMN not in test_data.columns:
        st.error(
            f"The uploaded CSV must contain the target column "
            f"'{TARGET_COLUMN}'."
        )
        st.stop()

    X_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN].map({"B": 0, "M": 1})

    if y_test.isna().any():
        st.error("Diagnosis must contain only 'B' or 'M'.")
        st.stop()

    # -----------------------------
    # Model selection
    # -----------------------------
    selected_model = st.selectbox(
        "Select a classification model",
        list(MODEL_PATHS.keys()),
    )

    # -----------------------------
    # Load model
    # -----------------------------
    model = joblib.load(MODEL_PATHS[selected_model])

    # Models requiring scaling
    scaled_models = {
        "Logistic Regression",
        "K-Nearest Neighbor",
        "Support Vector Machine",
    }

    if selected_model in scaled_models:
        scaler = joblib.load(SCALER_PATH)
        X_model = scaler.transform(X_test)
    else:
        X_model = X_test

    # -----------------------------
    # Prediction
    # -----------------------------
    y_pred = model.predict(X_model)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_model)[:, 1]
    else:
        y_score = model.decision_function(X_model)

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")
    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"],
    )
    display.plot(ax=ax)
    ax.set_title(selected_model)
    st.pyplot(fig)

    # -----------------------------
    # Classification report
    # -----------------------------
    st.subheader("Prediction Summary")

    result = pd.DataFrame(
        {
            "Actual": y_test.map({0: "Benign", 1: "Malignant"}),
            "Predicted": pd.Series(y_pred, index=y_test.index).map(
                {0: "Benign", 1: "Malignant"}
            ),
        }
    )

    st.dataframe(result)