import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Assignment 2", layout="centered")
st.title("ML Assignment 2 - Classification Model Evaluation App")

st.write("Upload your TEST dataset (CSV).")
st.write("⚠️ Make sure the last column is the target variable.")

# -----------------------------
# Function to load model
# -----------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # -----------------------------
    # Model Selection
    # -----------------------------
    model_option = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    # -----------------------------
    # Load Selected Model
    # -----------------------------
    if model_option == "Logistic Regression":
        model = load_model("Model/logistic.pkl")

    elif model_option == "Decision Tree":
        model = load_model("Model/decision_tree.pkl")

    elif model_option == "KNN":
        model = load_model("Model/knn.pkl")

    elif model_option == "Naive Bayes":
        model = load_model("Model/naive_bayes.pkl")

    elif model_option == "Random Forest":
        model = load_model("Model/random_forest.pkl")

    elif model_option == "XGBoost":
        model = load_model("Model/xgboost.pkl")

    # -----------------------------
    # Make Predictions
    # -----------------------------
    y_pred = model.predict(X)

    # -----------------------------
    # Evaluation Metrics
    # -----------------------------
    st.subheader("Evaluation Metrics")

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    ax.matshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center')

    st.pyplot(fig)

    # -----------------------------
    # Classification Report
    # -----------------------------
    st.subheader("Classification Report")

    report = classification_report(y, y_pred)
    st.text(report)

