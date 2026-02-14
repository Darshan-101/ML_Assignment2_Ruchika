import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.title("ML Assignment 2 - Mobile Price Prediction App")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

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

    y_pred = model.predict(X)

    st.subheader("Evaluation Metrics")
    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred, average="weighted"))
    st.write("Recall:", recall_score(y, y_pred, average="weighted"))
    st.write("F1 Score:", f1_score(y, y_pred, average="weighted"))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.dataframe(pd.DataFrame(cm))

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
