import streamlit as st
import joblib
import numpy as np

st.title("Heart Disease Prediction")
st.write("Predict whether a patient has heart disease.")

# Load model và scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")
# tạo ô nhập thông tin 
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)
sex = 1 if sex == "Male" else 0
