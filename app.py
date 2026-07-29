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

cp = st.selectbox( # vì là biến categorical ( phân loại) nên ta ưu tiên sử dụng select box
    "Chest Pain Type",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    [0, 1, 2, 3]
)
features = np.array([
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]).reshape(1, -1)

features_scaled = scaler.transform(features)

if st.button("Predict"):

    prediction = model.predict(features_scaled)[0]

    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error("❤️ Heart Disease Detected")

    else:
        st.success("💚 No Heart Disease")

    st.write(f"Probability of Heart Disease: {probability*100:.2f}%")