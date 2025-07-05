import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("insurance_rf_model.pkl")

# Page title
st.title("💰 Insurance Charges Predictor")
st.write("Predict insurance cost based on user details")

# Inputs
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode categorical inputs
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Create input DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'children': children,
    'smoker': smoker,
    'region_northwest': region_northwest,
    'region_southeast': region_southeast,
    'region_southwest': region_southwest
}])

# Predict
if st.button("Predict Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"💵 Estimated Insurance Charges: ₹ {round(prediction, 2)}")
