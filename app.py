import streamlit as st
import pandas as pd
import joblib

# Load the saved model, scaler, and column names
model = joblib.load('knn_heart_model.pkl')  
scaler = joblib.load('heart_scaler.pkl')
expected_columns = joblib.load('heart_columns.pkl')

#Collect the data
st.title("Heart Stroke Risk Prediction by Tanmay_Sharma 💖 ")
st.markdown("Please enter the following details to predict the risk of heart stroke:")

# Create input fields for each feature
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ['Male', 'Female'])
chest_pain_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)  # Added range and default value
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)  # Added range and default value
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['True', 'False'])
resting_ecg = st.selectbox("Resting ECG", ['Normal',"ST","LVH"])
max_heart_rate = st.number_input("Max Heart Rate Achieved", 60, 220, 150)  #    Added range and default value
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ['Yes', 'No'])
st_depression = st.number_input("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 1.0)  # Added range and default value
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Up', 'Flat', 'Down']) # Added options for slope

#prediction
if st.button("Predict"):
    raw_input={
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'resting_ecg': resting_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression': st_depression,
        'st_slope': st_slope
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)  
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️High Risk of Heart Stroke! Please consult a doctor immediately.")
    else:
        st.success("✅ Low Risk of Heart Stroke.")