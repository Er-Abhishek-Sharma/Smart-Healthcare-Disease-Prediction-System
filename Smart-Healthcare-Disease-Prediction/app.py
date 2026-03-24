import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("diabetes_model.keras")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("AI Healthcare System - Diabetes Prediction")
st.write("Enter patient health details to predict diabetes risk")

# Input fields
Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
Glucose = st.number_input("Glucose Level", 0, 200, 100)
BloodPressure = st.number_input("Blood Pressure", 0, 150, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 79)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
Age = st.number_input("Age", 1, 120, 30)

# Predict button
if st.button("Predict Diabetes"):

    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    risk_percentage = float(prediction[0][0] * 100)

    # Display risk as text
    if risk_percentage > 50:
        st.error(f"High Risk of Diabetes ({risk_percentage:.2f}%)")
    else:
        st.success(f"Low Risk of Diabetes ({risk_percentage:.2f}%)")

    # Plot risk as a bar chart
    fig, ax = plt.subplots()
    ax.bar(["Diabetes Risk"], [risk_percentage], color='red' if risk_percentage > 50 else 'green')
    ax.set_ylim([0, 100])
    ax.set_ylabel("Risk (%)")
    ax.set_title("Diabetes Risk Visualization")
    st.pyplot(fig)