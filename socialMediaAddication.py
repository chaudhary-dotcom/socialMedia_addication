import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or pickle
from sklearn.linear_model import LinearRegression

# Load your trained pipeline (assume it's saved as 'model_pipeline.pkl')
pipeline = joblib.load("model_pipeline.pkl")  # Change the path as needed

# Title
st.title("Social Media Impact on Academic Performance")

# User Inputs
st.header("Enter Student Information")

age = st.number_input("Age", min_value=10, max_value=100, value=21)
gender = st.selectbox("Gender", ["Male", "Female"])
academic_level = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
country = st.selectbox("Country", ["Nepal", "Bangladesh", "India", "USA", "UK", "Canada", 'Finland'])
avg_daily_usage = st.slider("Avg Daily Social Media Usage (hours)", 0.0, 10.0, 4.0)
platform = st.selectbox("Most Used Platform", ["Facebook", "Instagram", "YouTube", "TikTok", "Twitter"])
affects_academic = st.selectbox("Affects Academic Performance?", ["Yes", "No"])
sleep_hours = st.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
mental_health = st.slider("Mental Health Score", 1, 10, 5)
relationship = st.selectbox("Relationship Status", ["Single", "In Relationship", "Complicated"])
conflicts = st.slider("Conflicts Over Social Media", 0, 10, 2)

# Convert to DataFrame
user_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Academic_Level": [academic_level],
    "Country": [country],
    "Avg_Daily_Usage_Hours": [avg_daily_usage],
    "Most_Used_Platform": [platform],
    "Affects_Academic_Performance": [affects_academic],
    "Sleep_Hours_Per_Night": [sleep_hours],
    "Mental_Health_Score": [mental_health],
    "Relationship_Status": [relationship],
    "Conflicts_Over_Social_Media": [conflicts]
})

# Predict Button
if st.button("Predict Addicted Score"):
    prediction = pipeline.predict(user_data)
    st.success(f"Predicted Addicted Score: {prediction[0]:.2f}")
