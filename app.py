import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Machine Learning Prediction Portal")

@st.cache_resource
def load_model():
    try:
        # Make sure the file on GitHub is exactly 'model.pkl'
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model:
    st.success("Model loaded!")
    # ... (rest of your input fields and prediction logic)
    age = st.number_input("Age", value=30)
    gender = st.number_input("Gender (0/1)", value=0)
    region = st.number_input("Region", value=0)
    occ = st.number_input("Occupation", value=0)
    inc = st.number_input("Income", value=50000)

    if st.button("Predict"):
        input_data = pd.DataFrame([[age, gender, region, occ, inc]], 
                                 columns=['Age', 'Gender', 'Region', 'Occupation', 'Income'])
        prediction = model.predict(input_data)
        st.write(f"Prediction: {prediction[0]}")
