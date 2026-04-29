import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn  # Add this to force-check if it's installed

# Set up the page
st.set_page_config(page_title="Logistic Regression Predictor", layout="centered")
st.title("Machine Learning Model Predictor")
st.write("Enter the details below to get a prediction.")

# Load the model
# Ensure 'model (5).pkl' is in the same directory as this script
@st.cache_resource
def load_model():
    with open('model (5).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()

    # Create input fields based on model features: 
    # Age, Gender, Region, Occupation, Income
    st.header("Input Features")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    # Note: Since the model expects 5 features, categorical inputs 
    # (Gender, Region, Occupation) must be provided in the numerical 
    # format the model was trained on (e.g., 0 or 1).
    gender = st.number_input("Gender (Numeric)", min_value=0, max_value=1, value=0, help="0 for Female, 1 for Male (depending on training)")
    region = st.number_input("Region (Numeric Code)", min_value=0, value=0)
    occupation = st.number_input("Occupation (Numeric Code)", min_value=0, value=0)
    income = st.number_input("Income", min_value=0, value=50000)

    # Create a button for prediction
    if st.button("Predict"):
        # Arrange inputs into the format the model expects
        input_data = pd.DataFrame([[age, gender, region, occupation, income]], 
                                 columns=['Age', 'Gender', 'Region', 'Occupation', 'Income'])
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display results
        st.subheader("Result")
        result_text = "Yes" if prediction[0] == "yes" else "No"
        st.success(f"The predicted outcome is: **{result_text}**")
        
        # Show probabilities
        st.write("### Prediction Probability")
        prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.bar_chart(prob_df.T)

except FileNotFoundError:
    st.error("Error: 'model (5).pkl' not found. Please ensure the file is in the same directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
