import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Configuration
st.set_page_config(page_title="Prediction App", layout="centered")
st.title("Machine Learning Model Predictor")

# Load the model
@st.cache_resource
def load_model():
    # Ensure the filename here matches your GitHub filename exactly
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
    st.success("Model loaded successfully!")

    st.header("Input Features")
    
    # Input fields for the 5 required features
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", options=[0, 1], help="0 for Female, 1 for Male")
    region = st.number_input("Region Code", min_value=0, value=0)
    occupation = st.number_input("Occupation Code", min_value=0, value=0)
    income = st.number_input("Income", min_value=0, value=30000)

    if st.button("Predict"):
        # Create DataFrame with the exact feature names the model expects
        features = pd.DataFrame([[age, gender, region, occupation, income]], 
                               columns=['Age', 'Gender', 'Region', 'Occupation', 'Income'])
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        st.subheader("Prediction Result")
        # Model classes from your file: ['no', 'yes']
        st.write(f"The model predicts: **{prediction[0].upper()}**")
        
        # Display probability
        prob_df = pd.DataFrame(probability, columns=model.classes_)
        st.write("Confidence Level:")
        st.bar_chart(prob_df.T)

except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Check your GitHub file name.")
except Exception as e:
    st.error(f"Error: {e}")
