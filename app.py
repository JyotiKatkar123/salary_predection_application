import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Page Setup
st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("Machine Learning Prediction Portal")

# Load the model - using the exact name from your pkl content
@st.cache_resource
def load_model():
    # Make sure your file on GitHub is named 'model.pkl'
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("⚠️ 'model.pkl' not found! Please rename your .pkl file to 'model.pkl' on GitHub.")
else:
    st.success("✅ Model loaded successfully!")

    st.header("Enter Details for Prediction")
    
    # These inputs match the 'feature_names_in_' found in your pkl file
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.number_input("Gender (0 for Female, 1 for Male)", min_value=0, max_value=1, value=0)
        region = st.number_input("Region Code", min_value=0, step=1)
        
    with col2:
        occupation = st.number_input("Occupation Code", min_value=0, step=1)
        income = st.number_input("Income", min_value=0, step=1000, value=50000)

    if st.button("Generate Prediction"):
        # Create DataFrame with exact column names from your model data
        input_df = pd.DataFrame(
            [[age, gender, region, occupation, income]], 
            columns=['Age', 'Gender', 'Region', 'Occupation', 'Income']
        )
        
        try:
            prediction = model.predict(input_df)
            prediction_prob = model.predict_proba(input_df)

            st.markdown("---")
            st.subheader("Results")
            
            # The classes in your model are 'no' and 'yes'
            result = "Positive (Yes)" if prediction[0] == "yes" else "Negative (No)"
            st.metric(label="Prediction Result", value=result)
            
            # Show probability bar chart
            st.write("### Prediction Probability")
            prob_data = pd.DataFrame(prediction_prob, columns=model.classes_)
            st.bar_chart(prob_data.T)
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
