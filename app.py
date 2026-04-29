import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page setup
st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("📊 Machine Learning Prediction App")

# Load model
@st.cache_resource
def load_model():
    try:
        # Ensure your file on GitHub is named exactly "model.pkl"
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# If model loaded
if model is not None:
    st.success("✅ Model loaded successfully!")

    st.subheader("Enter Input Values")

    # Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
        region = st.number_input("Region", value=0)
        
    with col2:
        occupation = st.number_input("Occupation", value=0)
        income = st.number_input("Income", value=50000)

    # Prediction
    if st.button("Predict"):
        try:
            # Create dataframe with exact column names expected by model 
            input_data = pd.DataFrame(
                [[age, gender, region, occupation, income]],
                columns=['Age', 'Gender', 'Region', 'Occupation', 'Income']
            )

            # 1. Get the class prediction (yes/no) 
            prediction = model.predict(input_data)

            # 2. Get the numerical probability 
            prediction_proba = model.predict_proba(input_data)

            # Output results
            st.markdown("---")
            st.subheader("Results")
            
            # Display Prediction
            result_color = "green" if prediction[0] == "yes" else "red"
            st.markdown(f"### 🎯 Prediction: :{result_color}[{prediction[0].upper()}]")

            # Display Probabilities
            st.write("### Confidence Levels")
            
            # Access probabilities for 'no' (index 0) and 'yes' (index 1) 
            prob_no = prediction_proba[0][0] * 100
            prob_yes = prediction_proba[0][1] * 100

            c1, c2 = st.columns(2)
            c1.metric("Probability of NO", f"{prob_no:.2f}%")
            c2.metric("Probability of YES", f"{prob_yes:.2f}%")

            # Visual Bar Chart for clarity
            prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
            st.bar_chart(prob_df.T)

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.warning("⚠️ Model not loaded. Check model.pkl file.")
