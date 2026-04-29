import streamlit as st
import numpy as np
import pickle

# Page setup
st.set_page_config(page_title="Linear Regression Predictor", layout="centered")

st.title("📈 Linear Regression Predictor")
st.write("Enter a value to get prediction")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Input (single feature)
input_value = st.number_input("Enter input value", value=0.0)

# Predict button
if st.button("Predict"):
    if model is not None:
        try:
            input_data = np.array([[input_value]])  # 2D array required
            prediction = model.predict(input_data)

            st.success(f"🎯 Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("Model not loaded!")
