import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💰 Salary Prediction App")

# Load model
@st.cache_resource
def load_model():
    try:
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

    st.subheader("Enter Input Data")

    # Inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", [0, 1])  # 0 = Female, 1 = Male (example)
    region = st.number_input("Region", value=0)
    occupation = st.number_input("Occupation", value=0)
    income = st.number_input("Income", value=50000)

    # Prediction button
    if st.button("Predict"):
        try:
            input_data = pd.DataFrame(
                [[age, gender, region, occupation, income]],
                columns=['Age', 'Gender', 'Region', 'Occupation', 'Income']
            )

            prediction = model.predict(input_data)

            st.success(f"🎯 Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.warning("⚠️ Model not loaded. Please check model.pkl file.")
