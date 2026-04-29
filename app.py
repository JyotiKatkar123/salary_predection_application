import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Prediction App", layout="centered")

st.title("📊 ML Prediction App")

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

if model is not None:
    st.success("✅ Model loaded!")

    st.subheader("Enter Input Values")

    # 👇 5 inputs (must match training)
    age = st.number_input("Age", value=30)
    gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
    region = st.number_input("Region", value=0)
    occupation = st.number_input("Occupation", value=0)
    income = st.number_input("Income", value=50000)

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
    st.warning("Model not loaded!")
