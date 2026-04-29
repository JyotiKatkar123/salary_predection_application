import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="ML Prediction App", layout="centered")

st.title("📊 Machine Learning Prediction App")

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

    st.subheader("Enter Input Values")

    # Inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1])
    region = st.number_input("Region", value=0)
    occupation = st.number_input("Occupation", value=0)
    income = st.number_input("Income", value=50000)

    # Show modified occupation (for clarity)
    st.write(f"👉 Occupation used in model: {occupation + 1000}")

    # Prediction
    if st.button("Predict"):
        try:
            # Modify occupation
            occupation_modified = occupation + 1000

            # Create dataframe
            input_data = pd.DataFrame(
                [[age, gender, region, occupation_modified, income]],
                columns=['Age', 'Gender', 'Region', 'Occupation', 'Income']
            )

            # Predict
            prediction = model.predict(input_data)

            # Output
            st.success(f"🎯 Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

else:
    st.warning("⚠️ Model not loaded. Check model.pkl file.")
