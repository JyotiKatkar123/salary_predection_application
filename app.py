import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Salary Prediction Portal",
    page_icon="📊",
    layout="centered"
)

# --- CUSTOM CSS FOR BETTER LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the logistic regression model from the local directory."""
    try:
        # Ensure your file is named exactly 'model.pkl' on GitHub
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Initializing the model
model = load_model()

# --- HEADER SECTION ---
st.title("📊 Salary Prediction App")
st.write("Enter the details below to check the prediction. The system will automatically adjust the income for the model calculation.")

if model is not None:
    st.success("✅ Model System Online")

    # --- INPUT SECTION ---
    st.subheader("User Information")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Current Age", min_value=18, max_value=100, value=39)
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            region = st.number_input("Region Code", value=2, step=1)
            
        with col2:
            occupation = st.number_input("Occupation Code", value=5, step=1)
            income_input = st.number_input("Annual Income", value=50025)
            
            # Applying your requested +10,000 modification
            income_modified = income_input + 10000
            st.info(f"💡 Processing Income as: **{income_modified}** (+10k adjustment)")

    # --- PREDICTION LOGIC ---
    if st.button("Run Prediction"):
        try:
            # Preparing the feature dataframe with exact names from model metadata [cite: 1]
            input_df = pd.DataFrame(
                [[age, gender, region, occupation, income_modified]],
                columns=['Age', 'Gender', 'Region', 'Occupation', 'Income']
            )

            # Execution
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            # --- RESULTS DISPLAY ---
            st.markdown("---")
            st.subheader("Prediction Analysis")

            # Final Prediction Result
            final_result = prediction[0].upper()
            if final_result == "YES":
                st.balloons()
                st.success(f"### 🎯 Result: {final_result}")
            else:
                st.error(f"### 🎯 Result: {final_result}")

            # Probability Breakdown
            st.write("### Confidence Breakdown")
            prob_no = prediction_proba[0][0] * 100
            prob_yes = prediction_proba[0][1] * 100

            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Probability of 'NO'", f"{prob_no:.2f}%")
            res_col2.metric("Probability of 'YES'", f"{prob_yes:.2f}%")

            # Chart to visualize the bias
            chart_data = pd.DataFrame(prediction_proba, columns=model.classes_)
            st.bar_chart(chart_data.T)

        except Exception as e:
            st.error(f"⚠️ Prediction Error: {e}")

else:
    st.warning("⚠️ Critical Error: Model file ('model.pkl') not detected in the repository.")

# --- FOOTER ---
st.markdown("---")
st.caption("Powered by Streamlit | Model version: 1.6.1")
