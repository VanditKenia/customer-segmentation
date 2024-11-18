import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load the pre-trained model, scaler, and training columns
try:
    model = joblib.load('optimized_random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    training_columns = joblib.load('training_columns.pkl')
except Exception as e:
    st.error(f"Error loading model or other files: {e}")

# Apply custom CSS for the gradient background and styled input fields
st.markdown("""
    <style>
        /* Apply a gradient background to the entire app */
        html, body, .block-container {
            height: 100%;
            background: linear-gradient(to right, #ff7e5f, #feb47b) !important;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #333;
        }

        .css-18e3th9, .css-1dp5vir, .stApp {
            background-color: transparent !important;
            color: inherit !important;
        }

        .block-container {
            flex: 1;
            width: 100%;
            max-width: 800px;
            text-align: center;
            padding: 20px;
        }

        h1 {
            color: #ffffff !important;
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            padding-bottom: 20px;
        }

        /* Input fields */
        input, select, textarea {
            background-color: rgba(255, 255, 255, 0.8) !important; /* Light semi-transparent background */
            border: 2px solid #ffffff !important;
            color: #333 !important;
            padding: 10px !important;
            border-radius: 8px !important;
            font-size: 1.1rem !important;
            width: 100% !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #feb47b !important;
            color: white !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }

        .stButton > button:hover {
            background-color: #ff7e5f !important;
        }

        /* Headers */
        h2 {
            color: #ffffff !important;
            font-size: 1.8rem !important;
            margin-bottom: 20px;
        }

        /* Radio buttons */
        .stRadio > label {
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }

        /* Spacing for form elements */
        .stSelectbox, .stNumberInput, .stTextInput, .stRadio {
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state if not already set
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'input_data' not in st.session_state:
    st.session_state.input_data = {}

# Step 1: Customer Basic Details
if st.session_state.step == 1:
    st.header("Step 1: Customer Basic Details")

    # Input fields for basic details
    income = st.number_input("Income (in USD)", min_value=0.0, step=1000.0)
    education = st.selectbox("Education", ["Graduation", "PhD", "High School", "Masters", "Doctorate"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

    # "Next" button
    if st.button("Next: Customer Tenure & Spending Details"):
        st.session_state.input_data['Income'] = income
        st.session_state.input_data['Education'] = education
        st.session_state.input_data['Marital_Status'] = marital_status
        st.session_state.step = 2

# Step 2: Customer Tenure and Spending Features
elif st.session_state.step == 2:
    st.header("Step 2: Customer Tenure & Spending Details")

    # Input fields for customer tenure and spending
    customer_tenure = st.number_input("Customer Tenure (days)", min_value=0, step=1)
    spending_features = {
        "MntWines": st.number_input("Amount Spent on Wine (USD)", min_value=0.0, step=100.0),
        "MntFruits": st.number_input("Amount Spent on Fruits (USD)", min_value=0.0, step=100.0),
        "MntMeatProducts": st.number_input("Amount Spent on Meat Products (USD)", min_value=0.0, step=100.0),
    }

    if st.button("Next: Additional Spending Features"):
        st.session_state.input_data['Customer_Tenure'] = customer_tenure
        st.session_state.input_data.update(spending_features)
        st.session_state.step = 3

# Step 3: Additional Spending Features
elif st.session_state.step == 3:
    st.header("Step 3: Additional Spending Features")

    spending_features = {
        "MntFishProducts": st.number_input("Amount Spent on Fish Products (USD)", min_value=0.0, step=100.0),
        "MntSweetProducts": st.number_input("Amount Spent on Sweets (USD)", min_value=0.0, step=100.0),
        "MntGoldProds": st.number_input("Amount Spent on Gold Products (USD)", min_value=0.0, step=100.0),
    }

    if st.button("Next: Make Prediction"):
        st.session_state.input_data.update(spending_features)
        st.session_state.step = 4

# Step 4: Make Prediction and Show Results
elif st.session_state.step == 4:
    st.header("Step 4: Prediction Result")

    input_df = pd.DataFrame([st.session_state.input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    input_prepared = scaler.transform(input_encoded)
    predicted_purchases = model.predict(input_prepared)[0]

    income_threshold = 50000
    purchase_threshold = 10

    if st.session_state.input_data['Income'] < income_threshold and predicted_purchases > purchase_threshold:
        label = "Low Income, High Buy"
    elif st.session_state.input_data['Income'] < income_threshold and predicted_purchases <= purchase_threshold:
        label = "Low Income, Low Buy"
    elif st.session_state.input_data['Income'] >= income_threshold and predicted_purchases > purchase_threshold:
        label = "High Income, High Buy"
    else:
        label = "High Income, Low Buy"

    st.write(f"### Predicted Number of Purchases: {predicted_purchases:.2f}")
    st.write(f"### Customer Category: {label}")

    if st.button("Start Over"):
        st.session_state.step = 1
        st.session_state.input_data = {}
