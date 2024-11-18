import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide",  # Use the entire width of the screen
    initial_sidebar_state="collapsed"
)

# Load the pre-trained model, scaler, and training columns
try:
    model = joblib.load('optimized_random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    training_columns = joblib.load('training_columns.pkl')
except Exception as e:
    st.error(f"Error loading model or other files: {e}")

# Apply custom CSS for styling
st.markdown("""
    <style>
        /* Full-screen white background */
        html, body, .block-container {
            height: 100%;
            background-color: #ffffff !important;  /* White background */
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
            width: 90%;
            text-align: center;
            padding: 20px;
            background-color: transparent !important;
        }

        h1, h2 {
            color: #333 !important;  /* Dark text for the heading */
        }

        /* Input fields with blended background */
        input, select, textarea {
            background: linear-gradient(to right, #f8f8f8, #e0e0e0) !important;  /* Blended gradient */
            border: 1px solid #d3d3d3 !important;  /* Subtle border */
            color: #333 !important;  /* Dark text for readability */
            padding: 12px !important;
            border-radius: 8px !important;
            font-size: 1.2rem !important;
            width: 100% !important;  /* Full-width input fields */
            margin-bottom: 20px !important;  /* Space between input fields */
            box-sizing: border-box !important;
            display: inline-block;
            transition: all 0.3s ease !important;  /* Smooth transition for all properties */
        }

        /* Focus effect with stronger blending */
        input:focus, select:focus, textarea:focus {
            outline: none !important;
            border-color: #feb47b !important;  /* Focus border color */
            background: linear-gradient(to right, #ffe0cc, #ffd6ba) !important;  /* Slightly darker gradient */
        }

        /* Make input field labels bigger and bolder */
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label {
            font-size: 1.3rem !important;  /* Larger font for labels */
            font-weight: bold !important;
            color: #333 !important;  /* Dark labels */
        }

        /* Dropdown fields styled similarly */
        .stSelectbox select {
            background: linear-gradient(to right, #f8f8f8, #e0e0e0) !important;  /* Gradient */
            border: 1px solid #d3d3d3 !important;  /* Subtle border */
            color: #333 !important;  /* Dark text */
            font-size: 1.2rem !important;
            width: 100% !important;
            padding: 12px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
            box-sizing: border-box !important;
        }

        /* Button styling */
        .stButton > button {
            background-color: #ff7e5f !important;  /* Button background */
            color: white !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease !important;  /* Smooth transition */
        }

        .stButton > button:hover {
            background-color: #e96b56 !important;  /* Darker shade on hover */
        }

        /* Form element spacing */
        .stNumberInput, .stTextInput, .stSelectbox {
            margin-bottom: 20px !important;
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
