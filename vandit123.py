import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model, scaler, and training columns
model = joblib.load('optimized_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
training_columns = joblib.load('training_columns.pkl')

# Set the title of the app
st.title("Customer Segmentation Prediction")

# Custom CSS to style the page
st.markdown("""
    <style>
        body {
            background-color: #F0F8FF;
        }
        .main {
            padding: 20px;
        }
        h1 {
            color: #004d99;
        }
        h2 {
            color: #004d99;
        }
        .stButton > button {
            background-color: #0066cc;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #005bb5;
        }
        .stRadio > label {
            color: #004d99;
        }
        .stSelectbox, .stNumberInput, .stTextInput {
            background-color: #e6f2ff;
        }
        .stSelectbox, .stNumberInput, .stTextInput, .stButton {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Use session_state to store intermediate values
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'input_data' not in st.session_state:
    st.session_state.input_data = {}

# Step 1: Customer Basic Details (Income, Education, Marital Status)
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
        st.experimental_rerun()

# Step 2: Customer Tenure and Spending Features
elif st.session_state.step == 2:
    st.header("Step 2: Customer Tenure & Spending Details")

    # Input fields for customer tenure and spending
    customer_tenure = st.number_input("Customer Tenure (days)", min_value=0, step=1)
    spending_threshold = 5000
    spending_features = {
        "MntWines": st.number_input("Amount Spent on Wine (USD)", min_value=0.0, step=100.0),
        "MntFruits": st.number_input("Amount Spent on Fruits (USD)", min_value=0.0, step=100.0),
        "MntMeatProducts": st.number_input("Amount Spent on Meat Products (USD)", min_value=0.0, step=100.0),
    }

    # Collect all data into session_state
    if st.button("Next: Spending Products"):
        st.session_state.input_data['Customer_Tenure'] = customer_tenure
        st.session_state.input_data.update(spending_features)
        st.session_state.step = 3
        st.experimental_rerun()

# Step 3: Additional Spending Features (Fish, Sweets, Gold)
elif st.session_state.step == 3:
    st.header("Step 3: Additional Spending Features")

    spending_features = {
        "MntFishProducts": st.number_input("Amount Spent on Fish Products (USD)", min_value=0.0, step=100.0),
        "MntSweetProducts": st.number_input("Amount Spent on Sweets (USD)", min_value=0.0, step=100.0),
        "MntGoldProds": st.number_input("Amount Spent on Gold Products (USD)", min_value=0.0, step=100.0),
    }

    # Collect data and go to the prediction step
    if st.button("Next: Make Prediction"):
        st.session_state.input_data.update(spending_features)
        st.session_state.step = 4
        st.experimental_rerun()

# Step 4: Make Prediction and Display Results
elif st.session_state.step == 4:
    st.header("Step 4: Prediction Result")

    # Collect all user inputs into a DataFrame
    input_df = pd.DataFrame([st.session_state.input_data])

    # Encode categorical data (e.g., Education, Marital Status)
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    # Scale the input data
    input_prepared = scaler.transform(input_encoded)

    # Predict the purchases using the trained model
    predicted_purchases = model.predict(input_prepared)[0]

    # Define thresholds for categorization
    income_threshold = 50000  # Example threshold for income
    purchase_threshold = 10  # Example threshold for purchase amount

    if st.session_state.input_data['Income'] < income_threshold and predicted_purchases > purchase_threshold:
        label = "Low Income, High Buy"
    elif st.session_state.input_data['Income'] < income_threshold and predicted_purchases <= purchase_threshold:
        label = "Low Income, Low Buy"
    elif st.session_state.input_data['Income'] >= income_threshold and predicted_purchases > purchase_threshold:
        label = "High Income, High Buy"
    else:
        label = "High Income, Low Buy"

    # Display the prediction result
    st.write(f"### Predicted Number of Purchases: {predicted_purchases:.2f}")
    st.write(f"### Customer Category: {label}")

    # Option to reset and start over
    if st.button("Start Over"):
        st.session_state.step = 1
        st.experimental_rerun()
