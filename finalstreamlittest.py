import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model = joblib.load('optimized_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
training_columns = joblib.load('training_columns.pkl')

st.title("Random Forest Regressor Prediction with Labels")

st.header("Enter Customer Details")

income = st.number_input("Income", min_value=0.0, step=1000.0)
mnt_wines = st.number_input("Amount Spent on Wine", min_value=0.0, step=10.0)
mnt_fruits = st.number_input("Amount Spent on Fruits", min_value=0.0, step=10.0)
mnt_meat = st.number_input("Amount Spent on Meat Products", min_value=0.0, step=10.0)
mnt_fish = st.number_input("Amount Spent on Fish Products", min_value=0.0, step=10.0)
mnt_sweets = st.number_input("Amount Spent on Sweets", min_value=0.0, step=10.0)
mnt_gold = st.number_input("Amount Spent on Gold Products", min_value=0.0, step=10.0)

if st.button("Predict"):

    input_data = {
        'Income': [income],
        'MntWines': [mnt_wines],
        'MntFruits': [mnt_fruits],
        'MntMeatProducts': [mnt_meat],
        'MntFishProducts': [mnt_fish],
        'MntSweetProducts': [mnt_sweets],
        'MntGoldProds': [mnt_gold]
    }

    import pandas as pd
    input_df = pd.DataFrame(input_data)
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

    input_prepared = scaler.transform(input_encoded)

    predicted_purchases = model.predict(input_prepared)[0]

    income_threshold = 50000  
    purchase_threshold = 10  

    if income < income_threshold and predicted_purchases > purchase_threshold:
        label = "Low Income, High Buy"
    elif income < income_threshold and predicted_purchases <= purchase_threshold:
        label = "Low Income, Low Buy"
    elif income >= income_threshold and predicted_purchases > purchase_threshold:
        label = "High Income, High Buy"
    else:
        label = "High Income, Low Buy"

    st.write(f"Predicted NumWebPurchases: {predicted_purchases:.2f}")
    st.write(f"Category: {label}")