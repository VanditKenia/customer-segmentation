import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model = joblib.load('optimized_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

training_columns = joblib.load('training_columns.pkl')

st.title("Random Forest Regressor Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    data_encoded = pd.get_dummies(data, drop_first=True)

    data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)

    data_prepared = scaler.transform(data_encoded)

    predictions = model.predict(data_prepared)

    data['Predicted NumWebPurchases'] = predictions.round()
    st.write("Predictions:")
    st.dataframe(data[['Predicted NumWebPurchases']])

    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='predictions.csv',
        mime='text/csv'
    )