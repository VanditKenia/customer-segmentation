import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model = joblib.load('optimized_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
training_columns = joblib.load('training_columns.pkl')

st.title("Random Forest Regressor Prediction with Labels")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    data_encoded = pd.get_dummies(data, drop_first=True)
    data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
    data_prepared = scaler.transform(data_encoded)

    predictions = model.predict(data_prepared)
    data['Predicted NumWebPurchases'] = predictions

    income_threshold = data['Income'].quantile(0.5)  
    purchase_threshold = data['Predicted NumWebPurchases'].quantile(0.5)  

    def assign_label(row):
        if row['Income'] < income_threshold and row['Predicted NumWebPurchases'] > purchase_threshold:
            return "Low Income, High Buy"
        elif row['Income'] < income_threshold and row['Predicted NumWebPurchases'] <= purchase_threshold:
            return "Low Income, Low Buy"
        elif row['Income'] >= income_threshold and row['Predicted NumWebPurchases'] > purchase_threshold:
            return "High Income, High Buy"
        else:
            return "High Income, Low Buy"

    data['Category'] = data.apply(assign_label, axis=1)

    st.write("Labeled Predictions:")
    st.dataframe(data[['Income', 'Predicted NumWebPurchases', 'Category']])

    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Labeled Predictions as CSV",
        data=csv,
        file_name='labeled_predictions.csv',
        mime='text/csv'
    )