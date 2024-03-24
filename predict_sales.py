import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the ARIMA model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the predictor variables
predictors = ['Harga Min (Rp)', 'Harga Max (Rp)', 'Harga Modal (Rp)']

# Create Streamlit app
st.title('ARIMA Model Inference')

# Input fields for exogenous variables
st.sidebar.header('Input Data for Inference')
input_data = {}
for predictor in predictors:
    input_data[predictor] = st.sidebar.number_input(f'Enter {predictor}', step=0.01)

# Perform inference
if st.sidebar.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Perform prediction using the loaded model
    prediction = model.forecast(steps=1, exog=input_df)
    
    # Display the prediction
    st.subheader('Prediction')
    st.write(f'Predicted Penjualan: {prediction[0]}')
