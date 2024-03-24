import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the ARIMA model
with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the predictor variable
predictor = 'Penjualan'

# Create Streamlit app
st.title('ARIMA Model Inference')

# Input field for the exogenous variable
st.sidebar.header('Input Data for Inference')
input_data = {}
input_data[predictor] = st.sidebar.number_input('Enter Previous Penjualan', step=0.01)

# Perform inference
if st.sidebar.button('Predict'):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Perform prediction using the loaded model
    prediction = model.forecast(steps=1, exog=input_df)
    
    # Display the prediction
    st.subheader('Prediction')
    st.write(f'Predicted Penjualan: {prediction[0]}')
