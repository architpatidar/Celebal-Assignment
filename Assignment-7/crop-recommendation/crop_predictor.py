import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Crop label dictionary
crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}

# Function to predict crop
def predict_crop(features):
    # Scale the features
    scaled_features = scaler.transform([features])

    # Make prediction
    prediction = model.predict(scaled_features)

    # Get the crop label
    label = list(crop_dict.keys())[list(crop_dict.values()).index(prediction[0])]

    return label

# Streamlit User Interface
st.title("Crop Recommendation System")

# Input fields for the features
nitrogen = st.number_input("Enter Nitrogen", min_value=0.0, max_value=100.0, value=20.0)
phosphorus = st.number_input("Enter Phosphorus", min_value=0.0, max_value=100.0, value=15.0)
potassium = st.number_input("Enter Potassium", min_value=0.0, max_value=100.0, value=30.0)
temperature = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
ph = st.number_input("Enter pH level", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0.0, max_value=3000.0, value=100.0)

# When the button is pressed
if st.button("Predict Crop"):
    features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
    predicted_crop = predict_crop(features)
    st.success(f"The recommended crop is: {predicted_crop}")
