# script.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model and mappings
with open('model_pickle', 'rb') as f:
    model = pickle.load(f)

with open('mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

property_mapping = mappings['property']
region_mapping = mappings['region']

property_names = list(property_mapping.keys())
region_names = list(region_mapping.keys())

df = pd.read_csv("data.csv")

# UI
st.title("🏠 Property Price Predictor")
st.write("Predict the price of a property based on its features.")

# User inputs
latitude = st.number_input("Latitude", float(df['latitude'].min()), float(df['latitude'].max()), float(df['latitude'].mean()))
longitude = st.number_input("Longitude", float(df['longitude'].min()), float(df['longitude'].max()), float(df['longitude'].mean()))
bedrooms = st.number_input("Number of Bedrooms", int(df['bedrooms'].min()), int(df['bedrooms'].max()), 3)
bathrooms = st.number_input("Number of Bathrooms", int(df['bathrooms'].min()), int(df['bathrooms'].max()), 2)
parking = st.number_input("Number of Parking Spaces", int(df['parkingSpaces'].min()), int(df['parkingSpaces'].max()), 1)

property_display = [f"{name} ({property_mapping[name]})" for name in property_names]
region_display = [f"{name} ({region_mapping[name]})" for name in region_names]

property_selection = st.selectbox("Property Type", property_display)
region_selection = st.selectbox("Region", region_display)

property_name = property_selection.split(" (")[0]
region_name = region_selection.split(" (")[0]



# Prediction Function
def predict_price(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    log_pred = model.predict(input_array)
    return np.expm1(log_pred)[0]

if st.button("Predict Price"):
    property_encoded = property_mapping[property_name]
    region_encoded = region_mapping[region_name]

    features = [
        latitude, longitude, bedrooms, bathrooms, parking,
        property_encoded, region_encoded
    ]

    price = predict_price(model, features)
    st.success(f"🏷️ Predicted Price: ${price:,.0f}")


# Data set preview
if st.checkbox("Show dataset preview"):
    st.dataframe(df.head(10))
