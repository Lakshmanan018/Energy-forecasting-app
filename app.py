import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("energy_model.pkl")

st.title("⚡ Smart Energy Consumption Forecasting")

st.markdown("### Enter Input Features")

# Weather inputs
temperature = st.number_input("Temperature", value=20.0)
humidity = st.number_input("Humidity", value=60.0)
windspeed = st.number_input("Wind Speed", value=2.0)
general_diffuse = st.number_input("General Diffuse Flows", value=100.0)
diffuse = st.number_input("Diffuse Flows", value=50.0)

# Zone consumption inputs (since model was trained with them)
zone1 = st.number_input("PowerConsumption_Zone1", value=30000.0)
zone2 = st.number_input("PowerConsumption_Zone2", value=20000.0)
zone3 = st.number_input("PowerConsumption_Zone3", value=15000.0)

# Time features
hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)

# Lag features
lag_1 = st.number_input("Previous Hour Consumption (lag_1)", value=70000.0)
lag_24 = st.number_input("Previous Day Same Hour (lag_24)", value=68000.0)
rolling_mean_24 = st.number_input("Rolling Mean 24", value=69000.0)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "WindSpeed": windspeed,
        "GeneralDiffuseFlows": general_diffuse,
        "DiffuseFlows": diffuse,
        "PowerConsumption_Zone1": zone1,
        "PowerConsumption_Zone2": zone2,
        "PowerConsumption_Zone3": zone3,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "rolling_mean_24": rolling_mean_24
    }])

    prediction = model.predict(input_data)

    st.success(f"Predicted Total Consumption: {prediction[0]:.2f}")
