import streamlit as st
import numpy as np
import joblib

model = joblib.load("energy_model.pkl")

st.title("⚡ Smart Energy Consumption Forecasting")

temperature = st.number_input("Temperature", value=20.0)
humidity = st.number_input("Humidity", value=60.0)
windspeed = st.number_input("Wind Speed", value=2.0)

hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of Week", 0, 6, 3)

lag_1 = st.number_input("Previous Hour Consumption", value=70000.0)
lag_24 = st.number_input("Previous Day Same Hour", value=68000.0)

if st.button("Predict"):
    features = np.array([[temperature, humidity, windspeed,
                          hour, day, month, day_of_week,
                          lag_1, lag_24]])

    prediction = model.predict(features)
    st.success(f"Predicted Consumption: {prediction[0]:.2f}")
