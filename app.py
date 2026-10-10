
import streamlit as st
import pandas as pd

from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(
    page_title="Electricity Consumption Prediction",
    page_icon="⚡"
)

st.title("⚡ Electricity Consumption Prediction")
st.write("Predict electricity consumption using Machine Learning.")

st.subheader("Enter Previous Electricity Consumption")

hour1 = st.number_input(
    "Consumption 1 hour ago (kW)",
    min_value=0.0,
    value=1.2
)

hour2 = st.number_input(
    "Consumption 2 hours ago (kW)",
    min_value=0.0,
    value=1.5
)

hour3 = st.number_input(
    "Consumption 3 hours ago (kW)",
    min_value=0.0,
    value=1.3
)

if st.button("Predict Electricity Consumption"):

    model = joblib.load("electricity_model.pkl")

    inputs = pd.DataFrame(
        [[hour1, hour2, hour3]],
        columns=[
            "previous_hour_1",
            "previous_hour_2",
            "previous_hour_3"
        ]
    )

    prediction = model.predict(inputs)[0]
    prediction = max(0, prediction)

    st.success(
        f"Predicted Average Power: {prediction:.3f} kW"
    )
