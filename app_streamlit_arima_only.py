
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Time Series Forecasting (ARIMA Only)", layout="wide")
st.title("📉 Time Series Forecasting App - ARIMA Only")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a datetime index and one value column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Using sample AirPassengers dataset")
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")

df.columns = ["Date", "Value"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

st.subheader("📊 Preview of Data")
st.write(df.head())

st.subheader("📈 Time Series Plot")
st.line_chart(df)

st.subheader("🧪 Augmented Dickey-Fuller (ADF) Test")
adf_result = adfuller(df["Value"])
st.write(f"ADF Statistic: {adf_result[0]:.4f}")
st.write(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    st.success("✅ Series is stationary (reject H0)")
else:
    st.warning("⚠️ Series is not stationary (fail to reject H0)")

st.sidebar.header("ARIMA Parameters")
p = st.sidebar.slider("p", 0, 5, 1)
d = st.sidebar.slider("d", 0, 2, 1)
q = st.sidebar.slider("q", 0, 5, 0)

if st.sidebar.button("Run ARIMA Forecast"):
    st.subheader("🔮 ARIMA Forecast")
    try:
        model = ARIMA(df["Value"], order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=12)
        forecast_index = pd.date_range(df.index[-1], periods=12, freq="MS")
        forecast_series = pd.Series(forecast, index=forecast_index)
        st.line_chart(pd.concat([df["Value"], forecast_series.rename("Forecast")]))
        st.write(forecast_series)
    except Exception as e:
        st.error(f"Model failed: {e}")
