
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="📊 Time Series Forecasting App", layout="wide")
st.title("📊 Time Series Forecasting App with ARIMA and LSTM")

st.sidebar.header("Step 1: Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Upload your time series CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.write("Using sample dataset (AirPassengers)")
    df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")

df.columns = ["Month", "Value"]
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)

st.subheader("📄 Preview of Data")
st.dataframe(df.head())

st.subheader("📈 Time Series Plot")
st.line_chart(df)

st.subheader("📊 ADF Test (Stationarity)")
adf_result = adfuller(df["Value"])
st.write(f"ADF Statistic: {adf_result[0]:.4f}")
st.write(f"p-value: {adf_result[1]:.4f}")
if adf_result[1] < 0.05:
    st.success("✅ Series is stationary (reject H0)")
else:
    st.warning("⚠️ Series is NOT stationary (fail to reject H0)")

model_choice = st.sidebar.selectbox("Step 2: Choose model", ["ARIMA", "LSTM"])

if model_choice == "ARIMA":
    st.sidebar.subheader("ARIMA Parameters")
    p = st.sidebar.slider("p", 0, 5, 1)
    d = st.sidebar.slider("d", 0, 2, 1)
    q = st.sidebar.slider("q", 0, 5, 0)

    model = ARIMA(df["Value"], order=(p, d, q))
    model_fit = model.fit()
    st.subheader("📈 ARIMA Forecast")
    forecast = model_fit.forecast(steps=12)
    st.line_chart(pd.concat([df["Value"], forecast.rename("Forecast")]))
    st.write(forecast)

elif model_choice == "LSTM":
    st.sidebar.subheader("LSTM Parameters")
    epochs = st.sidebar.slider("Epochs", 1, 100, 10)
    units = st.sidebar.slider("LSTM units", 10, 100, 50)

    data = df["Value"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_dataset(data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            Y.append(data[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 3
    X, y = create_dataset(data_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)

    last_sequence = data_scaled[-look_back:]
    predictions = []
    current_input = last_sequence.reshape(1, look_back, 1)

    for _ in range(12):
        next_val = model.predict(current_input, verbose=0)
        predictions.append(next_val[0, 0])
        current_input = np.append(current_input[:, 1:, :], [[[next_val[0, 0]]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    st.subheader("📈 LSTM Forecast")
    future_dates = pd.date_range(start=df.index[-1], periods=13, freq="MS")[1:]
    forecast_series = pd.Series(predictions, index=future_dates)

    st.line_chart(pd.concat([df["Value"], forecast_series.rename("Forecast")]))
    st.write(forecast_series)
