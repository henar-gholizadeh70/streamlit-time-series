
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Time Series Analysis with ARIMA and ARIMAX")

uploaded_file = st.file_uploader("Upload CSV data file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    df = df.dropna(subset=['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic'])

    st.subheader("Data Preview")
    st.dataframe(df.head())

    participants = df['participant_id'].unique()
    selected_id = st.selectbox("Select Participant", participants)

    participant_data = df[df['participant_id'] == selected_id].sort_values('date').set_index('date')

    ts = participant_data['time_to_event']

    st.subheader("Stationarity Test (ADF)")
    adf_result = adfuller(ts.dropna())
    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
    st.write(f"p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        st.success("The time series is stationary.")
    else:
        st.warning("The time series is non-stationary.")

    
    st.subheader("ACF and PACF Plots")
    if len(ts.dropna()) > 10:
        fig1 = plt.figure()
        plot_acf(ts.dropna(), lags=10)
        st.pyplot(fig1)

        fig2 = plt.figure()
        plot_pacf(ts.dropna(), lags=10)
        st.pyplot(fig2)
    else:
        st.warning("Not enough data to plot ACF and PACF.")
    

    model_type = st.radio("Select Model Type", ["ARIMA", "ARIMAX"])

    if model_type == "ARIMAX":
        st.subheader("ARIMAX Model")

        participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
        participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

        y = participant_data['time_to_event']
        X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                              'lag_1_systolic', 'lag_1_diastolic']]

        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined.drop('time_to_event', axis=1)

        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        model = ARIMA(endog=y_train, exog=X_train, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)
    else:
        st.subheader("ARIMA Model (no exogenous variables)")

        ts = participant_data['time_to_event'].dropna()
        ts_diff = ts.diff().dropna()

        split_idx = int(len(ts) * 0.8)
        train, test = ts[:split_idx], ts[split_idx:]

        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        y_test, forecast = test, forecast

    mae = mean_absolute_error(y_test, forecast)
    mse = mean_squared_error(y_test, forecast)
    rss = np.sum(np.square(y_test - forecast))

    st.write("**Model Evaluation:**")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RSS: {rss:.2f}")

    fig3 = plt.figure(figsize=(10,4))
    plt.plot(y_test.index, y_test.values, label='Actual')
    plt.plot(y_test.index, forecast, label='Forecast', linestyle='--')
    plt.legend()
    plt.title(f"Forecast با مدل {model_type}")
    st.pyplot(fig3)

else:
    st.info("Please upload a dataset to proceed.")
