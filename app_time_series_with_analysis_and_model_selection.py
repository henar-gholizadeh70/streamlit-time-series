
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📊 Time Series Forecasting and Analysis")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.sort_values(by=["participant_id", "date"])
    grouped = df.groupby("participant_id")
    participant_ids = list(grouped.groups.keys())
    selected_pid = st.selectbox("Select a participant ID:", participant_ids)

    if selected_pid:
        participant_data = grouped.get_group(selected_pid).sort_values("date").set_index("date")
        participant_data = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
        participant_data['time_to_event'] = participant_data['time_to_event'].astype(float)

        st.subheader("📈 Time to Event and Blood Pressure Trends")

        # Rolling average
        participant_data['rolling_mean_7_systolic'] = participant_data['blood_pressure_systolic'].rolling(window=7).mean()
        participant_data['rolling_mean_7_diastolic'] = participant_data['blood_pressure_diastolic'].rolling(window=7).mean()

        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(participant_data.index, participant_data['blood_pressure_systolic'], alpha=0.4, label='Original Systolic')
        ax1.plot(participant_data.index, participant_data['rolling_mean_7_systolic'], label='Smoothed (7-day MA)', linewidth=2)
        ax1.set_title('Systolic BP: Original vs Smoothed')
        ax1.legend(); ax1.grid(True); fig1.tight_layout()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(participant_data.index, participant_data['blood_pressure_diastolic'], alpha=0.4, label='Original Diastolic')
        ax2.plot(participant_data.index, participant_data['rolling_mean_7_diastolic'], label='Smoothed (7-day MA)', linewidth=2)
        ax2.set_title('Diastolic BP: Original vs Smoothed')
        ax2.legend(); ax2.grid(True); fig2.tight_layout()
        st.pyplot(fig2)

        st.subheader("📉 ADF Stationarity Test and ACF")

        ts = participant_data['time_to_event']
        adf_result = adfuller(ts.dropna())
        st.write(f"**ADF Statistic**: {adf_result[0]:.4f}")
        st.write(f"**p-value**: {adf_result[1]:.4f}")
        for key, value in adf_result[4].items():
            st.write(f"**Critical Value ({key})**: {value:.4f}")
        if adf_result[1] < 0.05:
            st.success("✅ Series is stationary (reject H0)")
        else:
            st.warning("❌ Series is non-stationary (fail to reject H0)")

        fig3, ax3 = plt.subplots(figsize=(8, 3))
        plot_acf(ts.dropna(), lags=5, ax=ax3)
        ax3.set_title("Autocorrelation (ACF) - time_to_event")
        ax3.grid(True)
        fig3.tight_layout()
        st.pyplot(fig3)

        # Model selection
        st.subheader("⚙️ Model Selection")
        model_type = st.selectbox("Select forecasting model:", ["ARIMA", "ARIMAX"])

        if model_type == "ARIMAX":
            st.subheader("🔍 Granger Causality Tests (ARIMAX)")
            st.info("Based on Granger Causality, both systolic and diastolic pressures have a causal effect on time_to_event.")

            df_granger = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
            df_granger['diff_time_to_event'] = df_granger['time_to_event'].diff()
            df_granger.dropna(inplace=True)

            with st.expander("Systolic → time_to_event"):
                st.text("Granger test: systolic → time_to_event")
                grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5)

            with st.expander("Diastolic → time_to_event"):
                st.text("Granger test: diastolic → time_to_event")
                grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5)
else:
    st.info("👈 Please upload a dataset to begin.")
