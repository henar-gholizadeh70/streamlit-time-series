
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("📊 Time Series Analysis with ARIMA / ARIMAX")

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

        st.subheader("📦 Select Time Series Model")
        model_type = st.selectbox("Choose model type:", ["ARIMA", "ARIMAX"])

        if model_type == "ARIMAX":
            st.markdown("### 🔍 Granger Causality Test")
            granger_df = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
            granger_df['diff_time_to_event'] = granger_df['time_to_event'].diff()
            granger_df.dropna(inplace=True)

            st.write("Granger causality: systolic → time_to_event")
            grangercausalitytests(granger_df[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5)

            st.write("Granger causality: diastolic → time_to_event")
            grangercausalitytests(granger_df[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5)

else:
    st.info("👈 Please upload a dataset to begin.")
