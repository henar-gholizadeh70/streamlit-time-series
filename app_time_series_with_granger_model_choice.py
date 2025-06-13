
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf

st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("📊 Time Series Viewer for Participants")

# Upload section
uploaded_file = st.file_uploader("Please upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.sort_values(by=['participant_id', 'date'])
        grouped = df.groupby('participant_id')

        # Selection and filtering
        st.subheader("🔢 Sorting by participant and date:")
        st.dataframe(df.head())
        participant_series = {
            pid: group.set_index('date')['time_to_event'].sort_index()
            for pid, group in grouped
            if len(group) >= 12
        }

        st.write(f"✅ Total participants with ≥ 12 visits: {len(participant_series)}")
        selected_pid = st.selectbox("Select a participant ID:", list(participant_series.keys()))
        participant_data = grouped.get_group(selected_pid).set_index('date')

        # Model selection
        st.subheader("⚙️ Select Time Series Model")
        model_type = st.selectbox("Choose model:", ["ARIMA", "ARIMAX"])

        if model_type == "ARIMAX":
            st.subheader("📌 Granger Causality Test (for ARIMAX only)")
            df_granger = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
            df_granger['diff_time_to_event'] = df_granger['time_to_event'].diff()
            df_granger.dropna(inplace=True)

            with st.expander("Systolic → time_to_event", expanded=True):
                st.text("Granger Test: Systolic → time_to_event")
                grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5)

            with st.expander("Diastolic → time_to_event", expanded=True):
                st.text("Granger Test: Diastolic → time_to_event")
                grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👈 Please upload a dataset to get started.")
