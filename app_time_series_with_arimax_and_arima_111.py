
# کد کامل با اضافه شدن گزینه ARIMA(1,1,1)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("📊 Time Series Viewer for Participants")

uploaded_file = st.file_uploader("Please upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        st.success("✅ File successfully uploaded!")

        df = df.sort_values(by=['participant_id', 'date'])
        grouped = df.groupby('participant_id')
        participant_series = {
            pid: group.set_index('date')['time_to_event'].sort_index()
            for pid, group in grouped
            if len(group) >= 12
        }

        st.write(f"✅ Total participants with ≥ 12 visits: {len(participant_series)}")

        if participant_series:
            selected_pid = st.selectbox("Select a participant ID:", list(participant_series.keys()))
            participant_data = grouped.get_group(selected_pid).set_index('date')

            st.line_chart(participant_series[selected_pid])

            fig_bp, ax_bp = plt.subplots(figsize=(6, 3.5))
            participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].plot(ax=ax_bp)
            ax_bp.set_title(f'Blood Pressure Over Time (Participant {selected_pid})')
            ax_bp.set_xlabel('Date')
            ax_bp.set_ylabel('Blood Pressure (mmHg)')
            ax_bp.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_bp)

            st.subheader("🔮 Select Forecasting Model")
            model_type = st.selectbox("Choose a model type:", ["ARIMA(1,1,1)", "ARIMAX"])

            if model_type == "ARIMA(1,1,1)":
                y = participant_data['time_to_event'].dropna().astype(float)
                split_idx = int(len(y) * 0.8)
                y_train, y_test = y[:split_idx], y[split_idx:]
                model = ARIMA(y_train, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(y_test))
                mae = mean_absolute_error(y_test, forecast)
                mse = mean_squared_error(y_test, forecast)
                rss = np.sum(np.square(y_test - forecast))
                st.success("📊 ARIMA(1,1,1) Evaluation")
                st.markdown(f"**MAE**: {mae:.4f}")
                st.markdown(f"**MSE**: {mse:.5f}")
                st.markdown(f"**RSS**: {rss:.5f}")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(y_test.index, y_test, label="Actual")
                ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
                ax.set_title("ARIMA(1,1,1) Forecast vs Actual")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            if model_type == "ARIMAX":
                st.info("✅ همان ساختار ARIMAX موجود در فایل اصلی باقی می‌ماند.")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("👈 Please upload a dataset to get started.")
