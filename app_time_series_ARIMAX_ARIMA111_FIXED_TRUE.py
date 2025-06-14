
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

            # ایستایی و ACF
            st.subheader("📊 Stationarity Analysis (ADF + ACF + Differencing)")
            selected_series_name = st.selectbox(
                "Select a time series to analyze:",
                ["time_to_event", "blood_pressure_systolic", "blood_pressure_diastolic"]
            )

            ts_data = participant_data[selected_series_name]
            adf_result = adfuller(ts_data.dropna())
            st.write(f"**ADF Statistic**: {adf_result[0]:.4f}")
            st.write(f"**p-value**: {adf_result[1]:.4f}")
            for key, value in adf_result[4].items():
                st.write(f"Critical Value ({key}): {value:.4f}")
            if adf_result[1] < 0.05:
                st.success("✅ Series is stationary (reject H₀)")
            else:
                st.error("❌ Series is non-stationary (fail to reject H₀)")

            fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
            plot_acf(ts_data.dropna(), lags=5, ax=ax_acf)
            st.pyplot(fig_acf)

            if selected_series_name == "time_to_event":
                diff_ts = ts_data.diff().dropna()
                fig_diff, ax_diff = plt.subplots(figsize=(8, 3))
                ax_diff.plot(diff_ts, color="mediumseagreen")
                st.pyplot(fig_diff)

            # انتخاب مدل
            st.subheader("🔮 Select Forecasting Model")
            model_type = st.selectbox("Choose a model type:", ["ARIMAX", "ARIMA"])

            if model_type == "ARIMAX":
                st.subheader("📍 Granger Causality Test")
                df_granger = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
                df_granger['diff_time_to_event'] = df_granger['time_to_event'].diff()
                df_granger.dropna(inplace=True)
                st.markdown("**Granger test: systolic → time_to_event**")
                grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5, verbose=True)
                st.markdown("**Granger test: diastolic → time_to_event**")
                grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5, verbose=True)

                st.subheader("⚡ ARIMAX Modes")

                if st.checkbox("Run ARIMAX(1,1,1) with lag"):
                    participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
                    participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)
                    y = participant_data['time_to_event'].astype(float)
                    X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic', 'lag_1_systolic', 'lag_1_diastolic']]
                    combined = pd.concat([y, X], axis=1).dropna()
                    y, X = combined['time_to_event'], combined.drop(columns=['time_to_event'])
                    split_idx = int(len(y) * 0.8)
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    model = ARIMA(y_train, exog=X_train, order=(1, 1, 1)).fit()
                    forecast = model.forecast(steps=len(y_test), exog=X_test)
                    st.write(f"MAE: {mean_absolute_error(y_test, forecast):.4f}")

                if st.checkbox("Run ARIMAX(3,1,0) with lag"):
                    # مشابه بالا فقط order=(3,1,0)
                    pass

                if st.checkbox("Run ARIMAX PCA"):
                    # مشابه قبل، اول PCA و سپس مدل ARIMAX
                    pass

                # و بقیه حالات ARIMAX مثل فایل اصلی، اینجا قرار می‌گیرد.

            if model_type == "ARIMA":
                st.subheader("📈 ARIMA Forecasting")
                arima_option = st.selectbox("Choose ARIMA configuration:", ["Select an option", "ARIMA(1,1,1)"])
                if arima_option == "ARIMA(1,1,1)":
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
                    st.markdown(f"**MAE:** {mae:.4f}")
                    st.markdown(f"**MSE:** {mse:.5f}")
                    st.markdown(f"**RSS:** {rss:.5f}")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(y_test.index, y_test, label="Actual")
                    ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
                    ax.set_title("ARIMA(1,1,1) Forecast vs Actual")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("👈 Please upload a dataset to get started.")
