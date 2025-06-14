
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("📊 Time Series Viewer for Participants")

# Upload section
uploaded_file = st.file_uploader("Please upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
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

        participant_data = grouped.get_group(selected_pid).sort_values('date').set_index('date')

        st.line_chart(participant_series[selected_pid])

        st.subheader("🔮 Select Forecasting Model")
        model_type = st.selectbox("Choose a model type:", ["ARIMA", "ARIMAX"])

        if model_type == "ARIMA":
            st.subheader("📈 ARIMA Forecasting")

            arima_option = st.selectbox(
                "Choose ARIMA configuration:",
                ["Select an option", "ARIMA(1,1,1)", "ARIMA(3,1,0)"],
                key="arima_option_main"
            )

            if arima_option != "Select an option":
                y = participant_data['time_to_event'].dropna().astype(float)
                split_idx = int(len(y) * 0.8)
                y_train, y_test = y[:split_idx], y[split_idx:]

                if arima_option == "ARIMA(1,1,1)":
                    order = (1, 1, 1)
                else:
                    order = (3, 1, 0)

                model = ARIMA(y_train, order=order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(y_test))
                mae = mean_absolute_error(y_test, forecast)
                mse = mean_squared_error(y_test, forecast)
                rss = np.sum((y_test - forecast) ** 2)

                st.write(f"### Evaluation for {arima_option}")
                st.write(f"**MAE:** {mae:.4f}")
                st.write(f"**MSE:** {mse:.5f}")
                st.write(f"**RSS:** {rss:.5f}")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(y_test.index, y_test, label="Actual")
                ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
                ax.set_title(f"{arima_option} Forecast vs Actual")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

else:
    st.info("👈 Please upload a dataset to get started.")
