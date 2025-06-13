
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Analysis", layout="wide")
st.title("📊 Time Series Viewer for Participants")

uploaded_file = st.file_uploader("Please upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.sort_values(by=['participant_id', 'date'])
    st.success("✅ File successfully uploaded!")

    grouped = df.groupby('participant_id')
    participant_series = {pid: group.set_index('date')['time_to_event'].sort_index()
                          for pid, group in grouped if len(group) >= 12}
    selected_pid = st.selectbox("Select a participant ID:", list(participant_series.keys()))
    participant_data = grouped.get_group(selected_pid).sort_values('date').set_index('date')

    st.subheader("📊 Stationarity Analysis")
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf
    ts_data = participant_data['time_to_event']
    adf_result = adfuller(ts_data.dropna())
    st.write(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    fig_acf, ax_acf = plt.subplots()
    plot_acf(ts_data.dropna(), lags=5, ax=ax_acf)
    st.pyplot(fig_acf)

    st.subheader("🔮 Select Forecasting Model")
    model_type = st.selectbox("Choose a model:", ["ARIMAX(1,1,1) BP+Lag", "ARIMAX(3,1,0) BP+Lag",
                                                  "ARIMAX(1,1,1) No Lag", "ARIMAX(3,1,0) No Lag",
                                                  "ARIMAX(1,1,1) PCA+Lag"])

    y = participant_data['time_to_event'].astype(float)

    if model_type == "ARIMAX(1,1,1) PCA+Lag":
        st.subheader("📈 ARIMAX(1,1,1) with PCA Component + Lag")
        X_vif = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_vif)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled)
        pc1_df = pd.DataFrame(pc1, index=X_vif.index, columns=['pc1'])
        participant_data.loc[pc1_df.index, 'pc1'] = pc1_df['pc1']
        participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)
        X = participant_data[['pc1_lag1']]
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['pc1_lag1']]
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]
        model = ARIMA(endog=y_train, exog=X_train, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum((y_test - forecast) ** 2)
        st.write(f"MAE: {mae:.4f}, MSE: {mse:.5f}, RSS: {rss:.5f}")
        fig, ax = plt.subplots()
        ax.plot(y_test.index, y_test, label='Actual')
        ax.plot(y_test.index, forecast, label='Forecast', linestyle='--')
        ax.set_title("ARIMAX(1,1,1) with PCA+Lag Forecast vs Actual")
        ax.legend()
        st.pyplot(fig)

else:
    st.info("👈 Please upload a dataset to get started.")
