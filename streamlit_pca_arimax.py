
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Analysis with PCA-ARIMAX", layout="wide")
st.title("📊 Time Series with ARIMAX and PCA")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    st.success("✅ File uploaded successfully!")

    df = df.sort_values(by=['participant_id', 'date'])

    st.subheader("🔢 Data Preview")
    st.dataframe(df.head())

    sample_pid = df['participant_id'].unique()[0]
    participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

    st.write(f"### 📌 Participant: {sample_pid}")

    X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bp)

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)

    X_pca_df = pd.DataFrame(X_pca, index=X_bp.index, columns=['pc1'])
    participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']
    participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)

    st.write("### 🎛 PCA Component and Lag")
    st.dataframe(participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic', 'pc1', 'pc1_lag1']].head())

    data = participant_data[['time_to_event', 'pc1_lag1']].dropna()
    y = data['time_to_event']
    X = data[['pc1_lag1']]

    split_idx = int(len(y) * 0.8)
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]

    st.subheader("📈 ARIMAX(1,1,1) with PCA Component and Lag")
    model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

    mae = mean_absolute_error(y_test, forecast)
    mse = mean_squared_error(y_test, forecast)
    rss = np.sum((y_test - forecast) ** 2)

    st.write(f"**MAE**: {mae:.4f}")
    st.write(f"**MSE**: {mse:.5f}")
    st.write(f"**RSS**: {rss:.5f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.index, y_test, label='Actual')
    ax.plot(y_test.index, forecast, label='Forecast', linestyle='--')
    ax.set_title(f"ARIMAX(1,1,1) Forecast vs Actual with PCA-Lag for Participant {sample_pid}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.success("✅ ARIMAX(1,1,1) with PCA-lag completed!")
else:
    st.info("👈 Upload a dataset to get started.")
