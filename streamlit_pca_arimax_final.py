
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

st.set_page_config(page_title="Time Series Analysis with PCA ARIMAX", layout="wide")
st.title("📊 Time Series Analysis with ARIMAX and PCA (Reduced Multicollinearity)")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.sort_values(by=['participant_id', 'date'])
    st.success("✅ File uploaded and cleaned")

    sample_pid = df['participant_id'].unique()[0]
    participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

    st.write(f"## Selected Participant: {sample_pid}")

    # VIF Check
    X_vif = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    st.write("### 📊 VIF Check for Multicollinearity")
    st.dataframe(vif_data)

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vif)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, index=X_vif.index, columns=['pc1'])
    participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']
    participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)

    st.write("### 🎛 PCA Component and Lag")
    st.dataframe(participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic', 'pc1', 'pc1_lag1']].head())

    model_options = ["ARIMA", "ARIMAX(1,1,1) with BP", "ARIMAX(3,1,0) with BP", "ARIMAX(1,1,1) with PCA Component"]

    selected_model = st.selectbox("Select Forecasting Model:", model_options)

    y = participant_data['time_to_event'].astype(float)

    if selected_model == "ARIMAX(1,1,1) with PCA Component":
        st.subheader("📈 ARIMAX(1,1,1) with PCA Component (Reduced Multicollinearity)")
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
        st.write(f"**MAE**: {mae:.4f}")
        st.write(f"**MSE**: {mse:.5f}")
        st.write(f"**RSS**: {rss:.5f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label='Actual')
        ax.plot(y_test.index, forecast, label='Forecast', linestyle='--')
        ax.set_title("ARIMAX(1,1,1) with PCA Component Forecast vs Actual")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

else:
    st.info("👈 Upload a dataset to get started.")
