
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
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

        st.success("✅ File successfully uploaded!")

        # Sort by participant and date
        st.subheader("🔢 Sorting by participant and date:")
        df = df.sort_values(by=['participant_id', 'date'])
        st.dataframe(df.head())

        # 📊 Overview: Daily average and smoothed blood pressure
        st.subheader("📉 Blood Pressure Trends (All Participants Combined)")
        try:
            sd_df = df.groupby('date')[['blood_pressure_systolic', 'blood_pressure_diastolic']].mean()
            sd_df_smooth = sd_df.rolling(window=7).mean()

            fig, axes = plt.subplots(1, 2, figsize=(18, 5))
            sd_df.plot(ax=axes[0], title='Original Blood Pressure (Daily Average)')
            sd_df_smooth.plot(ax=axes[1], title='Smoothed Blood Pressure (7-day MA)')
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("""These plots show overall trends to help visually assess whether a clear pattern exists in the blood pressure data over time.""")
        except Exception as plot_err:
            st.warning(f"Could not generate trend plots: {plot_err}")

        # Grouping for time series
        st.subheader("📈 Time series per participant (≥ 12 visits):")
        grouped = df.groupby('participant_id')
        participant_series = {
            pid: group.set_index('date')['time_to_event'].sort_index()
            for pid, group in grouped
            if len(group) >= 12
        }

        st.write(f"✅ Total participants with ≥ 12 visits: {len(participant_series)}")

        if participant_series:
            selected_pid = st.selectbox("Select a participant ID:", list(participant_series.keys()))
            selected_series = participant_series[selected_pid]

            st.subheader(f"📊 Time Series for Participant {selected_pid}")
            st.line_chart(selected_series)


        # 🩺 اضافه کردن نمودار فشار خون برای فرد انتخاب‌شده (فقط کوچک‌تر شده)
        st.subheader(f"💓 Blood Pressure for Participant {selected_pid}")
        participant_data = grouped.get_group(selected_pid).set_index('date')

        fig_bp, ax_bp = plt.subplots(figsize=(6, 3.5))  # فقط کوچکتر از نسخه اولیه

        participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].plot(ax=ax_bp)
        ax_bp.set_title(f'Blood Pressure Over Time (Participant {selected_pid})')
        ax_bp.set_xlabel('Date')
        ax_bp.set_ylabel('Blood Pressure (mmHg)')
        ax_bp.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_bp)


        # 📉 نمودارهای هموارشده فشار خون با میانگین متحرک ۷ روزه
        st.subheader(f"🌀 Smoothed Blood Pressure Trends (7-day Rolling Average) for Participant {selected_pid}")

        # محاسبه rolling mean
        participant_data['rolling_mean_7_systolic'] = participant_data['blood_pressure_systolic'].rolling(window=7).mean()
        participant_data['rolling_mean_7_diastolic'] = participant_data['blood_pressure_diastolic'].rolling(window=7).mean()

        # سیستولیک
        fig_systolic, ax_sys = plt.subplots(figsize=(8, 4))
        ax_sys.plot(participant_data.index, participant_data['blood_pressure_systolic'], label='Original Systolic', alpha=0.4)
        ax_sys.plot(participant_data.index, participant_data['rolling_mean_7_systolic'], label='Smoothed (7-day MA)', linewidth=2)
        ax_sys.set_title('Systolic Blood Pressure: Original vs Smoothed')
        ax_sys.set_xlabel('Date')
        ax_sys.set_ylabel('Systolic BP (mmHg)')
        ax_sys.legend()
        ax_sys.grid(True)
        ax_sys.tick_params(axis='x', rotation=45)
        fig_systolic.tight_layout()
        st.pyplot(fig_systolic)

        # دیاستولیک
        fig_diastolic, ax_dia = plt.subplots(figsize=(8, 4))
        ax_dia.plot(participant_data.index, participant_data['blood_pressure_diastolic'], label='Original Diastolic', alpha=0.4)
        ax_dia.plot(participant_data.index, participant_data['rolling_mean_7_diastolic'], label='Smoothed (7-day MA)', linewidth=2)
        ax_dia.set_title('Diastolic Blood Pressure: Original vs Smoothed')
        ax_dia.set_xlabel('Date')
        ax_dia.set_ylabel('Diastolic BP (mmHg)')
        ax_dia.legend()
        ax_dia.grid(True)
        ax_dia.tick_params(axis='x', rotation=45)
        fig_diastolic.tight_layout()
        st.pyplot(fig_diastolic)



    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👈 Please upload a dataset to get started.")



# 📌 تحلیل ایستایی فقط با ADF + ACF (lags=5) + Differencing (فقط برای time_to_event)
st.subheader("📊 Stationarity Analysis (ADF + ACF + Differencing)")

selected_series_name = st.selectbox(
    "Select a time series to analyze:",
    ["time_to_event", "blood_pressure_systolic", "blood_pressure_diastolic"]
)

if selected_pid:
    participant_data = grouped.get_group(selected_pid).sort_values('date').set_index('date')
    ts_data = participant_data[selected_series_name]

    # ADF Test
    from statsmodels.tsa.stattools import adfuller
    st.markdown("### 📈 ADF Test (Augmented Dickey-Fuller)")
    adf_result = adfuller(ts_data.dropna())
    st.write(f"**ADF Statistic**: {adf_result[0]:.4f}")
    st.write(f"**p-value**: {adf_result[1]:.4f}")
    for key, value in adf_result[4].items():
        st.write(f"Critical Value ({key}): {value:.4f}")
    if adf_result[1] < 0.05:
        st.success("✅ Series is stationary (reject H₀)")
    else:
        st.error("❌ Series is non-stationary (fail to reject H₀)")

    # ACF Plot (lags=5)
    from statsmodels.graphics.tsaplots import plot_acf
    st.markdown("### 🔁 Autocorrelation (ACF)")
    fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
    plot_acf(ts_data.dropna(), lags=5, ax=ax_acf)
    ax_acf.set_title(f"Autocorrelation (ACF) - {selected_series_name} for participant {selected_pid}")
    ax_acf.grid(True)
    fig_acf.tight_layout()
    st.pyplot(fig_acf)

    # Differencing only for time_to_event
    if selected_series_name == "time_to_event":
        st.markdown("### 🛠 Differencing to Achieve Stationarity")
        diff_ts = ts_data.diff().dropna()
        fig_diff, ax_diff = plt.subplots(figsize=(8, 3))
        ax_diff.plot(diff_ts, color="mediumseagreen")
        ax_diff.set_title(f"First Difference of {selected_series_name} for Participant {selected_pid}")
        ax_diff.set_ylabel("Differenced Value")
        ax_diff.set_xlabel("Date")
        ax_diff.grid(True)
        ax_diff.tick_params(axis='x', rotation=45)
        fig_diff.tight_layout()
        st.pyplot(fig_diff)


# 🔮 انتخاب مدل پیش‌بینی (فقط لیست کشویی)
st.subheader("🔮 Select Forecasting Model")
model_type = st.selectbox("Choose a model type:", ["ARIMA", "ARIMAX"])
st.info(f"You selected: **{model_type}** model.")


# 📍 Granger Causality Test: Only when ARIMAX is selected
if model_type == "ARIMAX":
    st.subheader("📍 Granger Causality Test (for ARIMAX)")
    st.markdown("""
If ARIMAX model is selected, we check if blood pressure (systolic and diastolic)
granger-causes `time_to_event` using maxlag = 5.
""")
    df_granger = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
    df_granger['diff_time_to_event'] = df_granger['time_to_event'].diff()
    df_granger.dropna(inplace=True)

    from statsmodels.tsa.stattools import grangercausalitytests

    st.markdown("**Granger test: systolic → time_to_event**")
    grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5, verbose=True)

    st.markdown("**Granger test: diastolic → time_to_event**")
    grangercausalitytests(df_granger[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5, verbose=True)

    st.success("✅ Based on the Granger test, both systolic and diastolic pressure have a causal effect on time_to_event.")

    st.subheader("📈 ARIMAX(1,1,1) Forecasting")

    confirm_arimax_model = st.selectbox("Confirm ARIMAX model with BP + lag-1 as exogenous variables?", ["No", "Yes"])

    if confirm_arimax_model == "Yes":
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # ساخت lag-1
        participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
        participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

        # تعریف y و X
        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                              'lag_1_systolic', 'lag_1_diastolic']]

        # حذف NaNها
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic',
                      'lag_1_systolic', 'lag_1_diastolic']]

        # تقسیم به train و test
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        # ساخت مدل ARIMAX(1,1,1)
        model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model.fit()

        # پیش‌بینی
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        # ارزیابی
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(1,1,1) Evaluation")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        # رسم نمودار
        import matplotlib.pyplot as plt
        fig_forecast, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(1,1,1) Forecast vs Actual")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast)

    st.subheader("📈 ARIMAX(3,1,0) Forecasting")

    confirm_arimax_310 = st.selectbox("Run ARIMAX(3,1,0) model?", ["No", "Yes"], key="arimax_310")

    if confirm_arimax_310 == "Yes":
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # ساخت lag-1
        participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
        participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

        # تعریف y و X
        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                              'lag_1_systolic', 'lag_1_diastolic']]

        # حذف NaNها
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic',
                      'lag_1_systolic', 'lag_1_diastolic']]

        # تقسیم به train و test
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        # ساخت مدل ARIMAX(3,1,0)
        model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
        model_fit = model.fit()

        # پیش‌بینی
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        # ارزیابی
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(3,1,0) Evaluation")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        # رسم نمودار
        import matplotlib.pyplot as plt
        fig_forecast_310, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(3,1,0) Forecast vs Actual")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_310)

    st.subheader("📈 ARIMAX(1,1,1) Forecasting (No Lag Variables)")

    confirm_arimax_111_nolag = st.selectbox("Run ARIMAX(1,1,1) without lag variables?", ["No", "Yes"], key="arimax_111_nolag")

    if confirm_arimax_111_nolag == "Yes":
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # فقط متغیرهای فشار خون بدون lag
        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']]

        # حذف NaNها
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic']]

        # تقسیم به train و test
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        # ساخت مدل ARIMAX(1,1,1)
        model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model.fit()

        # پیش‌بینی
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        # ارزیابی
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(1,1,1) Evaluation (No Lags)")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        # رسم نمودار
        import matplotlib.pyplot as plt
        fig_forecast_nolag, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(1,1,1) Forecast vs Actual (No Lags)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_nolag)

    st.subheader("📈 ARIMAX(3,1,0) Forecasting (No Lag Variables)")

    confirm_arimax_310_nolag = st.selectbox("Run ARIMAX(3,1,0) without lag variables?", ["No", "Yes"], key="arimax_310_nolag")

    if confirm_arimax_310_nolag == "Yes":
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # فقط متغیرهای فشار خون بدون lag
        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']]

        # حذف NaNها
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic']]

        # تقسیم به train و test
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        # ساخت مدل ARIMAX(3,1,0)
        model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
        model_fit = model.fit()

        # پیش‌بینی
        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        # ارزیابی
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(3,1,0) Evaluation (No Lags)")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        # رسم نمودار
        import matplotlib.pyplot as plt
        fig_forecast_nolag_310, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(3,1,0) Forecast vs Actual (No Lags)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_nolag_310)



    st.subheader("📈 ARIMAX(1,1,1) Forecasting (with PCA Component)")

    confirm_arimax_111_pca = st.selectbox("Run ARIMAX(1,1,1) with PCA Component?", ["No", "Yes"], key="arimax_111_pca")

    if confirm_arimax_111_pca == "Yes":
        st.markdown("""
        🔑 **Explanation:** Since there was high multicollinearity between systolic and diastolic blood pressure (checked with VIF),
        they were combined into a single principal component (PC1) using PCA, then its lag-1 was used as the exogenous variable.
        """)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # VIF calculation
        X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_bp.columns
        vif_data['VIF'] = [variance_inflation_factor(X_bp.values, i) for i in range(X_bp.shape[1])]
        st.write("### 📊 VIF Check for Systolic and Diastolic BP")
        st.dataframe(vif_data)

        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bp)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled)
        participant_data.loc[X_bp.index, 'pc1'] = pc1
        participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)

        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['pc1_lag1']]

        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['pc1_lag1']]

        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(1,1,1) with PCA Evaluation")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        import matplotlib.pyplot as plt
        fig_forecast_pca, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(1,1,1) Forecast vs Actual (with PCA Component)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_pca)



    st.subheader("📈 ARIMAX(3,1,0) Forecasting (with PCA Component)")

    confirm_arimax_310_pca = st.selectbox("Run ARIMAX(3,1,0) with PCA Component?", ["No", "Yes"], key="arimax_310_pca")

    if confirm_arimax_310_pca == "Yes":
        st.markdown("""
        🔑 **Explanation:** Due to high multicollinearity between systolic and diastolic blood pressure (checked by VIF),
        they were combined into one principal component (PC1) using PCA, and its lag-1 was used as the exogenous variable
        for an ARIMAX(3,1,0) model.
        """)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # VIF calculation
        X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_bp.columns
        vif_data['VIF'] = [variance_inflation_factor(X_bp.values, i) for i in range(X_bp.shape[1])]
        st.write("### 📊 VIF Check for Systolic and Diastolic BP")
        st.dataframe(vif_data)

        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bp)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled)
        participant_data.loc[X_bp.index, 'pc1'] = pc1
        participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)

        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['pc1_lag1']]

        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['pc1_lag1']]

        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(3,1,0) with PCA Evaluation")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        import matplotlib.pyplot as plt
        fig_forecast_pca_310, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(3,1,0) Forecast vs Actual (with PCA Component)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_pca_310)



    st.subheader("📈 ARIMAX(1,1,1) Forecasting (with PCA, No Lag)")

    confirm_arimax_111_pca_nolag = st.selectbox("Run ARIMAX(1,1,1) with PCA Component (No Lag)?", ["No", "Yes"], key="arimax_111_pca_nolag")

    if confirm_arimax_111_pca_nolag == "Yes":
        st.markdown("""
        🔑 **Explanation:** Due to high multicollinearity, systolic and diastolic blood pressure were combined into one principal component (PC1) using PCA.
        This time, no lag is used — the raw PC1 is used directly as exogenous variable for an ARIMAX(1,1,1) model.
        """)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # VIF calculation
        X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_bp.columns
        vif_data['VIF'] = [variance_inflation_factor(X_bp.values, i) for i in range(X_bp.shape[1])]
        st.write("### 📊 VIF Check for Systolic and Diastolic BP")
        st.dataframe(vif_data)

        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bp)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled)
        participant_data.loc[X_bp.index, 'pc1'] = pc1

        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['pc1']]

        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['pc1']]

        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(1,1,1) with PCA (No Lag) Evaluation")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        import matplotlib.pyplot as plt
        fig_forecast_pca_111_nolag, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(1,1,1) Forecast vs Actual (with PCA, No Lag)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_pca_111_nolag)



    st.subheader("📈 ARIMAX(3,1,0) Forecasting (with PCA, No Lag)")

    confirm_arimax_310_pca_nolag = st.selectbox("Run ARIMAX(3,1,0) with PCA Component (No Lag)?", ["No", "Yes"], key="arimax_310_pca_nolag")

    if confirm_arimax_310_pca_nolag == "Yes":
        st.markdown("""
        🔑 **Explanation:** Due to high multicollinearity, systolic and diastolic blood pressure were combined into one principal component (PC1) using PCA.
        This time, no lag is used — the raw PC1 is used directly as exogenous variable for an ARIMAX(3,1,0) model.
        """)

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        # VIF calculation
        X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_bp.columns
        vif_data['VIF'] = [variance_inflation_factor(X_bp.values, i) for i in range(X_bp.shape[1])]
        st.write("### 📊 VIF Check for Systolic and Diastolic BP")
        st.dataframe(vif_data)

        # PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_bp)
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled)
        participant_data.loc[X_bp.index, 'pc1'] = pc1

        y = participant_data['time_to_event'].astype(float)
        X = participant_data[['pc1']]

        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['pc1']]

        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.success("📊 ARIMAX(3,1,0) with PCA (No Lag) Evaluation")
        st.markdown(f"**MAE**: {mae:.4f}")
        st.markdown(f"**MSE**: {mse:.5f}")
        st.markdown(f"**RSS**: {rss:.5f}")

        import matplotlib.pyplot as plt
        fig_forecast_pca_310_nolag, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMAX(3,1,0) Forecast vs Actual (with PCA, No Lag)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_forecast_pca_310_nolag)



# ✅ ARIMA(1,1,1) Forecasting (No Exogenous Variables)




if model_type == "ARIMA":
    st.subheader("📈 ARIMA Forecasting")
    arima_option = st.selectbox("Choose ARIMA configuration:", ["Select an option", "ARIMA(1,1,1)", "ARIMA(3,1,0)"])

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

    elif arima_option == "ARIMA(3,1,0)":
        y = participant_data['time_to_event'].dropna().astype(float)
        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        model = ARIMA(y_train, order=(3, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(y_test))
        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))
        st.success("📊 ARIMA(3,1,0) Evaluation")
        st.markdown(f"**MAE:** {mae:.4f}")
        st.markdown(f"**MSE:** {mse:.5f}")
        st.markdown(f"**RSS:** {rss:.5f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test.index, y_test, label="Actual")
        ax.plot(y_test.index, forecast, label="Forecast", linestyle="--")
        ax.set_title("ARIMA(3,1,0) Forecast vs Actual")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
