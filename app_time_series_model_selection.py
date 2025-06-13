
st.subheader("🤖 Model Selection and Forecasting")

model_type = st.selectbox("Select model type:", ["ARIMA", "ARIMAX (1,1,1)"])

if selected_pid:
    participant_data = grouped.get_group(selected_pid).sort_values('date').set_index('date')
    participant_data = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
    participant_data['time_to_event'] = participant_data['time_to_event'].astype(float)

    if model_type == "ARIMAX (1,1,1)":
        st.markdown("### 🔍 Granger Causality Test")

        from statsmodels.tsa.stattools import grangercausalitytests
        granger_df = participant_data[['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic']].copy()
        granger_df['diff_time_to_event'] = granger_df['time_to_event'].diff()
        granger_df.dropna(inplace=True)

        st.write("Granger causality: systolic → time_to_event")
        grangercausalitytests(granger_df[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5, verbose=True)

        st.write("Granger causality: diastolic → time_to_event")
        grangercausalitytests(granger_df[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5, verbose=True)

        st.markdown("### 🧠 ARIMAX(1,1,1) Forecasting")

        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np

        participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
        participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

        y = participant_data['time_to_event']
        X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                              'lag_1_systolic', 'lag_1_diastolic']]
        combined = pd.concat([y, X], axis=1).dropna()
        y = combined['time_to_event']
        X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic',
                      'lag_1_systolic', 'lag_1_diastolic']]

        split_idx = int(len(y) * 0.8)
        y_train, y_test = y[:split_idx], y[split_idx:]
        X_train, X_test = X[:split_idx], X[split_idx:]

        model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

        mae = mean_absolute_error(y_test, forecast)
        mse = mean_squared_error(y_test, forecast)
        rss = np.sum(np.square(y_test - forecast))

        st.markdown("#### 📊 ARIMAX Model Evaluation")
        st.write(f"**MAE**: {mae:.4f}")
        st.write(f"**MSE**: {mse:.5f}")
        st.write(f"**RSS**: {rss:.5f}")

        fig_pred, ax_pred = plt.subplots(figsize=(8, 3))
        ax_pred.plot(y_test.index, y_test, label='Actual')
        ax_pred.plot(y_test.index, forecast, label='Forecast', linestyle='--')
        ax_pred.set_title('ARIMAX Forecast vs Actual')
        ax_pred.legend()
        ax_pred.grid(True)
        fig_pred.tight_layout()
        st.pyplot(fig_pred)
