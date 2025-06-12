
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Streamlit Light App", layout="wide")
st.title("📊 نسخه سبک اپلیکیشن Streamlit")

st.write("✅ این نسخه بدون پکیج‌های سنگین اجرا شده است.")

# --- Cleaned code from notebook (no sklearn, tensorflow, etc.) ---
import pandas as pd

df = pd.read_csv("dataset_to_nonhyper.csv")
df

if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

df['date'] = pd.to_datetime(df['date'])

# Sort by participant and date
df = df.sort_values(by=['participant_id', 'date'])
df

# check for resonal variable
print(df[['blood_pressure_systolic', 'blood_pressure_diastolic']].describe())

original_shape = df.shape

#check for missing value
df = df.dropna(subset=['time_to_event', 'blood_pressure_systolic', 'blood_pressure_diastolic'])

print("Original shape:", original_shape)
print("New shape after dropna:", df.shape)

print("Unique participants:", df['participant_id'].nunique())

# 1. میانگین‌گیری روزانه فشار سیستولیک و دیاستولیک
sd_df = df.groupby('date')[['blood_pressure_systolic', 'blood_pressure_diastolic']].mean()

# 2. هموارسازی با میانگین متحرک ۷ روزه
sd_df_smooth = sd_df.rolling(window=7).mean()

# 3. رسم نمودار اصلی و نمودار هموارشده در کنار هم
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

sd_df.plot(ax=axes[0], title='Original Blood Pressure (Daily Average)')
sd_df_smooth.plot(ax=axes[1], title='Smoothed Blood Pressure (7-day MA)')

plt.tight_layout()
plt.show()

grouped = df.groupby('participant_id')
participant_series = {
    pid: group.set_index('date')['time_to_event'].sort_index()
    for pid, group in grouped
    if len(group) >= 12
}

print("Participants with ≥ 10 visits:", len(participant_series))

sample_pid = list(participant_series.keys())[0]
sample_series = participant_series[sample_pid]
print(f"\nSample time series for participant {sample_pid}:\n")
print(sample_series.head())

import matplotlib.pyplot as plt

sample_ids = list(participant_series.keys())[:3]

plt.figure(figsize=(16, 6))  # فضای افقی بیشتر برای برچسب‌ها

for pid in sample_ids:
    series = participant_series[pid]
    plt.plot(series.index, series.values, marker='o', label=f'Participant {pid}')

plt.title("Time to Event Countdown for Sample Participants")
plt.xlabel("Date")
plt.ylabel("Time to Event")

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)


# دیکشنری برای نگهداری تاریخ رویداد هر فرد
event_dates = {}

for pid, series in participant_series.items():
    zero_event = series[series == 0]
    if not zero_event.empty:
        event_dates[pid] = zero_event.index[0]  # اولین تاریخ که time_to_event = 0 شده

bp_by_person = {}

for pid, event_date in event_dates.items():
    sub_df = df[df['participant_id'] == pid].copy()
    sub_df['date'] = pd.to_datetime(sub_df['date'])
    sub_df = sub_df.sort_values('date')

    # انتخاب داده‌های فشار خون از 14 روز قبل تا روز event
    bp_window = sub_df[(sub_df['date'] <= event_date) &
                       (sub_df['date'] >= event_date - pd.Timedelta(days=14))]

    bp_by_person[pid] = bp_window[['date', 'blood_pressure_systolic', 'blood_pressure_diastolic']]

import matplotlib.pyplot as plt

for pid, bp_data in list(bp_by_person.items())[:3]:  # فقط برای 3 نفر اول
    plt.figure(figsize=(10, 4))
    plt.plot(bp_data['date'], bp_data['blood_pressure_systolic'], label='Systolic', marker='o')
    plt.plot(bp_data['date'], bp_data['blood_pressure_diastolic'], label='Diastolic', marker='x')

    plt.axvline(event_dates[pid], color='red', linestyle='--', label='Event (TTE = 0)')
    plt.title(f'Blood Pressure Before Event - Participant {pid}')
    plt.xlabel('Date')
    plt.ylabel('Blood Pressure')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# for example choose the first one and give its data from the main dataset and provide the date as a index for time series analysis.
sample_pid = sample_ids[0]
participant_data = df[df['participant_id'] == sample_pid].copy()
participant_data = participant_data.sort_values(by='date')
participant_data.set_index('date', inplace=True)

# Rolling averages
# 7-day rolling averages for deeper trend smoothing
participant_data['rolling_mean_7_systolic'] = participant_data['blood_pressure_systolic'].rolling(window=7).mean()
participant_data['rolling_mean_7_diastolic'] = participant_data['blood_pressure_diastolic'].rolling(window=7).mean()

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# نمودار اصلی فشار خون سیستولیک
plt.plot(participant_data.index, participant_data['blood_pressure_systolic'], label='Original Systolic', alpha=0.4)

# نمودار هموارشده با میانگین ۷ روزه
plt.plot(participant_data.index, participant_data['rolling_mean_7_systolic'], label='Smoothed (7-day MA)', linewidth=2)

plt.title('Systolic Blood Pressure: Original vs Smoothed')
plt.xlabel('Date')
plt.ylabel('Systolic BP (mmHg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

# نمودار اصلی فشار خون دیاستولیک
plt.plot(participant_data.index, participant_data['blood_pressure_diastolic'], label='Original Diastolic', alpha=0.4)

# نمودار هموارشده با میانگین ۷ روزه
plt.plot(participant_data.index, participant_data['rolling_mean_7_diastolic'], label='Smoothed (7-day MA)', linewidth=2)

plt.title('Diastolic Blood Pressure: Original vs Smoothed')
plt.xlabel('Date')
plt.ylabel('Diastolic BP (mmHg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

sample_pid = sample_ids[0]
participant_data = df[df['participant_id'] == sample_pid].copy()
participant_data = participant_data.sort_values(by='date')
participant_data.set_index('date', inplace=True)

# Rolling averages
# 7-day rolling averages for deeper trend smoothing
participant_data['rolling_mean_7_systolic'] = participant_data['blood_pressure_systolic'].rolling(window=7).mean()
participant_data['rolling_mean_7_diastolic'] = participant_data['blood_pressure_diastolic'].rolling(window=7).mean()

# Applying a 7-day moving average to align noise and indicate the real trends
# Use the columns with the rolling averages from participant_data
participant_data[['rolling_mean_7_systolic', 'rolling_mean_7_diastolic']].plot(figsize=(12, 5), title='7-Day Moving Average of Blood Pressure')
plt.ylabel('mmHg')
plt.grid(True)
plt.show()

# Use the columns with the rolling averages from participant_data which we used in for moving average
# Store the rolling average data in a variable
rolling = participant_data['rolling_mean_7_systolic']
# Now calculate residuals using participant_data and the rolling average
# Note: This calculates residuals for a single participant (sample_pid)
residuals = participant_data['blood_pressure_systolic'] - rolling
threshold = 1.5 * residuals.std()
anomalies = residuals[abs(residuals) > threshold]
plt.figure(figsize=(12, 5))
# Plot the original systolic BP for the participant
plt.plot(participant_data.index, participant_data['blood_pressure_systolic'], label='blood_pressure_systolic')
# Scatter plot anomalies
plt.scatter(anomalies.index, participant_data.loc[anomalies.index]['blood_pressure_systolic'], color='red', label='Anomalies')
plt.title(f'Anomalies in Systolic BP (based on moving average) - Participant {sample_pid}')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# فرض: participant_id های زیادی داری و می‌خوای اولی رو بررسی کنی
sample_pid = df['participant_id'].unique()[0]

# استخراج داده‌های مربوط به آن شرکت‌کننده
participant_data = df[df['participant_id'] == sample_pid].sort_values('date')
participant_data = participant_data.set_index('date')

# انتخاب سری زمانی مورد بررسی: time_to_event
ts = participant_data['blood_pressure_systolic']

# اجرای آزمون دیکی فولر
adf_result = adfuller(ts.dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")

# تفسیر نتیجه
if adf_result[1] < 0.05:
    print("\n✅ Series is stationary (reject H0)")
else:
    print("\n❌ Series is non-stationary (fail to reject H0)")

import matplotlib.pyplot as plt

# سری زمانی time_to_event
ts = participant_data['blood_pressure_systolic']

# رسم نمودار ACF
plt.figure(figsize=(10, 5))
plot_acf(ts.dropna(), lags=5)  # بررسی تا 30 lag
plt.title(f"Autocorrelation (ACF) - blood_pressure_systolic for participant {sample_pid}")
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# فرض: participant_id های زیادی داری و می‌خوای اولی رو بررسی کنی
sample_pid = df['participant_id'].unique()[0]

# استخراج داده‌های مربوط به آن شرکت‌کننده
participant_data = df[df['participant_id'] == sample_pid].sort_values('date')
participant_data = participant_data.set_index('date')

# انتخاب سری زمانی مورد بررسی: time_to_event
ts = participant_data['blood_pressure_diastolic']

# اجرای آزمون دیکی فولر
adf_result = adfuller(ts.dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")

# تفسیر نتیجه
if adf_result[1] < 0.05:
    print("\n✅ Series is stationary (reject H0)")
else:
    print("\n❌ Series is non-stationary (fail to reject H0)")

import matplotlib.pyplot as plt

# سری زمانی time_to_event
ts = participant_data['blood_pressure_diastolic']

# رسم نمودار ACF
plt.figure(figsize=(10, 5))
plot_acf(ts.dropna(), lags=5)  # بررسی تا 30 lag
plt.title(f"Autocorrelation (ACF) - blood_pressure_diastolic for participant {sample_pid}")
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# فرض: participant_id های زیادی داری و می‌خوای اولی رو بررسی کنی
sample_pid = df['participant_id'].unique()[0]

# استخراج داده‌های مربوط به آن شرکت‌کننده
participant_data = df[df['participant_id'] == sample_pid].sort_values('date')
participant_data = participant_data.set_index('date')

# انتخاب سری زمانی مورد بررسی: time_to_event
ts = participant_data['time_to_event']

# اجرای آزمون دیکی فولر
adf_result = adfuller(ts.dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value}")

# تفسیر نتیجه
if adf_result[1] < 0.05:
    print("\n✅ Series is stationary (reject H0)")
else:
    print("\n❌ Series is non-stationary (fail to reject H0)")

import matplotlib.pyplot as plt

# سری زمانی time_to_event
ts = participant_data['time_to_event']

# رسم نمودار ACF
plt.figure(figsize=(10, 5))
plot_acf(ts.dropna(), lags=5)  # بررسی تا 30 lag
plt.title(f"Autocorrelation (ACF) - time_to_event for participant {sample_pid}")
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd

# فرض: df شامل ستون‌های ['time_to_event', 'systolic', 'diastolic']

# ایستاسازی time_to_event
df['diff_time_to_event'] = df['time_to_event'].diff()
df.dropna(inplace=True)

# تست گرنجر بین systolic → time_to_event
print("Granger test: systolic → time_to_event")
grangercausalitytests(df[['diff_time_to_event', 'blood_pressure_systolic']], maxlag=5)

# تست گرنجر بین diastolic → time_to_event
print("Granger test: diastolic → time_to_event")
grangercausalitytests(df[['diff_time_to_event', 'blood_pressure_diastolic']], maxlag=5)

import matplotlib.pyplot as plt

# انتخاب یک participant (مثلاً اولین نفر در دیتافریم)
sample_pid = df['participant_id'].unique()[0]

# فیلتر داده‌ها برای آن فرد خاص
participant_data = df[df['participant_id'] == sample_pid].sort_values('date')
participant_data = participant_data.set_index('date')

# حذف مقادیر گمشده برای دقت بیشتر
systolic = participant_data['blood_pressure_systolic'].dropna()
diastolic = participant_data['blood_pressure_diastolic'].dropna()

# PACF برای فشار خون سیستولیک
plt.figure(figsize=(10, 4))
plot_pacf(systolic, lags=3)
plt.title(f'PACF – Systolic BP for Participant {sample_pid}')
plt.grid(True)
plt.show()

# PACF برای فشار خون دیاستولیک
plt.figure(figsize=(10, 4))
plot_pacf(diastolic, lags=3)
plt.title(f'PACF – Diastolic BP for Participant {sample_pid}')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# انتخاب فرد خاص (مثلاً اولین نفر در دیتافریم)
sample_pid = df['participant_id'].unique()[0]

# فیلتر داده‌ها برای همان فرد
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# تفاضل‌گیری (برای ایستا کردن)
participant_data['diff_time_to_event'] = participant_data['time_to_event'].diff()

# حذف NaNهای حاصل از تفاضل‌گیری
ts_diff = participant_data['diff_time_to_event'].dropna()

# رسم نمودار ACF برای تعیین q
plt.figure(figsize=(10, 4))
plot_acf(ts_diff, lags=5)
plt.title(f"ACF – time_to_event (Differenced) for Participant {sample_pid}")
plt.grid(True)
plt.show()

# رسم نمودار PACF برای تعیین p
plt.figure(figsize=(10, 4))
plot_pacf(ts_diff, lags=3)
plt.title(f"PACF – time_to_event (Differenced) for Participant {sample_pid}")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# انتخاب یک شرکت‌کننده خاص
sample_pid = df['participant_id'].unique()[0]

# فیلتر داده‌ها فقط برای آن فرد
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# ایجاد lag-1 برای فشار خون
participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

# متغیر وابسته
y = participant_data['time_to_event'].astype(float)

# متغیرهای توضیحی شامل مقدار اصلی و lag-1
X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                      'lag_1_systolic', 'lag_1_diastolic']]

# حذف ردیف‌هایی که NaN دارند (به دلیل shift)
combined = pd.concat([y, X], axis=1).dropna()
y = combined['time_to_event']
X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic',
              'lag_1_systolic', 'lag_1_diastolic']]

# تقسیم داده به train و test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# ساخت مدل ARIMAX با پارامترهای دلخواه
model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
model_fit = model.fit()

# پیش‌بینی
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# ارزیابی مدل
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum(np.square(y_test - forecast))

print("📊 ARIMAX Model Evaluation (Main + Lagged BP):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# انتخاب participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# ایجاد lag-1 برای فشار خون
participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

# متغیر هدف
y = participant_data['time_to_event'].astype(float)

# متغیرهای توضیحی: اصلی و lag-1
X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                      'lag_1_systolic', 'lag_1_diastolic']]

# ترکیب و حذف NaNها
combined = pd.concat([y, X], axis=1).dropna()
y = combined['time_to_event']
X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic', 'lag_1_systolic', 'lag_1_diastolic']]

# محدوده p، d، q
p_range = range(0, 5)
d_range = [1]
q_range = range(0, 5)

best_aic = np.inf
best_order = None
results = []

# Grid Search روی (p,d,q)
for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(endog=y, exog=X, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                results.append((p, d, q, aic))
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue

# نتایج
print(f"\n✅ Best ARIMAX Order: {best_order} with AIC = {best_aic:.2f}")
result_df = pd.DataFrame(results, columns=['p', 'd', 'q', 'AIC']).sort_values('AIC')
display(result_df)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# انتخاب یک شرکت‌کننده خاص
sample_pid = df['participant_id'].unique()[0]

# فیلتر داده‌ها فقط برای آن فرد
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# ایجاد lag-1 برای فشار خون
participant_data['lag_1_systolic'] = participant_data['blood_pressure_systolic'].shift(1)
participant_data['lag_1_diastolic'] = participant_data['blood_pressure_diastolic'].shift(1)

# متغیر وابسته
y = participant_data['time_to_event'].astype(float)

# متغیرهای توضیحی شامل مقدار اصلی و lag-1
X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic',
                      'lag_1_systolic', 'lag_1_diastolic']]

# حذف ردیف‌هایی که NaN دارند (به دلیل shift)
combined = pd.concat([y, X], axis=1).dropna()
y = combined['time_to_event']
X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic',
              'lag_1_systolic', 'lag_1_diastolic']]

# تقسیم داده به train و test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# ساخت مدل ARIMAX با پارامترهای دلخواه
model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
model_fit = model.fit()

# پیش‌بینی
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# ارزیابی مدل
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum(np.square(y_test - forecast))

print("📊 ARIMAX Model Evaluation (Main + Lagged BP):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# متغیر هدف
y = participant_data['time_to_event'].astype(float)

# فقط خود فشار خون‌ها (بدون lag)
X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']]

# حذف مقادیر NaN (در صورتی که وجود داشته باشد)
combined = pd.concat([y, X], axis=1).dropna()
y = combined['time_to_event']
X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic']]

# تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# ساخت مدل ARIMAX (بدون lagها)
model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
model_fit = model.fit()

# پیش‌بینی
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# ارزیابی
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum(np.square(y_test - forecast))

print("📊 ARIMAX بدون lag:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# انتخاب participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# متغیر هدف
y = participant_data['time_to_event'].astype(float)

# فقط متغیرهای فشار خون بدون lag
X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']]

# حذف NaN
combined = pd.concat([y, X], axis=1).dropna()
y = combined['time_to_event']
X = combined[['blood_pressure_systolic', 'blood_pressure_diastolic']]

# تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# محدوده برای Grid Search
p_range = range(0, 4)
d_range = [1]  # چون می‌دونیم سری بعد از یک تفاضل ایستا شده
q_range = range(0, 4)

best_aic = np.inf
best_order = None
results = []

# Grid Search
for p in p_range:
    for d in d_range:
        for q in q_range:
            try:
                model = ARIMA(endog=y_train, exog=X_train, order=(p, d, q))
                model_fit = model.fit()
                aic = model_fit.aic
                results.append((p, d, q, aic))
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue

# نمایش بهترین مدل
print(f"✅ Best ARIMAX order (no lag): {best_order} with AIC = {best_aic:.2f}")

# ارزیابی نهایی مدل
best_model = ARIMA(endog=y_train, exog=X_train, order=best_order)
best_model_fit = best_model.fit()

forecast = best_model_fit.forecast(steps=len(y_test), exog=X_test)

mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum(np.square(y_test - forecast))

print("\n📊 Evaluation of Best ARIMAX Model (no lag):")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# متغیر هدف فقط time_to_event
y = participant_data['time_to_event'].dropna().astype(float)

# تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]

# ساخت مدل ARIMA با وقفه (1,1,1)
model = ARIMA(y_train, order=(1, 1, 1))
model_fit = model.fit()

# پیش‌بینی
forecast = model_fit.forecast(steps=len(y_test))

# ارزیابی
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum(np.square(y_test - forecast))

print("📊 ARIMA(1,1,1) فقط با time_to_event:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# متغیر هدف فقط time_to_event
y = participant_data['time_to_event'].dropna().astype(float)

# تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]

# ساخت مدل ARIMA با وقفه (1,1,1)
model = ARIMA(y_train, order=(3, 1, 0))
model_fit = model.fit()

# پیش‌بینی
forecast = model_fit.forecast(steps=len(y_test))

# ارزیابی
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum(np.square(y_test - forecast))

print("📊 ARIMA(3,1,0) فقط با time_to_event:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd

# انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# انتخاب دو ستون فشار خون و حذف NaN (برای محاسبه VIF)
X = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()

# محاسبه VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# نمایش نتایج
print("📊 بررسی هم‌خطی بین فشار خون سیستولیک و دیاستولیک برای participant:", sample_pid)
print(vif_data)


import pandas as pd

# انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# انتخاب فقط دو ستون فشار خون و حذف NaN
X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()

# مقیاس‌گذاری (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bp)

# اعمال PCA (فقط مؤلفه اول)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# تبدیل به DataFrame با حفظ تاریخ
X_pca_df = pd.DataFrame(X_pca, index=X_bp.index, columns=['pc1'])

# اضافه کردن pc1 به دیتافریم participant
participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']

# نمایش چند سطر اول برای بررسی
print(participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic', 'pc1']].head())


print(f"Explained variance ratio by PC1: {pca.explained_variance_ratio_[0]:.4f}")


# اضافه کردن به دیتافریم اصلی (در صورت نیاز)
participant_data['pc1'] = X_pca_df


import matplotlib.pyplot as plt

# انتخاب سری زمانی
ts = participant_data['pc1'].dropna()

# اجرای آزمون ADF
adf_result = adfuller(ts)
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

# تفسیر نتیجه
if adf_result[1] < 0.05:
    print("\n✅ Series is stationary (reject H0)")
else:
    print("\n❌ Series is non-stationary (fail to reject H0)")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. انتخاب participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# 2. اجرای PCA روی فشار خون‌ها
X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bp)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# اضافه کردن pc1 به participant_data
X_pca_df = pd.DataFrame(X_pca, index=X_bp.index, columns=['pc1'])
participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']

# 3. اضافه کردن lag-1 از pc1
participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)

# 4. ساخت دیتای نهایی
data = participant_data[['time_to_event', 'pc1_lag1']].dropna()
y = data['time_to_event']
X = data[['pc1_lag1']]

# 5. تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# 6. ساخت و آموزش مدل ARIMAX(1,1,1)
model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
model_fit = model.fit()

# 7. پیش‌بینی روی داده تست
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# 8. ارزیابی مدل
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum((y_test - forecast) ** 2)

print(f"\n📊 ARIMAX(1,1,1) با pc1_lag1 برای participant {sample_pid}:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. انتخاب participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# 2. اجرای PCA روی فشار خون‌ها
X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bp)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# اضافه کردن pc1 به participant_data
X_pca_df = pd.DataFrame(X_pca, index=X_bp.index, columns=['pc1'])
participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']

# 3. اضافه کردن lag-1 از pc1
participant_data['pc1_lag1'] = participant_data['pc1'].shift(1)

# 4. ساخت دیتای نهایی
data = participant_data[['time_to_event', 'pc1_lag1']].dropna()
y = data['time_to_event']
X = data[['pc1_lag1']]

# 5. تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# 6. ساخت و آموزش مدل ARIMAX(1,1,1)
model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
model_fit = model.fit()

# 7. پیش‌بینی روی داده تست
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# 8. ارزیابی مدل
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum((y_test - forecast) ** 2)

print(f"\n📊 ARIMAX(3,1,0) با pc1_lag1 برای participant {sample_pid}:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# 2. اجرای PCA روی فشار خون‌ها
X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bp)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# اضافه کردن pc1 به participant_data
X_pca_df = pd.DataFrame(X_pca, index=X_bp.index, columns=['pc1'])
participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']

# 3. ساخت دیتای نهایی بدون lag
data = participant_data[['time_to_event', 'pc1']].dropna()
y = data['time_to_event']
X = data[['pc1']]

# 4. تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# 5. ساخت و آموزش مدل ARIMAX(1,1,1)
model = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
model_fit = model.fit()

# 6. پیش‌بینی روی داده تست
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# 7. ارزیابی مدل
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum((y_test - forecast) ** 2)

print(f"\n📊 ARIMAX(1,1,1) با pc1 بدون lag برای participant {sample_pid}:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. انتخاب یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# 2. اجرای PCA روی فشار خون‌ها
X_bp = participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bp)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)

# اضافه کردن pc1 به participant_data
X_pca_df = pd.DataFrame(X_pca, index=X_bp.index, columns=['pc1'])
participant_data.loc[X_pca_df.index, 'pc1'] = X_pca_df['pc1']

# 3. ساخت دیتای نهایی بدون lag
data = participant_data[['time_to_event', 'pc1']].dropna()
y = data['time_to_event']
X = data[['pc1']]

# 4. تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]
X_train, X_test = X[:split_idx], X[split_idx:]

# 5. ساخت و آموزش مدل ARIMAX(1,1,1)
model = ARIMA(endog=y_train, exog=X_train, order=(3, 1, 0))
model_fit = model.fit()

# 6. پیش‌بینی روی داده تست
forecast = model_fit.forecast(steps=len(y_test), exog=X_test)

# 7. ارزیابی مدل
mae = mean_absolute_error(y_test, forecast)
mse = mean_squared_error(y_test, forecast)
rss = np.sum((y_test - forecast) ** 2)

print(f"\n📊 ARIMAX(3,1,0) با pc1 بدون lag برای participant {sample_pid}:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.5f}")
print(f"RSS: {rss:.5f}")




import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import random

# 🔹 1. تنظیم seed برای reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 🔹 2. انتخاب فقط یک participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# 🔹 3. استخراج سری زمانی time_to_event
ts = participant_data['time_to_event'].dropna().sort_index()

# 🔹 4. نرمال‌سازی داده‌ها با MinMaxScaler در بازه [0,1]
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

# 🔹 5. ساخت دنباله‌های ورودی/خروجی برای RNN
def create_sequences(data, seq_len=2):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_length = 2
X_all, y_all = create_sequences(ts_scaled, seq_length)
X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))  # (samples, timesteps, features)

# 🔹 6. تقسیم به train و test
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

# 🔹 7. ساخت و آموزش مدل RNN
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0)

# 🔹 8. پیش‌بینی و بازگرداندن مقیاس اصلی
rnn_pred_scaled = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test)
rnn_pred_inv = scaler.inverse_transform(rnn_pred_scaled)

# 🔹 9. ارزیابی عملکرد مدل
rnn_mae = mean_absolute_error(y_test_inv, rnn_pred_inv)
rnn_mse = mean_squared_error(y_test_inv, rnn_pred_inv)
rnn_rss = np.sum((y_test_inv - rnn_pred_inv) ** 2)

print(f"📊 نتایج RNN برای participant {sample_pid}:")
print(f"MAE: {rnn_mae:.2f}")
print(f"MSE: {rnn_mse:.2f}")
print(f"RSS: {rnn_rss:.2f}")

import numpy as np
import pandas as pd

# 1. انتخاب همان participant خاص
sample_pid = df['participant_id'].unique()[0]
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')

# 2. تعریف سری زمانی time_to_event (بدون NaN)
y = participant_data['time_to_event'].dropna().astype(float)

# 3. تقسیم به train/test
split_idx = int(len(y) * 0.8)
y_train, y_test = y[:split_idx], y[split_idx:]

# 4. ساخت و آموزش مدل ARIMA(3,1,0)
arima_model = ARIMA(y_train, order=(3, 1, 0))
arima_result = arima_model.fit()

# 5. پیش‌بینی روی داده تست
arima_forecast = arima_result.forecast(steps=len(y_test))

# 6. محاسبه خطاها و log-likelihood
arima_mae = mean_absolute_error(y_test, arima_forecast)
arima_mse = mean_squared_error(y_test, arima_forecast)
arima_rss = np.sum((y_test - arima_forecast) ** 2)
arima_llf = arima_result.llf  # log-likelihood of ARIMA

# 7. مقایسه با مدل RNN
# فرض: متغیرهای زیر از اجرای RNN قبلی موجود هستن:
# rnn_mae, rnn_mse, rnn_rss, y_test_inv

# 8. محاسبه log-likelihood تقریبی برای RNN و آزمون نسبت درست‌نمایی
log_likelihood_rnn = -0.5 * len(y_test_inv) * np.log(rnn_mse)
log_likelihood_ratio = 2 * (arima_llf - log_likelihood_rnn)

# 9. چاپ نتایج
print("===== ARIMA Model (3,1,0) =====")
print(f"MAE: {arima_mae:.2f}, MSE: {arima_mse:.2f}, RSS: {arima_rss:.2f}, Log-Likelihood: {arima_llf:.2f}")

print("\n===== RNN Model =====")
print(f"MAE: {rnn_mae:.2f}, MSE: {rnn_mse:.2f}, RSS: {rnn_rss:.2f}")

print("\n===== Log Likelihood Ratio (ARIMA vs RNN) =====")
print(f"LLR: {log_likelihood_ratio:.2f}")


import matplotlib.pyplot as plt

# اطمینان از اینکه همه چیز در قالب سری زمانی تنظیم شده
# y_test: مشاهدات واقعی time_to_event برای تست
# arima_forecast: پیش‌بینی ARIMA روی داده‌های تست
# rnn_pred_inv: پیش‌بینی مدل RNN (در مقیاس اصلی)
# ts: سری زمانی اصلی کامل time_to_event

# ایندکس داده‌های تست (همان بازه‌ای که پیش‌بینی کردیم)
plot_index = y_test.index

plt.figure(figsize=(12, 6))

# خط مشاهدات واقعی
plt.plot(plot_index, y_test.values, label="Actual", marker='o')

# پیش‌بینی ARIMA
plt.plot(plot_index, arima_forecast.values, label="ARIMA Forecast", marker='x')

# پیش‌بینی RNN (با ایندکس جدید ساخته‌شده از همان طول)
rnn_plot_index = plot_index[-len(rnn_pred_inv):]  # در صورت اختلاف طول
plt.plot(rnn_plot_index, rnn_pred_inv.flatten(), label="RNN Forecast", marker='^')

plt.title(f"Forecast Comparison – ARIMA vs RNN (Participant {sample_pid})")
plt.xlabel("Date")
plt.ylabel("Time to Event")
plt.gca().invert_yaxis()  # چون کاهش مقدار time_to_event یعنی نزدیک‌تر شدن به event
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# تنظیمات
seq_length = 5  # طول توالی‌ها، در صورت نیاز قابل تغییر
sample_pid = df['participant_id'].unique()[0]

# فیلتر کردن داده شرکت‌کننده
participant_data = df[df['participant_id'] == sample_pid].sort_values('date').set_index('date')
ts = participant_data['time_to_event'].astype(float)

# نرمال‌سازی
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

# تابع ساخت توالی X و y
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# ساخت توالی‌ها
X_train, y_train = create_sequences(ts_scaled, seq_length)
X_train = X_train.reshape((X_train.shape[0], seq_length, 1))

# اطمینان از وجود داده کافی
if len(X_train) == 0:
    print("❗ داده کافی برای آموزش LSTM وجود ندارد. seq_length را کاهش دهید یا شرکت‌کننده‌ی دیگری انتخاب کنید.")
else:
    # تعریف مدل LSTM
    lstm_model = Sequential([
        LSTM(50, activation='tanh', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=50, verbose=1)

    # پیش‌بینی
    lstm_pred_scaled = lstm_model.predict(X_train)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    actual_y = scaler.inverse_transform(y_train)

    # رسم نمودار
    plt.figure(figsize=(12, 6))
    plt.plot(actual_y, label='Actual', marker='o')
    plt.plot(lstm_pred, label='LSTM Prediction', marker='^')
    plt.title(f'LSTM Forecast vs Actual – Participant {sample_pid}')
    plt.xlabel('Time step')
    plt.ylabel('Time to Event (days)')
    plt.gca().invert_yaxis()  # چون شمارش معکوسه
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Predict on test set
lstm_pred_scaled = lstm_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, lstm_pred_scaled)
mse = mean_squared_error(y_test, lstm_pred_scaled)
rss = np.sum((y_test - lstm_pred_scaled) ** 2)
print("===== LSTM Model (Global) =====")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RSS: {rss:.4f}")

#Prepare global LSTM sequences
seq_length = 5
grouped = df.groupby('participant_id')
all_sequences = []

for pid, group in grouped:
    if len(group) >= seq_length + 1:
        series = group.sort_values('date')['time_to_event'].values
        scaler = MinMaxScaler()
        scaled_series = scaler.fit_transform(series.reshape(-1, 1))
        for i in range(len(scaled_series) - seq_length):
            X_seq = scaled_series[i:i+seq_length]
            y_seq = scaled_series[i+seq_length]
            all_sequences.append((X_seq, y_seq))

X_all = np.array([seq[0] for seq in all_sequences])
y_all = np.array([seq[1] for seq in all_sequences])
X_all = X_all.reshape((X_all.shape[0], seq_length, 1))

#Training our LSTM model
model = Sequential([
    LSTM(50, activation='tanh', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_all, y_all, epochs=50, verbose=0)
#Prediction
def predict_non_hypertensive_date(df, participant_id, model, seq_length=5):
    df_part = df[df['participant_id'] == participant_id].sort_values('date')
    ts = df_part.set_index('date')['time_to_event'].astype(float)

    if len(ts) <= seq_length + 1:
        return None

    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

    X_p = np.array([ts_scaled[i:i+seq_length] for i in range(len(ts_scaled) - seq_length)])
    X_p = X_p.reshape((X_p.shape[0], seq_length, 1))

    preds_scaled = model.predict(X_p)
    preds = scaler.inverse_transform(preds_scaled)

    dates = ts.index[seq_length:]
    for date, val in zip(dates, preds.flatten()):
        if val <= 0.5:  # Threshold for predicting recovery
            return date

    return dates[-1]

#to all participant applying
results = []
for pid in df['participant_id'].unique():
    pred_date = predict_non_hypertensive_date(df, pid, model)
    if pred_date is not None:
        results.append({
            'participant_id': pid,
            'predicted_non_hypertensive_date': pred_date
        })

# Create and save our results
forecast_df = pd.DataFrame(results)
forecast_df = forecast_df.sort_values("predicted_non_hypertensive_date")

# View the result
print(forecast_df.head())

def plot_participant_forecast(df, participant_id, model, seq_length=5):
    df_part = df[df['participant_id'] == participant_id].sort_values('date')
    ts = df_part.set_index('date')['time_to_event'].astype(float)

    if len(ts) <= seq_length + 1:
        print(f"Participant {participant_id} has too few records.")
        return

    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

    X_p = np.array([ts_scaled[i:i+seq_length] for i in range(len(ts_scaled) - seq_length)])
    X_p = X_p.reshape((X_p.shape[0], seq_length, 1))
    preds_scaled = model.predict(X_p)
    preds = scaler.inverse_transform(preds_scaled)

    dates = ts.index[seq_length:]
    actual = ts[seq_length:]

    plt.figure(figsize=(10, 5))
    plt.plot(ts.index, ts.values, label="Actual Time to Event", marker='o')
    plt.plot(dates, preds.flatten(), label="LSTM Prediction", linestyle='--', marker='x')
    plt.title(f"Participant {participant_id} – LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Time to Event")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
for pid in df['participant_id'].unique()[:30]:  # first 5 participants if we want to see for more participants we can adjust it
    plot_participant_forecast(df, pid, model)

def plot_forecast_for_participant(df, participant_id, model, seq_length=5):
    df_part = df[df['participant_id'] == participant_id].sort_values('date')
    ts = df_part.set_index('date')['time_to_event'].astype(float)

    if len(ts) <= seq_length + 1:
        return

    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))
    X = np.array([ts_scaled[i:i+seq_length] for i in range(len(ts_scaled) - seq_length)])
    X = X.reshape((X.shape[0], seq_length, 1))

    pred_scaled = model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)

    dates = ts.index[seq_length:]

    plt.plot(ts.index, ts.values, label="Actual", marker='o')
    plt.plot(dates, pred.flatten(), label="LSTM Forecast", linestyle='--', marker='x')
    plt.title(f"Participant {participant_id}")
    plt.xlabel("Date")
    plt.ylabel("Time to Event")
    plt.legend()
    plt.grid(True)

# Creating dashboard for 6 participants
sample_participants = df['participant_id'].value_counts().loc[lambda x: x >= 10].index[:6]

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for idx, pid in enumerate(sample_participants):
    plt.sca(axes[idx])
    plot_forecast_for_participant(df, pid, model)

plt.suptitle("LSTM Forecast Dashboard – Time to Non-Hypertensive State", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
