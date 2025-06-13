
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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


        # 🩺 اضافه کردن نمودار فشار خون برای فرد انتخاب‌شده
        st.subheader(f"💓 Blood Pressure for Participant {selected_pid}")
        participant_data = grouped.get_group(selected_pid).set_index('date')

        fig_bp, ax_bp = plt.subplots(figsize=(10, 5))
        participant_data[['blood_pressure_systolic', 'blood_pressure_diastolic']].plot(ax=ax_bp)
        plt.title(f'Blood Pressure Over Time (Participant {selected_pid})')
        plt.xlabel('Date')
        plt.ylabel('Blood Pressure (mmHg)')
        plt.grid(True)
        st.pyplot(fig_bp)


    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👈 Please upload a dataset to get started.")
