
import streamlit as st
import pandas as pd

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

        # Remove unnamed columns (e.g., 'Unnamed: 0')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        st.success("✅ File successfully uploaded!")

        # Sort by participant and date
        st.subheader("🔢 Sorting by participant and date:")
        df = df.sort_values(by=['participant_id', 'date'])
        st.dataframe(df.head())

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

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👈 Please upload a dataset to get started.")
