
import streamlit as st
import pandas as pd

st.set_page_config(page_title="تحلیل داده‌ها", layout="wide")

st.title("📊 اپلیکیشن Streamlit برای بارگذاری و نمایش دیتاست")

# مرحله 1: گرفتن فایل از کاربر
uploaded_file = st.file_uploader("لطفاً فایل CSV یا Excel خود را آپلود کنید:", type=["csv", "xlsx"])

# مرحله 2: خواندن و نمایش داده‌ها
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ فایل با موفقیت بارگذاری شد!")
        st.subheader("پیش‌نمایش داده‌ها:")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ خطا در خواندن فایل: {e}")
else:
    st.info("👈 لطفاً یک فایل دیتاست آپلود کنید.")
