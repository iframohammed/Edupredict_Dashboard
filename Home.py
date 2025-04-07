import streamlit as st
from PIL import Image

st.set_page_config(page_title="EduPredict - Home", layout="wide")

# Background color
st.markdown("""
    <style>
        .stApp {
            background-color: #f4f6fa;
        }
    </style>
""", unsafe_allow_html=True)

# Header layout
col1, col2 = st.columns([1, 6])
with col1:
    st.image("university_logo.png", width=100)
with col2:
    st.markdown("<h1 style='font-size: 38px;'>EduPredict Tool for Enrollment Trends</h1>", unsafe_allow_html=True)

st.markdown("---")

st.subheader("Welcome to EduPredict!")
st.write("""
This tool helps forecast student enrollment trends across various categories like gender, visa types, and enrollment status using machine learning and time series forecasting.
""")

st.markdown("### Navigate to the **Status** tab to explore the dashboard.")