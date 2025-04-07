import streamlit as st

st.set_page_config(page_title="About EduPredict", layout="wide")

st.title("About the Project")

st.write("""
**EduPredict Tool for Enrollment Trends** is designed to assist universities, policymakers, and stakeholders in understanding and forecasting student enrollment patterns.

### Key Features:
- Forecast future student enrollment using ARIMA (time series model)
- Visualize key metrics like gender, visa status, and enrollment types
- Real-time dashboard with automated insights

### Technology Used:
- Python
- Streamlit (for dashboard)
- Statsmodels (ARIMA model)
- Matplotlib, Plotly
""")