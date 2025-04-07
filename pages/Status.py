import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Status Dashboard", layout="wide")
st.title("Status Dashboard")

uploaded_file = st.file_uploader("Upload your status.csv file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
    df.dropna(subset=['year'], inplace=True)
    df['year'] = df['year'].astype(int)

    def forecast_column(df, column, forecast_years=10):
        ts = df.groupby("year")[column].sum()
        ts = ts[ts > 0]
        ts = ts.rolling(window=3, min_periods=1).mean()
        log_ts = np.log(ts)
        model = ARIMA(log_ts, order=(1, 1, 1))
        model_fit = model.fit()
        log_forecast = model_fit.forecast(steps=forecast_years)
        forecast = np.exp(log_forecast)
        forecast_years = list(range(ts.index.max() + 1, ts.index.max() + 1 + forecast_years))
        forecast_df = pd.DataFrame({"year": forecast_years, column: forecast})
        full_df = pd.concat([ts.reset_index(), forecast_df], ignore_index=True)
        return full_df[(full_df['year'] >= 2007) & (full_df['year'] <= 2032)]

    columns_to_forecast = ['female', 'male', 'single', 'married', 'full_time', 'part_time', 'visa_f', 'visa_j', 'visa_other']
    forecasts = {col: forecast_column(df, col) for col in columns_to_forecast}

    # KPI Section
    st.markdown("### Key Performance Indicators")
    kpi_cols = st.columns(3)
    for i, col in enumerate(columns_to_forecast[:9]):
        last_known = forecasts[col][forecasts[col]['year'] == 2022][col].values[0]
        future_val = forecasts[col][forecasts[col]['year'] == 2023][col].values[0]
        delta = ((future_val - last_known) / last_known) * 100
        kpi_cols[i % 3].metric(label=f"{col.capitalize()} (Forecasted)", value=f"{int(future_val):,}", delta=f"{delta:.2f}%")

    # Ratio Charts
    st.markdown("### Ratio Charts")
    col1, col2 = st.columns(2)
    latest = {col: forecasts[col][forecasts[col]['year'] == 2023][col].values[0] for col in columns_to_forecast}

    with col1:
        gender_df = pd.DataFrame({"Gender": ["Female", "Male"], "Count": [latest['female'], latest['male']]})
        fig_gender = px.pie(gender_df, names="Gender", values="Count", hole=0.4, title="Gender Ratio (Donut)")
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        visa_df = pd.DataFrame({
            "Visa Type": ["Visa F", "Visa J", "Other"],
            "Count": [latest['visa_f'], latest['visa_j'], latest['visa_other']]
        })
        fig_visa = px.pie(visa_df, names="Visa Type", values="Count", title="Visa Type Ratio")
        st.plotly_chart(fig_visa, use_container_width=True)

    # Time-Based Trends
    st.markdown("### Time-Based Trends")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Marital Status Over Years")
        marital_df = forecasts['single'].merge(forecasts['married'], on='year', suffixes=('_single', '_married'))
        marital_df = marital_df[(marital_df['year'] >= 2007) & (marital_df['year'] <= 2032)]
        marital_df = marital_df.melt(id_vars='year', value_vars=['single', 'married'], var_name='Status', value_name='Count')
        fig_marital = px.bar(marital_df, x='year', y='Count', color='Status', barmode='stack')
        st.plotly_chart(fig_marital, use_container_width=True)

    with col4:
        st.subheader("Full-time vs Part-time Over Years")
        full_df = forecasts['full_time'].merge(forecasts['part_time'], on='year')
        full_df = full_df[(full_df['year'] >= 2007) & (full_df['year'] <= 2032)]
        fig_full = px.line(full_df, x='year', y=['full_time', 'part_time'], markers=True)
        fig_full.update_layout(yaxis_title="Enrollment")
        st.plotly_chart(fig_full, use_container_width=True)

else:
    st.info("Please upload status.csv to begin.")
