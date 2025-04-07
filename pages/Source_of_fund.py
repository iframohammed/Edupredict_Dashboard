import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Source of Fund Dashboard", layout="wide")
st.title("Source of Fund Dashboard")

uploaded_file = st.file_uploader("Upload source_of_fund.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
    df.dropna(subset=['year'], inplace=True)
    df['year'] = df['year'].astype(int)

    def format_millions(n):
        return f"{n/1e6:.0f}M"

    def forecast_series(data, group_col, value_col='students', forecast_years=9, start_year=2007, end_year=2032):
        all_forecasts = []
        for val in data[group_col].unique():
            subset = data[data[group_col] == val]
            ts = subset.groupby("year")[value_col].sum()
            ts = ts[ts > 0].rolling(window=3, min_periods=1).mean()
            try:
                log_ts = np.log(ts)
                model = ARIMA(log_ts, order=(1, 1, 1))
                model_fit = model.fit()
                log_forecast = model_fit.forecast(steps=forecast_years)
                future_vals = np.exp(log_forecast)
                future_years = list(range(ts.index.max() + 1, ts.index.max() + 1 + forecast_years))
                forecast_df = pd.DataFrame({"year": future_years, group_col: val, value_col: future_vals})
                actual_df = subset[["year", group_col, value_col]]
                full_df = pd.concat([actual_df, forecast_df])
                full_df = full_df[(full_df["year"] >= start_year) & (full_df["year"] <= end_year)]
                all_forecasts.append(full_df)
            except:
                subset = subset[(subset["year"] >= start_year) & (subset["year"] <= end_year)]
                all_forecasts.append(subset[["year", group_col, value_col]])
        return pd.concat(all_forecasts)

    # ---------- ROW 1: KPI Cards ----------
    st.markdown("### Overview Metrics")
    col1, col2, col3 = st.columns(3)

    total_students = df['students'].sum()
    top_source = df.groupby("source_of_fund")["students"].sum().idxmax()
    top_type = df.groupby("academic_type")["students"].sum().idxmax()

    col1.metric("Total Students", format_millions(total_students))
    col2.metric("Top Source of Fund", top_source)
    col3.metric("Most Popular Academic Type", top_type)

    # ---------- ROW 2: Breakdown Visuals ----------
    st.markdown("### Funding Distribution Insights")
    col4, col5 = st.columns(2)

    with col4:
        fund_df = df.groupby("source_type")["students"].sum().reset_index()
        fig_fund = px.pie(fund_df, names="source_type", values="students", title="Students by Source Type")
        st.plotly_chart(fig_fund, use_container_width=True)

    with col5:
        top_sources = df.groupby("source_of_fund")["students"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top = px.bar(top_sources[::-1], x="students", y="source_of_fund", orientation="h", title="Top 10 Sources of Fund")
        st.plotly_chart(fig_top, use_container_width=True)

    # ---------- ROW 3: Forecasted Trends ----------
    st.markdown("### Forecasted Funding Trends")
    col6, col7 = st.columns(2)

    with col6:
        forecast_type = forecast_series(df, "academic_type")
        fig_type = px.line(forecast_type, x="year", y="students", color="academic_type", markers=True,
                           title="Forecast by Academic Type")
        fig_type.update_layout(xaxis=dict(range=[2007, 2032]))
        st.plotly_chart(fig_type, use_container_width=True)

    with col7:
        forecast_fund = forecast_series(df, "source_type")
        fig_fund_forecast = px.area(forecast_fund, x="year", y="students", color="source_type",
                                    title="Forecast by Source Type")
        fig_fund_forecast.update_layout(xaxis=dict(range=[2007, 2032]))
        st.plotly_chart(fig_fund_forecast, use_container_width=True)

else:
    st.info("Please upload source_of_fund.csv to begin.")