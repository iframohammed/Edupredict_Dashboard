import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Academic Dashboard", layout="wide")
st.title("Academic Overview")

uploaded_files = st.file_uploader("Upload academic.csv, academic_detail.csv, and/or field_of_study.csv", type="csv", accept_multiple_files=True)

if uploaded_files:
    academic_df = academic_detail_df = field_df = pd.DataFrame()

    for file in uploaded_files:
        if "academic.csv" in file.name:
            academic_df = pd.read_csv(file)
        elif "academic_detail.csv" in file.name:
            academic_detail_df = pd.read_csv(file)
        elif "field_of_study.csv" in file.name:
            field_df = pd.read_csv(file)

    for df in [academic_df, academic_detail_df, field_df]:
        if not df.empty:
            df.columns = df.columns.str.strip().str.lower()
            df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
            df.dropna(subset=['year'], inplace=True)
            df['year'] = df['year'].astype(int)

    def format_millions(n):
        return f"{n/1e6:.1f}M"

    def forecast_series(data, group_col, value_col, forecast_years=9, plot_start=2007, plot_end=2032):
        all_forecasts = []
        for val in data[group_col].unique():
            subset = data[data[group_col] == val]
            ts = subset.groupby("year")[value_col].sum()
            ts = ts[ts > 0]
            try:
                ts = ts.rolling(window=3, min_periods=1).mean()
                log_ts = np.log(ts)
                model = ARIMA(log_ts, order=(1, 1, 1))
                model_fit = model.fit()
                log_forecast = model_fit.forecast(steps=forecast_years)
                future = np.exp(log_forecast)
                future_years = list(range(ts.index.max() + 1, ts.index.max() + 1 + forecast_years))
                forecast_df = pd.DataFrame({
                    "year": future_years,
                    group_col: val,
                    value_col: future.values
                })
                actual_df = subset[["year", group_col, value_col]]
                full_df = pd.concat([actual_df, forecast_df])
                full_df = full_df[(full_df["year"] >= plot_start) & (full_df["year"] <= plot_end)]
                all_forecasts.append(full_df)
            except:
                filtered = subset[["year", group_col, value_col]]
                filtered = filtered[(filtered["year"] >= plot_start) & (filtered["year"] <= plot_end)]
                all_forecasts.append(filtered)
        return pd.concat(all_forecasts)

    # ---------- ROW 1: KPI + Field of Study Breakdown ----------
    st.markdown("### Summary")
    col1, col2, col3 = st.columns(3)

    if not academic_df.empty:
        total_students = academic_df['students'].sum()
        col1.metric("Total Students", format_millions(total_students))
    else:
        col1.metric("Total Students", "N/A")

    if not academic_detail_df.empty:
        top_level = academic_detail_df.groupby("academic_level")["students"].sum().idxmax()
        col2.metric("Top Academic Level", top_level)

    if not field_df.empty:
        top_field = field_df.groupby("field_of_study")["students"].sum().idxmax()
        col3.metric("Top Field of Study", top_field)

    # ---------- ROW 2: Pie Chart + Bar Chart ----------
    col4, col5 = st.columns(2)

    if not academic_detail_df.empty:
        level_summary = academic_detail_df.groupby("academic_level")["students"].sum().reset_index()
        fig_level = px.pie(level_summary, names="academic_level", values="students", title="Academic Level Breakdown")
        col4.plotly_chart(fig_level, use_container_width=True)

    if not field_df.empty:
        field_summary = field_df.groupby("field_of_study")["students"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_fields = px.bar(field_summary, x="students", y="field_of_study", orientation="h", title="Top Fields of Study")
        col5.plotly_chart(fig_fields, use_container_width=True)

    # ---------- ROW 3: Forecasted Trends ----------
    st.markdown("### Academic Type Popularity (Forecasted)")
    if not academic_detail_df.empty:
        forecast_type = forecast_series(academic_detail_df, "academic_type", "students")
        fig_type = px.line(forecast_type, x="year", y="students", color="academic_type", markers=True)
        fig_type.update_layout(xaxis=dict(range=[2007, 2032]))
        st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("### Field of Study Trends (Forecasted)")
    if not field_df.empty:
        top_fields = field_df.groupby("field_of_study")["students"].sum().sort_values(ascending=False).head(6).index
        top_field_df = field_df[field_df["field_of_study"].isin(top_fields)]
        forecast_fields = forecast_series(top_field_df, "field_of_study", "students")
        fig_field = px.area(forecast_fields, x="year", y="students", color="field_of_study")
        fig_field.update_layout(xaxis=dict(range=[2007, 2032]))
        st.plotly_chart(fig_field, use_container_width=True)

else:
    st.info("Please upload one or more academic-related CSV files to begin.")
