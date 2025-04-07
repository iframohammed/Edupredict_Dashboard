import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

st.set_page_config(page_title="Origin Dashboard", layout="wide")
st.title("Origin Dashboard")

uploaded_file = st.file_uploader("Upload origin.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    def format_millions(n):
        return f"{n/1e6:.0f}M"

    def forecast_series(data, group_col, value_col, forecast_years=9, plot_start=2007, plot_end=2032):
        all_forecasts = []
        for val in data[group_col].unique():
            subset = data[data[group_col] == val]
            ts = subset.groupby("year")[value_col].sum()
            ts = ts[ts > 0]  # remove zero values to allow log
            try:
                ts = ts.rolling(window=3, min_periods=1).mean()  # smooth post-COVID data
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

    # ---------- ROW 1: KPI Metrics ----------
    st.markdown("### Key Metrics")
    col1, col2, col3 = st.columns(3)
    total_students = df['students'].sum()
    top_origin = df.groupby("origin")["students"].sum().idxmax()
    top_type = df.groupby("academic_type")["students"].sum().idxmax()

    col1.metric("Total Students", format_millions(total_students))
    col2.metric("Top Origin Country", top_origin)
    col3.metric("Most Popular Academic Type", top_type)

    # ---------- ROW 2: Map + Top 10 Countries ----------
    st.markdown("### Distribution Insights")
    col4, col5 = st.columns(2)

    region_summary = df.groupby("origin_region")["students"].sum().reset_index()
    region_coords = {
        "Africa, Subaharan": (0, 10), "Asia": (100, 60), "Caribbean": (-75, 18),
        "Central Africa": (20, 2), "East Africa": (40, -1), "East Asia": (120, 35),
        "Europe": (15, 50), "Latin America and Caribbean": (-65, -10),
        "Mexico and Central America": (-90, 15), "Middle East": (45, 30),
        "North Africa": (10, 30), "North America": (-100, 40), "Oceania": (140, -25),
        "South America": (-60, -15), "South and Central Asia": (75, 25),
        "Southeast Asia": (105, 10), "Southern Africa": (25, -25),
        "West Africa": (0, 10), "Stateless": (0, 0)
    }
    region_summary["lon"] = region_summary["origin_region"].map(lambda x: region_coords.get(x, (0, 0))[0])
    region_summary["lat"] = region_summary["origin_region"].map(lambda x: region_coords.get(x, (0, 0))[1])

    with col4:
        fig_map = px.scatter_geo(
            region_summary,
            lat="lat",
            lon="lon",
            size="students",
            hover_name="origin_region",
            size_max=50,
            projection="natural earth",
            title="Students by Region"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col5:
        top_countries = df.groupby("origin")["students"].sum().sort_values(ascending=False).head(10)
        fig_bar = px.bar(
            x=top_countries.values[::-1],
            y=top_countries.index[::-1],
            orientation='h',
            title="Top 10 Origin Countries"
        )
        fig_bar.update_layout(xaxis_title="Students", yaxis_title="Country")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ---------- ROW 3: Yearly Trends with Forecast ----------
    st.markdown("### Yearly Trends")
    col6, col7 = st.columns(2)

    with col6:
        st.subheader("Yearly Students by Academic Type (Forecasted)")
        forecast_type = forecast_series(df, "academic_type", "students")
        fig_type = px.line(forecast_type, x="year", y="students", color="academic_type", markers=True)
        fig_type.update_layout(xaxis=dict(range=[2007, 2032]))
        st.plotly_chart(fig_type, use_container_width=True)

    with col7:
        st.subheader("Yearly Region Breakdown (Forecasted)")
        top_regions = df.groupby("origin_region")["students"].sum().sort_values(ascending=False).head(8).index
        top_df = df[df["origin_region"].isin(top_regions)]
        forecast_region = forecast_series(top_df, "origin_region", "students")
        fig_region = px.area(forecast_region, x="year", y="students", color="origin_region")
        fig_region.update_layout(xaxis=dict(range=[2007, 2032]))
        st.plotly_chart(fig_region, use_container_width=True)

else:
    st.info("Please upload origin.csv to begin.")
