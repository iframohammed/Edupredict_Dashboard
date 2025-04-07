import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

def run_forecasts(df):
    results = {}
    year_col = "year"
    for col in df.columns:
        if col != year_col:
            ts = df.set_index(year_col)[col]
            try:
                model = ARIMA(ts, order=(1, 1, 1))
                model_fit = model.fit()
                forecast_years = 5
                forecast = model_fit.forecast(steps=forecast_years)
                future_years = list(range(df[year_col].max() + 1, df[year_col].max() + 1 + forecast_years))
                forecast_df = pd.DataFrame({
                    year_col: future_years,
                    col: forecast
                })
                full_df = pd.concat([df[[year_col, col]], forecast_df], ignore_index=True)
                results[col] = full_df
            except:
                pass  # Skip if ARIMA fails
    return results

def plot_kpis(results):
    cols = list(results.keys())
    for i in range(0, len(cols), 2):
        col1, col2 = st.columns(2)
        for j, c in enumerate([col1, col2]):
            if i + j < len(cols):
                col = cols[i + j]
                full_df = results[col]
                latest = full_df[col].iloc[-6]
                predicted = full_df[col].iloc[-1]
                growth = ((predicted - latest) / latest) * 100
                c.metric(f"{col.capitalize()} (Forecasted)", f"{int(predicted):,}", f"{growth:.2f}%")


def plot_all_charts(results):
    for col, full_df in results.items():
        st.markdown(f"#### {col.capitalize()} Forecast")

        # Line Chart
        fig, ax = plt.subplots()
        ax.plot(full_df["year"][:-5], full_df[col][:-5], label="Actual", marker='o')
        ax.plot(full_df["year"][-5:], full_df[col][-5:], label="Forecast", linestyle="--", marker='x')
        ax.set_xlabel("Year")
        ax.set_ylabel(col.capitalize())
        ax.legend()
        st.pyplot(fig)

        # Pie Chart (Latest Year)
        latest_year = full_df["year"].max()
        latest_val = full_df[full_df["year"] == latest_year][col].values[0]
        fig2, ax2 = plt.subplots()
        ax2.pie([latest_val, full_df[col].sum() - latest_val], labels=["Latest", "Others"], autopct="%1.1f%%")
        st.pyplot(fig2)