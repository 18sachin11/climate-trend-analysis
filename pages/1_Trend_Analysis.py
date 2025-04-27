# pages/1_Trend_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import linregress, spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

st.title("📈 Trend Analysis")

uploaded_file = st.file_uploader("Upload your Climate CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.dataframe(df.head())

    datetime_col = st.selectbox("Select datetime column", df.columns)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    variable = st.selectbox("Select Climate Variable", [col for col in df.columns])

    agg_level = st.selectbox("Aggregation Level", ["Daily", "Monthly", "Seasonal", "Yearly"])

    if agg_level == "Monthly":
        df_agg = df.resample('M').sum() if "rain" in variable.lower() else df.resample('M').mean()
    elif agg_level == "Seasonal":
        df['season'] = (df.index.month%12 + 3)//3
        df_agg = df.groupby([df.index.year, 'season'])[variable].sum().unstack() if "rain" in variable.lower() else df.groupby([df.index.year, 'season'])[variable].mean().unstack()
    elif agg_level == "Yearly":
        df_agg = df.resample('Y').sum() if "rain" in variable.lower() else df.resample('Y').mean()
    else:
        df_agg = df

    st.subheader("Aggregated Data")
    st.dataframe(df_agg)

    trend_methods = st.multiselect("Select Trend Analysis Methods", 
        ["Mann-Kendall", "Sen's Slope", "Spearman's Rho", "LOESS Smoothing", "STL Decomposition"]
    )

    results = {}

    if "Mann-Kendall" in trend_methods:
        mk_test = mk.original_test(df_agg[variable].dropna())
        st.write(mk_test)
        results["Mann-Kendall"] = mk_test

    if "Sen's Slope" in trend_methods:
        x = np.arange(len(df_agg))
        y = df_agg[variable].dropna().values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        st.write(f"Slope: {slope:.5f}, p-value: {p_value:.5f}")
        results["Sen's Slope"] = (slope, p_value)

    if "Spearman's Rho" in trend_methods:
        rho, pval = spearmanr(df_agg[variable].dropna())
        st.write(f"Spearman’s Rho: {rho:.4f}, p-value: {pval:.5f}")
        results["Spearman's Rho"] = (rho, pval)

    if "LOESS Smoothing" in trend_methods:
        smoothed = lowess(df_agg[variable], np.arange(len(df_agg)), frac=0.1)
        fig = px.line(x=df_agg.index, y=df_agg[variable], labels={'x':'Date', 'y':variable})
        fig.add_scatter(x=df_agg.index, y=smoothed[:,1], mode='lines', name='LOESS Smoothed')
        st.plotly_chart(fig)

    if "STL Decomposition" in trend_methods:
        stl = STL(df_agg[variable].dropna(), period=12)
        res = stl.fit()
        fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
        ax[0].plot(res.trend); ax[0].set_title('Trend')
        ax[1].plot(res.seasonal); ax[1].set_title('Seasonal')
        ax[2].plot(res.resid); ax[2].set_title('Residual')
        st.pyplot(fig)

    # Download Results
    if results:
        result_df = pd.DataFrame.from_dict(results, orient='index', columns=["Details"])
        st.download_button("Download Results as CSV", result_df.to_csv().encode('utf-8'), file_name='trend_analysis_results.csv')

else:
    st.warning("Please upload a CSV file to proceed.")

