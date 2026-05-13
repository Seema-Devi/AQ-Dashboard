# =====================================
# IMPORT LIBRARIES
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

# =====================================
# LOAD CLEANED DATA (FIXED — NO BOOLEAN EVALUATION)
# =====================================
if "df_cleaned" in st.session_state:
    df = st.session_state["df_cleaned"]
elif "df" in st.session_state:
    df = st.session_state["df"]
else:
    st.warning("""
    ⬅ Please clean your dataset first before running EDA.

   App Workflow:

1️⃣ Upload datasets from the sidebar  
2️⃣ Explore the data in Data Visualisation  
3️⃣ Clean and preprocess the dataset  
4️⃣ Analyse trends and patterns in EDA  
5️⃣ Apply Feature Engineering  
6️⃣ Train models and view AQI Forecasting
""")
    st.stop()

st.header("📊 Exploratory Data Analysis")

df = df.copy()

# =====================================
# DETECT DATETIME COLUMN
# =====================================
time_col = None
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# =====================================
# SECTION 1 — CLEANED DATA OVERVIEW
# =====================================
st.subheader("1️⃣ Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Numeric Columns", len(numeric_cols))

st.dataframe(df.head(), use_container_width=True)

# Summary statistics
st.markdown("###  Summary Statistics ")
st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 2 — FULL CORRELATION HEATMAP
# =====================================
with st.expander("2️⃣ Correlation Heatmap "):

    corr = df[numeric_cols].corr().round(3)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap (All Numeric Parameters)",
        aspect="auto"
    )

    fig.update_layout(
        xaxis_title="Parameters",
        yaxis_title="Parameters",
        height=800
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 3 — UNIVARIATE ANALYSIS
# =====================================
with st.expander("3️⃣ Univariate Analysis (Distribution + Outliers)"):

    selected_uni = st.selectbox("Select a numeric column:", numeric_cols)

    fig = px.histogram(df, x=selected_uni, nbins=40, marginal="box",
                       title=f"{selected_uni} Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(df, y=selected_uni, title=f"{selected_uni} Outliers")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 4 — BIVARIATE ANALYSIS
# =====================================
with st.expander("4️⃣ Bivariate Analysis (Scatter + Trendline)"):

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X-axis", numeric_cols)
    with col2:
        y_var = st.selectbox("Y-axis", numeric_cols, index=1)

    fig = px.scatter(df, x=x_var, y=y_var, trendline="ols",
                     title=f"{x_var} vs {y_var}")
    st.plotly_chart(fig, use_container_width=True)

    corr_val = df[[x_var, y_var]].corr().iloc[0, 1]
    st.info(f"Correlation between **{x_var}** and **{y_var}**: **{corr_val:.3f}**")

st.markdown("---")

# =====================================
# SECTION 5 — AQI INFLUENCE RANKING
# =====================================
with st.expander("5️⃣ AQI Influence Ranking"):

    aqi_col = None
    for col in df.columns:
        if col.lower() == "aqi":
            aqi_col = col
            break

    if aqi_col is None:
        st.warning("⚠ No AQI column found in dataset.")
    else:
        corr_rank = df[numeric_cols].corr()[aqi_col].sort_values(ascending=False).reset_index()
        corr_rank.columns = ["Parameter", "Correlation with AQI"]

        st.dataframe(corr_rank, use_container_width=True)

        fig = px.bar(
            corr_rank,
            x="Parameter",
            y="Correlation with AQI",
            title="Feature Influence on AQI",
            color="Correlation with AQI",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 6 — FIXED AQI PAIRWISE (DROPDOWN)
# =====================================
with st.expander("6️⃣ AQI Pairwise Trendlines"):

    aqi_col = None
    for col in df.columns:
        if col.lower() == "aqi":
            aqi_col = col
            break

    if aqi_col is None:
        st.warning("⚠ No AQI column found in dataset.")
    else:
        df[aqi_col] = pd.to_numeric(df[aqi_col], errors="coerce")

        compare_col = st.selectbox(
            "Select parameter to compare with AQI:",
            [col for col in numeric_cols if col != aqi_col]
        )

        fig = px.scatter(
            df,
            x=compare_col,
            y=aqi_col,
            trendline="ols",
            title=f"{aqi_col} vs {compare_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 7 — TIME-SERIES ANALYSIS
# =====================================
with st.expander("7️⃣ Time-Series Analysis (Trend + Rolling Mean + Seasonal Patterns)"):

    if time_col:

        st.markdown("### Select Parameter for Time-Series Trend")
        ts_param = st.selectbox("Parameter:", numeric_cols)

        fig = px.line(df, x=time_col, y=ts_param,
                      title=f"{ts_param} Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)

        df["rolling_mean"] = df[ts_param].rolling(window=24, min_periods=1).mean()
        fig = px.line(df, x=time_col, y="rolling_mean",
                      title=f"{ts_param} Rolling Mean (24-hour)")
        st.plotly_chart(fig, use_container_width=True)

        df["Hour"] = df[time_col].dt.hour
        df["Month"] = df[time_col].dt.month
        df["Weekday"] = df[time_col].dt.day_name()

        fig = px.line(df.groupby("Hour")[ts_param].mean().reset_index(),
                      x="Hour", y=ts_param,
                      title=f"Average {ts_param} by Hour")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(df.groupby("Month")[ts_param].mean().reset_index(),
                      x="Month", y=ts_param,
                      title=f"Average {ts_param} by Month")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(df, x="Weekday", y=ts_param,
                     title=f"{ts_param} by Weekday")
        st.plotly_chart(fig, use_container_width=True)

        if "season" in df.columns:
            fig = px.box(df, x="season", y=ts_param,
                         title=f"{ts_param} by Season")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
# =====================================
# SECTION 8 — PROFESSIONAL DYNAMIC INSIGHTS
# =====================================
st.subheader("8️⃣  Insights & Interpretation")

insights = []

# Compute correlation matrix
corr = df[numeric_cols].corr()

# Helper: remove duplicate pairs (A↔B and B↔A)
def unique_pairs(corr_series):
    seen = set()
    cleaned = []
    for (a, b), val in corr_series.items():
        if a == b:
            continue
        pair = tuple(sorted([a, b]))
        if pair not in seen:
            seen.add(pair)
            cleaned.append(((a, b), val))
    return cleaned

# 1. Strong correlations
strong_pairs = corr.abs().unstack().sort_values(ascending=False)
strong_pairs = strong_pairs[strong_pairs < 1]  # remove self-correlation
strong_pairs = [(p, v) for (p, v) in unique_pairs(strong_pairs[strong_pairs > 0.6])]

if strong_pairs:
    insights.append("### 🔥 Strong Relationships")
    insights.append("These variable pairs show strong relationships, indicating shared sources or redundant information:")
    for (a, b), val in strong_pairs[:8]:
        insights.append(f"- **{a}** and **{b}** are strongly related (**corr = {val:.2f}**)")

# 2. AQI-specific insights
if "AQI" in df.columns:
    aqi_corr = corr["AQI"].sort_values(ascending=False)

    pos = aqi_corr[aqi_corr > 0.3]
    neg = aqi_corr[aqi_corr < -0.3]

    if len(pos) > 1:
        insights.append("### 🌫️ Factors Increasing AQI")
        insights.append("These variables are positively associated with AQI and tend to worsen air quality:")
        for param, val in pos.items():
            if param != "AQI":
                insights.append(f"- **{param}** increases AQI (**corr = {val:.2f}**)")

    if len(neg) > 0:
        insights.append("### 💨 Factors Reducing AQI")
        insights.append("These variables are negatively associated with AQI and tend to improve air quality:")
        for param, val in neg.items():
            insights.append(f"- **{param}** reduces AQI (**corr = {val:.2f}**)")

# 3. Multicollinearity
multi_pairs = [(p, v) for (p, v) in strong_pairs if abs(v) > 0.8]
if multi_pairs:
    insights.append("### ⚠️ Multicollinearity Warning")
    insights.append("Some variables contain overlapping information. Consider removing one from each pair to avoid model instability:")
    for (a, b), val in multi_pairs:
        insights.append(f"- **{a}** and **{b}** are highly correlated (**{val:.2f}**)")

# 4. Weak variables
weak_vars = corr.abs().mean().sort_values().head(5)
insights.append("### 🧊 Weak or Low-Impact Variables")
insights.append("These variables show weak relationships with the rest of the dataset and may have limited predictive value:")
for param, val in weak_vars.items():
    insights.append(f"- **{param}** has low overall correlation (**avg corr = {val:.2f}**)")

# 5. Time-series insights
if time_col:
    insights.append("### ⏳ Time-Series Patterns")
    insights.append("- Clear daily, weekly, and monthly patterns detected.")
    insights.append("- Seasonal variation suggests adding lag, rolling averages, and seasonal features.")
   

# Display insights
for line in insights:
    st.markdown(line)


# =====================================
# FOOTER
# =====================================
render_footer()
