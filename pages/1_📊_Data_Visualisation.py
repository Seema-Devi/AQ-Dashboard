# =====================================
# IMPORT LIBRARIES
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ui_components import load_full_ui, render_footer

# =====================================
# PAGE UI
# =====================================
st.write("")
load_full_ui()

px.defaults.template = "plotly_white"

config = {
    "displayModeBar": False,
    "scrollZoom": False
}

# =====================================
# LOAD DATA
# =====================================
df = st.session_state.get("df")

if df is None:
    st.warning("""
⬅ Please upload both datasets from the Home page to continue.

App Workflow:

1️⃣ Upload datasets from the sidebar  
2️⃣ Explore the data in Data Visualisation  
3️⃣ Clean and preprocess the dataset  
4️⃣ Analyse trends and patterns in EDA  
5️⃣ Apply Feature Engineering  
6️⃣ Train models and view AQI Forecasting
""")
    render_footer()
    st.stop()

st.header("📊 Raw Data Visualisation")

df_raw = df.copy()

# =====================================
# DETECT COLUMNS
# =====================================
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_raw.select_dtypes(exclude=np.number).columns.tolist()

time_col = None
for col in df_raw.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce")

# =====================================
# PAGE PURPOSE
# =====================================
st.markdown("""
<div style="background-color:#ffffff; padding:20px; border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08); margin-bottom:22px;">
<h4 style="color:#003d7a; margin-top:0;">Purpose of Data Visualisation</h4>
<p style="font-size:15px; color:#334155; line-height:1.6;">
This page visually checks whether the uploaded raw dataset is reliable enough for
air quality analysis, cleaning, feature engineering and forecasting model development.
</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# 1. DATASET OVERVIEW
# =====================================
st.subheader("1️⃣ Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{df_raw.shape[0]:,}")
c2.metric("Columns", df_raw.shape[1])
c3.metric("Numeric Columns", len(numeric_cols))
c4.metric("Categorical Columns", len(categorical_cols))

if time_col:
    date_min = df_raw[time_col].min()
    date_max = df_raw[time_col].max()
    duration_days = (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Start Date", str(date_min.date()) if pd.notna(date_min) else "Not available")
    c2.metric("End Date", str(date_max.date()) if pd.notna(date_max) else "Not available")
    c3.metric("Monitoring Period", f"{duration_days} days")
else:
    st.warning("⚠ No datetime column detected.")

type_df = pd.DataFrame({
    "Column Type": ["Numeric", "Categorical", "Datetime"],
    "Count": [
        len(numeric_cols),
        len(categorical_cols),
        1 if time_col else 0
    ]
})

fig = px.pie(
    type_df,
    names="Column Type",
    values="Count",
    hole=0.35,
    title="Dataset Column Type Distribution"
)

fig.update_traces(
    textinfo="percent+label",
    hovertemplate="Column Type: %{label}<br>Count: %{value}<extra></extra>"
)

fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True, config=config)

with st.expander("Optional: Preview first 5 rows"):
    st.dataframe(df_raw.head(), use_container_width=True)

st.markdown("---")

# =====================================
# 2. DATA COMPLETENESS
# =====================================
st.subheader("2️⃣ Data Completeness")

missing_df = df_raw.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Count"]
missing_df["Missing %"] = (missing_df["Missing Count"] / len(df_raw) * 100).round(2)

completeness_score = 100 - missing_df["Missing %"].mean()

c1, c2, c3 = st.columns(3)
c1.metric("Completeness Score", f"{completeness_score:.2f}%")
c2.metric("Total Missing Values", f"{missing_df['Missing Count'].sum():,}")
c3.metric("Columns with Missing Values", f"{(missing_df['Missing Count'] > 0).sum()}")

st.progress(completeness_score / 100)

missing_plot = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)

if not missing_plot.empty:
    fig = px.bar(
        missing_plot,
        x="Column",
        y="Missing %",
        color="Missing %",
        color_continuous_scale="Oranges",
        title="Missing Percentage by Column",
        text="Missing %"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="Column: %{x}<br>Missing: %{y:.2f}%<extra></extra>"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Dataset Columns",
        yaxis_title="Missing Percentage",
        xaxis_tickangle=45
    )

    st.plotly_chart(fig, use_container_width=True, config=config)
else:
    st.success("✅ No missing values detected.")

st.markdown("---")

# =====================================
# 3. DUPLICATE AND INVALID VALUE CHECK
# =====================================
st.subheader("3️⃣ Duplicate and Invalid Value Check")

duplicates = df_raw.duplicated().sum()

if numeric_cols:
    numeric_df = df_raw[numeric_cols]
    invalid_negative = (numeric_df < 0).sum()
    total_negative = invalid_negative.sum()
else:
    numeric_df = pd.DataFrame()
    invalid_negative = pd.Series(dtype=int)
    total_negative = 0

c1, c2 = st.columns(2)

with c1:
    st.metric("Duplicate Rows", duplicates)

    if duplicates > 0:
        st.warning("⚠ Duplicate rows detected and should be handled during cleaning.")
    else:
        st.success("✅ No duplicate rows found.")

with c2:
    st.metric("Negative Values", f"{total_negative:,}")

    negative_df = invalid_negative.reset_index()
    negative_df.columns = ["Column", "Negative Count"]
    negative_df = negative_df[negative_df["Negative Count"] > 0]

    if not negative_df.empty:
        fig = px.bar(
            negative_df,
            x="Column",
            y="Negative Count",
            color="Negative Count",
            color_continuous_scale="Oranges",
            title="Negative Values by Column",
            text="Negative Count"
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.success("✅ No invalid negative values found.")

st.markdown("---")

# =====================================
# 4. TIMESTAMP READINESS
# =====================================
st.subheader("4️⃣ Timestamp Readiness")

if time_col:
    missing_time = df_raw[time_col].isnull().sum()
    duplicate_time = df_raw[time_col].duplicated().sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Datetime Column", time_col)
    c2.metric("Missing Timestamps", missing_time)
    c3.metric("Duplicate Timestamps", duplicate_time)

    time_status = pd.DataFrame({
        "Timestamp Status": ["Valid Timestamps", "Missing Timestamps"],
        "Count": [len(df_raw) - missing_time, missing_time]
    })

    fig = px.pie(
        time_status,
        names="Timestamp Status",
        values="Count",
        hole=0.35,
        title="Timestamp Completeness"
    )

    fig.update_traces(
        textinfo="percent+label",
        hovertemplate="Status: %{label}<br>Count: %{value}<extra></extra>"
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("⚠ No datetime column detected.")

st.markdown("---")

# =====================================
# 5. MISSINGNESS BY MONTH
# =====================================
st.subheader("5️⃣ Missingness by Month")

if time_col:
    df_raw["Month"] = df_raw[time_col].dt.to_period("M").astype(str)

    monthly_missing = (
        df_raw.isnull()
        .groupby(df_raw["Month"])
        .mean() * 100
    ).round(2)

    monthly_missing.index = monthly_missing.index.astype(str)
    monthly_missing["Average Missing %"] = monthly_missing.mean(axis=1)

    high_missing_months = monthly_missing[monthly_missing["Average Missing %"] > 30]

    c1, c2 = st.columns(2)
    c1.metric("Months Analysed", monthly_missing.shape[0])
    c2.metric("High-Missing Months", high_missing_months.shape[0])

    fig = px.imshow(
        monthly_missing.drop(columns=["Average Missing %"]).T,
        aspect="auto",
        color_continuous_scale="Oranges",
        title="Missingness Heatmap by Month (%)",
        labels={"x": "Month", "y": "Columns", "color": "Missing %"}
    )

    fig.update_xaxes(side="bottom", tickangle=45)
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("⚠ No datetime column detected — cannot compute monthly missingness.")

st.markdown("---")

# =====================================
# 6. MISSING PERIOD SUMMARY
# =====================================
st.subheader("6️⃣ Missing Period Summary")

def missing_periods(series):
    series = series.isnull().astype(int)
    shifted = series.shift(1, fill_value=0)

    starts = (series == 1) & (shifted == 0)
    ends = (series == 0) & (shifted == 1)

    start_times = df_raw.loc[starts.index[starts], time_col].tolist()
    end_times = df_raw.loc[ends.index[ends], time_col].tolist()

    if len(end_times) < len(start_times):
        end_times.append(df_raw[time_col].iloc[-1])

    periods = pd.DataFrame({"Start": start_times, "End": end_times})
    periods["Duration"] = periods["End"] - periods["Start"]
    return periods

if time_col:
    col_to_check = st.selectbox(
        "Select column for missing period summary:",
        df_raw.columns,
        key="missing_period_col"
    )

    periods = missing_periods(df_raw[col_to_check])

    if len(periods) == 0:
        st.success("✅ No missing periods detected for this column.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Missing Periods", len(periods))
        c2.metric("Longest Missing Gap", str(periods["Duration"].max()))

else:
    st.warning("No datetime column found — cannot compute missing periods.")

st.markdown("---")

# =====================================
# 7. RAW VARIABLE BEHAVIOUR
# =====================================
st.subheader("7️⃣ Raw Variable Behaviour")

if numeric_cols:
    selected_col = st.selectbox(
        "Select numeric column:",
        numeric_cols,
        key="raw_visual_col"
    )

    fig = px.histogram(
        df_raw,
        x=selected_col,
        nbins=40,
        marginal="box",
        title=f"{selected_col} Distribution and Outlier Overview"
    )

    fig.update_traces(
        hovertemplate=f"{selected_col}: %{{x}}<br>Records: %{{y}}<extra></extra>"
    )

    fig.update_layout(
        height=460,
        xaxis_title=selected_col,
        yaxis_title="Number of Records"
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    if time_col:
        fig = px.line(
            df_raw,
            x=time_col,
            y=selected_col,
            title=f"{selected_col} Trend Over Time"
        )

        fig.update_traces(
            hovertemplate="Time: %{x}<br>Value: %{y}<extra></extra>"
        )

        fig.update_layout(
            height=460,
            xaxis_title="Time",
            yaxis_title=selected_col
        )

        st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("No numeric columns found for visualisation.")

st.markdown("---")

# =====================================
# 8. CORRELATION HEATMAP
# =====================================
st.subheader("8️⃣ Correlation Heatmap")

if len(numeric_cols) > 1:
    corr = df_raw[numeric_cols].corr().round(2)

    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    corr_masked = corr.mask(mask)

    fig = px.imshow(
        corr_masked,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="One-Sided Correlation Heatmap",
        aspect="auto"
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{x}</b> vs <b>%{y}</b><br>" +
        "Correlation: %{z}<extra></extra>"
    )

    fig.update_layout(
        height=650,
        xaxis_title="Parameters",
        yaxis_title="Parameters",
        coloraxis_colorbar=dict(
            title="Relationship",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=[
                "Strong Negative",
                "Moderate Negative",
                "No Relation",
                "Moderate Positive",
                "Strong Positive"
            ]
        )
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("Not enough numeric columns for correlation heatmap.")

# =====================================
# FOOTER
# =====================================
render_footer()