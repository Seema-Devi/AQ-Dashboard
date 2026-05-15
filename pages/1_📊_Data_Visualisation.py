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
# SECTION 1 — BASIC OVERVIEW
# =====================================
st.subheader("Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", df_raw.shape[0])
c2.metric("Columns", df_raw.shape[1])
c3.metric("Numeric Columns", len(df_raw.select_dtypes(include=np.number).columns))
c4.metric("Categorical Columns", len(df_raw.select_dtypes(exclude=np.number).columns))

st.dataframe(df_raw.head(), use_container_width=True)

st.markdown("---")

# =====================================
# SUMMARY STATISTICS
# =====================================
st.subheader("Summary Statistics")

summary_df = df_raw.describe().transpose()
summary_df = summary_df.fillna("-")

st.dataframe(summary_df, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 2 — DUPLICATE DETECTION
# =====================================
st.subheader(" Duplicate Detection")

duplicates = df_raw.duplicated().sum()
st.metric("Duplicate Rows", duplicates)

if duplicates > 0:
    st.warning("⚠ Duplicate rows detected. Consider removing them during cleaning.")
else:
    st.success("✅ No duplicate rows found.")

st.markdown("---")

# =====================================
# SECTION 3 — MISSING VALUE ANALYSIS
# =====================================
st.subheader(" Missing Value Analysis")

missing_df = df_raw.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Count"]
missing_df["Missing %"] = (missing_df["Missing Count"] / len(df_raw) * 100).round(2)

st.dataframe(missing_df, use_container_width=True)

fig = px.bar(
    missing_df,
    x="Column",
    y="Missing %",
    title="Missing Percentage by Column",
    color="Missing %",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 4 — INVALID VALUE DETECTION
# =====================================
st.subheader(" Invalid Value Detection")

numeric_df = df_raw.select_dtypes(include=np.number)
invalid_negative = (numeric_df < 0).sum()

st.write("### Negative Value Check")
st.dataframe(invalid_negative.to_frame("Negative Count"))

if invalid_negative.sum() > 0:
    st.warning("⚠ Negative values detected in numeric columns.")
else:
    st.success("✅ No invalid negative values found.")

st.markdown("---")

# =====================================
# SECTION 5 — DATETIME VALIDATION
# =====================================
st.subheader(" Datetime Validation")

time_col = None
for col in df_raw.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce")

    st.info(f"Detected datetime column: **{time_col}**")

    missing_time = df_raw[time_col].isnull().sum()
    duplicate_time = df_raw[time_col].duplicated().sum()

    c1, c2 = st.columns(2)
    c1.metric("Missing Timestamps", missing_time)
    c2.metric("Duplicate Timestamps", duplicate_time)

else:
    st.warning("⚠ No datetime column detected.")

st.markdown("---")

# =====================================
# SECTION 6 — COLUMN CLASSIFICATION
# =====================================
st.subheader("Column Classification")

numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_raw.select_dtypes(exclude=np.number).columns.tolist()

classification_df = pd.DataFrame({
    "Type": ["Numeric", "Categorical", "Datetime"],
    "Count": [
        len(numeric_cols),
        len(categorical_cols),
        1 if time_col else 0
    ]
})

st.dataframe(classification_df, use_container_width=True)

st.markdown("---")

# =====================================
# DATATYPE TABLE (FIXED)
# =====================================
st.subheader("Datatype Table")

dtype_df = pd.DataFrame({
    "Column": df_raw.columns,
    "Datatype": df_raw.dtypes.astype(str)
}).reset_index(drop=True)

st.dataframe(dtype_df, use_container_width=True)

st.markdown("---")

# =====================================
# SECTION 7 — DATA COMPLETENESS SCORE
# =====================================
st.subheader("Data Completeness Score")

completeness_score = 100 - missing_df["Missing %"].mean()

st.metric("Completeness Score", f"{completeness_score:.2f}%")
st.progress(completeness_score / 100)

st.markdown("---")


# ============================================================
#   NEW SECTION: MISSING PERIOD SUMMARY
# ============================================================
st.subheader("📊 Missing Period Summary")

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
    col_to_check2 = st.selectbox("Select column for missing period summary:", df_raw.columns)
    periods = missing_periods(df_raw[col_to_check2])

    if len(periods) == 0:
        st.success("No missing periods detected for this column.")
    else:
        st.dataframe(periods, use_container_width=True)
else:
    st.warning("No datetime column found — cannot compute missing periods.")

st.markdown("---")
# ============================================================
#   NEW SECTION: MISSINGNESS BY MONTH (MISSING MONTHS DETECTION)
# ============================================================
st.subheader("📅 Missing by Month (High‑Missing Months Detection)")

if time_col:
    # Convert to month-year STRING (fixes Period JSON error)
    df_raw["Month"] = df_raw[time_col].dt.to_period("M").astype(str)

    # Calculate missingness per month
    monthly_missing = (
        df_raw.isnull()
        .groupby(df_raw["Month"])
        .mean() * 100
    ).round(2)

    # Convert index to string (fixes Plotly JSON error)
    monthly_missing.index = monthly_missing.index.astype(str)

    st.write("### Missing Percentage by Month (All Columns)")
    st.dataframe(monthly_missing, use_container_width=True)

    # Identify bad months (average missingness > 30%)
    monthly_missing["Avg Missing %"] = monthly_missing.mean(axis=1)
    bad_months = monthly_missing[monthly_missing["Avg Missing %"] > 30]

    st.write("### 🚨 High‑Missing Months (Avg Missing > 30%)")
    if bad_months.empty:
        st.success("No High‑Missing months detected — dataset is stable across months.")
    else:
        st.error("⚠ These months have high missingness and may need to be dropped:")
        st.dataframe(bad_months, use_container_width=True)

    # Heatmap for monthly missingness
    st.write("### 🔥 Missingness Heatmap by Month (%)")
    fig = px.imshow(
        monthly_missing.drop(columns=["Avg Missing %"]).T,
        aspect="auto",
        color_continuous_scale="Reds",
        title="Missingness Heatmap by Month (%)",
        labels={"x": "Month", "y": "Columns"}
    )
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠ No datetime column detected — cannot compute monthly missingness.")
    
# =====================================
# SECTION 9 — BASIC RAW VISUALS
# =====================================
st.subheader("Basic Visualisations")

selected_col = st.selectbox("Select a numeric column:", numeric_cols)

fig = px.histogram(df_raw, x=selected_col, nbins=40, title=f"{selected_col} Distribution")
st.plotly_chart(fig, use_container_width=True)

fig = px.box(df_raw, y=selected_col, title=f"{selected_col} Outliers")
st.plotly_chart(fig, use_container_width=True)

if time_col:
    fig = px.line(df_raw, x=time_col, y=selected_col, title=f"{selected_col} Over Time")
    st.plotly_chart(fig, use_container_width=True)

# =====================================
# UPDATED CORRELATION HEATMAP
# =====================================
st.subheader("🔥 Correlation Heatmap")

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap",
    aspect="auto"
)

fig.update_layout(
    xaxis_title="Parameters",
    yaxis_title="Parameters",
    margin=dict(l=40, r=40, t=60, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# =====================================
# FOOTER
# =====================================
render_footer()
