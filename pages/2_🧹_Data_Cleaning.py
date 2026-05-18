import streamlit as st
import pandas as pd
import numpy as np
from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

# =========================
# LOAD DATA
# =========================
df_before = st.session_state.get("df")

if df_before is None:
    st.warning("""
⬅ Please upload both datasets from the Home page to continue.

App Workflow:
1️⃣ Upload datasets  
2️⃣ Explore data  
3️⃣ Clean dataset  
4️⃣ EDA  
5️⃣ Feature engineering  
6️⃣ Forecasting
""")
    render_footer()
    st.stop()

df = df_before.copy()

st.header("🧹 Data Cleaning")

st.markdown("""
This page applies simple cleaning steps:
- replace infinite values
- fix datetime column
- remove duplicates
- replace negative numeric values
- optionally drop high-missing columns/months
- fill missing values using interpolation and median fallback
""")

# =========================
# CSS KPI CARDS
# =========================
st.markdown("""
<style>
.kpi-card {
    padding: 16px;
    border-radius: 14px;
    color: white;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.12);
}
.kpi-title {
    font-size: 14px;
    font-weight: 700;
}
.kpi-value {
    font-size: 25px;
    font-weight: 900;
    margin-top: 6px;
}
.blue-card { background: linear-gradient(135deg, #2563eb, #60a5fa); }
.green-card { background: linear-gradient(135deg, #059669, #34d399); }
.orange-card { background: linear-gradient(135deg, #ea580c, #fb923c); }
.red-card { background: linear-gradient(135deg, #dc2626, #f87171); }
.purple-card { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
</style>
""", unsafe_allow_html=True)


def kpi_card(col, title, value, css_class):
    col.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# INITIAL SUMMARY
# =========================
st.subheader("1️⃣ Dataset Before Cleaning")

missing_before_pct = (
    df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    if df.shape[0] > 0 and df.shape[1] > 0 else 0
)

duplicate_before = df.duplicated().sum()

c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Rows", f"{df.shape[0]:,}", "blue-card")
kpi_card(c2, "Columns", df.shape[1], "green-card")
kpi_card(c3, "Missing %", f"{missing_before_pct:.2f}%", "orange-card")
kpi_card(c4, "Duplicates", duplicate_before, "red-card")

with st.expander("📋 Open original data preview"):
    st.dataframe(df.head(5), use_container_width=True)

st.markdown("---")

# =========================
# BASIC CLEANING
# =========================
st.subheader("2️⃣ Basic Cleaning Setup")

df = df.replace([np.inf, -np.inf], np.nan)

time_col = None
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)
    st.success(f"Detected datetime column: {time_col}")
else:
    st.warning("No datetime column detected.")

# remove duplicates
rows_before_dup = len(df)
df = df.drop_duplicates()
duplicates_removed = rows_before_dup - len(df)

# replace negative numeric values with NaN
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

negative_counts = {}

for col in numeric_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        negative_counts[col] = int(neg_count)
        df.loc[df[col] < 0, col] = np.nan

st.info(f"Removed duplicate rows: {duplicates_removed}")

if negative_counts:
    with st.expander("📋 Open negative value replacement table"):
        st.dataframe(
            pd.DataFrame({
                "Column": list(negative_counts.keys()),
                "Negative Values Replaced": list(negative_counts.values())
            }),
            use_container_width=True,
            hide_index=True
        )
else:
    st.success("No negative numeric values detected.")

st.markdown("---")

# =========================
# HIGH MISSING COLUMNS
# =========================
st.subheader("3️⃣ Optional: Drop High-Missing Columns")

missing_percent = df.isnull().mean() * 100

threshold = st.slider(
    "Drop columns with missing percentage above:",
    min_value=10,
    max_value=90,
    value=40,
    step=5
)

high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()

with st.expander("📋 Open high-missing columns table"):
    high_missing_df = pd.DataFrame({
        "Column": high_missing_cols,
        "Missing %": missing_percent[high_missing_cols].round(2).values
    })

    st.dataframe(
        high_missing_df,
        use_container_width=True,
        hide_index=True
    )

drop_cols = st.checkbox(
    "Drop high-missing columns",
    value=False
)

if drop_cols and high_missing_cols:
    df = df.drop(columns=high_missing_cols)
    st.success(f"Dropped {len(high_missing_cols)} high-missing columns.")
elif drop_cols and not high_missing_cols:
    st.info("No columns above the selected threshold.")

st.markdown("---")

# =========================
# HIGH MISSING MONTHS
# =========================
st.subheader("4️⃣ Optional: Drop High-Missing Months")

drop_months = False
bad_months = []

if time_col:

    df["_Month"] = df[time_col].dt.to_period("M").astype(str)

    monthly_missing = (
        df.drop(columns=["_Month"], errors="ignore")
        .isnull()
        .groupby(df["_Month"])
        .mean() * 100
    ).round(2)

    monthly_missing["Average Missing %"] = monthly_missing.mean(axis=1)

    month_threshold = st.slider(
        "Drop months with average missing percentage above:",
        min_value=10,
        max_value=90,
        value=30,
        step=5
    )

    bad_months = monthly_missing[
        monthly_missing["Average Missing %"] > month_threshold
    ].index.tolist()

    with st.expander("📋 Open monthly missingness table"):
        st.dataframe(monthly_missing, use_container_width=True)

    if bad_months:
        st.warning(f"High-missing months detected: {bad_months}")
    else:
        st.success("No high-missing months detected.")

    drop_months = st.checkbox(
        "Drop high-missing months",
        value=False
    )

    if drop_months and bad_months:
        df = df[~df["_Month"].isin(bad_months)]
        st.success(f"Dropped {len(bad_months)} high-missing months.")

    df = df.drop(columns=["_Month"], errors="ignore")

else:
    st.warning("Cannot check monthly missingness because no datetime column was detected.")

st.markdown("---")

# =========================
# CLEANING STRATEGY
# =========================
st.subheader("5️⃣ Cleaning Strategy")

strategy_df = pd.DataFrame({
    "Column Group": [
        "Datetime",
        "Duplicate rows",
        "Infinite values",
        "Negative numeric values",
        "Pollutants",
        "Weather variables",
        "Traffic / pedestrian variables"
    ],
    "Cleaning Method": [
        "Parse to datetime and sort chronologically",
        "Remove duplicate rows",
        "Replace with NaN",
        "Replace with NaN",
        "Time interpolation + median fallback",
        "Time interpolation + median fallback",
        "Median fallback"
    ]
})

with st.expander("📋 Open cleaning strategy table"):
    st.dataframe(
        strategy_df,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# =========================
# FINAL CLEANING
# =========================
st.subheader("6️⃣ Run Final Cleaning")

if st.button("🧹 Run Final Data Cleaning"):

    df_cleaned = df.copy()

    numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

    pollutant_cols = [
        col for col in ["AQI", "PM2.5", "PM10", "NO", "NO2", "NOX"]
        if col in df_cleaned.columns
    ]

    weather_cols = [
        col for col in ["TEMP", "RH", "WS"]
        if col in df_cleaned.columns
    ]

    traffic_cols = [
        col for col in ["TRAFFICV", "TOTAL_PEDESTRIANS", "CITY_CENTRE_TVCOUNT"]
        if col in df_cleaned.columns
    ]

    # time interpolation for pollutants and weather
    if time_col:
        df_cleaned = df_cleaned.set_index(time_col)

        for col in pollutant_cols + weather_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
            df_cleaned[col] = df_cleaned[col].interpolate(method="time")
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        df_cleaned = df_cleaned.reset_index()

    else:
        for col in pollutant_cols + weather_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # wind direction: forward/back fill, then median fallback
    if "WD" in df_cleaned.columns:
        df_cleaned["WD"] = pd.to_numeric(df_cleaned["WD"], errors="coerce")
        df_cleaned["WD"] = df_cleaned["WD"].ffill().bfill()
        df_cleaned["WD"] = df_cleaned["WD"].fillna(df_cleaned["WD"].median())

    # traffic / pedestrian median
    for col in traffic_cols:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # remaining numeric columns median fallback
    final_numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

    for col in final_numeric_cols:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # remaining non-numeric columns
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == "object":
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)

    st.session_state["df_cleaned"] = df_cleaned

    st.success("✅ Dataset cleaned successfully and saved for EDA / Feature Engineering.")

    # =========================
    # AFTER CLEANING SUMMARY
    # =========================
    st.subheader("7️⃣ Cleaning Results")

    missing_after_pct = (
        df_cleaned.isna().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1]) * 100
        if df_cleaned.shape[0] > 0 and df_cleaned.shape[1] > 0 else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    kpi_card(c1, "Final Rows", f"{df_cleaned.shape[0]:,}", "blue-card")
    kpi_card(c2, "Final Columns", df_cleaned.shape[1], "green-card")
    kpi_card(c3, "Final Missing %", f"{missing_after_pct:.2f}%", "orange-card")
    kpi_card(c4, "Duplicates Removed", duplicates_removed, "red-card")

    before_missing = df_before.isnull().sum()
    after_missing = df_cleaned.isnull().sum()

    comparison_df = pd.DataFrame({
        "Column": df_cleaned.columns,
        "Before Missing": before_missing.reindex(df_cleaned.columns).fillna(0).astype(int).values,
        "After Missing": after_missing.reindex(df_cleaned.columns).fillna(0).astype(int).values
    })

    with st.expander("📋 Open before vs after missing values table"):
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    with st.expander("📋 Open cleaned data preview"):
        st.dataframe(
            df_cleaned.head(5),
            use_container_width=True
        )

    summary_text = f"""
Data Cleaning Summary:
- Infinite values replaced with NaN
- Datetime column detected: {time_col}
- Duplicate rows removed: {duplicates_removed}
- Negative numeric values replaced with NaN
- High-missing column threshold: {threshold}%
- High-missing columns dropped: {drop_cols}
- High-missing months dropped: {drop_months}
- Pollutants and weather variables cleaned using time interpolation + median fallback
- Traffic variables cleaned using median fallback
- Cleaned dataset saved as st.session_state["df_cleaned"]
"""

    with st.expander("📋 Open cleaning summary"):
        st.code(summary_text, language="text")

    csv = df_cleaned.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_aqi_dataset.csv",
        mime="text/csv"
    )

else:
    st.warning("Final cleaning has not been applied yet.")

render_footer()