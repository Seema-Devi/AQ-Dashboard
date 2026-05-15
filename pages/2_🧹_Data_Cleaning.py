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

1️⃣ Upload datasets from the sidebar  
2️⃣ Explore the data in Data Visualisation  
3️⃣ Clean and preprocess the dataset  
4️⃣ Analyse trends and patterns in EDA  
5️⃣ Apply Feature Engineering  
6️⃣ Train models and view AQI Forecasting
""")
    render_footer()
    st.stop()

df = df_before.copy()

st.header("🧹 Data Cleaning")

# =========================
# EXPLANATION SECTION
# =========================
st.markdown("""
### Why Data Cleaning Matters  
Real-world datasets contain **missing values**, **sensor errors**, and **inconsistent timestamps**.  
Cleaning ensures your dataset becomes:

- ✔ Reliable for analysis  
- ✔ Accurate for visualisation  
- ✔ Ready for machine learning  
""")

# =========================
# 1. FIX INF VALUES
# =========================
st.subheader("Handling Infinite Values")
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# 2. IDENTIFY DATETIME COLUMN
# =========================
st.subheader(" Datetime Cleaning")
time_col = None
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col)
    st.info(f"Detected datetime column: **{time_col}**")
else:
    st.warning("⚠ No datetime column detected.")

# =========================
# 3. FIX NEGATIVE VALUES
# =========================
st.subheader("Fixing Negative Values")
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].apply(lambda x: np.nan if x is not None and x < 0 else x)

# =========================
# 4. DROP HIGH-MISSING COLUMNS (DYNAMIC)
# =========================
st.subheader(" Drop High‑Missing Columns")

missing_percent = df.isnull().mean() * 100
threshold = st.slider("Select missing % threshold for dropping columns:", 10, 90, 40)

high_missing_cols = missing_percent[missing_percent > threshold].index.tolist()

st.write("### Columns above threshold:")
st.dataframe(
    pd.DataFrame({
        "Column": high_missing_cols,
        "Missing %": missing_percent[high_missing_cols].round(2)
    }),
    use_container_width=True
)

if st.button("🗑 Drop High‑Missing Columns"):
    df = df.drop(columns=high_missing_cols)
    st.success(f"Dropped columns: {high_missing_cols}")

# =========================
# 5. DROP BAD MONTHS (OPTIONAL)
# =========================
st.subheader(" Drop High‑Missing Months")

if time_col:
    df["Month"] = df[time_col].dt.to_period("M").astype(str)

    monthly_missing = (df.isnull().groupby(df["Month"]).mean() * 100).round(2)
    monthly_missing["Avg Missing %"] = monthly_missing.mean(axis=1)

    st.write("### Monthly Missingness Table")
    st.dataframe(monthly_missing, use_container_width=True)

    bad_months = monthly_missing[monthly_missing["Avg Missing %"] > 30].index.tolist()

    st.write("### High‑Missing Months Detected")
    if len(bad_months) == 0:
        st.success("No High‑Missing months detected.")
    else:
        st.error(f"High‑Missing months (Avg Missing > 30%): {bad_months}")

    if st.button("🗑 Drop High‑Missing Months Automatically"):
        df = df[~df["Month"].isin(bad_months)]
        st.success(f"Removed {len(bad_months)} high‑missing months: {bad_months}")

else:
    st.warning("Cannot compute high‑missing months — no datetime column found.")

# =========================
# 6. RUN FINAL CLEANING BUTTON
# =========================
st.subheader("Final Data Cleaning")

st.info("After selecting/removing high-missing columns or months, click the button below to apply final cleaning.")

if st.button("🧹 Run Final Data Cleaning"):

    # TEMP — time interpolation
    if "TEMP" in df.columns and time_col:
        df = df.set_index(time_col)
        df["TEMP"] = df["TEMP"].interpolate(method="time")
        df["TEMP"] = df["TEMP"].fillna(df["TEMP"].median())
        df = df.reset_index()

    # AQI — median
    if "AQI" in df.columns:
        df["AQI"] = df["AQI"].fillna(df["AQI"].median())

    # NO, NO2, NOX — median
    pollutants = ["NO", "NO2", "NOX"]
    existing_pollutants = [col for col in pollutants if col in df.columns]

    if existing_pollutants:
        df[existing_pollutants] = df[existing_pollutants].fillna(
            df[existing_pollutants].median()
        )

    # PM10, PM2.5 — median
    for col in ["PM10", "PM2.5"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Weather variables — median
    for col in ["RH", "WS", "WD"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Traffic columns — median
    traffic_cols = ["TRAFFICV", "TOTAL_PEDESTRIANS", "CITY_CENTRE_TVCOUNT"]

    for col in traffic_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Remove temporary Month column
    if "Month" in df.columns:
        df = df.drop(columns=["Month"])

    # Save cleaned dataset
    df_cleaned = df.copy()
    st.session_state["df_cleaned"] = df_cleaned

    st.success("✅ Dataset cleaned successfully.")

    # BEFORE/AFTER COMPARISON
    st.subheader("Before vs After Cleaning Comparison")

    before_missing = df_before.isnull().sum()
    after_missing = df_cleaned.isnull().sum()

    comparison_df = pd.DataFrame({
        "Column": df_cleaned.columns,
        "Before Missing": before_missing.reindex(df_cleaned.columns).values,
        "After Missing": after_missing.reindex(df_cleaned.columns).values
    })

    st.dataframe(comparison_df, use_container_width=True)

    # CLEANING SUMMARY
    st.subheader("📝 Cleaning Summary")

    summary_text = f"""
### Data Cleaning Summary

- Infinite values replaced with NaN
- Negative pollutant values converted to NaN
- Datetime column detected: **{time_col}**
- Dataset sorted chronologically
- High-missing columns dropped above selected threshold: {threshold}%
- TEMP cleaned using time interpolation + median fallback
- AQI, pollutants, weather, and traffic columns cleaned using median imputation
"""

    st.code(summary_text, language="markdown")

    # DOWNLOAD CLEANED DATASET
    st.subheader("📥 Download Cleaned Dataset")

    csv = df_cleaned.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_aqi_dataset.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠ Final cleaning has not been applied yet. Please review columns/months first, then click the cleaning button.")


# =========================
# FOOTER
# =========================
render_footer()
