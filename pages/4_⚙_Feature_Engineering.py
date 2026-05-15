import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

# ======================================================
# DATA STORAGE ACCESSIBILITY
# ======================================================
df_clean = st.session_state.get("df_cleaned")
if df_clean is None:
    
    st.warning("""
    ⬅ Please upload and clean your dataset before engineering features.

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

if "df_fe_base" not in st.session_state:
    st.session_state["df_fe_base"] = df_clean.copy()

df_fe_base = st.session_state["df_fe_base"]

st.header("⚙️ Advanced Feature Engineering")
st.markdown("---")

# ======================================================
# SECTION 1: REDUNDANCY CHECK INFORMATION ONLY
# ======================================================
st.subheader(" Multicollinearity & Redundancy Check")

numeric_cols = df_fe_base.select_dtypes(include=np.number).columns
corr_matrix = df_fe_base[numeric_cols].corr().abs()

protected_targets = ["AQI", "PM2.5", "NO2"]

redundant_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        if corr_matrix.iloc[i, j] > 0.80:
            col_a = numeric_cols[i]
            col_b = numeric_cols[j]
            corr_value = corr_matrix.iloc[i, j]

            redundant_pairs.append({
                "Feature 1": col_a,
                "Feature 2": col_b,
                "Correlation": corr_value
            })

if redundant_pairs:
    st.info("""
    Some variables show high correlation (>0.80), meaning they may contain overlapping information.
    These variables are kept in the dataset to maintain consistent modelling results.
    Feature importance and model performance will be used later to understand their contribution.
    """)

    redundant_df = pd.DataFrame(redundant_pairs)
    st.dataframe(
        redundant_df.style.format({"Correlation": "{:.2f}"}),
        use_container_width=True
    )
else:
    st.success("✅ Multicollinearity check passed. No highly redundant numeric features detected.")

st.markdown("---")

# ======================================================
# SECTION 2: ADVANCED ATMOSPHERIC RUN ENGINEERING
# ======================================================
st.subheader(" Feature Extraction")
run_fe = st.button("🔧 Generate Advanced Feature Set")

if run_fe or "df_engineered" in st.session_state:
    if run_fe:
        with st.spinner("Processing deep architectural features..."):
            progress = st.progress(0)
            df_temp = df_fe_base.copy()

            if "DATETIME_HOUR" not in df_temp.columns:
                st.error("DATETIME_HOUR column is missing. Please check your cleaned dataset.")
                render_footer()
                st.stop()

            df_temp["DATETIME_HOUR"] = pd.to_datetime(df_temp["DATETIME_HOUR"], errors="coerce")
            df_temp = df_temp.dropna(subset=["DATETIME_HOUR"]).sort_values("DATETIME_HOUR")

            # A. Cyclical Time Transformations
            progress.progress(20)
            df_temp["hour"] = df_temp["DATETIME_HOUR"].dt.hour
            df_temp["day_of_week"] = df_temp["DATETIME_HOUR"].dt.dayofweek
            df_temp["month"] = df_temp["DATETIME_HOUR"].dt.month

            df_temp["hour_sin"] = np.sin(2 * np.pi * df_temp["hour"] / 24)
            df_temp["hour_cos"] = np.cos(2 * np.pi * df_temp["hour"] / 24)

            df_temp["day_sin"] = np.sin(2 * np.pi * df_temp["day_of_week"] / 7)
            df_temp["day_cos"] = np.cos(2 * np.pi * df_temp["day_of_week"] / 7)

            df_temp["month_sin"] = np.sin(2 * np.pi * df_temp["month"] / 12)
            df_temp["month_cos"] = np.cos(2 * np.pi * df_temp["month"] / 12)

            def get_season(month):
                if month in [12, 1, 2]:
                    return "Summer"
                elif month in [3, 4, 5]:
                    return "Autumn"
                elif month in [6, 7, 8]:
                    return "Winter"
                else:
                    return "Spring"

            df_temp["season"] = df_temp["month"].apply(get_season)

            # B. Pollution-Weather Interaction
            progress.progress(50)

            if "TRAFFICV" in df_temp.columns and "WS" in df_temp.columns:
                df_temp["traffic_dispersion"] = df_temp["TRAFFICV"] / (df_temp["WS"] + 1)

            if "TEMP" in df_temp.columns and "RH" in df_temp.columns:
                df_temp["thermal_humidity_index"] = df_temp["TEMP"] * (df_temp["RH"] / 100)

            if "PM2.5" in df_temp.columns and "NO2" in df_temp.columns:
                df_temp["pm25_no2_ratio"] = df_temp["PM2.5"] / (df_temp["NO2"] + 1)

            if "WS" in df_temp.columns and "PM2.5" in df_temp.columns:
                df_temp["pm25_wind_dispersion"] = df_temp["PM2.5"] / (df_temp["WS"] + 1)

            # C. Time-Series Lag and Rolling Features
            progress.progress(80)
            lag_steps = [1, 2, 3, 6, 12, 24]

            for t in [t for t in protected_targets if t in df_temp.columns]:
                for lag in lag_steps:
                    df_temp[f"{t}_lag_{lag}"] = df_temp[t].shift(lag)

                # shift(1) prevents current-value leakage
                df_temp[f"{t}_roll_mean_6"] = df_temp[t].shift(1).rolling(6).mean()
                df_temp[f"{t}_roll_std_6"] = df_temp[t].shift(1).rolling(6).std()

                df_temp[f"{t}_roll_mean_24"] = df_temp[t].shift(1).rolling(24).mean()
                df_temp[f"{t}_roll_std_24"] = df_temp[t].shift(1).rolling(24).std()

            progress.progress(100)

            df_temp = df_temp.dropna().reset_index(drop=True)
            st.session_state["df_engineered"] = df_temp

            st.success("✅ Advanced feature set generated successfully.")

    df_display = st.session_state["df_engineered"]
    
    st.subheader("📋 Engineered Dataset Preview")
    st.dataframe(df_display.head(10), use_container_width=True)
    st.markdown("---")

    # ======================================================
    # SECTION 3: PRO VISUALISATION & EXPLAINABILITY
    # ======================================================
    st.subheader("📊 Statistical Mapping & Matrix Analysis")
    
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        st.write("**Complete Correlation Matrix**")
        final_numeric = df_display.select_dtypes(include=np.number)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            final_numeric.corr(),
            annot=False,
            cmap="coolwarm",
            center=0,
            ax=ax
        )
        plt.xticks(fontsize=8, rotation=45)
        plt.yticks(fontsize=8)
        st.pyplot(fig)
        
    with col_table:
        st.write("**Target Dependency Values**")
        available_targets = [t for t in protected_targets if t in df_display.columns]

        if available_targets:
            target_corr = df_display.select_dtypes(include=np.number).corr()[available_targets]

            if "AQI" in available_targets:
                target_corr = target_corr.sort_values(by="AQI", ascending=False)

            st.dataframe(
                target_corr.style.format("{:.2f}"),
                use_container_width=True
            )
        else:
            st.caption("No primary target parameters available.")

    
    # DOWNLOAD UTILITY
    st.subheader("📥 Production-Ready Dataset (CSV)")
    st.download_button(
        label=" Download Dataset", 
        data=df_display.to_csv(index=False), 
        file_name="auckland_council_engineered_data.csv", 
        mime="text/csv"
    )

render_footer()