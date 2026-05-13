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
    st.stop()

if "df_fe_base" not in st.session_state:
    st.session_state["df_fe_base"] = df_clean.copy()

df_fe_base = st.session_state["df_fe_base"]

st.header("⚙️ Advanced Atmospheric Feature Engineering")
st.markdown("---")

# ======================================================
# SECTION 1: REDUNDANCY MANAGEMENT (EXCLUDING TARGETS)
# ======================================================
st.subheader("🗑️ 1. Multicollinearity & Redundancy Filter")

numeric_cols = df_fe_base.select_dtypes(include=np.number).columns
corr_matrix = df_fe_base[numeric_cols].corr().abs()

protected_targets = ["AQI", "PM2.5", "NO2"]

redundant_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        if corr_matrix.iloc[i, j] > 0.80:
            redundant_pairs.append((numeric_cols[i], numeric_cols[j]))

suggested_to_remove = set()
for a, b in redundant_pairs:
    if a not in protected_targets: suggested_to_remove.add(a)
    elif b not in protected_targets: suggested_to_remove.add(b)

suggested_list = sorted(list(suggested_to_remove))

if sorted(list(suggested_to_remove)):
    st.info("The following features show high redundancy (>0.80 correlation). Select features to prune:")
    to_delete = st.multiselect("Select features to delete:", options=suggested_list)
    
    if st.button("🗑️ Purge Selected Features"):
        if to_delete:
            st.session_state["df_fe_base"].drop(columns=to_delete, inplace=True)
            st.success(f"Successfully pruned: {', '.join(to_delete)}")
            st.rerun()
        else:
            st.warning("Please pick at least one feature to drop.")
else:
    st.success("✅ Multi-collinearity check passed. No redundant non-target features detected.")

st.markdown("---")

# ======================================================
# SECTION 2: ADVANCED ATMOSPHERIC RUN ENGINEERING
# ======================================================
st.subheader("🚀 2. Atmospheric & Cyclical Feature Extraction")
run_fe = st.button("🔧 Generate Advanced Feature Set")

if run_fe or "df_engineered" in st.session_state:
    if run_fe:
        with st.spinner("Processing deep architectural features..."):
            progress = st.progress(0)
            df_temp = df_fe_base.copy().sort_values("DATETIME_HOUR")

            # A. Cyclical Time Transformations (Continuous Time Loop)
            progress.progress(20)
            if "DATETIME_HOUR" in df_temp.columns:
                df_temp["hour"] = df_temp["DATETIME_HOUR"].dt.hour
                df_temp["hour_sin"] = np.sin(2 * np.pi * df_temp["hour"] / 24)
                df_temp["hour_cos"] = np.cos(2 * np.pi * df_temp["hour"] / 24)
                df_temp["day_sin"] = np.sin(2 * np.pi * df_temp["DATETIME_HOUR"].dt.dayofweek / 7)
                df_temp["day_cos"] = np.cos(2 * np.pi * df_temp["DATETIME_HOUR"].dt.dayofweek / 7)

            # B. Pollution-Weather Interaction (Atmospheric Dispersion Logic)
            progress.progress(50)
            if "TRAFFICV" in df_temp.columns and "WS" in df_temp.columns:
                df_temp["traffic_dispersion"] = df_temp["TRAFFICV"] / (df_temp["WS"] + 1)
            if "TEMP" in df_temp.columns and "RH" in df_temp.columns:
                df_temp["thermal_humidity_index"] = df_temp["TEMP"] * (df_temp["RH"] / 100)

            # C. Multi-Step Deep Lags & Moving Statistics (Persistence Modeling)
            progress.progress(80)
            for t in [t for t in protected_targets if t in df_temp.columns]:
                df_temp[f"{t}_lag_1"] = df_temp[t].shift(1)
                df_temp[f"{t}_lag_24"] = df_temp[t].shift(24)
                df_temp[f"{t}_roll_mean_24"] = df_temp[t].rolling(24).mean()
                df_temp[f"{t}_roll_std_24"] = df_temp[t].rolling(24).std()

            progress.progress(100)
            df_temp = df_temp.dropna().reset_index(drop=True)
            st.session_state["df_engineered"] = df_temp

    df_display = st.session_state["df_engineered"]
    
    st.subheader("📋 Engineered Dataset Preview")
    st.dataframe(df_display.head(10), use_container_width=True)
    st.markdown("---")

    # ======================================================
    # SECTION 3: PRO VISUALISATION & EXPLAINABILITY
    # ======================================================
    st.subheader("📊 3. Statistical Mapping & Matrix Analysis")
    
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        st.write("**Complete Correlation Matrix**")
        final_numeric = df_display.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(final_numeric.corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
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
            st.dataframe(target_corr.style.format("{:.2f}"), use_container_width=True)
        else:
            st.caption("No primary target parameters available.")

    # ======================================================
    # SECTION 4: EXECUTIVE SUMMARY & ENVIRONMENTAL CARDS
    # ======================================================
    st.markdown("---")
    st.subheader("🏥 4. Data-Driven Environmental Risk Assessment")
    
    if "AQI" in df_display.columns:
        avg_aqi = df_display["AQI"].mean()
        max_aqi = df_display["AQI"].max()
        
        c_risk, c_stats = st.columns(2)
        with c_risk:
            if avg_aqi <= 50:
                st.success(f"🍃 **Auckland Baseline: Good** (Avg AQI: {avg_aqi:.1f}). Ambient values sit safely below standard thresholds.")
            elif avg_aqi <= 100:
                st.warning(f"⚠️ **Auckland Baseline: Moderate** (Avg AQI: {avg_aqi:.1f}). Potential risks present for highly sensitive communities.")
            else:
                st.error(f"🚨 **Auckland Baseline: Poor** (Avg AQI: {avg_aqi:.1f}). Action recommended for public health mitigation.")
                
        with c_stats:
            st.markdown(f"""
            **Dataset Intelligence Briefing:**
            *   **Maximum Recorded Peak:** `{max_aqi:.1f}` units.
            *   **Engineered Dimensions:** `{df_display.shape[1]}` columns prepared for training.
            *   **Time Horizon Continuity:** Temporal features fully mapped mathematically using cyclical coordinates.
            """)

    # DOWNLOAD UTILITY
    st.download_button(
        label="📥 Download Production-Ready Dataset (CSV)", 
        data=df_display.to_csv(index=False), 
        file_name="auckland_council_engineered_data.csv", 
        mime="text/csv"
    )

render_footer()
