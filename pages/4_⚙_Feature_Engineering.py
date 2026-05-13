import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

# ======================================================
# LOAD CLEANED DATA 
# ======================================================
df_clean = st.session_state.get("df_cleaned")
if df_clean is None:
    st.warning("""
    ⬅ Please clean your dataset first before running Feature Engineering.

   App Workflow:

1️⃣ Upload datasets from the Home page  
2️⃣ Explore data in Data Visualisation  
3️⃣ Clean and preprocess the dataset  
4️⃣ Perform EDA and analyse trends  
5️⃣ Apply Feature Engineering  
6️⃣ Train models and view AQI Forecasting
""")
    st.stop()

if "df_fe_base" not in st.session_state:
    st.session_state["df_fe_base"] = df_clean.copy()

df_fe_base = st.session_state["df_fe_base"]

st.header("⚙ Feature Engineering")

# ======================================================
# REDUNDANCY MANAGEMENT (PROTECTED TARGETS FIXED)
# ======================================================
st.subheader("🗑 Redundancy Management")

numeric_cols = df_fe_base.select_dtypes(include=np.number).columns
corr_matrix = df_fe_base[numeric_cols].corr().abs()

# 1. Define the protected targets that should NEVER be deleted
protected_targets = ["AQI", "PM2.5", "NO2"]

# 2. Find redundant pairs (> 0.80)
redundant_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        if corr_matrix.iloc[i, j] > 0.80:
            redundant_pairs.append((numeric_cols[i], numeric_cols[j]))

# 3. Identify potential columns to remove, EXCLUDING protected targets
suggested_to_remove = set()
for a, b in redundant_pairs:
    if a not in protected_targets:
        suggested_to_remove.add(a)
    elif b not in protected_targets:
        suggested_to_remove.add(b)

suggested_list = sorted(list(suggested_to_remove))

if suggested_list:
    st.info("The following features show high multicollinearity. Select those you wish to remove:")
    
    # Multi-select list for redundancy
    to_delete = st.multiselect("Select redundant features to delete:", options=suggested_list)
    
    # Single delete button for the selection
    if st.button("🗑 Delete Selected Features"):
        if to_delete:
            st.session_state["df_fe_base"].drop(columns=to_delete, inplace=True)
            st.success(f"Successfully removed: {', '.join(to_delete)}")
            st.rerun()
        else:
            st.warning("Please select at least one feature to delete.")
else:
    st.success("No redundant features detected (Correlation < 0.80) or all high-correlation features are protected targets.")

st.markdown("---")


# ======================================================
# RUN FEATURE ENGINEERING
# ======================================================
run_fe = st.button("🚀 Run Feature Engineering")

if run_fe or "df_engineered" in st.session_state:
    if run_fe:
        with st.spinner("Generating engineered features..."):
            progress = st.progress(0)
            df_temp = df_fe_base.copy().sort_values("DATETIME_HOUR")

            # 1. TIME FEATURES
            progress.progress(25)
            if "DATETIME_HOUR" in df_temp.columns:
                df_temp["hour_sin"] = np.sin(2 * np.pi * df_temp["DATETIME_HOUR"].dt.hour / 24)
                df_temp["hour_cos"] = np.cos(2 * np.pi * df_temp["DATETIME_HOUR"].dt.hour / 24)
                df_temp["day_sin"] = np.sin(2 * np.pi * df_temp["DATETIME_HOUR"].dt.dayofweek / 7)
                df_temp["day_cos"] = np.cos(2 * np.pi * df_temp["DATETIME_HOUR"].dt.dayofweek / 7)

            # 2. LAGS & ROLLING
            progress.progress(75)
            targets = [t for t in protected_targets if t in df_temp.columns]
            for t in targets:
                df_temp[f"{t}_lag_1"] = df_temp[t].shift(1)
                df_temp[f"{t}_lag_24"] = df_temp[t].shift(24)
                df_temp[f"{t}_roll_mean_24"] = df_temp[t].rolling(24).mean()

            progress.progress(100)
            df_temp = df_temp.dropna().reset_index(drop=True)
            st.session_state["df_engineered"] = df_temp

    df_display = st.session_state["df_engineered"]
    st.success("✅ Engineering Complete")
    
    st.subheader("📋 Engineered Dataset Preview")
    st.dataframe(df_display.head(10), use_container_width=True)
    
    st.subheader("📊 Feature Correlation Matrix")
    final_numeric = df_display.select_dtypes(include=np.number)
    final_corr = final_numeric.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(final_corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🎯 Feature Relevance (Target Correlations)")
    
    available_targets = [t for t in protected_targets if t in df_display.columns]
    
    if available_targets:
        target_corr = df_display.select_dtypes(include=np.number).corr()[available_targets]
        if "AQI" in available_targets:
            target_corr = target_corr.sort_values(by="AQI", ascending=False)
        
        st.write(f"Numerical correlation coefficients for **{', '.join(available_targets)}**.")
        st.dataframe(target_corr.style.format("{:.2f}"), use_container_width=True)
        
        with st.expander("💡 Note on Correlation Values"):
            st.write("""
            - Values range from -1 to +1.
            - **Positive:** Feature and pollution increase together.
            - **Negative:** Feature increases while pollution decreases.
            """)
    else:
        st.info("Target variables not found.")

    st.download_button(
        label="📥 Download Dataset", 
        data=df_display.to_csv(index=False), 
        file_name="engineered_data.csv", 
        mime="text/csv"
    )

render_footer()
