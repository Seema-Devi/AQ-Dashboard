import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

px.defaults.template = "plotly_white"

config = {
    "displayModeBar": False,
    "scrollZoom": False
}

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

df_fe_base = st.session_state["df_fe_base"].copy()

st.header("⚙️ Feature Engineering")

st.markdown("""
<div style="
background-color:white;
padding:20px;
border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08);
margin-bottom:20px;
">
This page converts the cleaned dataset into a model-ready forecasting dataset by creating
time features, wind-direction features, weather interaction features, lag features and rolling statistics.
</div>
""", unsafe_allow_html=True)

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>
.kpi-card {
    padding: 18px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.12);
}
.kpi-title {
    font-size: 14px;
    font-weight: 700;
}
.kpi-value {
    font-size: 28px;
    font-weight: 900;
    margin-top: 6px;
}
.blue-card {
    background: linear-gradient(135deg, #2563eb, #60a5fa);
}
.green-card {
    background: linear-gradient(135deg, #059669, #34d399);
}
.orange-card {
    background: linear-gradient(135deg, #ea580c, #fb923c);
}
.purple-card {
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
}
.red-card {
    background: linear-gradient(135deg, #dc2626, #f87171);
}
</style>
""", unsafe_allow_html=True)


def kpi_card(col, title, value, css_class):
    col.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# DETECT TIME COLUMN
# ======================================================
time_col = None

for col in df_fe_base.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col is None:
    st.error("No datetime column found. Please check your cleaned dataset.")
    render_footer()
    st.stop()

df_fe_base[time_col] = pd.to_datetime(df_fe_base[time_col], errors="coerce")
df_fe_base = df_fe_base.dropna(subset=[time_col]).sort_values(time_col)

protected_targets = ["AQI", "PM2.5", "NO2"]
available_targets = [t for t in protected_targets if t in df_fe_base.columns]

# ======================================================
# 1. BASE DATASET SUMMARY
# ======================================================
st.subheader("1️⃣ Base Dataset Summary")

c1, c2, c3 = st.columns(3)

kpi_card(c1, "Base Rows", f"{len(df_fe_base):,}", "blue-card")
kpi_card(c2, "Base Columns", df_fe_base.shape[1], "green-card")
kpi_card(c3, "Forecast Targets", len(available_targets), "orange-card")

with st.expander("📋 Open base dataset preview"):
    st.dataframe(
        df_fe_base.head(10),
        use_container_width=True
    )

st.markdown("---")

# ======================================================
# 2. MULTICOLLINEARITY CHECK
# ======================================================
st.subheader("2️⃣ Multicollinearity & Redundancy Check")

numeric_cols = df_fe_base.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) > 1:
    corr_matrix = df_fe_base[numeric_cols].corr().abs()

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
                    "Correlation": round(corr_value, 2)
                })

    if redundant_pairs:
        st.info("""
Some variables show high correlation above 0.80, meaning they may contain overlapping information.
They are kept for now because tree-based models can handle correlated predictors, and feature importance
will help identify useful variables later.
""")

        redundant_df = pd.DataFrame(redundant_pairs)

        with st.expander("📋 Open redundant feature pairs table"):
            st.dataframe(
                redundant_df,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.success("✅ No highly redundant numeric feature pairs detected.")
else:
    st.warning("Not enough numeric columns available for redundancy analysis.")

st.markdown("---")

# ======================================================
# 3. FEATURE GENERATION
# ======================================================
st.subheader("3️⃣ Generate Forecasting Features")

st.info("""
Leakage control:
Rolling features are created using `shift(1)` before rolling calculations.
This prevents the current target value from leaking into predictor features.
""")

run_fe = st.button("🔧 Generate Model-Ready Feature Set")

if run_fe or "df_engineered" in st.session_state:

    if run_fe:

        with st.spinner("Generating model-ready features..."):

            progress = st.progress(0)
            df_temp = df_fe_base.copy()

            # ======================================================
            # A. TIME FEATURES
            # ======================================================
            progress.progress(15)

            df_temp["hour"] = df_temp[time_col].dt.hour
            df_temp["day_of_week"] = df_temp[time_col].dt.dayofweek
            df_temp["month"] = df_temp[time_col].dt.month
            df_temp["year"] = df_temp[time_col].dt.year
            df_temp["is_weekend"] = df_temp["day_of_week"].isin([5, 6]).astype(int)

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

            # ======================================================
            # B. WIND DIRECTION CYCLICAL FEATURES
            # ======================================================
            progress.progress(30)

            if "WD" in df_temp.columns:
                df_temp["WD"] = pd.to_numeric(df_temp["WD"], errors="coerce")
                df_temp["wind_dir_sin"] = np.sin(2 * np.pi * df_temp["WD"] / 360)
                df_temp["wind_dir_cos"] = np.cos(2 * np.pi * df_temp["WD"] / 360)

            # ======================================================
            # C. INTERACTION FEATURES
            # ======================================================
            progress.progress(50)

            if "TRAFFICV" in df_temp.columns and "WS" in df_temp.columns:
                df_temp["traffic_dispersion"] = (
                    pd.to_numeric(df_temp["TRAFFICV"], errors="coerce") /
                    (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)
                )

            if "TEMP" in df_temp.columns and "RH" in df_temp.columns:
                df_temp["thermal_humidity_index"] = (
                    pd.to_numeric(df_temp["TEMP"], errors="coerce") *
                    (pd.to_numeric(df_temp["RH"], errors="coerce") / 100)
                )

            if "PM2.5" in df_temp.columns and "NO2" in df_temp.columns:
                df_temp["pm25_no2_ratio"] = (
                    pd.to_numeric(df_temp["PM2.5"], errors="coerce") /
                    (pd.to_numeric(df_temp["NO2"], errors="coerce") + 1)
                )

            if "WS" in df_temp.columns and "PM2.5" in df_temp.columns:
                df_temp["pm25_wind_dispersion"] = (
                    pd.to_numeric(df_temp["PM2.5"], errors="coerce") /
                    (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)
                )

            if "WS" in df_temp.columns and "NO2" in df_temp.columns:
                df_temp["no2_wind_dispersion"] = (
                    pd.to_numeric(df_temp["NO2"], errors="coerce") /
                    (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)
                )

            if "AQI" in df_temp.columns and "WS" in df_temp.columns:
                df_temp["aqi_wind_dispersion"] = (
                    pd.to_numeric(df_temp["AQI"], errors="coerce") /
                    (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)
                )

            # ======================================================
            # D. LAG AND ROLLING FEATURES
            # ======================================================
            progress.progress(75)

            lag_steps = [1, 2, 3, 6, 12, 24]

            for target in available_targets:
                df_temp[target] = pd.to_numeric(df_temp[target], errors="coerce")

                for lag in lag_steps:
                    df_temp[f"{target}_lag_{lag}"] = df_temp[target].shift(lag)

                df_temp[f"{target}_roll_mean_6"] = (
                    df_temp[target]
                    .shift(1)
                    .rolling(6)
                    .mean()
                )

                df_temp[f"{target}_roll_std_6"] = (
                    df_temp[target]
                    .shift(1)
                    .rolling(6)
                    .std()
                )

                df_temp[f"{target}_roll_mean_24"] = (
                    df_temp[target]
                    .shift(1)
                    .rolling(24)
                    .mean()
                )

                df_temp[f"{target}_roll_std_24"] = (
                    df_temp[target]
                    .shift(1)
                    .rolling(24)
                    .std()
                )

            # ======================================================
            # E. SEASON ENCODING
            # ======================================================
            progress.progress(90)

            df_temp = pd.get_dummies(
                df_temp,
                columns=["season"],
                drop_first=True
            )

            # Drop rows created by lag/rolling NA only
            lag_roll_cols = [
                col for col in df_temp.columns
                if "_lag_" in col or "_roll_" in col
            ]

            rows_before_drop = len(df_temp)

            if lag_roll_cols:
                df_temp = df_temp.dropna(subset=lag_roll_cols)

            rows_after_drop = len(df_temp)

            df_temp = df_temp.reset_index(drop=True)

            progress.progress(100)

            st.session_state["df_engineered"] = df_temp
            st.session_state["df_model_ready"] = df_temp

            st.session_state["fe_summary"] = {
                "base_rows": len(df_fe_base),
                "base_cols": df_fe_base.shape[1],
                "engineered_rows": rows_after_drop,
                "engineered_cols": df_temp.shape[1],
                "rows_removed": rows_before_drop - rows_after_drop,
                "features_added": df_temp.shape[1] - df_fe_base.shape[1]
            }

            st.success("✅ Model-ready feature set generated successfully.")

    df_display = st.session_state["df_engineered"]
    fe_summary = st.session_state.get("fe_summary", {})

    # ======================================================
    # 4. FEATURE ENGINEERING SUMMARY
    # ======================================================
    st.subheader("4️⃣ Feature Engineering Summary")

    c1, c2, c3, c4, c5 = st.columns(5)

    kpi_card(c1, "Original Rows", f"{fe_summary.get('base_rows', len(df_fe_base)):,}", "blue-card")
    kpi_card(c2, "Final Rows", f"{fe_summary.get('engineered_rows', len(df_display)):,}", "green-card")
    kpi_card(c3, "Original Columns", fe_summary.get("base_cols", df_fe_base.shape[1]), "orange-card")
    kpi_card(c4, "Final Columns", fe_summary.get("engineered_cols", df_display.shape[1]), "purple-card")
    kpi_card(c5, "Features Added", fe_summary.get("features_added", df_display.shape[1] - df_fe_base.shape[1]), "red-card")

    with st.expander("📋 Open engineered dataset preview"):
        st.dataframe(
            df_display.head(10),
            use_container_width=True
        )

    # ======================================================
    # 5. CREATED FEATURE GROUPS
    # ======================================================
    st.subheader("5️⃣ Created Feature Groups")

    feature_groups = []

    for col in df_display.columns:
        if col in df_fe_base.columns:
            continue

        if "_lag_" in col:
            group = "Lag Feature"
        elif "_roll_" in col:
            group = "Rolling Feature"
        elif col in ["hour", "day_of_week", "month", "year", "is_weekend"]:
            group = "Time Feature"
        elif col.endswith("_sin") or col.endswith("_cos"):
            group = "Cyclical Feature"
        elif "dispersion" in col or "ratio" in col or "index" in col:
            group = "Interaction Feature"
        elif col.startswith("season_"):
            group = "Season Encoding"
        else:
            group = "Other Feature"

        feature_groups.append({
            "Feature": col,
            "Feature Group": group
        })

    feature_group_df = pd.DataFrame(feature_groups)

    if not feature_group_df.empty:
        group_counts = (
            feature_group_df["Feature Group"]
            .value_counts()
            .reset_index()
        )

        group_counts.columns = ["Feature Group", "Count"]

        fig = px.bar(
            group_counts,
            x="Feature Group",
            y="Count",
            color="Feature Group",
            text="Count",
            title="Created Feature Groups"
        )

        fig.update_traces(textposition="outside")
        fig.update_layout(height=450, showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        with st.expander("📋 Open created feature list"):
            st.dataframe(
                feature_group_df,
                use_container_width=True,
                hide_index=True
            )

    # ======================================================
    # 6. TARGET DEPENDENCY / CORRELATION
    # ======================================================
    st.subheader("6️⃣ Target Dependency Analysis")

    final_numeric = df_display.select_dtypes(include=np.number)

    if len(final_numeric.columns) > 1 and available_targets:

        corr = final_numeric.corr().round(2)

        selected_target_corr = st.selectbox(
            "Select forecast target for dependency ranking",
            available_targets
        )

        corr_rank = (
            corr[selected_target_corr]
            .drop(selected_target_corr)
            .sort_values(key=abs, ascending=False)
            .reset_index()
        )
        # Remove year / YEAR columns
        corr_rank = corr_rank[~corr_rank["index"].str.lower().eq("year")]

        corr_rank.columns = ["Feature", "Correlation"]

        # Sort by absolute correlation
        corr_rank = corr_rank.reindex(corr_rank["Correlation"].abs().sort_values(ascending=False).index)

        corr_rank.columns = ["Feature", "Correlation"]

        fig = px.bar(
            corr_rank.head(20),
            x="Correlation",
            y="Feature",
            orientation="h",
            color="Correlation",
            color_continuous_scale="RdBu_r",
            title=f"Top Engineered Features Related to {selected_target_corr}"
        )

        fig.update_layout(
            height=620,
            yaxis=dict(autorange="reversed")
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        with st.expander("📋 Open full target dependency table"):
            st.dataframe(
                corr_rank.round(2),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("📋 Open full engineered correlation matrix"):
            st.dataframe(
                corr,
                use_container_width=True
            )

    else:
        st.warning("Not enough numeric engineered features for dependency analysis.")

    # ======================================================
    # 7. DOWNLOAD MODEL-READY DATASET
    # ======================================================
    st.subheader("7️⃣ Download Model-Ready Dataset")

    st.download_button(
        label="⬇️ Download Engineered Dataset",
        data=df_display.to_csv(index=False).encode("utf-8"),
        file_name="auckland_council_engineered_data.csv",
        mime="text/csv"
    )

render_footer()