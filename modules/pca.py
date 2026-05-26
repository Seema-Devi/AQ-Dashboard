import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ui_components import render_footer

px.defaults.template = "plotly_white"
CONFIG = {"displayModeBar": False, "scrollZoom": False}


# ======================================================
# MAIN RENDER FUNCTION
# ======================================================
def render_pca():
    """Combined Feature Engineering + PCA page.

    Layout is intentionally vertical:
    1. Feature Engineering first
    2. PCA second
    This avoids the congested dashboard look.
    """

    # ======================================================
    # CSS - LIGHT DASHBOARD STYLE
    # ======================================================
    st.markdown(
        """
    <style>
    .block-container {
        padding-top: 1rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
        background: #f7faff;
    }

    .page-title {
        font-size: 30px;
        font-weight: 950;
        color: #07184a;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 14px;
        font-weight: 600;
        color: #475569;
        margin-bottom: 16px;
    }

    .filter-card {
        background: #ffffff;
        border: 1px solid #dbe7ff;
        border-radius: 14px;
        padding: 9px 12px 0 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #dbe7ff;
        border-radius: 14px;
        padding: 10px 12px;
        min-height: 74px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05);
    }

    .metric-row {
        display: flex;
        align-items: center;
        gap: 9px;
    }

    .metric-icon {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 950;
        flex-shrink: 0;
    }

    .icon-blue {background:#e8f1ff; color:#2563eb;}
    .icon-green {background:#e9f9ef; color:#16a34a;}
    .icon-purple {background:#f2eaff; color:#7c3aed;}
    .icon-orange {background:#fff4df; color:#f59e0b;}
    .icon-red {background:#fff0f4; color:#e11d48;}
    .icon-cyan {background:#e6fbff; color:#0891b2;}

    .metric-label {
        font-size: 11px;
        color: #0f172a;
        font-weight: 850;
        margin-bottom: 2px;
    }

    .metric-value {
        font-size: 20px;
        line-height: 1.05;
        color: #061345;
        font-weight: 950;
    }

    .metric-note {
        font-size: 10.5px;
        color: #64748b;
        font-weight: 700;
        margin-top: 2px;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #dbe7ff;
        border-radius: 16px;
        padding: 12px 14px 10px 14px;
        box-shadow: 0 4px 13px rgba(15, 23, 42, 0.05);
        margin-bottom: 14px;
    }

    .section-title {
        font-size: 15px;
        font-weight: 950;
        color: #0057e7;
        margin-bottom: 8px;
    }

    .mini-title {
        font-size: 13px;
        font-weight: 900;
        color: #07184a;
        margin-bottom: 8px;
    }

    .check-row {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #0f172a;
        font-size: 13px;
        font-weight: 650;
        margin: 8px 0;
    }

    .check-icon {
        color: #16a34a;
        font-weight: 950;
    }

    .note-box {
        background: linear-gradient(90deg, #eef6ff 0%, #ffffff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 13px 16px;
        color: #07184a;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 13px 16px;
        color: #14532d;
        font-size: 13px;
        font-weight: 750;
        margin-bottom: 14px;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }

    .info-item {
        background: #ffffff;
        border: 1px solid #dbe7ff;
        border-radius: 16px;
        padding: 14px 15px;
        min-height: 105px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    .info-heading {
        font-size: 13px;
        font-weight: 950;
        color: #07184a;
        margin-bottom: 6px;
    }

    .info-text {
        font-size: 12.5px;
        color: #334155;
        font-weight: 650;
        line-height: 1.45;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        font-weight: 850 !important;
    }

    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 14px !important;
        border: 1px solid #bfdbfe !important;
        background: #ffffff !important;
        color: #0057e7 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ======================================================
    # HELPERS
    # ======================================================
    def clean_name(text):
        return (
            str(text).lower()
            .replace(" ", "")
            .replace("_", "")
            .replace(".", "")
            .replace("-", "")
            .replace("₂", "2")
        )

    def find_col(df, possible_names):
        for name in possible_names:
            n = clean_name(name)
            for col in df.columns:
                if n in clean_name(col):
                    return col
        return None

    def detect_time_column(df):
        for col in df.columns:
            cl = str(col).lower()
            if "date" in cl or "time" in cl or "timestamp" in cl:
                return col
        return None

    def is_time_or_id_column(col):
        col_clean = clean_name(col)
        exclude_contains = [
            "year", "date", "timestamp", "datetime", "index", "unnamed",
            "recordid", "sensorid", "stationid", "id"
        ]
        return any(k in col_clean for k in exclude_contains)

    def get_season(month):
        # Southern Hemisphere seasons for New Zealand
        if month in [12, 1, 2]:
            return "Summer"
        if month in [3, 4, 5]:
            return "Autumn"
        if month in [6, 7, 8]:
            return "Winter"
        return "Spring"

    def kpi_card(title, value, note, icon, icon_class):
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-row">
                    <div class="metric-icon {icon_class}">{icon}</div>
                    <div>
                        <div class="metric-label">{title}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-note">{note}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def card_start(title):
        st.markdown(f'<div class="section-card"><div class="section-title">{title}</div>', unsafe_allow_html=True)

    def card_end():
        st.markdown("</div>", unsafe_allow_html=True)

    def style_fig(fig, height=330, legend=True):
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=10, t=28, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11, color="#07184a"),
            legend=dict(orientation="h", y=1.13, x=0.5, xanchor="center") if legend else None,
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        return fig

    def build_feature_engineering(df_base, time_col, targets):
        df_temp = df_base.copy()
        df_temp[time_col] = pd.to_datetime(df_temp[time_col], errors="coerce")
        df_temp = df_temp.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

        # Time features
        df_temp["hour"] = df_temp[time_col].dt.hour
        df_temp["day_of_week"] = df_temp[time_col].dt.dayofweek
        df_temp["month"] = df_temp[time_col].dt.month
        df_temp["is_weekend"] = df_temp["day_of_week"].isin([5, 6]).astype(int)
        df_temp["hour_sin"] = np.sin(2 * np.pi * df_temp["hour"] / 24)
        df_temp["hour_cos"] = np.cos(2 * np.pi * df_temp["hour"] / 24)
        df_temp["day_sin"] = np.sin(2 * np.pi * df_temp["day_of_week"] / 7)
        df_temp["day_cos"] = np.cos(2 * np.pi * df_temp["day_of_week"] / 7)
        df_temp["month_sin"] = np.sin(2 * np.pi * df_temp["month"] / 12)
        df_temp["month_cos"] = np.cos(2 * np.pi * df_temp["month"] / 12)
        df_temp["season"] = df_temp["month"].apply(get_season)

        # Wind direction cyclic features
        if "WD" in df_temp.columns:
            df_temp["WD"] = pd.to_numeric(df_temp["WD"], errors="coerce")
            df_temp["wind_dir_sin"] = np.sin(2 * np.pi * df_temp["WD"] / 360)
            df_temp["wind_dir_cos"] = np.cos(2 * np.pi * df_temp["WD"] / 360)

        # Interaction / domain features
        if "TRAFFICV" in df_temp.columns and "WS" in df_temp.columns:
            df_temp["traffic_dispersion"] = pd.to_numeric(df_temp["TRAFFICV"], errors="coerce") / (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)

        if "TEMP" in df_temp.columns and "RH" in df_temp.columns:
            df_temp["thermal_humidity_index"] = pd.to_numeric(df_temp["TEMP"], errors="coerce") * (pd.to_numeric(df_temp["RH"], errors="coerce") / 100)

        if "PM2.5" in df_temp.columns and "NO2" in df_temp.columns:
            df_temp["pm25_no2_ratio"] = pd.to_numeric(df_temp["PM2.5"], errors="coerce") / (pd.to_numeric(df_temp["NO2"], errors="coerce") + 1)

        if "WS" in df_temp.columns and "PM2.5" in df_temp.columns:
            df_temp["pm25_wind_dispersion"] = pd.to_numeric(df_temp["PM2.5"], errors="coerce") / (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)

        if "WS" in df_temp.columns and "NO2" in df_temp.columns:
            df_temp["no2_wind_dispersion"] = pd.to_numeric(df_temp["NO2"], errors="coerce") / (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)

        if "WS" in df_temp.columns and "AQI" in df_temp.columns:
            df_temp["aqi_wind_dispersion"] = pd.to_numeric(df_temp["AQI"], errors="coerce") / (pd.to_numeric(df_temp["WS"], errors="coerce") + 1)

        # Lag and rolling features with leakage control
        lag_steps = [1, 2, 3, 6, 12, 24]
        for target in targets:
            df_temp[target] = pd.to_numeric(df_temp[target], errors="coerce")
            for lag in lag_steps:
                df_temp[f"{target}_lag_{lag}"] = df_temp[target].shift(lag)

            df_temp[f"{target}_roll_mean_6"] = df_temp[target].shift(1).rolling(6).mean()
            df_temp[f"{target}_roll_std_6"] = df_temp[target].shift(1).rolling(6).std()
            df_temp[f"{target}_roll_mean_24"] = df_temp[target].shift(1).rolling(24).mean()
            df_temp[f"{target}_roll_std_24"] = df_temp[target].shift(1).rolling(24).std()

        df_temp = pd.get_dummies(df_temp, columns=["season"], drop_first=True)

        lag_roll_cols = [c for c in df_temp.columns if "_lag_" in c or "_roll_" in c]
        before_drop = len(df_temp)
        if lag_roll_cols:
            df_temp = df_temp.dropna(subset=lag_roll_cols)

        df_temp = df_temp.reset_index(drop=True)
        summary = {
            "base_rows": len(df_base),
            "base_cols": df_base.shape[1],
            "engineered_rows": len(df_temp),
            "engineered_cols": df_temp.shape[1],
            "rows_removed": before_drop - len(df_temp),
            "features_added": df_temp.shape[1] - df_base.shape[1],
        }
        return df_temp, summary

    def classify_feature(col, original_cols):
        if col in original_cols:
            return None
        if "_lag_" in col:
            return "Lag Features"
        if "_roll_" in col:
            return "Rolling Statistics"
        if col in ["hour", "day_of_week", "month", "is_weekend"] or col.endswith("_sin") or col.endswith("_cos"):
            return "Time-based Features"
        if "dispersion" in col or "ratio" in col or "index" in col:
            return "Interaction Features"
        if col.startswith("season_"):
            return "Domain Features"
        return "Other Features"

    def feature_summary_table(df_engineered, original_cols):
        rows = []
        for col in df_engineered.columns:
            group = classify_feature(col, original_cols)
            if not group:
                continue
            desc, example = "Created feature", "-"
            if "_lag_" in col:
                desc = "Previous time-step value"
                example = "t-" + col.split("_lag_")[-1]
            elif "roll_mean" in col:
                desc = "Rolling mean with leakage control"
                example = "Previous window"
            elif "roll_std" in col:
                desc = "Rolling standard deviation"
                example = "Previous window"
            elif col.endswith("_sin") or col.endswith("_cos"):
                desc = "Cyclical time/wind encoding"
                example = "sin/cos transform"
            elif "ratio" in col:
                desc = "Pollutant ratio feature"
                example = "A / B"
            elif "dispersion" in col:
                desc = "Wind dispersion interaction"
                example = "value / wind"
            elif "index" in col:
                desc = "Weather interaction index"
                example = "TEMP × RH"
            rows.append({"Feature Name": col, "Type": group, "Description": desc, "Example": example})
        return pd.DataFrame(rows)

    def plot_feature_distribution(feature_df):
        if feature_df.empty:
            return None
        counts = feature_df["Type"].value_counts().reset_index()
        counts.columns = ["Feature Type", "Count"]
        fig = px.bar(counts, x="Feature Type", y="Count", color="Feature Type", text="Count", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        return style_fig(fig, height=285, legend=False)

    def prepare_pca_data(df, target_col, candidate_cols, max_features=24):
        exclude = [target_col]
        pollutant_names = ["AQI", "PM2.5", "NO2"]
        exclude += [c for c in pollutant_names if c in df.columns and c != target_col]

        numeric_cols = []
        for col in candidate_cols:
            if col in exclude or is_time_or_id_column(col):
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().sum() >= 20 and s.nunique(dropna=True) > 1:
                numeric_cols.append(col)

        if len(numeric_cols) < 2:
            return None, None, [], pd.DataFrame()

        corr_rows = []
        if target_col in df.columns:
            for col in numeric_cols:
                temp = df[[target_col, col]].copy()
                temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
                temp[col] = pd.to_numeric(temp[col], errors="coerce")
                valid = temp.dropna()
                if len(valid) >= 20 and valid[col].nunique() > 1:
                    r = valid[target_col].corr(valid[col])
                    if not pd.isna(r):
                        corr_rows.append({"Feature": col, "Correlation": r, "Absolute Correlation": abs(r)})

        corr_df = pd.DataFrame(corr_rows).sort_values("Absolute Correlation", ascending=False) if corr_rows else pd.DataFrame()
        selected = corr_df["Feature"].head(max_features).tolist() if not corr_df.empty else numeric_cols[:max_features]

        X = df[selected].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        X = X.dropna(axis=1, how="all")

        usable = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        X = X[usable]
        if X.shape[1] < 2:
            return None, None, [], corr_df

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X, X_scaled, usable, corr_df

    def run_pca(X_scaled, feature_names):
        n_components = min(len(feature_names), X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_scaled)
        pc_cols = [f"PC{i+1}" for i in range(n_components)]

        scores_df = pd.DataFrame(scores, columns=pc_cols)
        explained = pd.DataFrame({
            "Component": pc_cols,
            "Explained Variance %": pca.explained_variance_ratio_ * 100,
            "Cumulative Variance %": np.cumsum(pca.explained_variance_ratio_) * 100,
        })
        loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=pc_cols)
        return pca, scores_df, explained, loadings

    def plot_variance(explained, selected_components):
        plot_df = explained.head(min(20, len(explained))).copy()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=plot_df["Component"],
            y=plot_df["Explained Variance %"],
            name="Individual Explained Variance",
            marker_color="#60a5fa",
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["Component"],
            y=plot_df["Cumulative Variance %"],
            name="Cumulative Explained Variance",
            mode="lines+markers",
            line=dict(color="#22c55e", width=3),
            marker=dict(size=6),
            yaxis="y2",
        ))
        if selected_components <= len(plot_df):
            x_val = plot_df.loc[selected_components - 1, "Component"]
            y_val = plot_df.loc[selected_components - 1, "Cumulative Variance %"]
            fig.add_vline(x=x_val, line_dash="dash", line_color="#334155")
            fig.add_hline(y=y_val, line_dash="dash", line_color="#fb7185")
            fig.add_annotation(x=x_val, y=y_val, text=f"{y_val:.2f}%", showarrow=False, yshift=12, font=dict(color="#e11d48"))
        fig.update_layout(
            xaxis_title="Principal Components",
            yaxis=dict(title="Individual Variance (%)"),
            yaxis2=dict(title="Cumulative Variance (%)", overlaying="y", side="right", range=[0, 105]),
        )
        return style_fig(fig, height=360, legend=True)

    def plot_correlation_heatmap(df, features):
        cols = features[:8]
        if len(cols) < 2:
            return None
        corr = df[cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title=""))
        return style_fig(fig, height=330, legend=False)

    def plot_pca_scatter(scores_df, target_values=None):
        if scores_df.shape[1] < 2:
            return None
        temp = scores_df[["PC1", "PC2"]].copy()
        if target_values is not None:
            temp["Target"] = pd.to_numeric(target_values, errors="coerce").reset_index(drop=True)
            color = "Target"
        else:
            temp["Target"] = np.arange(len(temp))
            color = "Target"
        fig = px.scatter(temp, x="PC1", y="PC2", color=color, opacity=0.65, color_continuous_scale="Viridis")
        fig.update_traces(marker=dict(size=4))
        return style_fig(fig, height=330, legend=False)

    def plot_top_contributions(corr_df):
        if corr_df.empty:
            return None
        plot_df = corr_df.head(12).copy().sort_values("Absolute Correlation", ascending=True)
        fig = px.bar(
            plot_df,
            x="Absolute Correlation",
            y="Feature",
            orientation="h",
            text="Absolute Correlation",
            color="Absolute Correlation",
            color_continuous_scale="Blues",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Importance Score", yaxis_title="")
        return style_fig(fig, height=390, legend=False)

    # ======================================================
    # LOAD DATA
    # ======================================================
    df_clean = st.session_state.get("df_cleaned")
    if df_clean is None:
        df_clean = st.session_state.get("df")

    if df_clean is None:
        st.warning("⬅ Please upload and clean your dataset before using Feature Engineering & PCA.")
        render_footer()
        st.stop()

    df_base = df_clean.copy()
    time_col = detect_time_column(df_base)

    if time_col is None:
        st.error("No datetime column found. Please check your cleaned dataset.")
        render_footer()
        st.stop()

    df_base[time_col] = pd.to_datetime(df_base[time_col], errors="coerce")
    df_base = df_base.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    AQI_COL = find_col(df_base, ["AQI", "air quality index"])
    PM25_COL = find_col(df_base, ["PM2.5", "PM25", "PM 2.5"])
    NO2_COL = find_col(df_base, ["NO2", "NO₂", "nitrogen dioxide"])
    TARGETS = [c for c in [AQI_COL, NO2_COL, PM25_COL] if c]

    if not TARGETS:
        st.warning("AQI, NO2 or PM2.5 column was not found.")
        render_footer()
        st.stop()

    # ======================================================
    # HEADER
    # ======================================================
    h_left, h_target, h_date = st.columns([3.4, 1.1, 1.6], gap="large")

    with h_left:
        st.markdown('<div class="page-title">⚙️ Feature Engineering & PCA Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-subtitle">Clean feature creation followed by PCA dimensionality reduction for AQI, NO₂ and PM2.5 modelling.</div>',
            unsafe_allow_html=True,
        )

    with h_target:
        st.markdown('<div class="filter-card">', unsafe_allow_html=True)
        selected_target = st.selectbox("Parameter", TARGETS, index=0, key="fe_pca_target")
        st.markdown('</div>', unsafe_allow_html=True)

    with h_date:
        period = f"{df_base[time_col].min().strftime('%d %b %Y')} - {df_base[time_col].max().strftime('%d %b %Y')}"
        st.markdown(f'<div class="filter-card"><b>Date Range</b><br>{period}</div>', unsafe_allow_html=True)

    if st.button("🔧 Generate / Refresh Feature Set", use_container_width=False):
        df_engineered, fe_summary = build_feature_engineering(df_base, time_col, TARGETS)
        st.session_state["df_engineered"] = df_engineered
        st.session_state["fe_summary"] = fe_summary
        st.success("✅ Feature engineering refreshed successfully.")

    if "df_engineered" not in st.session_state:
        df_engineered, fe_summary = build_feature_engineering(df_base, time_col, TARGETS)
        st.session_state["df_engineered"] = df_engineered
        st.session_state["fe_summary"] = fe_summary
    else:
        df_engineered = st.session_state["df_engineered"].copy()
        fe_summary = st.session_state.get("fe_summary", {})

    original_cols = set(df_base.columns)
    feature_df = feature_summary_table(df_engineered, original_cols)

    # ======================================================
    # SECTION 1: FEATURE ENGINEERING FIRST
    # ======================================================
    st.markdown("### 1. Feature Engineering")
    st.markdown(
        '<div class="note-box">Feature engineering creates model-ready time, lag, rolling and interaction variables. Rolling features use previous values only.</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4, gap="medium")
    with f1:
        kpi_card("Original Columns", fe_summary.get("base_cols", df_base.shape[1]), "Before feature engineering", "▦", "icon-blue")
    with f2:
        kpi_card("Final Columns", fe_summary.get("engineered_cols", df_engineered.shape[1]), "After feature engineering", "⌘", "icon-green")
    with f3:
        kpi_card("Features Added", max(0, fe_summary.get("features_added", 0)), "New model inputs", "+", "icon-purple")
    with f4:
        kpi_card("Model Rows", f"{len(df_engineered):,}", "After lag/rolling rows removed", "↧", "icon-orange")

    fe_col1, fe_col2, fe_col3 = st.columns([0.9, 1.15, 1.55], gap="medium")

    with fe_col1:
        card_start("Feature Types Created")
        for item in [
            "Time-based features",
            "Lag features",
            "Rolling statistics",
            "Interaction features",
            "Ratio/domain features",
        ]:
            st.markdown(
                f'<div class="check-row"><span class="check-icon">✓</span>{item}</div>',
                unsafe_allow_html=True,
            )
        card_end()

    with fe_col2:
        card_start("Feature Type Distribution")
        fig_dist = plot_feature_distribution(feature_df)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True, config=CONFIG)
        else:
            st.info("No engineered features found.")
        card_end()

    with fe_col3:
        card_start("Sample of Engineered Features")
        if not feature_df.empty:
            st.dataframe(feature_df.head(8), use_container_width=True, hide_index=True, height=295)
        else:
            st.info("No new engineered features were created.")
        card_end()

    with st.expander("📋 Open engineered dataset preview"):
        st.dataframe(df_engineered.head(8), use_container_width=True)

    # ======================================================
    # SECTION 2: PCA SECOND
    # ======================================================
    candidate_cols = df_engineered.select_dtypes(include=np.number).columns.tolist()
    X, X_scaled, pca_features, corr_df = prepare_pca_data(df_engineered, selected_target, candidate_cols, max_features=24)

    if X is None:
        st.warning("Not enough valid numeric features are available for PCA.")
        render_footer()
        st.stop()

    pca, pca_scores, explained, loadings = run_pca(X_scaled, pca_features)
    selected_components = int((explained["Cumulative Variance %"] < 92).sum() + 1)
    selected_components = max(1, min(selected_components, len(explained)))
    variance_explained = explained.loc[selected_components - 1, "Cumulative Variance %"]
    features_removed = max(0, len(pca_features) - selected_components)

    pca_output = pd.concat(
        [df_engineered.reset_index(drop=True), pca_scores.iloc[:, :selected_components].reset_index(drop=True)],
        axis=1,
    )

    st.session_state["pca_focus_target"] = selected_target
    st.session_state["pca_scores"] = pca_scores.iloc[:, :selected_components]
    st.session_state["pca_explained_variance"] = explained
    st.session_state["pca_loadings"] = loadings
    st.session_state["pca_output"] = pca_output
    # Final single modelling dataset used by Modelling & Forecasting page.
    # It contains engineered features plus selected PCA component columns.
    st.session_state["df_model_ready"] = pca_output
    st.session_state["pca_features_used"] = pca_features
    st.session_state["pca_selected_components"] = selected_components
    st.session_state["pca_variance_explained"] = variance_explained

    st.markdown("### 2. PCA Dimensionality Reduction")
    st.markdown(
        '<div class="note-box">PCA is applied after feature engineering and scaling. PCA features are best for Random Forest and XGBoost; GRU should use sequential features.</div>',
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4, gap="medium")
    with p1:
        kpi_card("PCA Input Features", len(pca_features), "Numeric engineered drivers", "▣", "icon-blue")
    with p2:
        kpi_card("Selected Components", selected_components, "Retain ~92% variance", "PCA", "icon-purple")
    with p3:
        kpi_card("Variance Explained", f"{variance_explained:.2f}%", "Cumulative variance", "◔", "icon-orange")
    with p4:
        kpi_card("Features Reduced", features_removed, "Simplified feature space", "↘", "icon-green")

    pca_chart_left, pca_chart_right = st.columns(2, gap="medium")

    with pca_chart_left:
        card_start("Explained Variance by Principal Components")
        st.plotly_chart(plot_variance(explained, selected_components), use_container_width=True, config=CONFIG)
        card_end()

    with pca_chart_right:
        card_start("Top Original Features Contributing to PCA Input")
        contrib_fig = plot_top_contributions(corr_df)
        if contrib_fig:
            st.plotly_chart(contrib_fig, use_container_width=True, config=CONFIG)
        else:
            st.info("No contribution ranking available.")
        card_end()

    with st.expander("📋 Open PCA summary tables"):
        summary_df = pd.DataFrame({
            "Metric": [
                "PCA Input Features", "Selected Components", "Variance Explained",
                "Variance Remaining", "Method", "Strategy"
            ],
            "Value": [
                len(pca_features), selected_components, f"{variance_explained:.2f}%",
                f"{100 - variance_explained:.2f}%", "StandardScaler + PCA", "Retain ~92% variance"
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.dataframe(explained.round(3), use_container_width=True, hide_index=True)

    # ======================================================
    # KEY INFORMATION + DOWNLOAD
    # ======================================================
    st.markdown("### 3. Key Information")
    top_features = ", ".join(corr_df["Feature"].head(5).tolist()) if not corr_df.empty else "Not available"

    st.markdown(
        f"""
        <div class="info-grid">
            <div class="info-item">
                <div class="info-heading">Feature Engineering</div>
                <div class="info-text">{max(0, fe_summary.get('features_added', 0))} new features created from {fe_summary.get('base_cols', df_base.shape[1])} original columns.</div>
            </div>
            <div class="info-item">
                <div class="info-heading">PCA Components</div>
                <div class="info-text">{selected_components} components selected to retain {variance_explained:.2f}% of the information.</div>
            </div>
            <div class="info-item">
                <div class="info-heading">Important Drivers</div>
                <div class="info-text">{top_features} are strongest inputs for {selected_target}.</div>
            </div>
            <div class="info-item">
                <div class="info-heading">Next Step</div>
                <div class="info-text">Use PCA features for Random Forest/XGBoost and sequential engineered features for GRU.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="success-box">✅ Final modelling dataset saved as df_model_ready. It includes engineered features plus selected PCA components. Metadata is also saved separately.</div>',
        unsafe_allow_html=True,
    )

    csv = pca_output.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Engineered + PCA Data (CSV)",
        data=csv,
        file_name=f"engineered_pca_data_for_{clean_name(selected_target)}.csv",
        mime="text/csv",
        use_container_width=True,
    )

 
