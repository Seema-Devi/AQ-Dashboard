import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None

try:
    from sklearn.impute import KNNImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# FAST DATA CLEANING PAGE
# Main speed changes:
# 1. Compare imputation methods only on important scientific columns.
# 2. Use smaller sample for comparison.
# 3. Remove repeated RMSE/correlation masked validation loops.
# 4. Use light MissForest-style settings.
# 5. Cache comparison results.
# 6. Final cleaning runs only the selected method once.
# ============================================================


def render_cleaning():
    # =========================
    # LOAD DATA
    # =========================
    df_before = st.session_state.get("df")

    if df is None:
        st.warning(
            """
        ⬅ Please follow the application workflow before using this page:

        1️⃣ Go to the Home page  
        2️⃣ Upload the required datasets from the sidebar  
        3️⃣ Confirm the uploaded datasets on the Data Processing page   
        4️⃣ Continue with Data Insights & Cleaning
        """
        )
        st.stop()

    df = df_before.copy()

    st.markdown('<div class="main-title"> Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-banner">
        ℹ️This page compares <b>KNN</b>, <b>MissForest-style</b>, and <b>MICE</b>
        only on key air-quality variables using KS distribution similarity.
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # CSS
    # =========================
    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
        max-width: 100%;
    }

    .main-title {
        font-size: 30px;
        font-weight: 900;
        color: #020b4d;
        margin-bottom: 12px;
    }

    .info-banner {
        background: linear-gradient(90deg, #eaf3ff, #f8fbff);
        border: 1px solid #cfe2ff;
        border-radius: 9px;
        padding: 11px 14px;
        color: #061345;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 22px;
        font-weight: 900;
        color: #0433d9;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .badge {
        background: linear-gradient(135deg, #315dff, #6d5cff);
        color: white;
        font-weight: 900;
        border-radius: 8px;
        padding: 3px 10px;
        box-shadow: 0 2px 5px rgba(49,93,255,0.3);
    }

    .kpi-card {
        border-radius: 13px;
        padding: 12px 9px;
        text-align: center;
        min-height: 94px;
        border: 1.5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .kpi-title {
        font-size: 14px;
        font-weight: 900;
        margin-bottom: 4px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 950;
        color: #080826;
        line-height: 1.05;
        margin-top: 6px;
        word-break: break-word;
    }

    .blue-card {background: linear-gradient(135deg,#f4f8ff,#eef5ff); border-color:#74a7ff; color:#0645d8;}
    .green-card {background: linear-gradient(135deg,#f3fff7,#eefcf3); border-color:#79d49a; color:#078037;}
    .orange-card {background: linear-gradient(135deg,#fff9ef,#fff4e6); border-color:#ffa442; color:#ef6400;}
    .purple-card {background: linear-gradient(135deg,#fbf7ff,#f7efff); border-color:#b076ff; color:#6618c8;}
    .red-card {background: linear-gradient(135deg,#fff7f7,#fff0f1); border-color:#ff7a83; color:#d51b2c;}

    .compact-note {
        font-size: 13px;
        color: #051449;
        background: linear-gradient(90deg, #eaf4ff, #f5faff);
        border: 1px solid #d5e1ff;
        border-radius: 8px;
        padding: 8px 11px;
        font-weight: 600;
    }

    .stAlert {border-radius: 9px;}
    hr {margin: 0.4rem 0;}
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # HELPER FUNCTIONS
    # =========================
    def section_header(number, title):
        st.markdown(
            f"""
            <div class="section-header">
                <span class="badge">{number}</span>
                <span>{title}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    def kpi_card(col, title, value, css_class):
        col.markdown(f"""
        <div class="kpi-card {css_class}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    def detect_time_column(dataframe):
        possible_names = [
            "datetime", "date_time", "date time", "timestamp", "time", "date", "sample_time", "datetimelocal"
        ]
        lower_cols = {str(col).lower().strip(): col for col in dataframe.columns}
        for name in possible_names:
            if name in lower_cols:
                return lower_cols[name]
        for col in dataframe.columns:
            col_lower = str(col).lower()
            if "date" in col_lower or "time" in col_lower:
                return col
        return None

    def normalise_name(name):
        return str(name).lower().replace(" ", "").replace("_", "").replace(".", "")

    def find_existing_columns(dataframe, possible_names):
        lookup = {normalise_name(col): col for col in dataframe.columns}
        found = []
        for name in possible_names:
            key = normalise_name(name)
            if key in lookup and lookup[key] not in found:
                found.append(lookup[key])
        return found

    def scientific_negative_columns(dataframe):
        physically_non_negative_names = [
            "AQI", "PM2.5", "PM25", "PM_2_5", "PM10", "NO", "NO2", "NOX", "SO2", "CO", "O3",
            "WS", "WIND_SPEED", "WINDSPEED", "TRAFFICV", "TRAFFIC", "TOTAL_PEDESTRIANS",
            "CITY_CENTRE_TVCOUNT", "PEDESTRIANS", "COUNT", "VOLUME"
        ]
        return find_existing_columns(dataframe, physically_non_negative_names)

    def apply_physical_rules(dataframe):
        cleaned = dataframe.copy()
        issue_counts = {}

        numeric_part = cleaned.select_dtypes(include=np.number)
        inf_count = int(np.isinf(numeric_part).sum().sum()) if not numeric_part.empty else 0
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        issue_counts["Infinite values converted to NaN"] = inf_count

        neg_cols = scientific_negative_columns(cleaned)
        negative_counts = {}
        for col in neg_cols:
            s = pd.to_numeric(cleaned[col], errors="coerce")
            neg_count = int((s < 0).sum())
            if neg_count > 0:
                negative_counts[col] = neg_count
                cleaned.loc[s < 0, col] = np.nan
        issue_counts["Negative physical values converted to NaN"] = int(sum(negative_counts.values()))

        rh_cols = find_existing_columns(cleaned, ["RH", "RELATIVE_HUMIDITY", "HUMIDITY"])
        rh_invalid = {}
        for col in rh_cols:
            s = pd.to_numeric(cleaned[col], errors="coerce")
            bad = (s < 0) | (s > 100)
            bad_count = int(bad.sum())
            if bad_count > 0:
                rh_invalid[col] = bad_count
                cleaned.loc[bad, col] = np.nan
        issue_counts["Invalid RH values converted to NaN"] = int(sum(rh_invalid.values()))

        wd_cols = find_existing_columns(cleaned, ["WD", "WIND_DIRECTION", "WINDDIRECTION"])
        wd_invalid = {}
        for col in wd_cols:
            s = pd.to_numeric(cleaned[col], errors="coerce")
            bad = (s < 0) | (s > 360)
            bad_count = int(bad.sum())
            if bad_count > 0:
                wd_invalid[col] = bad_count
                cleaned.loc[bad, col] = np.nan
        issue_counts["Invalid WD values converted to NaN"] = int(sum(wd_invalid.values()))

        return cleaned, issue_counts, negative_counts, rh_invalid, wd_invalid

    def basic_preclean(dataframe):
        cleaned = dataframe.copy()

        time_col = detect_time_column(cleaned)
        invalid_time_count = 0
        if time_col:
            cleaned[time_col] = pd.to_datetime(cleaned[time_col], errors="coerce")
            invalid_time_count = int(cleaned[time_col].isna().sum())
            cleaned = cleaned.dropna(subset=[time_col])
            cleaned = cleaned.sort_values(time_col)

        rows_before_dup = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        duplicates_removed = rows_before_dup - len(cleaned)

        cleaned, issue_counts, negative_counts, rh_invalid, wd_invalid = apply_physical_rules(cleaned)
        issue_counts["Invalid timestamps removed"] = invalid_time_count
        issue_counts["Duplicate rows removed"] = int(duplicates_removed)

        return cleaned, time_col, duplicates_removed, issue_counts, negative_counts, rh_invalid, wd_invalid

    def get_model_numeric_columns(dataframe):
        excluded = []
        for col in dataframe.columns:
            col_lower = str(col).lower().strip()
            if col_lower in ["year", "month", "day", "hour"]:
                excluded.append(col)
        return [
            col for col in dataframe.select_dtypes(include=np.number).columns.tolist()
            if col not in excluded
        ]

    def get_fast_comparison_columns(dataframe, numeric_columns):
        """Use only important air-quality columns for method comparison."""
        priority_names = [
            "AQI", "PM2.5", "PM25", "PM_2_5", "PM10", "NO2", "NO", "NOX", "O3", "SO2", "CO",
            "TEMP", "TEMPERATURE", "RH", "HUMIDITY", "RELATIVE_HUMIDITY", "WS", "WIND_SPEED", "WINDSPEED",
            "WD", "WIND_DIRECTION", "TRAFFICV", "TRAFFIC", "TOTAL_PEDESTRIANS", "CITY_CENTRE_TVCOUNT"
        ]
        important_cols = find_existing_columns(dataframe, priority_names)
        important_cols = [col for col in important_cols if col in numeric_columns]

        # Prefer columns that actually contain missing values because they matter most for imputation.
        missing_important = [col for col in important_cols if dataframe[col].isna().sum() > 0]
        if len(missing_important) >= 2:
            selected = missing_important
        else:
            selected = important_cols

        # Fallback: first numeric columns with missing values, then first numeric columns.
        if not selected:
            selected = [col for col in numeric_columns if dataframe[col].isna().sum() > 0]
        if not selected:
            selected = numeric_columns[:8]

        return selected[:10]

    def add_time_features_for_imputation(dataframe, time_col):
        model_df = dataframe.copy()
        added_features = []

        if time_col and time_col in model_df.columns:
            dt = pd.to_datetime(model_df[time_col], errors="coerce")
            if dt.notna().sum() > 0:
                model_df["__hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
                model_df["__hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
                model_df["__dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
                model_df["__dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
                model_df["__month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
                model_df["__month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
                added_features = ["__hour_sin", "__hour_cos", "__dow_sin", "__dow_cos", "__month_sin", "__month_cos"]

        return model_df, added_features

    def apply_time_interpolation_first(dataframe, time_col):
        cleaned = dataframe.copy()
        interpolated_cols = []

        if not time_col or time_col not in cleaned.columns:
            return cleaned, interpolated_cols

        smooth_cols = find_existing_columns(cleaned, ["TEMP", "TEMPERATURE", "RH", "RELATIVE_HUMIDITY", "HUMIDITY"])
        if not smooth_cols:
            return cleaned, interpolated_cols

        cleaned = cleaned.sort_values(time_col)
        original_index = cleaned.index
        temp = cleaned.set_index(time_col)

        for col in smooth_cols:
            if col in temp.columns:
                before_missing = int(temp[col].isna().sum())
                temp[col] = pd.to_numeric(temp[col], errors="coerce").interpolate(method="time", limit_direction="both")
                after_missing = int(temp[col].isna().sum())
                if before_missing > after_missing:
                    interpolated_cols.append(col)

        cleaned = temp.reset_index()
        cleaned.index = original_index
        return cleaned, interpolated_cols

    def safe_ks_stat(original_values, imputed_values):
        original_values = pd.Series(original_values).dropna()
        imputed_values = pd.Series(imputed_values).dropna()
        if len(original_values) < 5 or len(imputed_values) < 5 or ks_2samp is None:
            return np.nan
        try:
            return float(ks_2samp(original_values, imputed_values).statistic)
        except Exception:
            return np.nan

    def make_imputer(method, max_iter=2):
        if method == "KNN":
            return KNNImputer(n_neighbors=4, weights="distance")

        if method == "MissForest-style":
            estimator = RandomForestRegressor(
                n_estimators=15,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
            return IterativeImputer(
                estimator=estimator,
                max_iter=max_iter,
                initial_strategy="median",
                random_state=42,
                skip_complete=True
            )

        if method == "MICE":
            return IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=max_iter,
                initial_strategy="median",
                random_state=42,
                skip_complete=True,
                sample_posterior=False
            )

        return KNNImputer(n_neighbors=4, weights="distance")

    def apply_imputation_method(dataframe, numeric_columns, method, time_col=None, max_iter=2):
        imputed_df = dataframe.copy()

        if not numeric_columns:
            return imputed_df

        model_df, time_features = add_time_features_for_imputation(imputed_df, time_col)
        model_columns = numeric_columns + [col for col in time_features if col not in numeric_columns]
        X = model_df[model_columns].apply(pd.to_numeric, errors="coerce")

        for col in model_columns:
            if X[col].isna().all():
                X[col] = 0

        if not SKLEARN_AVAILABLE:
            imputed_values = X.fillna(X.median(numeric_only=True)).fillna(0)
        else:
            try:
                imputer = make_imputer(method, max_iter=max_iter)
                imputed_array = imputer.fit_transform(X)
                imputed_values = pd.DataFrame(imputed_array, columns=model_columns, index=imputed_df.index)
            except Exception:
                imputed_values = X.fillna(X.median(numeric_only=True)).fillna(0)

        imputed_df[numeric_columns] = imputed_values[numeric_columns]
        return imputed_df

    @st.cache_data(show_spinner=False)
    def compare_imputation_methods_fast(dataframe, numeric_columns, time_col=None, sample_rows=800, max_iter=2):
        """
        Fast comparison:
        - small sample
        - important columns only
        - KS distribution similarity only
        """
        if not numeric_columns:
            return pd.DataFrame(), pd.DataFrame(), {}

        working_cols = numeric_columns + ([time_col] if time_col and time_col in dataframe.columns else [])
        working = dataframe[working_cols].copy()

        for col in numeric_columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

        if len(working) > sample_rows:
            working = working.sample(sample_rows, random_state=42).sort_index()

        methods = ["KNN", "MissForest-style", "MICE"]
        ks_table = pd.DataFrame(index=numeric_columns, columns=methods, dtype=float)
        density_data = {}

        for method in methods:
            imputed_df = apply_imputation_method(
                working,
                numeric_columns,
                method=method,
                time_col=time_col,
                max_iter=max_iter
            )

            for col in numeric_columns:
                original_non_missing = working[col].dropna()
                missing_mask = working[col].isna()

                if missing_mask.sum() > 0:
                    imputed_missing_values = imputed_df.loc[missing_mask, col]
                else:
                    # If no missing values, compare full imputed column with observed column.
                    imputed_missing_values = imputed_df[col].dropna()

                ks_table.loc[col, method] = safe_ks_stat(original_non_missing, imputed_missing_values)
                density_data[(method, col)] = {
                    "original": original_non_missing,
                    "imputed": pd.Series(imputed_missing_values).dropna()
                }

        method_scores = []
        for method in methods:
            avg_ks = ks_table[method].mean(skipna=True)
            rank_score = ks_table[method].rank(ascending=True, na_option="bottom").mean()
            method_scores.append({
                "Method": method,
                "Avg KS": avg_ks,
                "Rank Score": rank_score
            })

        summary_table = pd.DataFrame(method_scores).sort_values("Rank Score")
        return ks_table.round(3), summary_table.round(3), density_data

    def highlight_best_low_table(metric_table):
        def style_row(row):
            styles = []
            min_value = row.min(skipna=True)
            for value in row:
                if pd.notna(value) and value == min_value:
                    styles.append("background-color: #bbf7d0; color: #14532d; font-weight: 800;")
                else:
                    styles.append("background-color: #ffffff; color: #111827;")
            return styles

        return (
            metric_table.style
            .apply(style_row, axis=1)
            .format("{:.3f}")
            .set_table_styles([
                {"selector": "th", "props": [("background-color", "#f8fafc"), ("color", "#111827"), ("font-size", "10px"), ("font-weight", "700"), ("padding", "3px")]},
                {"selector": "td", "props": [("border", "1px solid #e5e7eb"), ("font-size", "10px"), ("padding", "2px 3px")]},
            ])
        )

    def get_best_imputation_method(summary_table):
        if summary_table is None or summary_table.empty:
            return "MICE"
        best = summary_table.iloc[0]["Method"]
        if pd.isna(best):
            return "MICE"
        return str(best)

    def plot_fast_density(density_data, columns, best_method):
        if not columns:
            return None

        methods = ["KNN", "MissForest-style", "MICE"]
        method_colours = {
            "KNN": "#dc2626",
            "MissForest-style": "#16a34a",
            "MICE": "#2563eb",
        }

        n_cols = 2
        n_rows = int(np.ceil(len(columns) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, max(3.3, n_rows * 2.8)), facecolor="white")

        if n_rows == 1:
            axes = np.array([axes])

        axes_flat = axes.flatten()

        for ax, col in zip(axes_flat, columns):
            ax.set_facecolor("white")
            plotted = False
            original_plotted = False

            for method in methods:
                key = (method, col)
                if key not in density_data:
                    continue

                original = density_data[key]["original"]
                imputed = density_data[key]["imputed"]

                if not original_plotted and len(original) > 5:
                    original.plot(kind="density", ax=ax, linewidth=1.4, label="Observed", color="#111827")
                    original_plotted = True
                    plotted = True

                if len(imputed) > 5:
                    line_width = 2.4 if method == best_method else 1.2
                    alpha = 1.0 if method == best_method else 0.65
                    imputed.plot(
                        kind="density",
                        ax=ax,
                        linewidth=line_width,
                        alpha=alpha,
                        label=f"{method}{' selected' if method == best_method else ''}",
                        color=method_colours.get(method, None)
                    )
                    plotted = True

            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.16)

            for spine in ax.spines.values():
                spine.set_color("#d1d5db")

            if plotted:
                ax.legend(fontsize=7, frameon=False)
            else:
                ax.text(0.5, 0.5, "Not enough data", transform=ax.transAxes, ha="center", va="center", fontsize=8)

        for ax in axes_flat[len(columns):]:
            ax.axis("off")

        fig.tight_layout()
        return fig

    def fill_object_columns(dataframe):
        cleaned = dataframe.copy()
        object_fill_summary = {}
        for col in cleaned.columns:
            if cleaned[col].dtype == "object" or str(cleaned[col].dtype) == "category":
                missing_count = int(cleaned[col].isna().sum())
                if missing_count > 0:
                    mode_values = cleaned[col].mode(dropna=True)
                    fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"
                    cleaned[col] = cleaned[col].fillna(fill_value)
                    object_fill_summary[col] = {"Missing Filled": missing_count, "Fill Value": fill_value}
        return cleaned, object_fill_summary

    def build_outlier_summary(dataframe, numeric_columns):
        rows = []
        for col in numeric_columns:
            s = pd.to_numeric(dataframe[col], errors="coerce").dropna()
            if len(s) < 10:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((s < lower) | (s > upper)).sum())
            rows.append({
                "Column": col,
                "Possible Outliers": count,
                "Lower Limit": round(lower, 3),
                "Upper Limit": round(upper, 3),
                "Action": "Flag only - not removed automatically"
            })
        return pd.DataFrame(rows)

    # =========================
    # 1. DATASET BEFORE CLEANING
    # =========================
    section_header("1", "Dataset Before Cleaning")

    missing_before_pct = (
        df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        if df.shape[0] > 0 and df.shape[1] > 0 else 0
    )
    duplicate_before = int(df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "Rows", f"{df.shape[0]:,}", "blue-card")
    kpi_card(c2, "Columns", df.shape[1], "green-card")
    kpi_card(c3, "Missing %", f"{missing_before_pct:.2f}%", "orange-card")
    kpi_card(c4, "Duplicates", duplicate_before, "red-card")

    with st.expander("📋 Open original data preview"):
        st.dataframe(df.head(5), use_container_width=True)

    df, time_col, duplicates_removed, issue_counts, negative_counts, rh_invalid, wd_invalid = basic_preclean(df)
    df, interpolated_cols = apply_time_interpolation_first(df, time_col)


    # =========================
    # 2. SCIENTIFIC CLEANING RULES
    # =========================
    section_header("2", "Cleaning Rules Applied Before Imputation")

    rule_rows = [{"Rule": key, "Count": value} for key, value in issue_counts.items()]
    rule_rows.append({
        "Rule": "Time interpolation applied before ML imputation",
        "Count": ", ".join(interpolated_cols) if interpolated_cols else "No suitable TEMP/RH columns found or no missing values reduced"
    })

    with st.expander("📋 Open cleaning rules table"):
        st.dataframe(pd.DataFrame(rule_rows), use_container_width=True, hide_index=True)

    
    # =========================
    # 3. FAST IMPUTATION COMPARISON
    # =========================
    section_header("3", " Imputation Method Comparison")

    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn is not available, so median fallback imputation will be used.")

    all_numeric_cols = get_model_numeric_columns(df)
    comparison_cols = get_fast_comparison_columns(df, all_numeric_cols)

    auto_best_method = "MICE"
    ks_table = pd.DataFrame()
    summary_table = pd.DataFrame()
    density_data = {}

    if comparison_cols:
        with st.spinner("Fast comparing KNN, MissForest-style, and MICE..."):
            ks_table, summary_table, density_data = compare_imputation_methods_fast(
                df,
                comparison_cols,
                time_col=time_col,
                sample_rows=800,
                max_iter=2
            )

        if not summary_table.empty:
            auto_best_method = get_best_imputation_method(summary_table)
            st.success(f"Best method selected automatically: {auto_best_method}")

            with st.expander("📋 Open overall method ranking table", expanded=True):
                st.caption("Lowest KS rank score is selected. KS checks distribution similarity; lower is better.")
                st.dataframe(summary_table, use_container_width=True, hide_index=True)

            st.markdown("**KS Distribution Similarity**")
            st.caption("Lower KS is better. Green = best method for each parameter.")
            st.dataframe(
                highlight_best_low_table(ks_table),
                use_container_width=True,
                height=min(420, 35 * len(ks_table) + 70)
            )

            with st.expander("📈 Open density plot comparison"):
                fig_density = plot_fast_density(density_data, comparison_cols, auto_best_method)
                if fig_density:
                    st.pyplot(fig_density, use_container_width=True)

           
        else:
            st.info("No numeric parameters available for imputation comparison.")
    else:
        st.info("No numeric parameters found for imputation.")

    # =========================
    # 4. OUTLIER REVIEW
    # =========================
    section_header("4", "Outlier and Sensor Spike Review")

    outlier_cols = comparison_cols if comparison_cols else all_numeric_cols[:10]
    outlier_df = build_outlier_summary(df, outlier_cols)
    if not outlier_df.empty:
        with st.expander("📋 Open outlier and sensor spike review table"):
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        st.info("Possible outliers are flagged only. They are not automatically removed because high pollution spikes may represent real environmental events.")
    else:
        st.info("Not enough numeric data available for outlier review.")

    
    # =========================
    # 5. RUN FINAL CLEANING
    # =========================
    section_header("5", "Run Final Cleaning")

    if st.button("Run Final Data Cleaning"):
        df_cleaned = df.copy()
        final_numeric_cols = get_model_numeric_columns(df_cleaned)

        for col in final_numeric_cols:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

        # Fast final pass: selected method only, lower iterations.
        df_cleaned = apply_imputation_method(
            df_cleaned,
            final_numeric_cols,
            method=auto_best_method,
            time_col=time_col,
            max_iter=3
        )

        # Final numeric fallback if any values remain missing.
        for col in final_numeric_cols:
            if df_cleaned[col].isna().any():
                median_value = df_cleaned[col].median()
                if pd.isna(median_value):
                    median_value = 0
                df_cleaned[col] = df_cleaned[col].fillna(median_value)

        df_cleaned, object_fill_summary = fill_object_columns(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
        st.session_state["df_cleaned"] = df_cleaned

        st.success(f"✅ Dataset cleaned successfully using {auto_best_method} and saved for EDA / Feature Engineering.")

        section_header("6", "Cleaning Results")

        missing_after_pct = (
            df_cleaned.isna().sum().sum() / (df_cleaned.shape[0] * df_cleaned.shape[1]) * 100
            if df_cleaned.shape[0] > 0 and df_cleaned.shape[1] > 0 else 0
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        kpi_card(c1, "Final Rows", f"{df_cleaned.shape[0]:,}", "blue-card")
        kpi_card(c2, "Final Columns", df_cleaned.shape[1], "green-card")
        kpi_card(c3, "Final Missing %", f"{missing_after_pct:.2f}%", "orange-card")
        kpi_card(c4, "Duplicates Removed", duplicates_removed, "red-card")
        kpi_card(c5, "Method", auto_best_method, "purple-card")

        before_missing = df_before.isnull().sum()
        after_missing = df_cleaned.isnull().sum()

        comparison_df = pd.DataFrame({
            "Column": df_cleaned.columns,
            "Before Missing": before_missing.reindex(df_cleaned.columns).fillna(0).astype(int).values,
            "After Missing": after_missing.reindex(df_cleaned.columns).fillna(0).astype(int).values
        })

        with st.expander("📋 Open before vs after missing values table"):
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        with st.expander("📋 Open categorical filling summary"):
            if object_fill_summary:
                st.dataframe(
                    pd.DataFrame([{"Column": col, **values} for col, values in object_fill_summary.items()]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No categorical/object missing values required filling.")

        with st.expander("📋 Open cleaned data preview"):
            st.dataframe(df_cleaned.head(5), use_container_width=True)

        summary_text = f"""
Data Cleaning Summary:
- Infinite values replaced with NaN: {issue_counts.get('Infinite values converted to NaN', 0)}
- Datetime column detected: {time_col}
- Invalid timestamps removed: {issue_counts.get('Invalid timestamps removed', 0)}
- Duplicate rows removed: {duplicates_removed}
- Negative values converted to NaN only for physically non-negative variables: {issue_counts.get('Negative physical values converted to NaN', 0)}
- RH outside 0-100 converted to NaN: {issue_counts.get('Invalid RH values converted to NaN', 0)}
- WD outside 0-360 converted to NaN: {issue_counts.get('Invalid WD values converted to NaN', 0)}
- Time interpolation applied first for smooth meteorological columns: {interpolated_cols if interpolated_cols else 'None'}
- Fast imputation comparison used columns: {comparison_cols}
- Compared KNN, MissForest-style, and MICE using KS distribution similarity only
- Automatically selected final imputation method: {auto_best_method}
- Final cleaning ran the selected method only once
- Possible outliers were flagged for review, not automatically removed
- Categorical missing values filled using mode where possible
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
