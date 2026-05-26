import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui_components import render_footer


def render_eda():

    st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
        background: #f5f7fb;
    }

    .eda-header {
        background: #ffffff;
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border: 1px solid #e5eaf5;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    .eda-title {
        font-size: 28px;
        font-weight: 950;
        color: #061345;
        margin-bottom: 2px;
    }

    .eda-subtitle {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
    }

    .metric-card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid #e5eaf5;
        padding: 18px 18px 14px 18px;
        height: 142px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    }

    .metric-label {
        color: #1e2a55;
        font-size: 12px;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 950;
        color: #061345;
        line-height: 1.05;
    }

    .metric-note {
        font-size: 12px;
        font-weight: 800;
        margin-top: 8px;
    }

    .good {color:#22c55e;}
    .warn {color:#f59e0b;}
    .bad {color:#ef4444;}
    .info {color:#3b82f6;}

    .chart-card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid #e5eaf5;
        padding: 14px 16px 8px 16px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    }

    .chart-title {
        color: #061345;
        font-size: 15px;
        font-weight: 950;
        margin-bottom: 8px;
    }

    .info-dot {
        float: right;
        color: #94a3b8;
        font-size: 15px;
    }

    .insight-card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid #e5eaf5;
        padding: 18px 20px;
        margin-top: 2px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
    }

    .insight-title {
        font-size: 17px;
        font-weight: 950;
        color: #6d28d9;
        margin-bottom: 6px;
    }

    .insight-text {
        font-size: 14px;
        color: #1e293b;
        font-weight: 600;
        line-height: 1.55;
    }

    div[data-testid="stSelectbox"] label {
        font-weight: 800;
        color: #061345;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5eaf5;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    px.defaults.template = "plotly_white"
    CONFIG = {"displayModeBar": False, "scrollZoom": False}

    # ======================================================
    # LOAD DATA
    # ======================================================
    if "df_cleaned" in st.session_state:
        df = st.session_state["df_cleaned"].copy()
    elif "df" in st.session_state:
        df = st.session_state["df"].copy()
    else:
        st.warning(
            """
        ⬅ Please follow the application workflow before using this page:

        1️⃣ Go to the Home page  
        2️⃣ Upload the required datasets from the sidebar  
        3️⃣ Confirm the uploaded dataset on the Data Processing page  
        4️⃣ Complete Data Insights & Cleaning  
        5️⃣ Return here for Exploratory Analysis & PCA
        """
        )
        render_footer()
        st.stop()

    # ======================================================
    # COLUMN DETECTION
    # ======================================================
    def clean_name(text):
        return (
            str(text).lower()
            .replace(" ", "")
            .replace("_", "")
            .replace(".", "")
            .replace("-", "")
            .replace("₂", "2")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
        )

    def find_col(possible_names):
        for name in possible_names:
            target = clean_name(name)
            for col in df.columns:
                if target in clean_name(col):
                    return col
        return None

    time_col = None
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or "time" in cl or "timestamp" in cl:
            time_col = c
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col)

    aqi_col = find_col(["AQI", "air quality index"])
    pm25_col = find_col(["PM2.5", "PM25", "PM 2.5"])
    pm10_col = find_col(["PM10", "PM 10"])
    no2_col = find_col(["NO2", "NO₂", "nitrogen dioxide"])
    temp_col = find_col(["temperature", "temp"])
    humidity_col = find_col(["humidity", "relative humidity", "rh"])
    wind_speed_col = find_col(["wind speed", "windspeed", "ws"])
    wind_dir_col = find_col(["wind direction", "winddirection", "wd"])
    traffic_col = find_col(["traffic volume", "traffic", "vehicle", "cars"])
    ped_col = find_col(["pedestrian", "pedestrians", "total pedestrians", "footfall"])

    main_targets = [c for c in [aqi_col, pm25_col, no2_col] if c]

    if not main_targets:
        st.warning("AQI, PM2.5, or NO2 columns were not found.")
        render_footer()
        st.stop()

    primary_col = aqi_col if aqi_col else main_targets[0]

    # ======================================================
    # HELPERS
    # ======================================================
    def num(col):
        if col and col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(dtype=float)

    def fmt(v, d=1):
        if pd.isna(v):
            return "N/A"
        return f"{v:,.{d}f}"

    def threshold_for(col):
        if col == aqi_col:
            return 100
        if col == pm25_col:
            return 15
        if col == no2_col:
            return 25
        return None

    def safe_corr(a, b):
        if not a or not b or a not in df.columns or b not in df.columns:
            return np.nan
        tmp = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) < 5:
            return np.nan
        return tmp.corr().iloc[0, 1]

    def get_daily(col):
        if not time_col:
            return pd.DataFrame()
        tmp = df[[time_col, col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            return tmp
        return tmp.set_index(time_col).resample("D").mean(numeric_only=True).reset_index()

    def style_fig(fig, height=300, legend=True):
        fig.update_layout(
            height=height,
            margin=dict(l=8, r=8, t=10, b=8),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11, color="#061345"),
            legend=dict(
                orientation="h",
                y=1.13,
                x=0.02,
                xanchor="left",
                font=dict(size=10)
            ) if legend else None,
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)", zeroline=False)
        return fig

    def chart_box(title):
        st.markdown(
            f'<div class="chart-card"><div class="chart-title">{title}<span class="info-dot">ⓘ</span></div>',
            unsafe_allow_html=True
        )

    def end_box():
        st.markdown("</div>", unsafe_allow_html=True)

    def metric_card(label, value, note, css_class="info"):
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note {css_class}">{note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    def status_for(col, mean_value):
        thr = threshold_for(col)
        if pd.isna(mean_value):
            return "N/A", "info"
        if thr is None:
            return "Observed", "info"
        if mean_value <= thr * 0.5:
            return "Good", "good"
        if mean_value <= thr:
            return "Moderate", "warn"
        return "High", "bad"

    # ======================================================
    # PLOTS
    # ======================================================
    def plot_trend_decomposition(col):
        daily = get_daily(col)
        if daily.empty:
            return go.Figure()

        daily["trend"] = daily[col].rolling(30, min_periods=5).mean()
        daily["month"] = daily[time_col].dt.month
        monthly_mean = daily.groupby("month")[col].transform("mean")
        overall_mean = daily[col].mean()
        daily["seasonal"] = monthly_mean - overall_mean

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Observed", "Trend", "Seasonal")
        )

        fig.add_trace(
            go.Scatter(
                x=daily[time_col],
                y=daily[col],
                mode="lines",
                name="Observed",
                line=dict(color="#2563eb", width=1.3)
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=daily[time_col],
                y=daily["trend"],
                mode="lines",
                name="Trend",
                line=dict(color="#1e3a8a", width=2.2)
            ),
            row=2,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=daily[time_col],
                y=daily["seasonal"],
                mode="lines",
                name="Seasonal",
                line=dict(color="#2dd4bf", width=1.8)
            ),
            row=3,
            col=1
        )

        fig.update_layout(showlegend=True)
        return style_fig(fig, height=300, legend=True)

    def plot_hour_day_heatmap(col):
        if not time_col:
            return go.Figure()

        tmp = df[[time_col, col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()

        if tmp.empty:
            return go.Figure()

        order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        tmp["day"] = tmp[time_col].dt.day_name().str[:3]
        tmp["hour"] = tmp[time_col].dt.hour

        pivot = (
            tmp.groupby(["day", "hour"])[col]
            .mean()
            .reset_index()
            .pivot(index="day", columns="hour", values=col)
            .reindex(order)
        )

        fig = px.imshow(
            pivot,
            aspect="auto",
            color_continuous_scale="Turbo",
            labels=dict(color=f"Avg {col}")
        )
        fig.update_xaxes(title="Hour of Day", dtick=3)
        fig.update_yaxes(title="Day of Week")
        return style_fig(fig, height=300, legend=False)

    def plot_parameter_category_donut(col):
        vals = num(col).dropna()

        if vals.empty:
            return go.Figure()

        # AQI uses standard AQI categories.
        if col == aqi_col:
            bins = [-np.inf, 50, 100, 150, 200, 300, np.inf]
            labels = [
                "Good (0-50)",
                "Moderate (51-100)",
                "Unhealthy Sensitive (101-150)",
                "Unhealthy (151-200)",
                "Very Unhealthy (201-300)",
                "Hazardous (301+)"
            ]
            colors = ["#22c55e", "#facc15", "#fb923c", "#ef4444", "#a855f7", "#be123c"]

        # PM2.5 uses common concentration bands.
        elif col == pm25_col:
            bins = [-np.inf, 5, 10, 15, 25, 35, np.inf]
            labels = [
                "Very Low (≤5)",
                "Low (5-10)",
                "Moderate (10-15)",
                "Elevated (15-25)",
                "High (25-35)",
                "Very High (>35)"
            ]
            colors = ["#22c55e", "#84cc16", "#facc15", "#fb923c", "#ef4444", "#be123c"]

        # NO2 uses practical analysis bands.
        elif col == no2_col:
            bins = [-np.inf, 10, 20, 25, 40, 60, np.inf]
            labels = [
                "Very Low (≤10)",
                "Low (10-20)",
                "Moderate (20-25)",
                "Elevated (25-40)",
                "High (40-60)",
                "Very High (>60)"
            ]
            colors = ["#22c55e", "#84cc16", "#facc15", "#fb923c", "#ef4444", "#be123c"]

        # Fallback for any other selected parameter.
        else:
            q = vals.quantile([0, 0.25, 0.5, 0.75, 0.9, 1]).values
            q = np.unique(q)

            if len(q) < 3:
                return go.Figure()

            bins = [-np.inf] + list(q[1:-1]) + [np.inf]
            labels = [f"Band {i+1}" for i in range(len(bins) - 1)]
            colors = ["#22c55e", "#84cc16", "#facc15", "#fb923c", "#ef4444", "#be123c"][:len(labels)]

        cat = pd.cut(vals, bins=bins, labels=labels, duplicates="drop")
        counts = cat.value_counts().reindex(cat.cat.categories).fillna(0)

        fig = go.Figure(go.Pie(
            labels=counts.index.astype(str),
            values=counts.values,
            hole=0.52,
            textinfo="percent",
            marker=dict(colors=colors[:len(counts)])
        ))

        fig.add_annotation(
            text=f"<b>{len(vals):,}</b><br>Total",
            showarrow=False,
            font=dict(size=15, color="#061345")
        )

        return style_fig(fig, height=300, legend=True)

    def plot_monthly_box(col):
        if not time_col:
            return go.Figure()

        tmp = df[[time_col, col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()

        if tmp.empty:
            return go.Figure()

        tmp["Month"] = tmp[time_col].dt.strftime("%b")
        month_order = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        fig = px.box(
            tmp,
            x="Month",
            y=col,
            category_orders={"Month": month_order},
            points="outliers"
        )
        fig.update_traces(marker_color="#3b82f6", line_color="#2563eb")
        fig.update_xaxes(title="")
        fig.update_yaxes(title=col)
        return style_fig(fig, height=300, legend=False)

    def plot_pollutant_bar():
        # Show all available air-quality parameters, including AQI.
        cols = [c for c in [aqi_col, pm25_col, pm10_col, no2_col] if c]

        if not cols:
            return go.Figure()

        values = [num(c).mean() for c in cols]

        fig = go.Figure(go.Bar(
            x=cols,
            y=values,
            text=[fmt(v, 1) for v in values],
            textposition="outside",
            marker_color=["#2563eb", "#ef4444", "#fb923c", "#22c55e", "#06b6d4", "#8b5cf6", "#3b82f6"][:len(cols)]
        ))

        fig.update_yaxes(title="Average value")
        fig.update_xaxes(title="")
        return style_fig(fig, height=300, legend=False)

    def plot_parameter_wind_rose(col):
        # Wind rose now changes with selected parameter.
        # It shows average selected parameter by wind direction and wind-speed band.
        if not wind_speed_col or not wind_dir_col or not col:
            return go.Figure()

        tmp = df[[wind_speed_col, wind_dir_col, col]].copy()
        tmp[wind_speed_col] = pd.to_numeric(tmp[wind_speed_col], errors="coerce")
        tmp[wind_dir_col] = pd.to_numeric(tmp[wind_dir_col], errors="coerce")
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna()

        if tmp.empty:
            return go.Figure()

        tmp = tmp[(tmp[wind_dir_col] >= 0) & (tmp[wind_dir_col] <= 360)]

        if tmp.empty:
            return go.Figure()

        speed_bins = [0, 2, 4, 6, 8, np.inf]
        speed_labels = ["0-2", "2-4", "4-6", "6-8", ">8"]
        tmp["speed_bin"] = pd.cut(tmp[wind_speed_col], bins=speed_bins, labels=speed_labels)

        dir_bins = np.arange(0, 361, 22.5)
        dir_labels = [f"{int(d)}°" for d in dir_bins[:-1]]
        tmp["dir_bin"] = pd.cut(
            tmp[wind_dir_col],
            bins=dir_bins,
            labels=dir_labels,
            include_lowest=True
        )

        grouped = (
            tmp.groupby(["dir_bin", "speed_bin"], observed=False)[col]
            .mean()
            .reset_index(name="avg_value")
        )

        fig = go.Figure()
        colors = ["#2563eb", "#22c55e", "#facc15", "#fb923c", "#ef4444"]

        for i, label in enumerate(speed_labels):
            sub = grouped[grouped["speed_bin"] == label]
            fig.add_trace(go.Barpolar(
                r=sub["avg_value"],
                theta=sub["dir_bin"].astype(str),
                name=f"{label} wind speed",
                marker_color=colors[i],
                opacity=0.82
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(showticklabels=True, ticks="", title=f"Avg {col}"),
                angularaxis=dict(direction="clockwise")
            )
        )

        return style_fig(fig, height=300, legend=True)

    def plot_scatter_driver(target, driver):
        if not driver:
            return go.Figure(), np.nan, np.nan

        tmp = df[[target, driver]].apply(pd.to_numeric, errors="coerce").dropna()

        if len(tmp) < 10:
            return go.Figure(), np.nan, np.nan

        corr = tmp[target].corr(tmp[driver])
        r2 = corr ** 2 if not pd.isna(corr) else np.nan

        sample_df = tmp.sample(min(len(tmp), 5000), random_state=42)

        fig = px.scatter(
            sample_df,
            x=driver,
            y=target,
            trendline="ols",
            opacity=0.55
        )

        fig.update_traces(marker=dict(size=5, color="#2563eb"))
        fig.update_xaxes(title=driver)
        fig.update_yaxes(title=target)

        return style_fig(fig, height=300, legend=False), corr, r2

    def plot_high_event_analysis(col):
        daily = get_daily(col)

        if daily.empty:
            return go.Figure(), {}

        q90 = daily[col].quantile(0.90)
        q95 = daily[col].quantile(0.95)
        q99 = daily[col].quantile(0.99)

        high_events = daily[daily[col] >= q90]
        extreme_events = daily[daily[col] >= q95]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily[time_col],
            y=daily[col],
            mode="lines",
            name=f"Daily {col}",
            line=dict(color="#2563eb", width=1.2)
        ))

        fig.add_trace(go.Scatter(
            x=high_events[time_col],
            y=high_events[col],
            mode="markers",
            name="High Event ≥90th",
            marker=dict(color="#f59e0b", size=6, opacity=0.85)
        ))

        fig.add_trace(go.Scatter(
            x=extreme_events[time_col],
            y=extreme_events[col],
            mode="markers",
            name="Extreme Event ≥95th",
            marker=dict(color="#ef4444", size=8, opacity=0.95)
        ))

        fig.add_hline(
            y=q90,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="90th percentile",
            annotation_position="top right"
        )

        fig.add_hline(
            y=q95,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="95th percentile",
            annotation_position="top right"
        )

        fig.update_xaxes(title="")
        fig.update_yaxes(title=col)

        stats = {
            "q90": q90,
            "q95": q95,
            "q99": q99,
            "extreme_days": len(extreme_events)
        }

        return style_fig(fig, height=300, legend=True), stats


    # ======================================================
    # HEADER
    # ======================================================
    header_left, header_right = st.columns([4, 1.35])

    with header_left:
        st.markdown(
            """
            <div class="eda-header">
                <div class="eda-title">EXPLORATORY DATA ANALYSIS</div>
                <div class="eda-subtitle">Understand patterns, trends and key drivers of air quality.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with header_right:
        if time_col and df[time_col].notna().any():
            period = f"{df[time_col].min().strftime('%b %Y')} – {df[time_col].max().strftime('%b %Y')}"
        else:
            period = "All Records"

        selected_col = st.selectbox(
            "Filters",
            main_targets,
            index=main_targets.index(primary_col)
        )
        st.caption(period)

    # ======================================================
    # KPI ROW
    # ======================================================
    series = num(selected_col)
    daily = get_daily(selected_col)
    thr = threshold_for(selected_col)

    mean_val = series.mean()
    max_val = series.max()
    stat, stat_cls = status_for(selected_col, mean_val)

    exceed_count = 0
    exceed_pct = np.nan

    if not daily.empty and thr is not None:
        exceed_count = int((daily[selected_col] > thr).sum())
        exceed_pct = exceed_count / max(1, daily[selected_col].count()) * 100

    corr_pm25 = safe_corr(aqi_col, pm25_col) if aqi_col and pm25_col else np.nan
    quality = 100 - df.isna().mean().mean() * 100

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        metric_card(f"Average {selected_col}", fmt(mean_val, 1), stat, stat_cls)

    with k2:
        pm25_mean = num(pm25_col).mean() if pm25_col else np.nan
        metric_card("Average PM2.5", fmt(pm25_mean, 1), "Key fine-particle pollutant", "bad")

    with k3:
        metric_card(f"Max {selected_col}", fmt(max_val, 0), "Peak observed value", "warn")

    with k4:
        metric_card(
            f"Days {selected_col} > {thr}" if thr else "High Days",
            f"{exceed_count:,}" if thr else "N/A",
            f"{fmt(exceed_pct, 1)}% of days" if thr else "No threshold",
            "warn"
        )

    with k5:
        metric_card("Correlation AQI vs PM2.5", fmt(corr_pm25, 2), "Relationship strength", "good")

    with k6:
        metric_card("Data Quality", f"{quality:.2f}%", "Completeness score", "good")

    # ======================================================
    # CHART GRID
    # ======================================================
    r1c1, r1c2  = st.columns([1, 1])

    with r1c1:
        chart_box(f"1. {selected_col} Trend with Seasonal Decomposition")
        st.plotly_chart(plot_trend_decomposition(selected_col), use_container_width=True, config=CONFIG)
        end_box()

    with r1c2:
        chart_box(f"2. {selected_col} by Day of Week & Hour of Day")
        st.plotly_chart(plot_hour_day_heatmap(selected_col), use_container_width=True, config=CONFIG)
        end_box()
    r2c1, r2c2 = st.columns([1, 1])

    with r2c1:
        chart_box(f"3. {selected_col} Distribution by Category")
        st.plotly_chart(plot_parameter_category_donut(selected_col), use_container_width=True, config=CONFIG)
        end_box()


    with r2c2:
        chart_box(f"4. Monthly {selected_col} Box Plot")
        st.plotly_chart(plot_monthly_box(selected_col), use_container_width=True, config=CONFIG)
        end_box()

    r3c1, r3c2 = st.columns([1, 1])


    with r3c1:
        chart_box("5. Air Quality Parameter Comparison")
        st.plotly_chart(plot_pollutant_bar(), use_container_width=True, config=CONFIG)
        end_box()

    with r3c2:
        chart_box(f"6. Wind Rose by {selected_col}")
        st.plotly_chart(plot_parameter_wind_rose(selected_col), use_container_width=True, config=CONFIG)
        end_box()

    # ======================================================
    # DRIVER RELATIONSHIP + HIGH EVENT ANALYSIS
    # Two charts side by side.
    # Missing values chart removed as requested.
    # ======================================================
    driver_options = {
        "Temperature": temp_col,
        "Humidity": humidity_col,
        "Wind Speed": wind_speed_col,
        "Traffic Volume": traffic_col,
        "Pedestrians": ped_col,
    }

    driver_options = {
        name: col for name, col in driver_options.items()
        if col and col in df.columns
    }

    r4c1, r4c2 = st.columns(2)

    with r4c1:
        chart_box(f"7. {selected_col} vs Selected Driver")

        if driver_options:
            selected_driver_name = st.selectbox(
                "Select driver parameter",
                list(driver_options.keys()),
                index=0,
                key="eda_driver_select_1"
            )

            selected_driver_col = driver_options[selected_driver_name]

            fig, corr, r2 = plot_scatter_driver(selected_col, selected_driver_col)
            st.plotly_chart(fig, use_container_width=True, config=CONFIG)
            st.caption(f"Correlation: {fmt(corr, 2)} | R² Score: {fmt(r2, 2)}")
        else:
            st.info("No driver columns found.")

        end_box()

    with r4c2:
        chart_box(f"8. High {selected_col} Event Analysis")

        event_fig, event_stats = plot_high_event_analysis(selected_col)
        st.plotly_chart(event_fig, use_container_width=True, config=CONFIG)

        if event_stats:
            st.caption(
                f"90th percentile: {fmt(event_stats['q90'], 1)} | "
                f"95th percentile: {fmt(event_stats['q95'], 1)} | "
                f"Extreme days: {event_stats['extreme_days']:,}"
            )

        end_box()


    # ======================================================
    # KEY INSIGHTS
    # ======================================================
    related = []

    for name, col in [
        ("PM2.5", pm25_col),
        ("NO2", no2_col),
        ("Traffic Volume", traffic_col),
        ("Temperature", temp_col),
        ("Wind Speed", wind_speed_col),
        ("Humidity", humidity_col),
        ("Pedestrians", ped_col),
    ]:
        r = safe_corr(selected_col, col) if col and col != selected_col else np.nan
        if not pd.isna(r) and abs(r) >= 0.25:
            related.append(name)

    related_text = ", ".join(related[:4]) if related else "available meteorological and activity variables"

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">KEY INSIGHTS</div>
            <div class="insight-text">
            {selected_col} shows clear time-based patterns across daily trends, weekday/hour behaviour and monthly variation.
            The strongest useful relationships are with {related_text}. These findings support feature selection before
            Random Forest, XGBoost and GRU forecasting.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    
