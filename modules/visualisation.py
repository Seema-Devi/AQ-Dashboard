import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_visualisation():
    # =========================================================
    # PAGE SETUP
    # =========================================================

    px.defaults.template = "plotly_white"

    config = {
        "displayModeBar": False,
        "scrollZoom": False
    }

    df = st.session_state.get("df")

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

    df_raw = df.copy()

    # =========================================================
    # COLUMN DETECTION
    # =========================================================
    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

    time_col = None
    for col in df_raw.columns:
        if "date" in col.lower() or "time" in col.lower():
            time_col = col
            break

    if time_col:
        df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce")


    # =========================================================
    # UNITS AND LABELS
    # =========================================================
    PARAM_UNITS = {
        "AQI": "AQI",
        "PM2.5": "µg/m³",
        "PM10": "µg/m³",
        "NO2": "µg/m³",
        "TEMP": "°C",
        "RH": "%",
        "WS": "m/s",
        "WD": "°",
        "TRAFFICV": "vehicle/hr",
        "TOTAL_PEDESTRIANS": "count/hr"
    }

    DISPLAY_NAMES = {
        "AQI": "AQI",
        "PM2.5": "PM2.5",
        "PM10": "PM10",
        "NO2": "NO2",
        "TEMP": "Temperature",
        "RH": "Relative Humidity",
        "WS": "Wind Speed",
        "WD": "Wind Direction",
        "TRAFFICV": "Traffic Volume",
        "TOTAL_PEDESTRIANS": "Total Pedestrians"
    }


    def label_with_unit(col):
        unit = PARAM_UNITS.get(col, "")
        return f"{col} ({unit})" if unit else col


    def mean_value(column):
        if column in df_raw.columns:
            value = pd.to_numeric(df_raw[column], errors="coerce").mean()
            return f"{value:.2f}" if pd.notna(value) else "N/A"
        return "N/A"


    def get_unit(column):
        return PARAM_UNITS.get(column, "")


    def available(cols):
        return [c for c in cols if c in df_raw.columns]


    # =========================================================
    # CSS - IMAGE STYLE DASHBOARD
    # =========================================================
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

    .section-card {
        background: #ffffff;
        border: 1px solid #cbdcff;
        border-radius: 16px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        padding: 12px 14px;
        margin-bottom: 14px;
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 22px;
        font-weight: 900;
        color: #0433d9;
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

    .small-note {
        background: linear-gradient(90deg, #eaf4ff, #f5faff);
        border-radius: 8px;
        padding: 8px 11px;
        color: #051449;
        font-size: 13px;
        font-weight: 600;
        margin-top: 7px;
    }

    .green-note {
        background: linear-gradient(90deg, #eafbe9, #f5fff5);
        border: 1px solid #bde9bd;
        border-radius: 8px;
        padding: 8px 11px;
        color: #0b4a16;
        font-size: 13px;
        font-weight: 700;
        margin-top: 8px;
    }

    .pink-note {
        background: linear-gradient(90deg, #fff1f4, #fff7f8);
        border-radius: 8px;
        padding: 8px 11px;
        color: #a01836;
        font-size: 13px;
        font-weight: 700;
        margin-top: 8px;
    }

    .kpi-wrap {
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
    }

    .kpi-unit {
        font-size: 12px;
        color: #111827;
        font-weight: 800;
        margin-top: 2px;
    }

    .kpi-blue {background: linear-gradient(135deg,#f4f8ff,#eef5ff); border-color:#74a7ff; color:#0645d8;}
    .kpi-green {background: linear-gradient(135deg,#f3fff7,#eefcf3); border-color:#79d49a; color:#078037;}
    .kpi-orange {background: linear-gradient(135deg,#fff9ef,#fff4e6); border-color:#ffa442; color:#ef6400;}
    .kpi-purple {background: linear-gradient(135deg,#fbf7ff,#f7efff); border-color:#b076ff; color:#6618c8;}
    .kpi-red {background: linear-gradient(135deg,#fff7f7,#fff0f1); border-color:#ff7a83; color:#d51b2c;}

    .plot-subtitle {
        font-weight: 900;
        color: #07104b;
        text-align: center;
        margin-bottom: 2px;
        font-size: 15px;
    }

    .legend-box {
        border: 1px solid #d5e1ff;
        border-radius: 10px;
        padding: 8px;
        background: #fbfdff;
        font-size: 12px;
        font-weight: 800;
    }

    hr {
        margin: 0.4rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


    # =========================================================
    # HELPER FUNCTIONS
    # =========================================================
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


    def kpi_card(col, title, value, unit, css_class, icon):
        col.markdown(
            f"""
            <div class="kpi-wrap {css_class}">
                <div class="kpi-title">{icon} {title}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-unit">({unit})</div>
            </div>
            """,
            unsafe_allow_html=True
        )


    def add_nz_season(df, time_col):
        temp = df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors="coerce")
        month = temp[time_col].dt.month

        conditions = [
            month.isin([12, 1, 2]),
            month.isin([3, 4, 5]),
            month.isin([6, 7, 8]),
            month.isin([9, 10, 11])
        ]

        labels = ["Summer (Dec–Feb)", "Autumn (Mar–May)", "Winter (Jun–Aug)", "Spring (Sep–Nov)"]

        temp["Season"] = np.select(conditions, labels, default="Unknown")
        temp = temp[temp["Season"] != "Unknown"]
        return temp


    def direction_from_degree(series):
        bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
        labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        return pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False, ordered=False)


    def make_wind_rose(base_df, value_col="WS", title="Wind Rose", colour_label="Wind Speed", unit="m/s", height=430):
        temp = base_df.copy()
        temp["WD"] = pd.to_numeric(temp["WD"], errors="coerce")
        temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
        temp = temp.dropna(subset=["WD", value_col])
        temp = temp[(temp["WD"] >= 0) & (temp["WD"] <= 360)]

        if temp.empty:
            return None

        temp["Direction"] = direction_from_degree(temp["WD"])
        direction_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        if value_col == "WS":
            temp = temp[temp[value_col] >= 0]
            bins = [0, 1, 3, 5, 8, np.inf]
            labels = ["0–1", "1–3", "3–5", "5–8", "8+"]
            temp["Range"] = pd.cut(temp[value_col], bins=bins, labels=labels, include_lowest=True)

            rose = (
                temp.groupby(["Direction", "Range"], observed=False)
                .size()
                .reset_index(name="Frequency")
            )

            fig = px.bar_polar(
                rose,
                r="Frequency",
                theta="Direction",
                color="Range",
                category_orders={"Direction": direction_order, "Range": labels},
                color_discrete_sequence=px.colors.sequential.Jet,
                title=title
            )

            fig.update_layout(
                legend_title=f"{colour_label}<br>({unit})"
            )

        else:
            temp["Angle Bin"] = (np.round(temp["WD"] / 10) * 10) % 360
            smooth = (
                temp.groupby("Angle Bin")
                .agg(
                    Mean_Value=(value_col, "mean"),
                    Count=(value_col, "size")
                )
                .reset_index()
            )

            # Use frequency percentage as radius and colour as mean pollutant.
            smooth["Frequency %"] = smooth["Count"] / smooth["Count"].sum() * 100

            fig = px.bar_polar(
                smooth,
                r="Frequency %",
                theta="Angle Bin",
                color="Mean_Value",
                color_continuous_scale="Jet",
                range_color=[
                    np.nanpercentile(smooth["Mean_Value"], 2),
                    np.nanpercentile(smooth["Mean_Value"], 98)
                ],
                title=title,
                labels={
                    "Mean_Value": f"Mean {value_col}<br>({unit})",
                    "Frequency %": "Frequency (%)",
                    "Angle Bin": "Wind Direction"
                }
            )

            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=f"Mean {value_col}<br>({unit})"
                )
            )

        fig.update_traces(marker_line_width=0, opacity=0.96)

        fig.update_layout(
            height=height,
            margin=dict(l=8, r=8, t=42, b=8),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=11, color="#061345"),
            polar=dict(
                angularaxis=dict(
                    rotation=90,
                    direction="clockwise",
                    tickmode="array",
                    tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                    ticktext=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                    gridcolor="rgba(120,130,150,0.35)",
                    linecolor="rgba(20,30,60,0.65)"
                ),
                radialaxis=dict(
                    ticksuffix="%",
                    gridcolor="rgba(120,130,150,0.35)",
                    linecolor="rgba(20,30,60,0.35)",
                    showticklabels=True
                )
            ),
            title=dict(x=0.5, xanchor="center", font=dict(size=14, color="#07104b"))
        )

        return fig


    def make_small_pollution_rose(base_df, pollutant, season_title):
        fig = make_wind_rose(
            base_df,
            value_col=pollutant,
            title=season_title,
            colour_label=pollutant,
            unit=get_unit(pollutant),
            height=250
        )
        if fig:
            fig.update_layout(
                margin=dict(l=2, r=2, t=32, b=2),
                font=dict(size=9),
                polar=dict(
                    angularaxis=dict(
                        rotation=90,
                        direction="clockwise",
                        tickmode="array",
                        tickvals=[0, 90, 180, 270],
                        ticktext=["N", "E", "S", "W"]
                    ),
                    radialaxis=dict(showticklabels=False, ticksuffix="%")
                )
            )
        return fig


    def make_correlation_heatmap():
        corr_cols = available([
            "AQI", "PM2.5", "PM10", "NO2", "TEMP",
            "RH", "WS", "TRAFFICV", "TOTAL_PEDESTRIANS"
        ])

        if len(corr_cols) < 2:
            return None

        corr_matrix = df_raw[corr_cols].apply(pd.to_numeric, errors="coerce").corr().round(2)

        labels =  corr_cols

        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            labels=dict(color="Correlation")
        )

        fig.update_xaxes(ticktext=labels, tickvals=list(range(len(corr_cols))), side="top")
        fig.update_yaxes(ticktext=labels, tickvals=list(range(len(corr_cols))))

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(size=10, color="#061345"),
            coloraxis_showscale=False
        )

        return fig


    # =========================================================
    # TITLE
    # =========================================================
    st.markdown(
        '<div class="main-title">📊 Raw Data Insights</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="info-banner">
            ℹ️ This page shows the raw behaviour and patterns of the dataset before preprocessing and cleaning.
        </div>
        """,
        unsafe_allow_html=True
    )


    # =========================================================
    # 1. KPI OVERVIEW
    # =========================================================
    with st.container(border=True):
        section_header("1", "KPI Overview (Average Values)")
        k1, k2, k3, k4, k5 = st.columns(5, gap="small")

        kpi_card(k1, "Avg AQI", mean_value("AQI"), "AQI", "kpi-blue", "☁️")
        kpi_card(k2, "Avg PM2.5", mean_value("PM2.5"), "µg/m³", "kpi-green", "●●")
        kpi_card(k3, "Avg NO2", mean_value("NO2"), "µg/m³", "kpi-orange", "🏭")
        kpi_card(k4, "Avg Wind Speed", mean_value("WS"), "m/s", "kpi-purple", "💨")
        kpi_card(k5, "Avg Temperature", mean_value("TEMP"), "°C", "kpi-red", "🌡️")

    # =========================================================
    # 2. RAW DAILY TREND ANALYSIS
    # =========================================================
    with st.container(border=True):
        section_header("2", " Daily Trend Analysis")

        if time_col and numeric_cols:
            trend_options = available([
                "AQI", "PM2.5", "NO2", "PM10",
                "TEMP", "WS", "TRAFFICV", "TOTAL_PEDESTRIANS"
            ])

            default_trend = available(["AQI", "PM2.5", "NO2"])
            if not default_trend:
                default_trend = trend_options[:3]

            selected_trend_cols = st.multiselect(
                "Select variables for trend",
                trend_options,
                default=default_trend,
                key="trend_cols"
            )

            if selected_trend_cols:
                trend_df = (
                    df_raw[[time_col] + selected_trend_cols]
                    .dropna(subset=[time_col])
                    .set_index(time_col)
                    .resample("D")
                    .mean(numeric_only=True)
                    .reset_index()
                )

                trend_long = trend_df.melt(
                    id_vars=time_col,
                    value_vars=selected_trend_cols,
                    var_name="Variable",
                    value_name="Daily Average"
                )
                trend_long["Variable Label"] = trend_long["Variable"].apply(label_with_unit)

                fig = px.line(
                    trend_long,
                    x=time_col,
                    y="Daily Average",
                    color="Variable Label",
                    title=None
                )
                fig.update_traces(line=dict(width=2.7))

                # Event labels make the plot look like the reference layout.
                events = {
                    "2020-03-25": "COVID lockdown",
                    "2020-08-12": "Auckland lockdown",
                    "2021-08-17": "Delta lockdown",
                    "2023-01-27": "Auckland floods",
                    "2023-02-13": "Cyclone Gabrielle"
                }
                max_y = trend_long["Daily Average"].max()
                if pd.isna(max_y):
                    max_y = 0
                for event_date, event_name in events.items():
                    event_dt = pd.to_datetime(event_date)
                    if trend_long[time_col].min() <= event_dt <= trend_long[time_col].max():
                        fig.add_vline(
                            x=event_dt,
                            line_width=1.5,
                            line_dash="dot",
                            line_color="#ff5a7a"
                        )
                        fig.add_annotation(
                            x=event_dt,
                            y=max_y,
                            text=event_name,
                            showarrow=False,
                            yshift=8,
                            font=dict(size=10, color="#7f1d1d"),
                            bgcolor="#ffe4e9",
                            bordercolor="#fda4af"
                        )

                fig.update_layout(
                    height=430,
                    xaxis_title="Date",
                    yaxis_title="Daily Average",
                    legend_title=None,
                    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
                    margin=dict(l=10, r=10, t=45, b=10),
                    font=dict(color="#061345")
                )
                st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(
                    '<div class="small-note">💡 Spikes, noisy behaviour and seasonal patterns are visible.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Select at least one variable to display the raw trend.")
        else:
            st.warning("A date/time column and numeric columns are required for the daily trend plot.")

    # =========================================================
    # 3. WIND ROSE ANALYSIS
    # =========================================================
    with st.container(border=True):
        section_header("3", "Wind Rose Analysis")

        if "WD" in df_raw.columns and "WS" in df_raw.columns:
            wind_base = df_raw[["WD", "WS"] + ([time_col] if time_col else [])].copy()
            wind_base["WD"] = pd.to_numeric(wind_base["WD"], errors="coerce")
            wind_base["WS"] = pd.to_numeric(wind_base["WS"], errors="coerce")
            wind_base = wind_base.dropna(subset=["WD", "WS"])

            w_left, w_right = st.columns([1.05, 1.35], gap="medium")

            with w_left:
                st.markdown('<div class="plot-subtitle">Overall Wind Rose</div>', unsafe_allow_html=True)
                fig = make_wind_rose(
                    wind_base,
                    value_col="WS",
                    title="",
                    colour_label="Wind Speed",
                    unit="m/s",
                    height=470
                )
                if fig:
                    fig.update_layout(title=None, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config=config)
                    st.caption("Wind speed unit: m/s")

            with w_right:
                st.markdown('<div class="plot-subtitle">Seasonal Wind Rose</div>', unsafe_allow_html=True)
                if time_col:
                    seasonal_wind = add_nz_season(wind_base, time_col)
                    seasons_order = [
                        "Summer (Dec–Feb)",
                        "Autumn (Mar–May)",
                        "Winter (Jun–Aug)",
                        "Spring (Sep–Nov)"
                    ]

                    from plotly.subplots import make_subplots
                    fig_grid = make_subplots(
                        rows=2,
                        cols=2,
                        specs=[[{"type": "polar"}, {"type": "polar"}], [{"type": "polar"}, {"type": "polar"}]],
                        subplot_titles=seasons_order,
                        vertical_spacing=0.18,
                        horizontal_spacing=0.12
                    )

                    speed_bins = [0, 1, 3, 5, 8, np.inf]
                    speed_labels = ["0–1", "1–3", "3–5", "5–8", "8+"]
                    direction_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                    colours = px.colors.sequential.Jet[:len(speed_labels)]

                    for idx, season_name in enumerate(seasons_order):
                        r = idx // 2 + 1
                        c = idx % 2 + 1
                        season_data = seasonal_wind[seasonal_wind["Season"] == season_name].copy()
                        if season_data.empty:
                            continue
                        season_data["Direction"] = direction_from_degree(season_data["WD"])
                        season_data["Range"] = pd.cut(season_data["WS"], bins=speed_bins, labels=speed_labels, include_lowest=True)
                        rose = (
                            season_data.groupby(["Direction", "Range"], observed=False)
                            .size()
                            .reset_index(name="Frequency")
                        )
                        for lab, colr in zip(speed_labels, colours):
                            sub = rose[rose["Range"] == lab]
                            fig_grid.add_trace(
                                go.Barpolar(
                                    r=sub["Frequency"],
                                    theta=sub["Direction"],
                                    name=lab,
                                    marker_color=colr,
                                    marker_line_width=0,
                                    opacity=0.95,
                                    showlegend=(idx == 0)
                                ),
                                row=r,
                                col=c
                            )
                        mean_ws = season_data["WS"].mean()
                        if pd.notna(mean_ws):
                            fig_grid.add_annotation(
                                x=0.24 if c == 1 else 0.76,
                                y=0.53 if r == 1 else 0.03,
                                xref="paper", yref="paper",
                                text=f"Mean WS: {mean_ws:.2f} m/s",
                                showarrow=False,
                                font=dict(size=10, color="#075985")
                            )

                    polar_layout = dict(
                        angularaxis=dict(
                            rotation=90,
                            direction="clockwise",
                            tickmode="array",
                            tickvals=[0, 90, 180, 270],
                            ticktext=["N", "E", "S", "W"],
                            tickfont=dict(size=9)
                        ),
                        radialaxis=dict(
                            showticklabels=False,
                            ticksuffix="%",
                            gridcolor="rgba(120,130,150,0.25)"
                        )
                    )
                    fig_grid.update_layout(
                        height=470,
                        margin=dict(l=10, r=25, t=45, b=15),
                        legend=dict(title="Wind Speed<br>(m/s)", x=1.02, y=0.5),
                        font=dict(size=10, color="#061345"),
                        polar=polar_layout,
                        polar2=polar_layout,
                        polar3=polar_layout,
                        polar4=polar_layout
                    )
                    st.plotly_chart(fig_grid, use_container_width=True, config=config)
                else:
                    st.info("A date/time column is required for seasonal wind rose analysis.")

            st.markdown(
                '<div class="small-note">ℹ️ Wind direction uses N, NE, E, SE, S, SW, W and NW. Colours show wind speed ranges.</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("WD and WS columns are required for wind rose analysis.")

    # =========================================================
    # 4. WIND DIRECTION AND POLLUTION ANALYSIS
    # =========================================================
    with st.container(border=True):
        top_left, top_right = st.columns([3, 1], gap="medium")
        with top_left:
            section_header("4", "Wind Direction and Pollution Analysis")
        pollutant_options = available(["PM2.5", "PM10", "AQI", "NO2"])
        with top_right:
            if pollutant_options:
                pollutant = st.selectbox(
                    "Select Pollutant:",
                    pollutant_options,
                    index=0,
                    key="pollutant"
                )
            else:
                pollutant = None

        if pollutant and "WD" in df_raw.columns:
            pollution_base = df_raw[["WD", pollutant] + ([time_col] if time_col else [])].copy()
            pollution_base["WD"] = pd.to_numeric(pollution_base["WD"], errors="coerce")
            pollution_base[pollutant] = pd.to_numeric(pollution_base[pollutant], errors="coerce")
            pollution_base = pollution_base.dropna(subset=["WD", pollutant])

            p_left, p_right = st.columns([1.05, 1.4], gap="medium")

            with p_left:
                st.markdown(
                    f'<div class="plot-subtitle">Overall Wind Direction & {label_with_unit(pollutant)}</div>',
                    unsafe_allow_html=True
                )
                fig = make_wind_rose(
                    pollution_base,
                    value_col=pollutant,
                    title="",
                    colour_label=pollutant,
                    unit=get_unit(pollutant),
                    height=500
                )
                if fig:
                    fig.update_layout(title=None)
                    st.plotly_chart(fig, use_container_width=True, config=config)
                    st.caption(f"Colour scale: mean {label_with_unit(pollutant)} by wind direction")

            with p_right:
                st.markdown(
                    f'<div class="plot-subtitle">Seasonal Wind Direction & {label_with_unit(pollutant)}</div>',
                    unsafe_allow_html=True
                )
                if time_col:
                    seasonal_pollution = add_nz_season(pollution_base, time_col)
                    seasons_order = [
                        "Summer (Dec–Feb)",
                        "Autumn (Mar–May)",
                        "Winter (Jun–Aug)",
                        "Spring (Sep–Nov)"
                    ]

                    from plotly.subplots import make_subplots
                    fig_grid = make_subplots(
                        rows=2,
                        cols=2,
                        specs=[[{"type": "polar"}, {"type": "polar"}], [{"type": "polar"}, {"type": "polar"}]],
                        subplot_titles=seasons_order,
                        vertical_spacing=0.18,
                        horizontal_spacing=0.12
                    )

                    global_min = np.nanpercentile(seasonal_pollution[pollutant], 2) if not seasonal_pollution.empty else 0
                    global_max = np.nanpercentile(seasonal_pollution[pollutant], 98) if not seasonal_pollution.empty else 1

                    for idx, season_name in enumerate(seasons_order):
                        r = idx // 2 + 1
                        c = idx % 2 + 1
                        season_data = seasonal_pollution[seasonal_pollution["Season"] == season_name].copy()
                        if season_data.empty:
                            continue
                        season_data["Angle Bin"] = (np.round(season_data["WD"] / 10) * 10) % 360
                        smooth = (
                            season_data.groupby("Angle Bin")
                            .agg(Mean_Value=(pollutant, "mean"), Count=(pollutant, "size"))
                            .reset_index()
                        )
                        smooth["Frequency %"] = smooth["Count"] / smooth["Count"].sum() * 100
                        fig_grid.add_trace(
                            go.Barpolar(
                                r=smooth["Frequency %"],
                                theta=smooth["Angle Bin"],
                                marker=dict(
                                    color=smooth["Mean_Value"],
                                    colorscale="Jet",
                                    cmin=global_min,
                                    cmax=global_max,
                                    colorbar=dict(title=f"{pollutant}<br>({get_unit(pollutant)})") if idx == 1 else None,
                                    showscale=(idx == 1)
                                ),
                                marker_line_width=0,
                                opacity=0.96,
                                showlegend=False
                            ),
                            row=r,
                            col=c
                        )
                        mean_val = season_data[pollutant].mean()
                        if pd.notna(mean_val):
                            fig_grid.add_annotation(
                                x=0.24 if c == 1 else 0.76,
                                y=0.53 if r == 1 else 0.03,
                                xref="paper", yref="paper",
                                text=f"Mean: {mean_val:.2f} {get_unit(pollutant)}",
                                showarrow=False,
                                font=dict(size=10, color="#047857")
                            )

                    polar_layout = dict(
                        angularaxis=dict(
                            rotation=90,
                            direction="clockwise",
                            tickmode="array",
                            tickvals=[0, 90, 180, 270],
                            ticktext=["N", "E", "S", "W"],
                            tickfont=dict(size=9)
                        ),
                        radialaxis=dict(
                            showticklabels=False,
                            ticksuffix="%",
                            gridcolor="rgba(120,130,150,0.25)"
                        )
                    )
                    fig_grid.update_layout(
                        height=500,
                        margin=dict(l=10, r=70, t=45, b=15),
                        font=dict(size=10, color="#061345"),
                        polar=polar_layout,
                        polar2=polar_layout,
                        polar3=polar_layout,
                        polar4=polar_layout
                    )
                    st.plotly_chart(fig_grid, use_container_width=True, config=config)
                else:
                    st.info("A date/time column is required for seasonal pollution analysis.")

            st.markdown(
                '<div class="pink-note">🔥 Warm colours (red/orange) indicate higher pollution concentrations from that wind direction.</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("WD and at least one pollutant column are required for wind-pollution analysis.")

    # =========================================================
    # 5. CORRELATION HEATMAP
    # =========================================================
    with st.container(border=True):
        h_left, h_right = st.columns([3, 1], gap="medium")
        with h_left:
            section_header("5", "Correlation Heatmap")
        with h_right:
            show_heatmap = st.toggle("Open correlation heatmap", value=True)

        if show_heatmap:
            fig = make_correlation_heatmap()
            if fig:
                st.plotly_chart(fig, use_container_width=True, config=config)
                st.markdown(
                    '<div class="green-note">✅ The heatmap shows relationships between variables and supports feature selection for modelling.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("At least two numeric key variables are required for correlation analysis.")
