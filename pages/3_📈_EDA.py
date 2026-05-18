import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

px.defaults.template = "plotly_white"

config = {
    "displayModeBar": False,
    "scrollZoom": False
}

# ======================================================
# LOAD DATA
# ======================================================
if "df_cleaned" in st.session_state:
    df = st.session_state["df_cleaned"]
elif "df" in st.session_state:
    df = st.session_state["df"]
else:
    st.warning("⬅ Please upload and process data first.")
    render_footer()
    st.stop()

df = df.copy()

st.header("📊 Exploratory Data Analysis")

# ======================================================
# DETECT COLUMNS
# ======================================================
time_col = None
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()


def find_col(names):
    for name in names:
        for col in df.columns:
            clean_col = (
                col.lower()
                .replace(" ", "")
                .replace("_", "")
                .replace(".", "")
                .replace("₂", "2")
            )

            clean_name = (
                name.lower()
                .replace(" ", "")
                .replace("_", "")
                .replace(".", "")
                .replace("₂", "2")
            )

            if clean_name in clean_col:
                return col

    return None


aqi_col = find_col(["AQI"])
pm25_col = find_col(["PM2.5", "PM25"])
no2_col = find_col(["NO2", "NO₂"])
pm10_col = find_col(["PM10"])

traffic_col = find_col(["traffic", "trafficv"])
ped_col = find_col(["pedestrian", "total pedestrians"])
temp_col = find_col(["temp", "temperature"])
humidity_col = find_col(["rh", "humidity"])
wind_speed_col = find_col(["ws", "wind speed"])
wind_dir_col = find_col(["wd", "wind direction"])

target_cols = [c for c in [aqi_col, pm25_col, no2_col] if c]
trend_cols = [c for c in [aqi_col, pm25_col, no2_col, pm10_col] if c]

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
.blue-card { background: linear-gradient(135deg, #2563eb, #60a5fa); }
.green-card { background: linear-gradient(135deg, #059669, #34d399); }
.orange-card { background: linear-gradient(135deg, #ea580c, #fb923c); }
.purple-card { background: linear-gradient(135deg, #7c3aed, #a78bfa); }
.red-card { background: linear-gradient(135deg, #dc2626, #f87171); }
</style>
""", unsafe_allow_html=True)


def kpi_card(col, title, value, css_class):
    col.markdown(f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def safe_mean(col):
    if col and col in df.columns:
        val = pd.to_numeric(df[col], errors="coerce").mean()
        return f"{val:.2f}" if pd.notna(val) else "N/A"
    return "N/A"


# ======================================================
# HELPER FUNCTIONS
# ======================================================
def make_binned_average_chart(data, factor_col, target_col, chart_title):
    temp = data[[factor_col, target_col]].copy()
    temp[factor_col] = pd.to_numeric(temp[factor_col], errors="coerce")
    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna()

    if temp.empty or temp[factor_col].nunique() < 3:
        st.warning("Not enough valid data for this influence chart.")
        return

    try:
        temp["Level"] = pd.qcut(
            temp[factor_col],
            q=3,
            labels=["Low", "Medium", "High"],
            duplicates="drop"
        )
    except Exception:
        temp["Level"] = pd.cut(
            temp[factor_col],
            bins=3,
            labels=["Low", "Medium", "High"]
        )

    grouped = (
        temp
        .groupby("Level", observed=False)
        .agg(
            Average_Target=(target_col, "mean"),
            Records=(target_col, "count"),
            Min_Factor=(factor_col, "min"),
            Max_Factor=(factor_col, "max")
        )
        .reset_index()
    )

    grouped["Factor Range"] = (
        grouped["Min_Factor"].round(2).astype(str)
        + " - "
        + grouped["Max_Factor"].round(2).astype(str)
    )

    fig = px.bar(
        grouped,
        x="Level",
        y="Average_Target",
        color="Average_Target",
        color_continuous_scale="Oranges",
        text=grouped["Average_Target"].round(2),
        title=chart_title,
        hover_data=["Records", "Factor Range"]
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=470,
        xaxis_title=f"{factor_col} Level",
        yaxis_title=f"Average {target_col}"
    )

    fig = add_threshold_lines(fig, target_col)

    st.plotly_chart(fig, use_container_width=True, config=config)

    with st.expander("📋 Open influence summary table"):
        st.dataframe(
            grouped.round(2),
            use_container_width=True,
            hide_index=True
        )


# ======================================================
# PAGE PURPOSE
# ======================================================
st.markdown("""
<div style="
background-color:white;
padding:20px;
border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08);
margin-bottom:20px;
">
This EDA page focuses on AQI, PM2.5 and NO2 behaviour after cleaning.
It analyses pollutant trends, distributions, outliers, weather influence,
traffic behaviour, temporal patterns and forecasting insights.
</div>
""", unsafe_allow_html=True)

# ======================================================
# 1. FORECAST TARGET OVERVIEW
# ======================================================
st.subheader("1️⃣ Forecast Target Overview")

k1, k2, k3 = st.columns(3)

kpi_card(k1, "Average AQI", safe_mean(aqi_col), "blue-card")
kpi_card(k2, "Average PM2.5", safe_mean(pm25_col), "green-card")
kpi_card(k3, "Average NO2", safe_mean(no2_col), "orange-card")

# ======================================================
# 2. THRESHOLD OPTIONS
# ======================================================
st.subheader("2️⃣ Threshold Reference")

threshold_standard = st.radio(
    "Select threshold reference",
    ["WHO Guidelines", "NZ / AQI Reference"],
    horizontal=True
)

thresholds = {
    "WHO Guidelines": {},
    "NZ / AQI Reference": {}
}

if aqi_col:
    thresholds["WHO Guidelines"][aqi_col] = [
        ("Good AQI", 50, "green"),
        ("Moderate AQI", 100, "orange"),
        ("Unhealthy AQI", 150, "red")
    ]

    thresholds["NZ / AQI Reference"][aqi_col] = [
        ("Good AQI", 50, "green"),
        ("Moderate AQI", 100, "orange"),
        ("Unhealthy AQI", 150, "red")
    ]

if pm25_col:
    thresholds["WHO Guidelines"][pm25_col] = [
        ("WHO annual PM2.5", 5, "green"),
        ("WHO 24h PM2.5", 15, "red")
    ]

    thresholds["NZ / AQI Reference"][pm25_col] = [
        ("PM2.5 low", 12, "green"),
        ("PM2.5 moderate", 35.4, "orange"),
        ("PM2.5 high", 55.4, "red")
    ]

if no2_col:
    thresholds["WHO Guidelines"][no2_col] = [
        ("WHO annual NO2", 10, "green"),
        ("WHO 24h NO2", 25, "red")
    ]

    thresholds["NZ / AQI Reference"][no2_col] = [
        ("NO2 low", 53, "green"),
        ("NO2 moderate", 100, "orange"),
        ("NO2 high", 360, "red")
    ]


def add_threshold_lines(fig, target_col):
    if target_col in thresholds.get(threshold_standard, {}):
        for label, value, color in thresholds[threshold_standard][target_col]:
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=color,
                opacity=0.75,
                annotation_text=label,
                annotation_position="top left"
            )
    return fig


def add_threshold_vlines(fig, target_col):
    if target_col in thresholds.get(threshold_standard, {}):
        for label, value, color in thresholds[threshold_standard][target_col]:
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=color,
                opacity=0.75,
                annotation_text=label,
                annotation_position="top right"
            )
    return fig


with st.expander("📋 Open Threshold Reference Table"):
    threshold_rows = []

    for reference, target_dict in thresholds.items():
        for pollutant, values in target_dict.items():
            for label, value, color in values:
                threshold_rows.append({
                    "Reference": reference,
                    "Pollutant": pollutant,
                    "Threshold": label,
                    "Value": value,
                    "Colour": color
                })

    st.dataframe(
        pd.DataFrame(threshold_rows),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trends",
    "🌦 Influences",
    "🕒 Time Behaviour",
    "📊 Forecast Insights"
])

# ======================================================
# TAB 1 - TRENDS
# ======================================================
with tab1:

    st.subheader("AQI & Pollutant Trend Analysis")

    if time_col and trend_cols:

        freq = st.selectbox(
            "Select Trend Frequency",
            ["Hourly", "Daily"],
            index=1
        )

        freq_map = {
            "Hourly": "h",
            "Daily": "D"
        }

        trend_df = (
            df[[time_col] + trend_cols]
            .dropna(subset=[time_col])
            .set_index(time_col)
            .resample(freq_map[freq])
            .mean(numeric_only=True)
            .reset_index()
        )

        trend_long = trend_df.melt(
            id_vars=time_col,
            value_vars=trend_cols,
            var_name="Pollutant",
            value_name="Average Value"
        )

        fig = px.line(
            trend_long,
            x=time_col,
            y="Average Value",
            color="Pollutant",
            title=f"{freq} AQI and Pollutant Trends"
        )

        fig.update_layout(
            height=520,
            xaxis_title="Date / Time",
            yaxis_title="Average Value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        st.info("""
How to get insights:
- Hourly view shows short-term peaks such as rush-hour behaviour.
- Daily view smooths the data and shows broader pollution changes.
- Similar movement between AQI and pollutants suggests related behaviour.
""")

        if target_cols:
            selected_threshold_trend = st.selectbox(
                "Select target for threshold trend",
                target_cols,
                key="threshold_trend_target"
            )

            target_trend_df = (
                df[[time_col, selected_threshold_trend]]
                .dropna(subset=[time_col])
                .set_index(time_col)
                .resample(freq_map[freq])
                .mean(numeric_only=True)
                .reset_index()
            )

            fig = px.line(
                target_trend_df,
                x=time_col,
                y=selected_threshold_trend,
                title=f"{selected_threshold_trend} {freq} Trend with {threshold_standard} Thresholds",
                color_discrete_sequence=["#2563eb"]
            )

            fig = add_threshold_lines(fig, selected_threshold_trend)

            fig.update_layout(height=500)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=config
            )

        with st.expander("📋 Open Trend Table"):
            st.dataframe(
                trend_df.round(2),
                use_container_width=True
            )

        # ==================================================
        # MONTHLY / YEARLY BAR + AQI LINE
        # ==================================================
        st.subheader("Monthly / Yearly Pollutant Comparison")

        period_choice = st.radio(
            "Select comparison period",
            ["Monthly", "Yearly"],
            horizontal=True
        )

        compare_cols = [c for c in [pm25_col, no2_col, pm10_col] if c]

        if aqi_col and compare_cols:

            temp_df = df[[time_col, aqi_col] + compare_cols].dropna(subset=[time_col]).copy()

            if period_choice == "Monthly":
                temp_df["Period"] = temp_df[time_col].dt.to_period("M").astype(str)
            else:
                temp_df["Period"] = temp_df[time_col].dt.year.astype(str)

            period_df = (
                temp_df
                .groupby("Period")[[aqi_col] + compare_cols]
                .mean()
                .reset_index()
            )

            bar_long = period_df.melt(
                id_vars="Period",
                value_vars=compare_cols,
                var_name="Pollutant",
                value_name="Average Value"
            )

            fig = go.Figure()

            for pollutant in compare_cols:
                temp_bar = bar_long[bar_long["Pollutant"] == pollutant]

                fig.add_trace(
                    go.Bar(
                        x=temp_bar["Period"],
                        y=temp_bar["Average Value"],
                        name=pollutant
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=period_df["Period"],
                    y=period_df[aqi_col],
                    name="AQI Line",
                    mode="lines+markers",
                    line=dict(width=3),
                    yaxis="y2"
                )
            )

            fig.update_layout(
                title=f"{period_choice} Pollutants as Bars with AQI Line",
                xaxis_title=period_choice,
                yaxis=dict(title="Average Pollutant Value"),
                yaxis2=dict(
                    title="Average AQI",
                    overlaying="y",
                    side="right"
                ),
                barmode="group",
                height=560,
                legend_title="Parameter"
            )

            st.plotly_chart(fig, use_container_width=True, config=config)

            with st.expander("📋 Open monthly/yearly comparison table"):
                st.dataframe(
                    period_df.round(2),
                    use_container_width=True,
                    hide_index=True
                )

            st.info("""
How to get insights:
- Bars compare pollutant averages across months or years.
- The AQI line shows whether overall air-quality risk follows pollutant changes.
- If AQI rises with PM2.5 or NO2, that pollutant may be strongly influencing AQI.
""")

        else:
            st.warning("AQI and at least one pollutant column are required for this comparison chart.")

    else:
        st.warning("Datetime and pollutant columns are required for trend analysis.")

    st.subheader("Pollution Distribution Analysis")

    if target_cols:

        selected_dist = st.selectbox(
            "Select Pollutant",
            target_cols
        )

        fig = px.histogram(
            df,
            x=selected_dist,
            nbins=40,
            marginal="box",
            color_discrete_sequence=["#2563eb"],
            title=f"{selected_dist} Distribution"
        )

        fig = add_threshold_vlines(fig, selected_dist)

        fig.update_layout(height=480)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        summary_df = (
            df[target_cols]
            .describe()
            .T
            .reset_index()
        )

        summary_df = summary_df.rename(
            columns={"index": "Pollutant"}
        )

        summary_df["Skewness"] = (
            df[target_cols]
            .skew(numeric_only=True)
            .values
        )

        with st.expander("📋 Open Distribution Summary Table"):

            st.dataframe(
                summary_df.round(2),
                use_container_width=True,
                hide_index=True
            )

        st.info("""
How to get insights:
- Right skew suggests occasional extreme pollution.
- Threshold lines show where values exceed selected reference levels.
- Boxplot outliers may represent pollution events.
""")

    else:
        st.warning("Forecast target columns not found.")

    st.subheader("Outlier Analysis")

    if target_cols and time_col:

        selected_box_target = st.selectbox(
            "Select target for monthly outlier view",
            target_cols,
            key="box_threshold_target"
        )

        outlier_df_plot = df[[time_col, selected_box_target]].copy()
        outlier_df_plot[selected_box_target] = pd.to_numeric(
            outlier_df_plot[selected_box_target],
            errors="coerce"
        )
        outlier_df_plot = outlier_df_plot.dropna()

        outlier_df_plot["Month Number"] = outlier_df_plot[time_col].dt.month
        outlier_df_plot["Month"] = outlier_df_plot[time_col].dt.month_name()

        month_order = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        fig = px.box(
            outlier_df_plot,
            x="Month",
            y=selected_box_target,
            color="Month",
            category_orders={"Month": month_order},
            title=f"Monthly Outlier Pattern for {selected_box_target}"
        )

        fig = add_threshold_lines(fig, selected_box_target)

        fig.update_layout(
            height=520,
            showlegend=False,
            xaxis_title="Month",
            yaxis_title=selected_box_target
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        outlier_rows = []

        for month in month_order:
            month_data = outlier_df_plot[outlier_df_plot["Month"] == month][selected_box_target]

            if month_data.empty:
                continue

            q1 = month_data.quantile(0.25)
            q3 = month_data.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            count = ((month_data < lower) | (month_data > upper)).sum()

            outlier_rows.append({
                "Month": month,
                "Records": len(month_data),
                "Outlier Count": count,
                "Outlier %": round(count / len(month_data) * 100, 2),
                "IQR Lower Limit": round(lower, 2),
                "IQR Upper Limit": round(upper, 2)
            })

        outlier_monthly_df = pd.DataFrame(outlier_rows)

        with st.expander("📋 Open Monthly Outlier Summary Table"):
            st.dataframe(
                outlier_monthly_df,
                use_container_width=True,
                hide_index=True
            )

        st.info("""
How to get insights:
- Months with taller boxes have greater pollutant variability.
- Points outside whiskers show possible pollution spikes.
- Comparing months helps identify seasonal extreme events.
""")

    else:
        st.warning("Datetime and forecast target columns are required for monthly outlier analysis.")

# ======================================================
# TAB 2 - INFLUENCES
# ======================================================
with tab2:

    st.subheader("Weather Influence Analysis")

    weather_options = [
        c for c in [
            temp_col,
            humidity_col,
            wind_speed_col
        ] if c
    ]

    if weather_options and target_cols:

        selected_target = st.selectbox(
            "Select Forecast Target",
            target_cols,
            key="weather_target"
        )

        selected_weather = st.selectbox(
            "Select Weather Variable",
            weather_options
        )

        make_binned_average_chart(
            df,
            selected_weather,
            selected_target,
            f"Average {selected_target} by Low / Medium / High {selected_weather}"
        )

        st.info("""
How to get insights:
- This chart is easier to explain than a scatter plot.
- If the average pollutant rises from Low to High weather level, the weather factor may increase pollution.
- If pollution falls at High wind speed, wind may be dispersing pollutants.
""")

    else:
        st.warning("Weather or target columns not found.")

    st.subheader("Traffic & Pedestrian Influence")

    activity_options = [
        c for c in [
            traffic_col,
            ped_col
        ] if c
    ]

    if activity_options and target_cols:

        selected_target = st.selectbox(
            "Select Target",
            target_cols,
            key="activity_target"
        )

        selected_activity = st.selectbox(
            "Select Activity Variable",
            activity_options
        )

        make_binned_average_chart(
            df,
            selected_activity,
            selected_target,
            f"Average {selected_target} by Low / Medium / High {selected_activity}"
        )

        st.info("""
How to get insights:
- This groups urban activity into Low, Medium and High levels.
- If NO2 or AQI increases from Low to High traffic, it suggests traffic-related pollution influence.
- This is useful for explaining Auckland city-centre activity effects.
""")

    else:
        st.warning("Traffic, pedestrian or target columns not found.")

# ======================================================
# TAB 3 - TIME BEHAVIOUR
# ======================================================
with tab3:

    st.subheader("Time-Based Behaviour")

    if time_col and target_cols:

        df["Hour"] = df[time_col].dt.hour
        df["Month"] = df[time_col].dt.month_name()
        df["Month Number"] = df[time_col].dt.month

        selected_target = st.selectbox(
            "Select Time Analysis Target",
            target_cols,
            key="time_target"
        )

        hourly_df = (
            df.groupby("Hour")[selected_target]
            .mean()
            .reset_index()
        )

        fig = px.line(
            hourly_df,
            x="Hour",
            y=selected_target,
            markers=True,
            color_discrete_sequence=["#7c3aed"],
            title=f"Hourly Average {selected_target}"
        )

        fig = add_threshold_lines(fig, selected_target)

        fig.update_layout(
            height=460,
            xaxis=dict(dtick=1)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        monthly_df = (
            df.groupby(["Month Number", "Month"])[selected_target]
            .mean()
            .reset_index()
            .sort_values("Month Number")
        )

        fig = px.bar(
            monthly_df,
            x="Month",
            y=selected_target,
            color=selected_target,
            color_continuous_scale="Blues",
            text=monthly_df[selected_target].round(1),
            title=f"Monthly Seasonality of {selected_target}"
        )

        fig = add_threshold_lines(fig, selected_target)

        fig.update_traces(textposition="outside")

        fig.update_layout(height=480)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        heat_df = (
            df.groupby(["Month", "Hour"])[selected_target]
            .mean()
            .reset_index()
        )

        pivot_df = heat_df.pivot(
            index="Month",
            columns="Hour",
            values=selected_target
        )

        month_order = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        pivot_df = pivot_df.reindex(month_order)

        fig = px.imshow(
            pivot_df,
            color_continuous_scale="Turbo",
            aspect="auto",
            title=f"{selected_target} Monthly-Hourly Heatmap"
        )

        fig.update_layout(height=520)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        st.info("""
How to get insights:
- Hourly peaks may indicate rush-hour pollution.
- Monthly differences suggest seasonality.
- Heatmaps reveal combined monthly and daily behaviour.
- Threshold lines in line/bar charts help identify exceedance periods.
""")

        with st.expander("📋 Open Time Behaviour Table"):
            st.dataframe(
                monthly_df.round(2),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning("Datetime or target columns not found.")

# ======================================================
# TAB 4 - FORECAST INSIGHTS
# ======================================================
with tab4:

    st.subheader("Correlation Analysis")

    exclude_words = [
        "year", "month", "day",
        "date", "hour", "minute"
    ]

    heatmap_cols = [
        col for col in numeric_cols
        if not any(
            word in col.lower()
            for word in exclude_words
        )
    ]

    if len(heatmap_cols) > 1:

        corr = (
            df[heatmap_cols]
            .corr()
            .round(2)
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Feature Relationship Heatmap"
        )

        fig.update_layout(height=720)

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )

        if target_cols:

            selected_target = st.selectbox(
                "Select Target for Contributor Analysis",
                target_cols,
                key="corr_target"
            )

            corr_rank = (
                df[heatmap_cols]
                .corr()[selected_target]
                .drop(selected_target)
                .sort_values(
                    key=abs,
                    ascending=False
                )
                .reset_index()
            )

            corr_rank.columns = [
                "Feature",
                "Correlation"
            ]

            fig = px.bar(
                corr_rank.head(10),
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Correlation",
                color_continuous_scale="RdBu_r",
                title=f"Top Contributors Associated with {selected_target}"
            )

            fig.update_layout(
                height=520,
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config=config
            )

            with st.expander("📋 Open Correlation Ranking Table"):

                st.dataframe(
                    corr_rank.round(2),
                    use_container_width=True,
                    hide_index=True
                )

        st.info("""
How to get insights:
- Strong positive correlation means variables rise together.
- Strong negative correlation means inverse behaviour.
- Important correlations guide ML feature selection.
""")

    else:
        st.warning("Not enough numeric columns for correlation analysis.")

    st.subheader("Most Polluted Days")

    if target_cols:

        most_polluted_target = st.selectbox(
            "Select target for most polluted days",
            target_cols,
            key="most_polluted_target"
        )

        if time_col and most_polluted_target:

            polluted_df = (
                df[[time_col, most_polluted_target]]
                .dropna()
                .sort_values(
                    most_polluted_target,
                    ascending=False
                )
                .head(10)
            )

            with st.expander("📋 Open Most Polluted Days Table"):

                st.dataframe(
                    polluted_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.success("""
Key Insight:
Extreme pollution days are important for understanding environmental risk and improving forecasting models.
""")

    else:
        st.warning("No target columns found for most polluted days.")

render_footer()