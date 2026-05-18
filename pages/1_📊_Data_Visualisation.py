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

df = st.session_state.get("df")

if df is None:
    st.warning("⬅ Please upload, process, and confirm data from the Home page first.")
    render_footer()
    st.stop()

df_raw = df.copy()

st.header("📊 Data Visualisation")

st.markdown("""
<div style="background-color:#ffffff; padding:18px; border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08); margin-bottom:20px;">
<h4 style="color:#003d7a; margin-top:0;">Purpose of This Page</h4>
<p style="font-size:15px; color:#334155; line-height:1.6;">
This page visualises the overall structure, quality, completeness, and general behaviour of the full dataset.
Detailed AQI, PM2.5 and NO2 forecasting analysis should be done on the EDA page.
</p>
</div>
""", unsafe_allow_html=True)

# =========================
# DETECT COLUMNS
# =========================
numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_raw.select_dtypes(exclude=np.number).columns.tolist()

time_col = None
for col in df_raw.columns:
    if "date" in col.lower() or "time" in col.lower():
        time_col = col
        break

if time_col:
    df_raw[time_col] = pd.to_datetime(df_raw[time_col], errors="coerce")

# =========================
# CSS
# =========================
st.markdown("""
<style>
.kpi-card {
    padding: 18px;
    border-radius: 16px;
    color: white;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.12);
    text-align: center;
}
.kpi-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 27px;
    font-weight: 900;
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


def mean_value(column):
    if column in df_raw.columns:
        value = pd.to_numeric(df_raw[column], errors="coerce").mean()
        return f"{value:.2f}" if pd.notna(value) else "N/A"
    return "N/A"


# =========================
# 1. KPI OVERVIEW
# =========================
st.subheader("1️⃣ General Dataset KPI Overview")

k1, k2, k3, k4, k5 = st.columns(5)

kpi_card(k1, "Avg AQI", mean_value("AQI"), "blue-card")
kpi_card(k2, "Avg PM2.5", mean_value("PM2.5"), "green-card")
kpi_card(k3, "Avg NO2", mean_value("NO2"), "orange-card")
kpi_card(k4, "Avg Wind Speed", mean_value("WS"), "purple-card")
kpi_card(k5, "Avg Temperature", mean_value("TEMP"), "red-card")

st.markdown("---")

# =========================
# 2. DATASET OVERVIEW
# =========================
st.subheader("2️⃣ Dataset Overview")

total_missing = df_raw.isnull().sum().sum()
missing_pct = (
    total_missing / (df_raw.shape[0] * df_raw.shape[1]) * 100
    if df_raw.shape[0] > 0 and df_raw.shape[1] > 0 else 0
)

start_date = "N/A"
end_date = "N/A"
duration = "N/A"

if time_col:
    start = df_raw[time_col].min()
    end = df_raw[time_col].max()

    if pd.notna(start):
        start_date = str(start.date())

    if pd.notna(end):
        end_date = str(end.date())

    if pd.notna(start) and pd.notna(end):
        duration = f"{(end - start).days} days"

overview_df = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Missing Values",
        "Missing %",
        "Start Date",
        "End Date",
        "Monitoring Duration"
    ],
    "Value": [
        f"{df_raw.shape[0]:,}",
        df_raw.shape[1],
        f"{total_missing:,}",
        f"{missing_pct:.2f}%",
        start_date,
        end_date,
        duration
    ]
})


def overview_style(row):
    colours = [
        "background-color:#dbeafe; color:#003d7a; font-weight:700;",
        "background-color:#dcfce7; color:#166534; font-weight:700;",
        "background-color:#fef3c7; color:#92400e; font-weight:700;",
        "background-color:#ede9fe; color:#5b21b6; font-weight:700;",
        "background-color:#fee2e2; color:#991b1b; font-weight:700;",
        "background-color:#ffedd5; color:#9a3412; font-weight:700;",
        "background-color:#e0f2fe; color:#075985; font-weight:700;",
        
    ]
    return [colours[row.name]] * len(row)


st.dataframe(
    overview_df.style.apply(overview_style, axis=1),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# =========================
# 3. COLUMN TYPE DISTRIBUTION
# =========================
st.subheader("3️⃣ Column Type Distribution")

type_df = pd.DataFrame({
    "Column Type": ["Numeric", "Categorical", "Datetime"],
    "Count": [
        len(numeric_cols),
        len(categorical_cols),
        1 if time_col else 0
    ]
})

fig = px.pie(
    type_df,
    names="Column Type",
    values="Count",
    hole=0.35,
    title="Column Type Distribution"
)

fig.update_traces(
    textinfo="percent+label",
    hovertemplate="Type: %{label}<br>Count: %{value}<extra></extra>"
)

fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True, config=config)

with st.expander("Open data types table"):
    dtype_df = pd.DataFrame({
        "Parameter": df_raw.columns,
        "Data Type": df_raw.dtypes.astype(str).values,
        "Non-Null Count": df_raw.notna().sum().values,
        "Missing Count": df_raw.isna().sum().values,
        "Missing %": (df_raw.isna().sum().values / len(df_raw) * 100).round(2)
    })

    st.dataframe(
        dtype_df,
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# =========================
# 4. DATA COMPLETENESS
# =========================
st.subheader("4️⃣ Data Completeness")

missing_df = df_raw.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Count"]

missing_df["Missing %"] = (
    missing_df["Missing Count"] / len(df_raw) * 100
).round(2)

missing_df["Completeness %"] = (
    100 - missing_df["Missing %"]
).round(2)

overall_completeness = missing_df["Completeness %"].mean()
total_missing_values = missing_df["Missing Count"].sum()
columns_with_missing = (missing_df["Missing Count"] > 0).sum()

c1, c2, c3 = st.columns(3)

kpi_card(c1, "Overall Completeness", f"{overall_completeness:.2f}%", "green-card")
kpi_card(c2, "Missing Values", f"{total_missing_values:,}", "red-card")
kpi_card(c3, "Columns With Missing", columns_with_missing, "purple-card")

st.progress(overall_completeness / 100)

missing_plot = missing_df[
    missing_df["Missing Count"] > 0
].sort_values("Missing %", ascending=False)

if not missing_plot.empty:
    fig = px.bar(
        missing_plot,
        x="Column",
        y="Missing %",
        color="Missing %",
        color_continuous_scale="Oranges",
        title="Missing Percentage by Column",
        text="Missing %"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="Column: %{x}<br>Missing: %{y:.2f}%<extra></extra>"
    )

    fig.update_layout(
        height=480,
        xaxis_title="Column",
        yaxis_title="Missing %",
        xaxis_tickangle=45
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    with st.expander("Open missing percentage table"):
        st.dataframe(
            missing_df,
            use_container_width=True,
            hide_index=True
        )
else:
    st.success("✅ No missing values detected.")

st.markdown("---")

# =========================
# 5. RAW VARIABLE DISTRIBUTION
# =========================
st.subheader("5️⃣ Raw Variable Distribution")

if numeric_cols:

    selected_distribution_col = st.selectbox(
        "Select a numeric variable to view distribution",
        numeric_cols,
        key="distribution_col"
    )

    fig = px.histogram(
        df_raw,
        x=selected_distribution_col,
        nbins=40,
        marginal="box",
        title=f"Distribution of {selected_distribution_col}"
    )

    fig.update_layout(
        height=450,
        xaxis_title=selected_distribution_col,
        yaxis_title="Number of Records"
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    with st.expander("📋 Open Distribution Summary Table"):

        summary_df = (
            df_raw[numeric_cols]
            .describe()
            .T
            .reset_index()
        )

        summary_df = summary_df.rename(columns={"index": "Parameter"})

        summary_df["Skewness"] = (
            df_raw[numeric_cols]
            .skew(numeric_only=True)
            .values
        )

        summary_df["Missing %"] = (
            df_raw[numeric_cols]
            .isna()
            .sum()
            .values / len(df_raw) * 100
        ).round(2)

        summary_df["IQR"] = summary_df["75%"] - summary_df["25%"]

        summary_df["Lower Outlier Limit"] = (
            summary_df["25%"] - 1.5 * summary_df["IQR"]
        )

        summary_df["Upper Outlier Limit"] = (
            summary_df["75%"] + 1.5 * summary_df["IQR"]
        )

        summary_df = summary_df[[
            "Parameter",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "IQR",
            "Skewness",
            "Lower Outlier Limit",
            "Upper Outlier Limit",
            "Missing %"
        ]]

        st.dataframe(
            summary_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.info("""
How to interpret:
- High skewness means an asymmetric distribution.
- Large IQR means higher variability.
- Values outside outlier limits may be abnormal or extreme observations.
- Compare mean and median to understand skewed behaviour.
""")

    st.info("""
How to read this distribution:
- A tall peak means many records have similar values.
- A long right tail may indicate high pollution or weather spikes.
- Boxplot outliers may indicate unusual events or sensor anomalies.
- Strongly skewed variables may need transformation before machine learning.
""")

else:
    st.warning("No numeric columns available for distribution chart.")

st.markdown("---")

# =========================
# 6. DAILY AVERAGE RAW TREND
# =========================
st.subheader("6️⃣ Daily Average Raw Data Trend")

if time_col and numeric_cols:

    trend_options = [
        col for col in [
            "AQI", "PM2.5", "NO2", "PM10",
            "TEMP", "RH", "WS",
            "TRAFFICV", "TOTAL_PEDESTRIANS"
        ]
        if col in df_raw.columns
    ]

    if not trend_options:
        trend_options = numeric_cols[:5]

    selected_trend_cols = st.multiselect(
        "Select variables to show daily trend",
        trend_options,
        default=trend_options[:3]
    )

    if selected_trend_cols:

        trend_df = df_raw[[time_col] + selected_trend_cols].copy()
        trend_df = trend_df.dropna(subset=[time_col])

        trend_df = (
            trend_df
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

        fig = px.line(
            trend_long,
            x=time_col,
            y="Daily Average",
            color="Variable",
            title="Daily Average Trend of Selected Variables"
        )

        events = {
            "2020-03-25": "COVID Lockdown",
            "2020-08-12": "Auckland Lockdown",
            "2021-08-17": "Delta Lockdown",
            "2023-01-27": "Auckland Floods",
            "2023-02-13": "Cyclone Gabrielle",
            "2023-07-20": "FIFA Women's World Cup"
        }

        event_y = trend_long["Daily Average"].max()

        if pd.isna(event_y):
            event_y = 0

        for event_date, event_name in events.items():

            event_dt = pd.to_datetime(event_date)

            fig.add_scatter(
                x=[event_dt],
                y=[event_y],
                mode="markers+text",
                marker=dict(
                    symbol="cross",
                    size=2,
                    color="red",
                    line=dict(
                        width=2,
                        color="darkred"
                    )
                ),
                text=["+"],
                textposition="bottom center",
                textfont=dict(
                    size=26,
                    color="red"
                ),
                name=event_name,
                showlegend=False,
                hovertemplate=(
                    f"<b>{event_name}</b><br>"
                    f"Date: {event_dt.date()}<extra></extra>"
                )
            )

            fig.add_annotation(
                x=event_dt,
                y=event_y,
                text=event_name,
                showarrow=False,
                yshift=30,
                font=dict(
                    size=10,
                    color="red"
                ),
                bgcolor="rgba(255,255,255,0.75)"
            )

        fig.update_layout(
            height=530,
            xaxis_title="Date",
            yaxis_title="Daily Average"
        )

        st.plotly_chart(fig, use_container_width=True, config=config)

        st.info("""
Event markers help explain unusual changes in pollution, weather, traffic, or pedestrian activity.
For example, lockdowns may reduce traffic-related NO2, while floods or cyclones may change wind, rain, and pollutant dispersion.
""")

    else:
        st.info("Select at least one variable to display the trend.")

else:
    st.warning("Datetime and numeric columns are required for trend visualisation.")

st.markdown("---")

# =========================
# 7. WIND ROSE + WIND POLLUTION POLAR
# =========================
st.subheader("7️⃣ Wind Rose and Wind-Pollution Analysis")

if "WD" in df_raw.columns and "WS" in df_raw.columns:

    wind_df = df_raw[["WD", "WS"]].copy()
    wind_df["WD"] = pd.to_numeric(wind_df["WD"], errors="coerce")
    wind_df["WS"] = pd.to_numeric(wind_df["WS"], errors="coerce")
    wind_df = wind_df.dropna()

    wind_df = wind_df[
        (wind_df["WD"] >= 0) &
        (wind_df["WD"] <= 360) &
        (wind_df["WS"] >= 0)
    ]

    if not wind_df.empty:

        direction_bins = np.arange(0, 361, 30)
        direction_labels = [
            "0°", "30°", "60°", "90°", "120°", "150°",
            "180°", "210°", "240°", "270°", "300°", "330°"
        ]

        wind_df["Direction Bin"] = pd.cut(
            wind_df["WD"],
            bins=direction_bins,
            labels=direction_labels,
            include_lowest=True,
            right=False
        )

        speed_bins = [0, 1, 3, 5, 8, np.inf]
        speed_labels = ["0–1", "1–3", "3–5", "5–8", "8+"]

        wind_df["Wind Speed Range"] = pd.cut(
            wind_df["WS"],
            bins=speed_bins,
            labels=speed_labels,
            include_lowest=True
        )

        wind_rose = (
            wind_df
            .groupby(["Direction Bin", "Wind Speed Range"], observed=False)
            .size()
            .reset_index(name="Frequency")
        )

        fig = px.bar_polar(
            wind_rose,
            r="Frequency",
            theta="Direction Bin",
            color="Wind Speed Range",
            title="Wind Rose: Wind Direction and Wind Speed Frequency",
            template="plotly_white"
        )

        fig.update_layout(
            height=560,
            polar=dict(
                radialaxis=dict(showticklabels=True),
                angularaxis=dict(direction="clockwise")
            )
        )

        st.plotly_chart(fig, use_container_width=True, config=config)

        st.info("""
Wind rose interpretation:
- Direction shows where wind is coming from.
- Colour shows wind speed range.
- Longer bars mean wind comes from that direction more often.
- This explains general wind behaviour before analysing pollutant influence.
""")

        pollutant_options = [
            col for col in ["AQI", "PM2.5", "NO2"]
            if col in df_raw.columns
        ]

        if pollutant_options:

            target_pollutant = st.selectbox(
                "Select pollutant for wind influence analysis",
                pollutant_options,
                key="wind_target"
            )

            wind_pollution_df = df_raw[["WD", "WS", target_pollutant]].copy()

            wind_pollution_df["WD"] = pd.to_numeric(
                wind_pollution_df["WD"],
                errors="coerce"
            )
            wind_pollution_df["WS"] = pd.to_numeric(
                wind_pollution_df["WS"],
                errors="coerce"
            )
            wind_pollution_df[target_pollutant] = pd.to_numeric(
                wind_pollution_df[target_pollutant],
                errors="coerce"
            )

            wind_pollution_df = wind_pollution_df.dropna()

            wind_pollution_df = wind_pollution_df[
                (wind_pollution_df["WD"] >= 0) &
                (wind_pollution_df["WD"] <= 360) &
                (wind_pollution_df["WS"] >= 0)
            ]

            if not wind_pollution_df.empty:

                plot_df = wind_pollution_df.sample(
                    min(3000, len(wind_pollution_df)),
                    random_state=42
                )

                fig = px.scatter_polar(
                    plot_df,
                    r="WS",
                    theta="WD",
                    color=target_pollutant,
                    color_continuous_scale="Turbo",
                    title=f"{target_pollutant} Concentration by Wind Direction and Wind Speed"
                )

                fig.update_layout(height=540)

                st.plotly_chart(fig, use_container_width=True, config=config)

                st.info(f"""
How to read this plot:
- Angle shows wind direction.
- Distance from centre shows wind speed.
- Colour shows {target_pollutant} concentration.
- High pollution at low wind speed may suggest pollutant accumulation.
- High pollution from one direction may suggest pollution transport from that direction.
""")
            else:
                st.warning("No valid wind and pollutant records available for polar pollution plot.")
        else:
            st.warning("AQI, PM2.5 or NO2 columns are required for wind-pollution polar analysis.")

    else:
        st.warning("Wind direction and wind speed columns exist but contain no valid data.")

else:
    st.warning("WD and WS columns are required for wind rose analysis.")

st.markdown("---")

# =========================
# 8. HOURLY OVERALL PATTERN
# =========================
st.subheader("8️⃣ Hourly Pattern of Key Variables")

if time_col and numeric_cols:

    df_raw["Hour"] = df_raw[time_col].dt.hour

    hourly_options = [
        col for col in [
            "AQI", "PM2.5", "NO2", "PM10",
            "TEMP", "RH", "WS",
            "TRAFFICV", "TOTAL_PEDESTRIANS"
        ]
        if col in df_raw.columns
    ]

    if not hourly_options:
        hourly_options = numeric_cols[:5]

    selected_hourly_cols = st.multiselect(
        "Select variables for hourly pattern",
        hourly_options,
        default=hourly_options[:3],
        key="hourly_cols"
    )

    if selected_hourly_cols:

        hourly_df = (
            df_raw
            .groupby("Hour")[selected_hourly_cols]
            .mean()
            .reset_index()
        )

        hourly_long = hourly_df.melt(
            id_vars="Hour",
            value_vars=selected_hourly_cols,
            var_name="Variable",
            value_name="Average Value"
        )

        fig = px.line(
            hourly_long,
            x="Hour",
            y="Average Value",
            color="Variable",
            markers=True,
            title="Average Hourly Pattern of Selected Variables"
        )

        fig.update_layout(
            height=480,
            xaxis_title="Hour of Day",
            yaxis_title="Average Value",
            xaxis=dict(dtick=1)
        )

        st.plotly_chart(fig, use_container_width=True, config=config)

        st.info("""
Hourly patterns help show daily behaviour such as morning peaks, evening peaks, or low overnight activity.
More detailed weekday and seasonal analysis should be placed on the EDA page.
""")

    else:
        st.info("Select at least one variable to display hourly pattern.")

else:
    st.warning("Datetime and numeric columns are required for hourly pattern analysis.")

render_footer()