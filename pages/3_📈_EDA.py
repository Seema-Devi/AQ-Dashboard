# =====================================
# IMPORT LIBRARIES
# =====================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

px.defaults.template = "plotly_white"

# Stops charts zooming in/out when mouse scrolls
config = {
    "displayModeBar": False,
    "scrollZoom": False
}

# =====================================
# LOAD DATA
# =====================================
if "df_cleaned" in st.session_state:
    df = st.session_state["df_cleaned"]
elif "df" in st.session_state:
    df = st.session_state["df"]
else:
    st.warning("""
⬅ Please upload both datasets from the Home page to continue.

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

df = df.copy()

st.header("📊 Business-Oriented Air Quality EDA")

# =====================================
# DETECT COLUMNS
# =====================================
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
pm25_col = find_col(["PM2.5", "PM25", "PM 2.5"])
no2_col = find_col(["NO2", "NO₂"])

traffic_col = find_col([
    "traffic", "traffic volume", "vehicle", "vehicles",
    "vehicle count", "traffic count"
])

ped_col = find_col([
    "pedestrian", "pedestrian count", "foot traffic",
    "people count", "pedestrian volume"
])

wind_speed_col = find_col(["wind speed", "windspeed", "ws"])
wind_dir_col = find_col(["wind direction", "winddirection", "wd"])

temp_col = find_col(["temperature", "temp"])
humidity_col = find_col(["humidity", "relative humidity"])
rain_col = find_col(["rain", "rainfall", "precipitation"])

# =====================================
# HELPER FUNCTIONS
# =====================================
def aqi_category(x):
    if pd.isna(x):
        return np.nan
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Moderate"
    elif x <= 150:
        return "Unhealthy for Sensitive"
    elif x <= 200:
        return "Unhealthy"
    elif x <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def pm25_category(x):
    if pd.isna(x):
        return np.nan
    if x <= 12:
        return "Low"
    elif x <= 35.4:
        return "Moderate"
    elif x <= 55.4:
        return "High"
    else:
        return "Very High"

def no2_category(x):
    if pd.isna(x):
        return np.nan
    if x <= 53:
        return "Low"
    elif x <= 100:
        return "Moderate"
    elif x <= 360:
        return "High"
    else:
        return "Very High"

def safe_qcut(series, labels):
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    except:
        return pd.cut(series, bins=len(labels), labels=labels)

category_order = [
    "Good", "Low", "Moderate", "Unhealthy for Sensitive",
    "High", "Unhealthy", "Very High", "Very Unhealthy", "Hazardous"
]

# =====================================
# PAGE PURPOSE
# =====================================
st.markdown("""
<div style="background-color:#ffffff; padding:20px; border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08); margin-bottom:22px;">
<h4 style="color:#003d7a; margin-top:0;">Business Focus</h4>
<p style="font-size:15px; color:#334155; line-height:1.6;">
This EDA page tells a business story: pollution status, key relationships,
urban activity pressure, time-based pollution risk, weather and wind behaviour,
and forecasting readiness for AQI, PM2.5 and NO2.
</p>
</div>
""", unsafe_allow_html=True)

# =====================================
# 1. AIR QUALITY SITUATION
# =====================================
st.subheader("1️⃣ Air Quality Situation")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", f"{len(df):,}")
c2.metric("Average AQI", f"{df[aqi_col].mean():.1f}" if aqi_col else "Not found")
c3.metric("Average PM2.5", f"{df[pm25_col].mean():.1f}" if pm25_col else "Not found")
c4.metric("Average NO2", f"{df[no2_col].mean():.1f}" if no2_col else "Not found")

st.markdown("---")

# =====================================
# 2. RIGHT-SIDE DIAGONAL HEATMAP
# =====================================
st.subheader("2️⃣ Key Relationship Heatmap")

exclude_words = [
    "year", "month", "day", "date", "hour", "minute", "second",
    "weekday", "week", "season"
]

heatmap_cols = [
    col for col in numeric_cols
    if not any(word in col.lower() for word in exclude_words)
]

if len(heatmap_cols) > 1:

    corr = df[heatmap_cols].corr().round(2)

    # Right-side / upper diagonal heatmap
    mask = np.tril(np.ones_like(corr, dtype=bool), k=-1)
    corr_masked = corr.mask(mask)

    fig = px.imshow(
        corr_masked,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    fig.update_traces(
        hovertemplate=
        "<b>%{x}</b> vs <b>%{y}</b><br>" +
        "Correlation: %{z}<extra></extra>"
    )

    fig.update_layout(
        title={
            "text": "Relationship Between Pollution, Traffic, Pedestrian and Weather Variables",
            "x": 0.02
        },
        height=720,
        font=dict(size=13),
        xaxis_title="Business Variables",
        yaxis_title="Business Variables",
        coloraxis_colorbar=dict(
            title="Relationship",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=[
                "Strong Negative",
                "Moderate Negative",
                "No Relation",
                "Moderate Positive",
                "Strong Positive"
            ]
        )
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("Not enough numeric columns available for heatmap.")

st.markdown("---")

# =====================================
# 3. POLLUTION CATEGORY STATUS
# =====================================
st.subheader("3️⃣ Pollution Category Status")

pollution_options = []

if aqi_col:
    df["AQI Category"] = df[aqi_col].apply(aqi_category)
    pollution_options.append(("AQI", "AQI Category"))

if pm25_col:
    df["PM2.5 Category"] = df[pm25_col].apply(pm25_category)
    pollution_options.append(("PM2.5", "PM2.5 Category"))

if no2_col:
    df["NO2 Category"] = df[no2_col].apply(no2_category)
    pollution_options.append(("NO2", "NO2 Category"))

if pollution_options:

    selected_pollutant = st.selectbox(
        "Select pollutant:",
        [x[0] for x in pollution_options]
    )

    selected_category_col = dict(pollution_options)[selected_pollutant]

    category_count = df[selected_category_col].value_counts().reset_index()
    category_count.columns = ["Pollution Category", "Records"]

    total_records = category_count["Records"].sum()
    category_count["Percentage"] = (
        category_count["Records"] / total_records * 100
    ).round(1)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            category_count,
            x="Pollution Category",
            y="Records",
            color="Pollution Category",
            text=category_count["Percentage"].astype(str) + "%",
            category_orders={"Pollution Category": category_order},
            title=f"{selected_pollutant} Pollution Severity Distribution"
        )

        fig.update_traces(
            textposition="outside",
            hovertemplate=
            "Category: %{x}<br>" +
            "Records: %{y}<br>" +
            "Share: %{text}<extra></extra>"
        )

        fig.update_layout(
            height=430,
            xaxis_title="Pollution Category",
            yaxis_title="Number of Records",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, config=config)

    with c2:
        fig = px.pie(
            category_count,
            names="Pollution Category",
            values="Records",
            hole=0.35,
            title=f"{selected_pollutant} Category Share"
        )

        fig.update_traces(
            textinfo="percent+label",
            hovertemplate=
            "Category: %{label}<br>" +
            "Records: %{value}<br>" +
            "Share: %{percent}<extra></extra>"
        )

        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("AQI, PM2.5 or NO2 column not found.")

st.markdown("---")

# =====================================
# 4. URBAN ACTIVITY PRESSURE
# =====================================
st.subheader("4️⃣ Urban Activity Pressure")

if traffic_col and ped_col and aqi_col:

    df["Traffic Level"] = safe_qcut(
        df[traffic_col],
        ["Low Traffic", "Medium Traffic", "High Traffic"]
    )

    df["Pedestrian Activity"] = safe_qcut(
        df[ped_col],
        ["Low Activity", "Medium Activity", "High Activity"]
    )

    activity_summary = (
        df.groupby(["Traffic Level", "Pedestrian Activity"], observed=False)[aqi_col]
        .mean()
        .reset_index()
    )

    fig = px.density_heatmap(
        activity_summary,
        x="Traffic Level",
        y="Pedestrian Activity",
        z=aqi_col,
        text_auto=".1f",
        color_continuous_scale="YlOrRd",
        title="Urban Activity Pressure on AQI"
    )

    fig.update_traces(
        hovertemplate=
        "Traffic Level: %{x}<br>" +
        "Pedestrian Activity: %{y}<br>" +
        "Average AQI: %{z:.1f}<extra></extra>"
    )

    fig.update_layout(
        height=520,
        xaxis_title="Traffic Activity",
        yaxis_title="Pedestrian Density",
        coloraxis_colorbar=dict(title="AQI Severity")
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

elif traffic_col and aqi_col:

    df["Traffic Level"] = safe_qcut(
        df[traffic_col],
        ["Low Traffic", "Medium Traffic", "High Traffic"]
    )

    traffic_summary = (
        df.groupby("Traffic Level", observed=False)[aqi_col]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        traffic_summary,
        x="Traffic Level",
        y=aqi_col,
        color="Traffic Level",
        text=traffic_summary[aqi_col].round(1),
        title="Average AQI by Traffic Activity"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="Traffic Level: %{x}<br>Average AQI: %{y:.1f}<extra></extra>"
    )

    fig.update_layout(height=430, yaxis_title="Average AQI", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=config)

elif ped_col and aqi_col:

    df["Pedestrian Activity"] = safe_qcut(
        df[ped_col],
        ["Low Activity", "Medium Activity", "High Activity"]
    )

    ped_summary = (
        df.groupby("Pedestrian Activity", observed=False)[aqi_col]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        ped_summary,
        x="Pedestrian Activity",
        y=aqi_col,
        color="Pedestrian Activity",
        text=ped_summary[aqi_col].round(1),
        title="Average AQI by Pedestrian Activity"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="Pedestrian Activity: %{x}<br>Average AQI: %{y:.1f}<extra></extra>"
    )

    fig.update_layout(height=430, yaxis_title="Average AQI", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("Traffic, pedestrian or AQI column not found.")

st.markdown("---")

# =====================================
# 5. POLLUTION RISK OVER TIME
# =====================================
st.subheader("5️⃣ Pollution Risk Over Time")

if time_col and aqi_col:

    df["Hour"] = df[time_col].dt.hour
    df["AQI Category"] = df[aqi_col].apply(aqi_category)

    hourly_risk = (
        df.groupby(["Hour", "AQI Category"], observed=False)
        .size()
        .reset_index(name="Records")
    )

    fig = px.area(
        hourly_risk,
        x="Hour",
        y="Records",
        color="AQI Category",
        category_orders={"AQI Category": category_order},
        title="Pollution Risk Pattern Throughout the Day"
    )

    fig.update_traces(
        hovertemplate=
        "Hour: %{x}:00<br>" +
        "Records: %{y}<extra></extra>"
    )

    fig.update_layout(
        height=500,
        xaxis_title="Hour of Day",
        yaxis_title="Pollution Events"
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    df["Month"] = df[time_col].dt.month_name()
    df["Month Number"] = df[time_col].dt.month

    monthly_avg = (
        df.groupby(["Month Number", "Month"])[aqi_col]
        .mean()
        .reset_index()
        .sort_values("Month Number")
    )

    monthly_avg["AQI Status"] = monthly_avg[aqi_col].apply(aqi_category)

    fig = px.bar(
        monthly_avg,
        x="Month",
        y=aqi_col,
        color="AQI Status",
        text=monthly_avg[aqi_col].round(1),
        category_orders={"AQI Status": category_order},
        title="Average AQI Status by Month"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=
        "Month: %{x}<br>" +
        "Average AQI: %{y:.1f}<extra></extra>"
    )

    fig.update_layout(
        height=460,
        yaxis_title="Average AQI",
        xaxis_title="Month"
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("Date/time or AQI column not found.")

st.markdown("---")

# =====================================
# 6. WEATHER AND WIND BEHAVIOUR
# =====================================
st.subheader("6️⃣ Weather and Wind Behaviour")

weather_options = []

for col in [temp_col, humidity_col, rain_col, wind_speed_col]:
    if col and col not in weather_options:
        weather_options.append(col)

if weather_options and aqi_col:

    selected_weather = st.selectbox(
        "Select weather factor for AQI:",
        weather_options
    )

    df["Weather Level"] = safe_qcut(
        df[selected_weather],
        ["Low", "Medium", "High", "Very High"]
    )

    weather_summary = (
        df.groupby("Weather Level", observed=False)[aqi_col]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        weather_summary,
        x="Weather Level",
        y=aqi_col,
        color="Weather Level",
        text=weather_summary[aqi_col].round(1),
        title=f"Average AQI by {selected_weather} Level"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=
        "Weather Level: %{x}<br>" +
        "Average AQI: %{y:.1f}<extra></extra>"
    )

    fig.update_layout(
        height=430,
        yaxis_title="Average AQI",
        xaxis_title=f"{selected_weather} Level",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("Weather columns or AQI column not found.")

if wind_speed_col and wind_dir_col and (pm25_col or no2_col):

    wind_pollutants = []

    if pm25_col:
        wind_pollutants.append("PM2.5")

    if no2_col:
        wind_pollutants.append("NO2")

    selected_wind_pollutant = st.selectbox(
        "Select pollutant for wind impact:",
        wind_pollutants
    )

    pollutant_col = pm25_col if selected_wind_pollutant == "PM2.5" else no2_col

    wind_df = df[[wind_speed_col, wind_dir_col, pollutant_col]].dropna().copy()

    wind_df["Wind Speed Level"] = pd.cut(
        wind_df[wind_speed_col],
        bins=[-np.inf, 2, 5, 8, np.inf],
        labels=["Calm", "Light", "Moderate", "Strong"]
    )

    fig = px.scatter_polar(
        wind_df,
        r=wind_speed_col,
        theta=wind_dir_col,
        color=pollutant_col,
        color_continuous_scale="YlOrRd",
        title=f"Wind Direction Impact on {selected_wind_pollutant}"
    )

    fig.update_traces(
        hovertemplate=
        "Wind Direction: %{theta}°<br>" +
        "Wind Speed: %{r:.1f}<br>" +
        f"{selected_wind_pollutant}: %{{marker.color:.1f}}<extra></extra>"
    )

    fig.update_layout(
        height=620,
        polar=dict(
            radialaxis=dict(title="Wind Speed")
        ),
        coloraxis_colorbar_title=selected_wind_pollutant
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    wind_summary = (
        wind_df.groupby("Wind Speed Level", observed=False)[pollutant_col]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        wind_summary,
        x="Wind Speed Level",
        y=pollutant_col,
        color="Wind Speed Level",
        text=wind_summary[pollutant_col].round(1),
        title=f"Average {selected_wind_pollutant} by Wind Speed Level"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=
        "Wind Speed Level: %{x}<br>" +
        f"Average {selected_wind_pollutant}: %{{y:.1f}}<extra></extra>"
    )

    fig.update_layout(
        height=430,
        yaxis_title=f"Average {selected_wind_pollutant}",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("Wind speed, wind direction, PM2.5 or NO2 column not found.")

st.markdown("---")

# =====================================
# 7. TOP AQI CONTRIBUTORS
# =====================================
st.subheader("7️⃣ Top AQI Contributors")

if aqi_col and len(heatmap_cols) > 1:

    corr_data = (
        df[heatmap_cols]
        .corr()[aqi_col]
        .drop(aqi_col)
        .sort_values(key=abs, ascending=False)
        .head(10)
        .reset_index()
    )

    corr_data.columns = ["Business Factor", "Relationship with AQI"]

    corr_data["Relationship Type"] = np.where(
        corr_data["Relationship with AQI"] > 0,
        "AQI Increases",
        "AQI Decreases"
    )

    fig = px.bar(
        corr_data,
        x="Relationship with AQI",
        y="Business Factor",
        orientation="h",
        color="Relationship Type",
        text=corr_data["Relationship with AQI"].round(2),
        title="Top Business Factors Associated with AQI"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=
        "Business Factor: %{y}<br>" +
        "Relationship with AQI: %{x:.2f}<extra></extra>"
    )

    fig.update_layout(
        height=520,
        yaxis=dict(autorange="reversed"),
        xaxis_title="Correlation with AQI",
        yaxis_title="Business Factor"
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

else:
    st.warning("AQI column or suitable numeric variables not available.")

st.markdown("---")

# =====================================
# 8. FORECASTING READINESS
# =====================================
st.subheader("8️⃣ Forecasting Readiness")

forecast_cols = [col for col in [aqi_col, pm25_col, no2_col] if col is not None]

if forecast_cols:

    readiness = pd.DataFrame({
        "Forecast Target": forecast_cols,
        "Available Records": [df[col].notna().sum() for col in forecast_cols],
        "Missing Records": [df[col].isna().sum() for col in forecast_cols],
        "Missing %": [(df[col].isna().mean() * 100).round(2) for col in forecast_cols]
    })

    fig = px.bar(
        readiness,
        x="Forecast Target",
        y="Available Records",
        text="Available Records",
        title="Target Availability for Forecasting"
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate=
        "Target: %{x}<br>" +
        "Available Records: %{y}<extra></extra>"
    )

    fig.update_layout(
        height=420,
        yaxis_title="Available Records"
    )

    st.plotly_chart(fig, use_container_width=True, config=config)

    st.dataframe(readiness, use_container_width=True)

else:
    st.warning("No forecast target columns found.")

# =====================================
# FOOTER
# =====================================
render_footer()