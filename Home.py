import streamlit as st
import pandas as pd
import numpy as np
from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

# =========================
# PAGE CONFIG CACHE
# =========================
st.cache_data.clear = False

# =========================
# PROJECT OVERVIEW
# =========================
st.markdown("""
<div style="
background-color:#ffffff;
padding:24px;
border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08);
margin-bottom:28px;
">

<h2 style="margin-top:0; color:#003d7a;">AI-Driven Air Quality Analysis & Forecasting</h2>

<p style="font-size:15px; line-height:1.6; color:#334155;">
This dashboard is part of an applied research project conducted in collaboration with 
<b>Auckland Council</b>. The project focuses on understanding pollution behaviour in 
Auckland’s city centre and developing AI-powered forecasting models.
</p>

<h4 style="margin-bottom:6px; color:#003d7a;">Team Members</h4>
<p style="font-size:14px; line-height:1.5; color:#475569;">
• Seema Devi<br>
• Deshika<br>
• Sarah
</p>

<h4 style="margin-bottom:6px; color:#003d7a;">Supervisors</h4>
<p style="font-size:14px; line-height:1.5; color:#475569;">
• Dr Louis Boamponsem (External – Auckland Council)<br>
• Dr Sara Zandi (Internal Supervisor)
</p>

<p style="font-size:14px; line-height:1.6; color:#334155;">
The project investigates how traffic activity, pedestrian movement, and meteorological 
conditions influence pollutant levels such as PM<sub>2.5</sub>, PM<sub>10</sub>, and NO₂. 
Using advanced machine learning and AI techniques, the system provides data-driven insights 
and short-term air quality forecasts for Auckland’s city centre.
</p>

</div>
""", unsafe_allow_html=True)

# =========================
# KEEP DATA ACROSS PAGES
# =========================
if "merged_df" in st.session_state:

    df = st.session_state["merged_df"]

else:

    # =========================
    # FILE CHECK
    # =========================
    file1 = st.session_state.get("file1")
    file2 = st.session_state.get("file2")

    if not file1 or not file2:
        st.warning("⬅ Please upload both datasets from the sidebar.")
        st.stop()

    # =========================
    # FAST FILE READING
    # =========================
    @st.cache_data(show_spinner=False)
    def read_any(file):

        if file.name.endswith(".csv"):
            return pd.read_csv(
                file,
                low_memory=False
            )

        else:
            return pd.read_excel(file)

    # =========================
    # PROCESS FUNCTION
    # =========================
    @st.cache_data(show_spinner=True)
    def process_and_merge(file1, file2):

        # ---------------------
        # READ FILES
        # ---------------------
        raw1 = read_any(file1)
        raw2 = read_any(file2)

        # ---------------------
        # PROCESS FUNCTION
        # ---------------------
        def process(df):

            df = df.copy()

            # CLEAN COLUMN NAMES
            df.columns = df.columns.str.strip()

            # CLEAN TIME
            df['Time'] = (
                df['Time']
                .astype(str)
                .str.lower()
                .str.strip()
            )

            # FIX am/pm ISSUE
            df['Time'] = df['Time'].str.replace(
                r'(am|pm)\s+\1',
                r'\1',
                regex=True
            )

            # CREATE DATETIME
            df['datetime'] = pd.to_datetime(
                df['Date'].astype(str) + " " + df['Time'].astype(str),
                errors='coerce',
                dayfirst=True
            )

            # CONVERT VALUE
            df['Value'] = pd.to_numeric(
                df['Value'],
                errors='coerce'
            )

            # FLOOR TO HOUR
            df['datetime_hour'] = df['datetime'].dt.floor('h')

            # GROUP
            df_hourly = (
                df.groupby(
                    ['Site', 'datetime_hour', 'Parameter']
                )['Value']
                .mean()
                .reset_index()
            )

            # PIVOT
            df_wide = df_hourly.pivot_table(
                index=['Site', 'datetime_hour'],
                columns='Parameter',
                values='Value'
            ).reset_index()

            # CLEAN COLUMN NAMES
            df_wide.columns.name = None
            df_wide = df_wide.rename(columns=str.upper)

            return df_wide

        # ---------------------
        # PROCESS BOTH
        # ---------------------
        df1 = process(raw1)
        df2 = process(raw2)

        # ---------------------
        # MERGE
        # ---------------------
        merged = df1.merge(
            df2,
            on=['SITE', 'DATETIME_HOUR'],
            how='outer',
            suffixes=('_1', '_2')
        )

        # REMOVE DUPLICATES
        merged = merged.loc[:, ~merged.columns.duplicated()]

        # ---------------------
        # COMBINE COLUMNS
        # ---------------------
        cols = [
            'AQI',
            'NO',
            'NO2',
            'NOX',
            'PM10',
            'PM2.5',
            'TEMP',
            'RH',
            'WS',
            'WD'
        ]

        for col in cols:

            col1 = f"{col}_1"
            col2 = f"{col}_2"

            if col1 in merged.columns and col2 in merged.columns:

                merged[col] = merged[col1].combine_first(
                    merged[col2]
                )

        # ---------------------
        # DROP EXTRA COLUMNS
        # ---------------------
        merged = merged.drop(
            columns=[
                c for c in merged.columns
                if c.endswith("_1") or c.endswith("_2")
            ],
            errors='ignore'
        )

        # SORT
        merged = merged.sort_values(
            ['SITE', 'DATETIME_HOUR']
        )

        # DATETIME TYPE
        merged['DATETIME_HOUR'] = pd.to_datetime(
            merged['DATETIME_HOUR']
        )

        return merged

    # =========================
    # PROCESS + CACHE
    # =========================
    with st.spinner("Processing and merging datasets..."):

        try:

            df = process_and_merge(file1, file2)

            # STORE PERMANENTLY
            st.session_state["merged_df"] = df

        except Exception as e:

            st.error(f"Error processing datasets: {e}")
            st.stop()

# =========================
# FILTERS
# =========================
st.markdown("### 📅 Filter Data by Year / Month")

df['YEAR'] = df['DATETIME_HOUR'].dt.year
df['MONTH'] = df['DATETIME_HOUR'].dt.month

years = sorted(df['YEAR'].dropna().unique())

col1, col2 = st.columns(2)

year_from = col1.selectbox(
    "From Year",
    years,
    index=0
)

year_to = col2.selectbox(
    "To Year",
    years,
    index=len(years)-1
)

months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

selected_months = st.multiselect(
    "Select Months (optional)",
    options=list(months.keys()),
    format_func=lambda x: months[x]
)

# =========================
# APPLY FILTERS
# =========================
filtered_df = df[
    (df['YEAR'] >= year_from) &
    (df['YEAR'] <= year_to)
]

if selected_months:

    filtered_df = filtered_df[
        filtered_df['MONTH'].isin(selected_months)
    ]

# =========================
# SAVE FILTERED DATA
# =========================
st.session_state["df"] = filtered_df

# =========================
# SUCCESS MESSAGE
# =========================
st.success("Datasets merged and filtered successfully.")

# =========================
# PREVIEW
# =========================
st.subheader("Filtered Dataset Preview")

st.dataframe(
    filtered_df.head(200),
    use_container_width=True
)
# =========================
# DATA PARAMETERS (CLEAN UI)
# =========================
st.markdown("""
<div style="
background-color:#ffffff;
padding:20px;
border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08);
margin-top:20px;
">
<h3 style="color:#003d7a;">📘 Data Parameters</h3>
<p style="color:gray; font-size:14px;">
Click to expand each parameter description
</p>
</div>
""", unsafe_allow_html=True)

with st.expander("🌫 Air Quality Parameters"):
    st.markdown("""
    - **AQI** → Air Quality Index  
    - **PM2.5** → Fine particulate matter  
    - **PM10** → Coarse particulate matter  
    - **NO, NO2, NOX** → Nitrogen-based pollutants  
    """)

with st.expander("🌦 Meteorological Data"):
    st.markdown("""
    - **TEMP** → Temperature (°C)  
    - **RH** → Relative Humidity (%)  
    - **WS** → Wind Speed  
    - **WD** → Wind Direction  
    """)

with st.expander("🚗 Traffic & Activity Data"):
    st.markdown("""
    - **TRAFFICV** → Traffic volume  
    - **TOTAL_PEDESTRIANS** → Pedestrian count  
    - **CITY_CENTRE_TVCOUNT** → City centre traffic volume  
    """)

with st.expander("📍 Metadata"):
    st.markdown("""
    - **SITE** → Monitoring location  
    - **DATETIME_HOUR** → Hourly timestamp  
    """)
# =========================
# FOOTER – ALWAYS AT BOTTOM
# =========================
render_footer()