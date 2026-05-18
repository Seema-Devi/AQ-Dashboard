import streamlit as st
import pandas as pd
from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()

# =========================
# ABOUT PROJECT + WORKFLOW
# =========================

col_left, col_right = st.columns([1.15, 0.85])

with col_left:
    st.markdown("""
<div style="
    background-color:#ffffff;
    padding:22px;
    border-radius:14px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    border:1px solid #e2e8f0;
">

<h3 style="color:#1e3a8a;">📘 About This Project</h3>

<p style="font-size:14px; color:#334155; line-height:1.4; text-align: justify;">
This dashboard is part of the <b>GDDA713 Capstone Project</b>, developed in collaboration with 
<b>Auckland Council</b>. All datasets used in this analysis were provided directly by Auckland Council.
The project focuses on understanding pollution behaviour in Auckland’s city centre and developing 
AI-powered forecasting models to support evidence-based urban planning and air-quality management.
</p>

</div>
""", unsafe_allow_html=True)

    col_team, col_supervisor = st.columns(2)

    with col_team:
        st.markdown("""
<div style="background-color:#f8fafc; padding:16px; border-radius:12px;
border:1px solid #e2e8f0; margin-top:12px;">
<h4 style="color:#1e3a8a;">👩‍💻 Team Members</h4>
<p style="font-size:14px; color:#475569; line-height:1.5;">
• Seema Devi<br>
• Deshika Jayatilaka<br>
• Yaqing Zhang (Sarah)
</p>
</div>
""", unsafe_allow_html=True)

    with col_supervisor:
       st.markdown("""
<div style="background-color:#f8fafc; padding:16px; border-radius:12px;
border:1px solid #e2e8f0; margin-top:12px;">
<h4 style="color:#1e3a8a;">🎓 Supervisors</h4>
<p style="font-size:14px; color:#475569; line-height:1.5;">
• Dr Louis Boamponsem (External Supervisor - Auckland Council)<br>
• Dr Sara Zandi (Internal Supervisor - NZSE)
</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div style="
    background-color:#f0fdf4;
    border-radius:10px;
    padding:12px;
    margin-top:14px;
    ">
<p style="font-size:14px; color:#334155; line-height:1.6; margin-bottom:20px; text-align: justify;
">
The project investigates how traffic activity, pedestrian movement, and meteorological 
conditions influence pollutant levels such as PM<sub>2.5</sub>, PM<sub>10</sub>, and NO₂. 
Using advanced machine learning and AI techniques, the system provides data-driven insights 
and short-term air quality forecasts for Auckland’s city centre.
</p>
</div>
""", unsafe_allow_html=True)

with col_right:
    st.markdown("""
<div style="background-color:#ffffff; padding:22px; border-radius:14px;
            box-shadow:0px 2px 8px rgba(0,0,0,0.08); border:1px solid #e2e8f0; margin-bottom:20px;
">

<h3 style="color:#1e3a8a;">🧭 App Workflow</h3>

<div style="font-size:15px; color:#334155; line-height:1.9;">

<div style="display:flex; align-items:center; margin-bottom:14px;">
    <div style="width:38px; height:38px; border-radius:50%; background:#22c55e;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">1</div>
    <div>
        <b>📂 Upload Datasets</b><br>
        <span style="font-size:13px; color:#64748b;">Start from the Home page</span>
    </div>
</div>

<div style="display:flex; align-items:center; margin-bottom:14px;">
    <div style="width:38px; height:38px; border-radius:50%; background:#3b82f6;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">2</div>
    <div>
        <b>📊 Explore Data</b><br>
        <span style="font-size:13px; color:#64748b;">Use the Data Visualisation page</span>
    </div>
</div>

<div style="display:flex; align-items:center; margin-bottom:14px;">
    <div style="width:38px; height:38px; border-radius:50%; background:#f97316;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">3</div>
    <div>
        <b>🧹 Clean & Preprocess</b><br>
        <span style="font-size:13px; color:#64748b;">Fix missing values and timestamps</span>
    </div>
</div>

<div style="display:flex; align-items:center; margin-bottom:14px;">
    <div style="width:38px; height:38px; border-radius:50%; background:#a855f7;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">4</div>
    <div>
        <b>📊 Perform EDA</b><br>
        <span style="font-size:13px; color:#64748b;">Analyse trends and correlations</span>
    </div>
</div>

<div style="display:flex; align-items:center; margin-bottom:14px;">
    <div style="width:38px; height:38px; border-radius:50%; background:#16a34a;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">5</div>
    <div>
        <b>⚙️ Feature Engineering</b><br>
        <span style="font-size:13px; color:#64748b;">Create lag features & rolling windows</span>
    </div>
</div>

<div style="display:flex; align-items:center;">
    <div style="width:38px; height:38px; border-radius:50%; background:#dc2626;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">6</div>
    <div>
        <b>🤖 Train Models & Forecast</b><br>
        <span style="font-size:13px; color:#64748b;">Generate 24‑hour AQI predictions</span>
    </div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

# =========================
# DATA PROCESSING + MERGING
# =========================

uploaded_files = st.session_state.get("uploaded_files", [])

st.markdown("## 1️⃣ Processing of Data")
# =========================
# SHOW UPLOADED DATASETS
# =========================

if uploaded_files:

    

    for i, file in enumerate(uploaded_files, start=1):

        file_size_mb = round(file.size / (1024 * 1024), 2)

        st.markdown(f"""
        
        <b>Dataset {i}:</b> {file.name}<br>
        <span style="color:#64748b; font-size:13px;">
        Size: {file_size_mb} MB
        </span>
        
        """, unsafe_allow_html=True)

if not uploaded_files:
    st.warning("⬅ Please upload datasets from the sidebar.")
    render_footer()
    st.stop()


@st.cache_data(show_spinner=False)
def read_any(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file, low_memory=False)
    return pd.read_excel(file)


@st.cache_data(show_spinner=False)
def process_single_file(file):
    raw = read_any(file)

    raw = raw.copy()
    raw.columns = raw.columns.str.strip()

    raw["Time"] = raw["Time"].astype(str).str.lower().str.strip()
    raw["Time"] = raw["Time"].str.replace(r"(am|pm)\s+\1", r"\1", regex=True)

    raw["datetime"] = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"],
        errors="coerce",
        dayfirst=True
    )

    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    raw["datetime_hour"] = raw["datetime"].dt.floor("h")

    df_hourly = (
        raw.groupby(["Site", "datetime_hour", "Parameter"])["Value"]
        .mean()
        .reset_index()
    )

    df_wide = (
        df_hourly
        .pivot_table(
            index=["Site", "datetime_hour"],
            columns="Parameter",
            values="Value"
        )
        .reset_index()
    )

    df_wide.columns.name = None
    df_wide = df_wide.rename(columns=str.upper)

    return df_wide


@st.cache_data(show_spinner=False)
def process_and_merge_multiple(files):

    processed_list = []

    for file in files:
        processed_df = process_single_file(file)

        # Remove duplicate columns inside each processed file
        processed_df = processed_df.loc[:, ~processed_df.columns.duplicated()]

        processed_list.append(processed_df)

    merged = processed_list[0]

    for next_df in processed_list[1:]:

        merged = merged.loc[:, ~merged.columns.duplicated()]
        next_df = next_df.loc[:, ~next_df.columns.duplicated()]

        merged = pd.merge(
            merged,
            next_df,
            on=["SITE", "DATETIME_HOUR"],
            how="outer",
            suffixes=("", "_NEW")
        )

        duplicate_cols = [c for c in merged.columns if c.endswith("_NEW")]

        for dup_col in duplicate_cols:
            original_col = dup_col.replace("_NEW", "")

            # Force both sides to be Series, not DataFrames
            left_series = merged[original_col]
            right_series = merged[dup_col]

            if isinstance(left_series, pd.DataFrame):
                left_series = left_series.iloc[:, 0]

            if isinstance(right_series, pd.DataFrame):
                right_series = right_series.iloc[:, 0]

            merged[original_col] = left_series.combine_first(right_series)

            merged = merged.drop(columns=[dup_col], errors="ignore")

        merged = merged.loc[:, ~merged.columns.duplicated()]

    merged["DATETIME_HOUR"] = pd.to_datetime(
        merged["DATETIME_HOUR"],
        errors="coerce"
    )

    merged = merged.dropna(subset=["DATETIME_HOUR"])

    merged = merged.sort_values(
        ["SITE", "DATETIME_HOUR"]
    )

    return merged


with st.spinner("Processing uploaded datasets..."):
    df = process_and_merge_multiple(uploaded_files)
    st.session_state["merged_df"] = df

st.success("✅ Data processed successfully.")

# =========================
# DATA AVAILABILITY BY YEAR / PARAMETER
# =========================

st.markdown("## 2️⃣ Data Availability by Year / Parameter")

df["YEAR"] = df["DATETIME_HOUR"].dt.year

exclude_cols = ["SITE", "DATETIME_HOUR", "YEAR", "MONTH"]
parameter_cols = [col for col in df.columns if col not in exclude_cols]

years = sorted(df["YEAR"].dropna().unique())

availability_table = pd.DataFrame(index=parameter_cols, columns=years)

availability_pct_table = pd.DataFrame(index=parameter_cols, columns=years)

for param in parameter_cols:
    for year in years:
        year_df = df[df["YEAR"] == year]
        total_count = len(year_df)
        available_count = year_df[param].notna().sum()

        available_pct = (
            (available_count / total_count) * 100
            if total_count > 0 else 0
        )

        availability_table.loc[param, year] = (
            f"{available_count}/{total_count} "
            f"({available_pct:.1f}%)"
        )

        availability_pct_table.loc[param, year] = available_pct


def colour_availability(value):
    try:
        pct = float(
            str(value).split("(")[1].replace("%)", "")
        )
    except:
        pct = 0

    if pct == 0:
        return "background-color: #e5e7eb;"   # grey
    elif pct <= 50:
        return "background-color: #fed7aa;"   # orange
    else:
        return "background-color: #bfdbfe;"   # blue


st.dataframe(
     availability_table.style.map(colour_availability),
    use_container_width=True
)

st.caption(
    "Grey = 0% available | Orange = 0–50% available | Blue = greater than 50% available"
)
# =========================
# YEAR RANGE SELECTION
# =========================

st.markdown("## 3️⃣ Data Selection")

years = sorted(df["YEAR"].dropna().unique())

col1, col2 = st.columns(2)

year_from = col1.selectbox(
    "From Year",
    years,
    index=0
)

year_to = col2.selectbox(
    "To Year",
    years,
    index=len(years) - 1
)

if year_from > year_to:
    st.error("From Year cannot be greater than To Year.")
    render_footer()
    st.stop()

selected_df = df[
    (df["YEAR"] >= year_from) &
    (df["YEAR"] <= year_to)
].copy()

st.info(f"Selected years: {year_from} to {year_to}")

# =========================
# PARAMETER AVAILABILITY FOR SELECTED YEARS
# =========================

st.markdown("## 4️⃣ Parameter Availability for Selected Years")

selected_summary = []

total_rows = len(selected_df)

for param in parameter_cols:

    available_count = selected_df[param].notna().sum()

    availability_pct = (
        (available_count / total_rows) * 100
        if total_rows > 0 else 0
    )

    selected_summary.append({
        "Parameter": param,
        "Availability":
            f"{available_count}/{total_rows} "
            f"({availability_pct:.1f}%)"
    })

selected_availability_df = pd.DataFrame(selected_summary)


def blue_cells(val):
    return "background-color: ; color: ; font-weight: 600;"


styled_selected_df = (
    selected_availability_df.style
    .map(blue_cells, subset=["Availability"])
)

st.dataframe(
    styled_selected_df,
    use_container_width=True
)

# =========================
# CONFIRM SELECTED DATA
# =========================

st.markdown("## 5️⃣ Confirm Selected Data")

if st.button("✅ Confirm This Data"):
    st.session_state["df"] = selected_df
    st.session_state["confirmed_data"] = selected_df
    st.success("✅ Selected data confirmed successfully.")

# =========================
# DOWNLOAD SELECTED DATA
# =========================

st.markdown("## 6️⃣ Download Selected Data")

csv_data = selected_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Selected Data as CSV",
    data=csv_data,
    file_name=f"selected_air_quality_data_{year_from}_{year_to}.csv",
    mime="text/csv"
)

# =========================
# SELECTED DATA PREVIEW
# =========================

st.subheader("Selected Dataset Preview")
st.dataframe(selected_df.head(5), use_container_width=True)

# =========================
# DATA PARAMETERS
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
    - **TEMP** → Temperature  
    - **RH** → Relative Humidity  
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

render_footer()