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

<p style="font-size:15px; color:#334155; line-height:1.6;">
This dashboard is part of an applied research project conducted in collaboration with 
<b>Auckland Council</b>. The project focuses on understanding pollution behaviour in 
Auckland’s city centre and developing AI-powered forecasting models.
</p>

</div>
""", unsafe_allow_html=True)

    col_team, col_supervisor = st.columns(2)

    with col_team:
        st.markdown("""
<div style="
    background-color:#f8fafc;
    padding:16px;
    border-radius:12px;
    border:1px solid #e2e8f0;
    margin-top:12px;
">
<h4 style="color:#1e3a8a;">👩‍💻 Team Members</h4>
<p style="font-size:14px; color:#475569; line-height:1.8;">
• Seema Devi<br>
• Deshika<br>
• Sarah
</p>
</div>
""", unsafe_allow_html=True)

    with col_supervisor:
        st.markdown("""
<div style="
    background-color:#f8fafc;
    padding:16px;
    border-radius:12px;
    border:1px solid #e2e8f0;
    margin-top:12px;
">
<h4 style="color:#1e3a8a;">🎓 Supervisors</h4>
<p style="font-size:14px; color:#475569; line-height:1.8;">
• Dr Louis Boamponsem (Auckland Council)<br>
• Dr Sara Zandi (NZSE)
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="
    background-color:#f0fdf4;
    border-radius:10px;
    padding:12px;
    margin-top:14px;
    border-left:4px solid #22c55e;
">
<p style="font-size:14px; color:#334155; line-height:1.6;">
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
            box-shadow:0px 2px 8px rgba(0,0,0,0.08); border:1px solid #e2e8f0;">

<h3 style="color:#1e3a8a;">🧭 App Workflow</h3>

<div style="font-size:15px; color:#334155; line-height:1.9;">

<div style="display:flex; align-items:center; margin-bottom:14px;">
    <div style="width:38px; height:38px; border-radius:50%; background:#22c55e;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">1</div>
    <div>
        <b>Upload Datasets</b><br>
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
        <b>Feature Engineering</b><br>
        <span style="font-size:13px; color:#64748b;">Create lag features & rolling windows</span>
    </div>
</div>

<div style="display:flex; align-items:center;">
    <div style="width:38px; height:38px; border-radius:50%; background:#dc2626;
                color:white; display:flex; align-items:center; justify-content:center;
                font-weight:700; margin-right:12px;">6</div>
    <div>
        <b>Train Models & Forecast</b><br>
        <span style="font-size:13px; color:#64748b;">Generate 24‑hour AQI predictions</span>
    </div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

# =========================
# DATA MERGING AND FILTERING LOGIC
# =========================
if "merged_df" in st.session_state:
    df = st.session_state["merged_df"]
else:
    file1 = st.session_state.get("file1")
    file2 = st.session_state.get("file2")

    if not file1 or not file2:
        st.warning("⬅ Please upload both datasets from the sidebar.")
        render_footer()

        st.stop()

    @st.cache_data(show_spinner=False)
    def read_any(file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file, low_memory=False)
        return pd.read_excel(file)

    @st.cache_data(show_spinner=True)
    def process_and_merge(file1, file2):
        raw1 = read_any(file1)
        raw2 = read_any(file2)

        def process(df):
            df = df.copy()
            df.columns = df.columns.str.strip()
            df['Time'] = df['Time'].astype(str).str.lower().str.strip()
            df['Time'] = df['Time'].str.replace(r'(am|pm)\s+\1', r'\1', regex=True)
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + " " + df['Time'], errors='coerce', dayfirst=True)
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df['datetime_hour'] = df['datetime'].dt.floor('h')

            df_hourly = df.groupby(['Site', 'datetime_hour', 'Parameter'])['Value'].mean().reset_index()
            df_wide = df_hourly.pivot_table(index=['Site', 'datetime_hour'], columns='Parameter', values='Value').reset_index()
            df_wide.columns.name = None
            df_wide = df_wide.rename(columns=str.upper)
            return df_wide

        df1 = process(raw1)
        df2 = process(raw2)

        merged = df1.merge(df2, on=['SITE', 'DATETIME_HOUR'], how='outer', suffixes=('_1', '_2'))
        merged = merged.loc[:, ~merged.columns.duplicated()]

        cols = ['AQI','NO','NO2','NOX','PM10','PM2.5','TEMP','RH','WS','WD']
        for col in cols:
            c1, c2 = f"{col}_1", f"{col}_2"
            if c1 in merged.columns and c2 in merged.columns:
                merged[col] = merged[c1].combine_first(merged[c2])

        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_1") or c.endswith("_2")], errors='ignore')
        merged = merged.sort_values(['SITE', 'DATETIME_HOUR'])
        merged['DATETIME_HOUR'] = pd.to_datetime(merged['DATETIME_HOUR'])
        return merged

    with st.spinner("Processing and merging datasets..."):
        df = process_and_merge(file1, file2)
        st.session_state["merged_df"] = df

# =========================
# FILTER SECTION
# =========================
st.markdown("""
<div style="
background:#ffffff;
padding:20px;
border-radius:14px;
box-shadow:0px 2px 8px rgba(0,0,0,0.08);
margin-top:20px;
">
<h3 style="color:#1e3a8a;">📅 Filter Data</h3>
</div>
""", unsafe_allow_html=True)

df['YEAR'] = df['DATETIME_HOUR'].dt.year
df['MONTH'] = df['DATETIME_HOUR'].dt.month

years = sorted(df['YEAR'].dropna().unique())

col1, col2 = st.columns(2)
year_from = col1.selectbox("From Year", years)
year_to = col2.selectbox("To Year", years, index=len(years)-1)

months = {
    1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
    7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"
}

selected_months = st.multiselect("Select Months (optional)", options=list(months.keys()), format_func=lambda x: months[x])

filtered_df = df[(df['YEAR'] >= year_from) & (df['YEAR'] <= year_to)]
if selected_months:
    filtered_df = filtered_df[filtered_df['MONTH'].isin(selected_months)]

st.session_state["df"] = filtered_df

st.success("Datasets merged and filtered successfully.")

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered_df.head(200), use_container_width=True)

render_footer()
