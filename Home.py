import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import textwrap
from matplotlib.colors import LinearSegmentedColormap
from ui_components import load_full_ui, render_footer

st.write("")
load_full_ui()


# ======================================================
# SAME COLOUR PALETTE AS DATA VISUALISATION PAGE
# ======================================================
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

.kpi-blue {background: linear-gradient(135deg,#f4f8ff,#eef5ff); border-color:#74a7ff; color:#0645d8;}
.kpi-green {background: linear-gradient(135deg,#f3fff7,#eefcf3); border-color:#79d49a; color:#078037;}
.kpi-orange {background: linear-gradient(135deg,#fff9ef,#fff4e6); border-color:#ffa442; color:#ef6400;}
.kpi-purple {background: linear-gradient(135deg,#fbf7ff,#f7efff); border-color:#b076ff; color:#6618c8;}
.kpi-red {background: linear-gradient(135deg,#fff7f7,#fff0f1); border-color:#ff7a83; color:#d51b2c;}
</style>
""", unsafe_allow_html=True)


uploaded_files = st.session_state.get("uploaded_files", [])

# Open Data Processing first after files are uploaded.
# Streamlit tabs always open the first tab by default, so the tab order is changed only after upload.
if uploaded_files:
    data_tab, about_tab = st.tabs(["📊 Data Processing", "📘 About Project"])
else:
    about_tab, data_tab = st.tabs(["📘 About Project", "📊 Data Processing"])


# =========================
# ABOUT PROJECT + WORKFLOW
# =========================
with about_tab:
    col_left, col_right = st.columns([1.15, 0.85])

    with col_left:
        st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:22px;
        border-radius:14px;
        box-shadow:0px 2px 8px rgba(0,0,0,0.08);
        border:1px solid #cbdcff;
    ">

    <h3 style="color:#0433d9;">📘 About This Project</h3>

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
    <div style="background-color:#fbfdff; padding:16px; border-radius:12px;
    border:1px solid #cbdcff; margin-top:12px;">
    <h4 style="color:#0433d9;">👩‍💻 Team Members</h4>
    <p style="font-size:14px; color:#475569; line-height:1.5;">
    • Seema Devi<br>
    • Deshika Jayatilaka<br>
    • Yaqing Zhang (Sarah)
    </p>
    </div>
    """, unsafe_allow_html=True)

        with col_supervisor:
            st.markdown("""
    <div style="background-color:#fbfdff; padding:16px; border-radius:12px;
    border:1px solid #cbdcff; margin-top:12px;">
    <h4 style="color:#0433d9;">🎓 Supervisors</h4>
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
                box-shadow:0px 2px 8px rgba(0,0,0,0.08); border:1px solid #cbdcff; margin-bottom:50px;
    ">

    <h3 style="color:#0433d9;">🧭 App Workflow</h3>

    <div style="font-size:18px; color:#334155; line-height:1.9;">

    <div style="display:flex; align-items:center; margin-bottom:20px;">
        <div style="width:38px; height:38px; border-radius:50%; background:#79d49a;
                    color:white; display:flex; align-items:center; justify-content:center;
                    font-weight:700; margin-right:12px;">1</div>
        <div>
            <b>🏠 Home</b><br>
            <span style="font-size:14px; color:#64748b;">Project overview, upload datasets, and data processing</span>
        </div>
    </div>

    <div style="display:flex; align-items:center; margin-bottom:20px;">
        <div style="width:38px; height:38px; border-radius:50%; background:#74a7ff;
                    color:white; display:flex; align-items:center; justify-content:center;
                    font-weight:700; margin-right:12px;">2</div>
        <div>
            <b>📊 Data Insights & Cleaning</b><br>
            <span style="font-size:14px; color:#64748b;">Data insights, analyse missing values, clean and preprocess data</span>
        </div>
    </div>

    <div style="display:flex; align-items:center; margin-bottom:20px;">
        <div style="width:38px; height:38px; border-radius:50%; background:#ffa442;
                    color:white; display:flex; align-items:center; justify-content:center;
                    font-weight:700; margin-right:12px;">3</div>
        <div>
            <b>📈 Exploratory Analysis & PCA</b><br>
            <span style="font-size:13px; color:#64748b;"> Analyse trends, correlations, feature engineering, and PCA optimisation</span>
        </div>
    </div>

    <div style="display:flex; align-items:center; margin-bottom:20px;">
        <div style="width:38px; height:38px; border-radius:50%; background:#b076ff;
                    color:white; display:flex; align-items:center; justify-content:center;
                    font-weight:700; margin-right:12px;">4</div>
        <div>
            <b>🤖 Train Models & Forecast</b><br>
            <span style="font-size:13px; color:#64748b;">Train RF, XGBoost, and GRU to generate AQI, NO2 & PM2.5 predictions </span>
        </div>
    </div>

    

    </div>
    </div>
    """, unsafe_allow_html=True)


    # =========================
    # DATA PARAMETERS
    # =========================
    st.markdown("""
    <div style="
    background-color:#ffffff;
    padding:1px;
    border-radius:14px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    border:1px solid #cbdcff;
    margin-top:10px;
    margin-bottom:10px;
    ">
    <h3 style="color:#0433d9;">📘 Data Parameters</h3>
    <p style="color:gray; font-size:12px;">
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

       
# =========================
# FUNCTIONS
# =========================
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
    raw["Parameter"] = raw["Parameter"].astype(str).str.strip().str.upper()
    raw["Site"] = raw["Site"].astype(str).str.strip().str.upper()
    
    

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
            left_series = merged[original_col]
            right_series = merged[dup_col]

            if isinstance(left_series, pd.DataFrame):
                left_series = left_series.iloc[:, 0]
            if isinstance(right_series, pd.DataFrame):
                right_series = right_series.iloc[:, 0]

            merged[original_col] = left_series.combine_first(right_series)
            merged = merged.drop(columns=[dup_col], errors="ignore")

        merged = merged.loc[:, ~merged.columns.duplicated()]

    merged["DATETIME_HOUR"] = pd.to_datetime(merged["DATETIME_HOUR"], errors="coerce")
    merged = merged.dropna(subset=["DATETIME_HOUR"])
    merged = merged.sort_values(["SITE", "DATETIME_HOUR"])

    return merged


def _format_num(value):
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.1f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)



def build_availability_heatmap_figure(input_df, parameter_cols):
    """Create a clear visual heatmap for yearly availability by parameter."""
    plot_df = input_df.copy()
    plot_df["DATETIME_HOUR"] = pd.to_datetime(plot_df["DATETIME_HOUR"], errors="coerce")
    plot_df = plot_df.dropna(subset=["DATETIME_HOUR"])
    plot_df["YEAR"] = plot_df["DATETIME_HOUR"].dt.year

    years = sorted(plot_df["YEAR"].dropna().unique())
    if len(years) == 0 or len(parameter_cols) == 0:
        return None

    heatmap_data = []
    for param in parameter_cols:
        row = []
        for year in years:
            year_df = plot_df[plot_df["YEAR"] == year]
            pct = year_df[param].notna().mean() * 100 if len(year_df) else 0
            row.append(pct)
        heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data, dtype=float)

    # Wider figure and higher DPI makes the cell numbers sharper in Streamlit.
    fig_width = max(11, len(years) * 1.05)
    fig_height = max(4.5, len(parameter_cols) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    soft_cmap = LinearSegmentedColormap.from_list(
        "soft_availability",
        ["#ffffff", "#f0f9ff", "#e0f2fe", "#bae6fd", "#7dd3fc"]
    )
    im = ax.imshow(heatmap_array, aspect="auto", vmin=0, vmax=100, cmap=soft_cmap, interpolation="nearest")

    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels([str(int(y)) for y in years], rotation=0, ha="center", fontsize=9)
    ax.set_yticks(np.arange(len(parameter_cols)))
    ax.set_yticklabels(parameter_cols, fontsize=8)
    ax.set_title("Data Availability by Year / Parameter", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Parameter")

    # Thin grid lines separate cells and prevent labels looking merged.
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(parameter_cols), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    label_font = 8 if len(years) <= 10 else 6.5
    # If there are many columns, show rounded values to avoid overlap.
    for i in range(len(parameter_cols)):
        for j in range(len(years)):
            value = heatmap_array[i, j]
            label = f"{value:.0f}%" if len(years) > 8 else f"{value:.1f}%"
            ax.text(
                j, i, label,
                ha="center", va="center",
                fontsize=label_font,
                fontweight="semibold",
                color="#0f172a"
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Availability (%)")
    fig.tight_layout()
    return fig

def _wrap_parameter_name(name, width=18):
    """Wrap long parameter names so labels stay outside the plot without overlapping."""
    return "\n".join(textwrap.wrap(str(name), width=width, break_long_words=False))


def build_parameter_timeline_figure(input_df, parameter_cols, selected_parameters=None, title="Parameter data availability timeline"):
    """Create a faceted timeline plot like the reference screenshot, but rows are parameters."""
    if selected_parameters is None:
        selected_parameters = parameter_cols

    plot_df = input_df.copy()
    plot_df["DATETIME_HOUR"] = pd.to_datetime(plot_df["DATETIME_HOUR"], errors="coerce")
    plot_df = plot_df.dropna(subset=["DATETIME_HOUR"])

    start_date = plot_df["DATETIME_HOUR"].min().floor("D")
    end_date = plot_df["DATETIME_HOUR"].max().ceil("D")

    if pd.isna(start_date) or pd.isna(end_date):
        return None

    full_index = pd.date_range(start=start_date, end=end_date, freq="h")
    years = list(range(start_date.year, end_date.year + 1))

    n_params = len(selected_parameters)
    fig_height = max(2.4, n_params * 1.28)
    fig, axes = plt.subplots(
        n_params,
        1,
        figsize=(12.5, fig_height),
        dpi=145,
        sharex=True,
        gridspec_kw={"hspace": 0.05}
    )

    if n_params == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    line_color = "#d8c515"       
    background_color = "#d9f3f7" 
    percent_color = "#2e8b57"    
    missing_color = "#b91c1c"    

    for ax, param in zip(axes, selected_parameters):
        if param not in plot_df.columns:
            continue

        hourly = (
            plot_df[["DATETIME_HOUR", param]]
            .groupby("DATETIME_HOUR")[param]
            .mean()
            .reindex(full_index)
        )

        valid = hourly.dropna()
        total_count = len(hourly)
        missing_count = int(hourly.isna().sum())
        missing_pct = (missing_count / total_count * 100) if total_count else 0

        ax.set_facecolor(background_color)
        ax.plot(hourly.index, hourly.values, color=line_color, linewidth=0.8)
        ax.grid(True, axis="x", color="#d1d5db", linewidth=0.6, alpha=0.8)
        ax.grid(False, axis="y")
        ax.set_xlim(start_date, end_date)

        missing_times = hourly.index[hourly.isna()]
        if len(valid) > 0:
            y_min = valid.min()
            y_max = valid.max()
            y_range = y_max - y_min if y_max != y_min else 1
            bottom = y_min - (0.08 * y_range)
            top = y_min + (0.12 * y_range)
        else:
            y_min, y_max, bottom, top = 0, 1, 0, 0.12

        if len(missing_times) > 0:
            step = max(1, len(missing_times) // 250)
            ax.vlines(missing_times[::step], ymin=bottom, ymax=top, color=missing_color, linewidth=0.45, alpha=0.75)

        # Keep parameter names outside the plotting area and wrap long names.
        ax.set_ylabel("")
        ax.text(
            -0.035, 0.5, _wrap_parameter_name(param, width=18),
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="center",
            color="#111827",
            clip_on=False
        )
        ax.tick_params(axis="y", labelleft=False, length=0)
        ax.tick_params(axis="x", labelsize=9)

        left_text = (
            f"missing = {missing_count} ({missing_pct:.1f}%)\n"
            f"min = {_format_num(valid.min() if len(valid) else np.nan)}\n"
            f"max = {_format_num(valid.max() if len(valid) else np.nan)}"
        )
        ax.text(0.012, 0.72, left_text, transform=ax.transAxes, fontsize=7.5, va="top", ha="left", color="black")

        right_text = (
            f"mean = {_format_num(valid.mean() if len(valid) else np.nan)}\n"
            f"median = {_format_num(valid.median() if len(valid) else np.nan)}\n"
            f"95th percentile = {_format_num(valid.quantile(0.95) if len(valid) else np.nan)}"
        )
        ax.text(0.985, 0.72, right_text, transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color="black")

        for year in years:
            year_start = pd.Timestamp(year=year, month=1, day=1)
            year_end = pd.Timestamp(year=year + 1, month=1, day=1)
            visible_start = max(year_start, start_date)
            visible_end = min(year_end, end_date)
            year_series = hourly[(hourly.index >= visible_start) & (hourly.index < visible_end)]
            if len(year_series) == 0:
                continue
            availability_pct = year_series.notna().mean() * 100
            mid_date = visible_start + (visible_end - visible_start) / 2
            label = f"{availability_pct:.1f}%".replace(".0%", "%")
            ax.text(
                mid_date, 0.93, label,
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color=percent_color,
                ha="center",
                va="top",
                fontweight="bold",
                clip_on=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.45)
            )

        for spine in ax.spines.values():
            spine.set_color("#4b5563")
            spine.set_linewidth(0.8)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    # Small outside label column + wide landscape plot area.
    fig.subplots_adjust(left=0.13, right=0.985, top=0.94, bottom=0.08)
    return fig


# =========================
# DATA PROCESSING + MERGING
# =========================
with data_tab:
    st.markdown("## 1️⃣ Processing of the Data")

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
        st.warning(
                """
             No dataset uploaded.

            ⬅ Please upload the required datasets from the sidebar to continue the application workflow.
            """
            )
        render_footer()
        st.stop()

    with st.spinner("Processing uploaded datasets..."):
        df = process_and_merge_multiple(uploaded_files)
        st.session_state["merged_df"] = df

    st.success("✅ Data processed successfully.")

    # =========================
    # DATA AVAILABILITY BY YEAR / PARAMETER
    # =========================
    st.markdown("## 2️⃣ Uploaded Data Availability")

    df["YEAR"] = df["DATETIME_HOUR"].dt.year
    exclude_cols = ["SITE", "DATETIME_HOUR", "YEAR", "MONTH"]
    parameter_cols = [col for col in df.columns if col not in exclude_cols]

    default_params = parameter_cols

    fig_heatmap = build_availability_heatmap_figure(df, parameter_cols)
    if fig_heatmap:
        st.pyplot(fig_heatmap, use_container_width=True)
        st.caption("Visual heatmap: light blue shows higher data availability. Each cell shows availability percentage for that parameter and year.")
    else:
        st.info("No parameter availability data found for heatmap.")

    # =========================
    # YEAR RANGE SELECTION
    # =========================
    st.markdown("## 3️⃣ Select Data for Analysis")

    years = sorted(df["YEAR"].dropna().unique())

    col1, col2 = st.columns(2)

    year_from = col1.selectbox("From Year", years, index=0)
    year_to = col2.selectbox("To Year", years, index=len(years) - 1)

    if year_from > year_to:
        st.error("From Year cannot be greater than To Year.")
        render_footer()
        st.stop()

    selected_df = df[(df["YEAR"] >= year_from) & (df["YEAR"] <= year_to)].copy()
    st.info(f"Selected years: {year_from} to {year_to}")

    # =========================
    # PARAMETER AVAILABILITY FOR SELECTED YEARS
    # =========================
    st.markdown("## 4️⃣ Data Quality & Statistics for Selected Years")

    selected_visual_params = st.multiselect(
        "Choose parameters for selected-year timeline",
        options=parameter_cols,
        default=default_params,
        key="selected_year_availability_params"
    )

    if selected_visual_params:
        fig_selected = build_parameter_timeline_figure(
            selected_df,
            parameter_cols,
            selected_parameters=selected_visual_params,
            title=f"Parameter Statistics for ({year_from}–{year_to})"
        )
        if fig_selected:
            st.pyplot(fig_selected, use_container_width=True)
        st.caption("This plot displays data quality timeline & statistics for selected years.")
    else:
        st.info("Please select at least one parameter to display the selected-year visual timeline.")

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
        label="⬇️ Download Data as CSV",
        data=csv_data,
        file_name=f"selected_air_quality_data_{year_from}_{year_to}.csv",
        mime="text/csv"
    )

    # =========================
    # SELECTED DATA PREVIEW
    # =========================
    st.subheader("Selected Dataset Preview")
    st.dataframe(selected_df.head(5), use_container_width=True)

render_footer()
