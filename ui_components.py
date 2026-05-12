import streamlit as st

def load_full_ui():

    st.markdown("""
    <style>

    /* ============================
   MAIN CONTENT AREA
   ============================ */
.main .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
    padding-bottom: 140px !important; /* space for footer */
    min-height: calc(100vh - 140px) !important;
}

/* ============================
   SIDEBAR – REMOVE TOP/BOTTOM MARGINS
   ============================ */
section[data-testid="stSidebar"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
    background: linear-gradient(180deg, #eef3f7 0%, #e1e8ef 100%);
    border-right: 1px solid #c9d3dd;
}

/* Remove internal sidebar padding */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ============================
   SIDEBAR CARDS
   ============================ */
.sidebar-card {
    background: #ffffff;
    border: 1px solid #d7e0ea;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 14px;
    box-shadow: 0px 1px 3px rgba(0,0,0,0.06);
                
}


.metric-title {
    font-size: 15px;
    font-weight: 600;
    color: #003d7a;
}

.small-text {
    font-size: 13px;
    color: #475569;
}

.upload-success {
    color: #0c7c59 !important;
    font-size: 13px;
    margin-top: 4px;
}

.upload-fail {
    color: #b91c1c !important;
    font-size: 13px;
    margin-top: 4px;
}

/* ============================
   HEADER – REMOVE TOP GAP
   ============================ */
.dashboard-header {
    background: linear-gradient(90deg, #003d7a, #0059a8);
    padding: 26px;
    border-radius: 14px;
    margin-bottom: 24px;
    margin-top: -100px !important; /* FIXED */
    box-shadow: 0px 4px 14px rgba(0,0,0,0.15);
}

.dashboard-header h1 {
    color: #ffffff;
    margin: 0;
    font-size: 32px;
    font-weight: 700;
}

.dashboard-header p {
    color: #dbeafe;
    margin-top: 6px;
    font-size: 15px;
}

/* ============================
   FILE UPLOADER
   ============================ */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #d7e0ea;
    border-radius: 10px;
    padding: 6px;
    width: 90%;
    font-size: 12px;
}

/* Reduce upload button size */
[data-testid="stFileUploader"] section {
    padding: 4px !important;
}

/* Smaller drag-drop area */
[data-testid="stFileUploaderDropzone"] {
    padding: 0.3rem !important;
}

/* Smaller uploaded filename text */
[data-testid="stFileUploaderFileName"] {
    font-size: 12px !important;
}

/* ============================
   HIDE STREAMLIT DEFAULTS
   ============================ */
#MainMenu, footer, header, [data-testid="stStatusWidget"] {
    visibility: hidden;
}


    </style>
    """, unsafe_allow_html=True)
    

    # HEADER
    st.markdown("""
    <div class="dashboard-header">
        <h1>Auckland Council AQI Dashboard</h1>
        <p>Air Quality Data Analytics | EDA | Feature Engineering | ML Models</p>
    </div>
    """, unsafe_allow_html=True)

    
      # ======================================================
    # 1. MODEL SETTINGS (NOW AT THE TOP)
    # ======================================================
    st.sidebar.markdown("""
    <div class="metric-title" style="font-size:17px; margin-bottom:8px;">
    🤖 ML Model Settings
    </div>
    """, unsafe_allow_html=True)

    # Selections stored in session_state
    st.session_state["selected_target"] = st.sidebar.selectbox("🎯 Select Target", ["AQI", "PM2.5", "NO2"], key="sidebar_target")
    st.session_state["selected_model"] = st.sidebar.selectbox("⚙️ Select Model", ["Random Forest", "XGBoost", "LSTM", "Compare All"], key="sidebar_model")
    st.session_state["run_prediction"] = st.sidebar.button("🚀 Run 24h Forecast", key="sidebar_run")

    st.sidebar.markdown("---")

    # ======================================================
    # 2. UPLOAD DATASETS (NOW BELOW SETTINGS)
    # ======================================================
    st.sidebar.markdown("""
    <div class="metric-title" style="font-size:17px; margin-bottom:8px;">
    📂 Upload Datasets
    </div>
    """, unsafe_allow_html=True)

    # FILE UPLOADERS
    uploaded1 = st.sidebar.file_uploader("Dataset 1", type=["csv", "xls", "xlsx"], key="combined_file1")
    if uploaded1 is not None:
        st.session_state["file1"] = uploaded1

    if "file1" in st.session_state:
        st.sidebar.markdown("<div class='upload-success'>✔ Dataset 1 uploaded successfully</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div class='upload-fail'>✖ Dataset 1 not uploaded</div>", unsafe_allow_html=True)

    uploaded2 = st.sidebar.file_uploader("Dataset 2", type=["csv", "xls", "xlsx"], key="combined_file2")
    if uploaded2 is not None:
        st.session_state["file2"] = uploaded2

    if "file2" in st.session_state:
        st.sidebar.markdown("<div class='upload-success'>✔ Dataset 2 uploaded successfully</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div class='upload-fail'>✖ Dataset 2 not uploaded</div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.caption("Auckland Council AQI Forecast Dashboard | GDDA713 | Team: Seema Devi • Deshika • Sarah")


# =========================
# PERFECTLY ALIGNED FOOTER (NOT BEHIND SIDEBAR)
# =========================
def render_footer():
    st.markdown("""
    <style>
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 300px; /* sidebar width */
        width: calc(100% - 300px);
        background: #e7f0fa;
        color: #003d7a;
        text-align: center;
        padding: 2px 0;
        font-size: 13px;
        border-top: 1px solid #c3d4e3;
        z-index: 9999;
    }
    </style>

    <div class="custom-footer">
        Auckland Council AQI Forecast Dashboard • GDDA713 • © 2026
    </div>
    """, unsafe_allow_html=True)