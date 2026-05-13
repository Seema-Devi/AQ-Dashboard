import streamlit as st
import base64

def load_full_ui():

    # =========================================================
    # GLOBAL CSS FIXES - FORCED FULL-PAGE MAX SIZE
    # =========================================================
    st.markdown("""
    <style>

/* ============================
   APP BACKGROUND
   ============================ */
.stApp {
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
}

/* ============================
   MAIN CONTENT AREA
   ============================ */
[data-testid="stAppViewContainer"] .main {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    max-width: 100% !important;
    padding-bottom: 70px !important;
    overflow-x: hidden !important;

}

/* ============================
   SIDEBAR
   ============================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #dbeafe 0%, #eff6ff 45%, #ffffff 100%) !important;
    border-right: 1px solid #bfdbfe;
    padding: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 12px !important;
}

/* ============================
   SIDEBAR LOGO
   ============================ */
[data-testid="stSidebarNav"]::before {
    content: "";
    display: block;
    margin: 10px auto 20px auto;
    height: 95px;
    width: 240px;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}

/* ============================
   IMAGE HANDLING
   ============================ */
div[data-testid="stImage"] img, .full-width img {
    width: 100% !important;
    height: auto !important;
    display: block !important;
}

.full-width {
     margin-left: 0rem !important;
    margin-right: 0rem !important;
    padding: 0rem !important;
    width: 100% !important;
    overflow-x: hidden !important;
}

/* ============================
   CARD SPACING
   ============================ */
div[style*="background-color:#ffffff"],
div[style*="background-color:#f8fafc"],
div[style*="background-color:#f0fdf4"] {
    padding: 26px !important;
    margin-bottom: 16px !important;
}

/* ============================
   SIDEBAR TEXT
   ============================ */
.sidebar-title {
    font-size: 18px;
    font-weight: 800;
    color: #003d7a;
    margin-bottom: 10px;
}

.upload-success {
    color: #047857 !important;
    font-size: 13px;
    font-weight: 600;
    margin-top: 4px;
}

.upload-fail {
    color: #b91c1c !important;
    font-size: 13px;
    font-weight: 600;
    margin-top: 4px;
}

/* ============================
   CUSTOM FOOTER
   ============================ */
.custom-footer {
    position: fixed;
    bottom: 0;
    left: 300px;
    width: calc(100% - 300px);
    background: linear-gradient(90deg, #dbeafe, #e0f2fe);
    color: #003d7a;
    text-align: center;
    padding: 6px 0;
    font-size: 13px;
    border-top: 1px solid #bfdbfe;
    z-index: 9999;
    font-weight: 600;
}

/* Responsive footer */
@media (max-width: 768px) {
    .custom-footer {
        left: 0;
        width: 100%;
    }
}

/* ============================
   HIDE STREAMLIT DEFAULTS
   ============================ */
#MainMenu, footer, header, [data-testid="stStatusWidget"] {
    visibility: hidden;
}
/* REMOVE ALL TOP SPACING ABOVE MAIN CONTENT */
.stApp {
    overflow-x: hidden !important;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stAppViewContainer"] .main {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stAppViewContainer"] .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* REMOVE TOP SPACING ABOVE SIDEBAR + LOGO */
section[data-testid="stSidebar"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stSidebarNav"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stSidebarNav"]::before {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
/* ============================================
   FORCE FULL-PAGE WIDTH (NO CENTER COLUMN)
   ============================================ */

/* Remove Streamlit's default centered container */
.block-container {
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 20px !important;
    margin-left: 20px !important;
    margin-right: 0 !important;
}

/* Remove padding from the main app container */
[data-testid="stAppViewContainer"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* Remove padding from the main content area */
[data-testid="stAppViewContainer"] .main {
    padding: 0 !important;
    margin: 0 !important;
}

/* Remove padding from the first wrapper Streamlit adds */
[data-testid="stAppViewContainer"] > div:first-child {
    padding: 0 !important;
    margin: 0 !important;
}

/* ============================================
   FULL-WIDTH IMAGES + HEADER
   ============================================ */

/* Skyline header full width + full height */
.full-width img {
    width: 100vw !important;
    height: 45vh !important;   /* Increase header height */
    object-fit: cover !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Info cards full width */
[data-testid="stImage"] img {
    width: 100% !important;
    height: 32vh !important;   /* Increase card image height */
    object-fit: cover !important;
}

/* ============================================
   REMOVE TOP MARGIN COMPLETELY
   ============================================ */
.stApp {
    margin-top: -60px !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

[data-testid="stSidebarNav"]::before {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ============================================
   OPTIONAL: Make columns stretch evenly
   ============================================ */
.css-1kyxreq, .css-1r6slb0, .css-1l269bu {
    width: 100% !important;
}

</style>

    """, unsafe_allow_html=True)
    
    # =========================================================
    # SIDEBAR LOGO BEFORE PAGES
    # =========================================================
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    try:
        logo_base64 = get_base64_image("assets/auckland_council_logo.png")
        st.markdown(f"""
        <style>
        [data-testid="stSidebarNav"]::before {{
            content: "";
            display: block;
            margin: 10px auto 20px auto;
            height: 95px;
            width: 240px;
            background-image: url("data:image/png;base64,{logo_base64}");
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # =========================================================
    # FULL-WIDTH SKYLINE HEADER
    # =========================================================
    st.markdown('<div class="full-width">', unsafe_allow_html=True)
    st.image("assets/auckland_skyline.jpg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # FULL-WIDTH INFO CARDS
    # =========================================================
    st.markdown('<div class="full-width" style="padding:0px 10px;">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("assets/pm25.jpg", use_container_width=True)
    with col2:
        st.image("assets/no2.jpg", use_container_width=True)
    with col3:
        st.image("assets/ml.jpg", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # SIDEBAR CONTENT (UNCHANGED)
    # =========================================================
    st.sidebar.markdown('<div class="sidebar-title">🤖 ML Model Settings</div>', unsafe_allow_html=True)

    st.session_state["selected_target"] = st.sidebar.selectbox(
        "🎯 Select Target",
        ["AQI", "PM2.5", "NO2"],
        key="sidebar_target"
    )

    st.session_state["selected_model"] = st.sidebar.selectbox(
        "⚙️ Select Model",
        ["Random Forest", "XGBoost", "LSTM", "Compare All"],
        key="sidebar_model"
    )

    st.session_state["run_prediction"] = st.sidebar.button(
        "🚀 Run 24h Forecast",
        key="sidebar_run"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-title">📂 Upload Datasets</div>', unsafe_allow_html=True)

    uploaded1 = st.sidebar.file_uploader("Dataset 1", type=["csv", "xls", "xlsx"], key="combined_file1")
    if uploaded1 is not None:
        st.session_state["file1"] = uploaded1

    st.sidebar.markdown(
        "<div class='upload-success'>✔ Dataset 1 uploaded</div>"
        if "file1" in st.session_state else
        "<div class='upload-fail'>✖ Dataset 1 missing</div>",
        unsafe_allow_html=True
    )

    uploaded2 = st.sidebar.file_uploader("Dataset 2", type=["csv", "xls", "xlsx"], key="combined_file2")
    if uploaded2 is not None:
        st.session_state["file2"] = uploaded2

    st.sidebar.markdown(
        "<div class='upload-success'>✔ Dataset 2 uploaded</div>"
        if "file2" in st.session_state else
        "<div class='upload-fail'>✖ Dataset 2 missing</div>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Auckland Council AQI Forecast Dashboard • GDDA713 • © 2026")


def render_footer():
    st.markdown("""
    <div class="custom-footer">
        🌿 Auckland Council AQI Forecast Dashboard • GDDA713 • © 2026
    </div>
    """, unsafe_allow_html=True)
