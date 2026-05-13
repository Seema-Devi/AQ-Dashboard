import streamlit as st
import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def load_full_ui():

    # Sidebar logo
    try:
        logo_base64 = get_base64_image("assets/auckland_council_logo.png")
    except FileNotFoundError:
        logo_base64 = ""

    st.markdown(f"""
<style>

/* ============================
   APP BACKGROUND
   ============================ */
.stApp {{
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
    overflow-x: hidden !important;
}}

/* ============================
   HIDE STREAMLIT DEFAULT HEADER
   ============================ */
header {{
    visibility: hidden;
    height: 0px !important;
}}

#MainMenu, footer, [data-testid="stStatusWidget"] {{
    visibility: hidden;
}}

[data-testid="stAppViewContainer"] {{
    padding: 0 !important;
    margin: 0 !important;
    overflow-x: hidden !important;
}}

/* ============================
   MAIN CONTENT AREA
   ============================ */
[data-testid="stAppViewContainer"] .main {{
    padding: 0 !important;
    margin: 0 !important;
    overflow-x: hidden !important;
}}

.block-container {{
    max-width: 100% !important;
    padding-top: 0 !important;
    padding-left: 16px !important;
    padding-right: 16px !important;
    padding-bottom: 90px !important;
    margin: 0 !important;
    overflow-x: hidden !important;
}}

/* ============================
   SIDEBAR
   ============================ */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #dbeafe 0%, #eff6ff 45%, #ffffff 100%) !important;
    border-right: 1px solid #bfdbfe;
    padding: 0 !important;
    margin: 0 !important;
}}

section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: 0 !important;
}}

/* ============================
   SIDEBAR LOGO BEFORE PAGES
   ============================ */
[data-testid="stSidebarNav"] {{
    padding-top: 0 !important;
    margin-top: 0 !important;
}}

[data-testid="stSidebarNav"]::before {{
    content: "";
    display: block;
    margin: 0 auto 14px auto;
    height: 95px;
    width: 235px;
    background-image: url("data:image/png;base64,{logo_base64}");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}}

/* ============================
   IMAGES
   ============================ */
.full-width {{
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
}}

.full-width img {{
    width: 100% !important;
    max-width: 100% !important;
    height: 42vh !important;
    object-fit: cover !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
}}

[data-testid="stImage"] img {{
    max-width: 100% !important;
    object-fit: cover !important;
}}

/* ============================
   SIDEBAR TITLES
   ============================ */
.sidebar-title {{
    background: #ffffff;
    color: #003d7a;
    font-size: 17px;
    font-weight: 800;
    padding: 10px 12px;
    border-radius: 10px;
    margin: 14px 0 10px 0;
    border-left: 5px solid #60a5fa;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.10);
}}

section[data-testid="stSidebar"] label {{
    color: #003d7a !important;
    font-weight: 700 !important;
}}

/* SELECTBOX */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background-color: #ffffff !important;
    border: 2px solid #93c5fd !important;
    border-radius: 10px !important;
    color: #003d7a !important;
}}

/* FILE UPLOADER */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: #ffffff !important;
    padding: 12px;
    border-radius: 12px;
    border: 2px dashed #93c5fd;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
    color: #003d7a !important;
}}

/* BUTTON */
section[data-testid="stSidebar"] .stButton > button {{
    background: linear-gradient(90deg, #3b82f6, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    width: 100%;
    padding: 0.6rem 1rem;
}}

section[data-testid="stSidebar"] .stButton > button:hover {{
    background: linear-gradient(90deg, #1e40af, #1e3a8a) !important;
    color: white !important;
}}

/* UPLOAD STATUS */
.upload-success {{
    background: #dbeafe;
    color: #1e40af !important;
    font-size: 13px;
    font-weight: 700;
    padding: 7px 10px;
    border-radius: 8px;
    margin-top: 6px;
}}

.upload-fail {{
    background: #fee2e2;
    color: #991b1b !important;
    font-size: 13px;
    font-weight: 700;
    padding: 7px 10px;
    border-radius: 8px;
    margin-top: 6px;
}}

/* ============================
   FOOTER ALWAYS AT BOTTOM
   ============================ */
.custom-footer {{
    position: fixed;
    bottom: 0;
    left: 300px;
    width: calc(100% - 300px);
    background: linear-gradient(90deg, #dbeafe, #e0f2fe);
    color: #003d7a;
    text-align: center;
    padding: 8px 0;
    font-size: 13px;
    border-top: 1px solid #bfdbfe;
    z-index: 999999;
    font-weight: 600;
}}

@media (max-width: 768px) {{
    .custom-footer {{
        left: 0;
        width: 100%;
    }}
}}

</style>
""", unsafe_allow_html=True)

    # =========================
    # HEADER IMAGE
    # =========================
    st.markdown('<div class="full-width">', unsafe_allow_html=True)
    st.image("assets/auckland_skyline.jpg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # THREE IMAGE CARDS
    # =========================
    st.markdown('<div class="full-width" style="padding:0px;">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("assets/pm25.jpg", use_container_width=True)

    with col2:
        st.image("assets/no2.jpg", use_container_width=True)

    with col3:
        st.image("assets/ml.jpg", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # =========================
    # SIDEBAR CONTENT
    # =========================
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

    uploaded1 = st.sidebar.file_uploader(
        "Dataset 1",
        type=["csv", "xls", "xlsx"],
        key="combined_file1"
    )

    if uploaded1 is not None:
        st.session_state["file1"] = uploaded1

    st.sidebar.markdown(
        "<div class='upload-success'>✔ Dataset 1 uploaded</div>"
        if "file1" in st.session_state else
        "<div class='upload-fail'>✖ Dataset 1 missing</div>",
        unsafe_allow_html=True
    )

    uploaded2 = st.sidebar.file_uploader(
        "Dataset 2",
        type=["csv", "xls", "xlsx"],
        key="combined_file2"
    )

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