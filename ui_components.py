# =========================
# ui_components.py
# =========================

import streamlit as st
import base64


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def load_full_ui():

    # =========================
    # LOAD SIDEBAR LOGO
    # =========================
    try:
        logo_base64 = get_base64_image("assets/auckland_council_logo.png")
    except:
        logo_base64 = ""

    st.markdown(f"""
<style>

/* ======================================================
   GLOBAL APP
   ====================================================== */

html, body, [class*="css"] {{
    margin: 0 !important;
    padding: 0 !important;
}}

.stApp {{
    background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%) !important;
    overflow-x: hidden !important;
}}

/* ======================================================
   REMOVE STREAMLIT DEFAULT HEADER
   ====================================================== */

header {{
    visibility: hidden;
    height: 0px !important;
}}

#MainMenu,
footer,
[data-testid="stStatusWidget"] {{
    visibility: hidden;
}}

[data-testid="stAppViewContainer"] {{
    padding: 0 !important;
    margin: 0 !important;
    overflow-x: hidden !important;
}}

[data-testid="stAppViewContainer"] .main {{
    padding: 0 !important;
    margin: 0 !important;
    overflow-x: hidden !important;
}}

/* ======================================================
   MAIN PAGE WIDTH
   ====================================================== */

.block-container {{
    max-width: 1200px !important;

    padding-top: 0px !important;
    padding-left: 16px !important;
    padding-right: 16px !important;

    padding-bottom: 20px !important;

    margin: -40px auto !important;
                

    overflow-x: hidden !important;
}}

/* ======================================================
   FULL WIDTH SECTIONS
   ====================================================== */

.full-width {{
    width: calc(100vw - 340px) !important;

    margin-left: calc(
        (1200px - (100vw - 340px)) / 2
    ) !important;

    padding: 0 !important;


    overflow-x: hidden !important;
}}

@media (max-width: 1200px) {{
    .full-width {{
        width: 100% !important;
        margin-left: 0 !important;
    }}
}}

/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg,
        #dbeafe 0%,
        #eff6ff 45%,
        #ffffff 100%
    ) !important;

    border-right: 1px solid #bfdbfe;

    padding: 0 !important;
    margin: 0 !important;

    min-height: 100vh !important;
}}

section[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: 0 !important;

    padding-bottom: 0 !important;
    margin-bottom: 0 !important;

    min-height: 100vh !important;
}}

[data-testid="stSidebarContent"] {{
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}}

/* ======================================================
   SIDEBAR LOGO
   ====================================================== */

[data-testid="stSidebarNav"] {{
    padding-top: 0 !important;
    margin-top: -70px !important;
}}

[data-testid="stSidebarNav"]::before {{
    content: "";

    display: block;

    margin: 12px auto 18px auto;

    height: 100px;
    width: 220px;

    background-image: url("data:image/png;base64,{logo_base64}");

    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
}}

/* ======================================================
   HEADER IMAGE
   ====================================================== */

.full-width > div:first-child img {{
    width: 100% !important;

    height: 42vh !important;

    object-fit: cover !important;

    border-radius: 0px 0px 22px 22px;
}}

/* ======================================================
   INFO IMAGE CARDS
   ====================================================== */

.full-width [data-testid="column"] {{
    background: #ffffff;

    padding: 8px;

    border-radius: 16px;

    border: 1px solid #e2e8f0;

    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);

    display: flex;
    align-items: stretch;
}}

.full-width [data-testid="column"] img {{
    width: 100% !important;

    height: 190px !important;

    object-fit: cover !important;

    border-radius: 14px !important;
}}

/* ======================================================
   SIDEBAR TITLES
   ====================================================== */

.sidebar-title {{
    background: #ffffff;

    color: #003d7a;

    font-size: 17px;

    font-weight: 800;

    padding: 10px 12px;

    border-radius: 10px;

    margin: -10px 0 8px 0;

    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}}

/* ======================================================
   LABELS
   ====================================================== */

section[data-testid="stSidebar"] label {{
    color: #003d7a !important;

    font-weight: 700 !important;
}}

/* ======================================================
   SELECTBOX
   ====================================================== */

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    background-color: #ffffff !important;

    border: 2px solid #93c5fd !important;

    border-radius: 10px !important;

    color: #003d7a !important;
}}

/* ======================================================
   FILE UPLOADER
   ====================================================== */

section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: #ffffff !important;

    padding: 2px;

    border-radius: 12px;

    border: 2px solid #93c5fd !important;

    box-shadow: 0px 2px 4px rgba(0,0,0,0.08);
}}

section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
    color: #003d7a !important;
}}

/* ======================================================
   BUTTON
   ====================================================== */

section[data-testid="stSidebar"] .stButton > button {{
    background: linear-gradient(
        90deg,
        #3b82f6,
        #1e40af
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 10px !important;

    font-weight: 700 !important;

    width: 100%;

    padding: 0.65rem 1rem;
}}

section[data-testid="stSidebar"] .stButton > button:hover {{
    background: linear-gradient(
        90deg,
        #1e40af,
        #1e3a8a
    ) !important;
}}

/* ======================================================
   STATUS BOXES
   ====================================================== */

.upload-success {{
    background: #dbeafe;

    color: #047857 !important;

    font-size: 13px;

    font-weight: 700;

    padding: 8px 10px;

    border-radius: 8px;

    margin-top: 2px;

    margin-bottom: 6px;
}}

.upload-fail {{
    background: #fee2e2;

    color: #991b1b !important;

    font-size: 13px;

    font-weight: 700;

    padding: 8px 10px;

    border-radius: 8px;

    margin-top: 2px;

    margin-bottom: 6px;
}}

/* ======================================================
   HOME PAGE CARDS
   ====================================================== */

.home-card {{
    background: #ffffff;

    padding: 22px;

    border-radius: 16px;

    border: 1px solid #e2e8f0;

    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}}

.info-card {{
    background: #f8fafc;

    padding: 16px;

    border-radius: 14px;

    border: 1px solid #e2e8f0;

    height: 100%;
}}

.highlight-box {{
    background: #f0fdf4;

    padding: 14px;

    border-radius: 12px;

    border-left: 5px solid #22c55e;
}}

/* ======================================================
   FOOTER
   SHOW ONLY AT PAGE BOTTOM
   ====================================================== */

.custom-footer {{
    position: relative;

    width: 100%;

    background: linear-gradient(
        90deg,
        #dbeafe,
        #e0f2fe
    );

    color: #003d7a;

    text-align: center;

    padding: 10px 0;

    font-size: 11px;

    border-top: 1px solid #bfdbfe;

    font-weight: 700;

    margin-top: 35px;

    border-radius: 12px 12px 0 0;
}}

</style>
""", unsafe_allow_html=True)
    # =========================
    # HEADER IMAGE
    # =========================

    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.image(
        "assets/auckland_skyline.jpg",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
     # =========================
    # Info cards
    # =========================

    st.markdown('<div class="full-width">', unsafe_allow_html=True)

    st.image(
        "assets/infocards.jpg",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    
    # =========================
    # SIDEBAR
    # =========================

    
    st.sidebar.markdown(
    '<div class="sidebar-title">📂 Upload Datasets</div>',
    unsafe_allow_html=True
    )

    uploaded_files = st.sidebar.file_uploader(
       "Upload datasets",
        type=["csv", "xls", "xlsx"],
        accept_multiple_files=True,
        key="combined_files"
        )

    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files

    st.sidebar.markdown(
        f"<div class='upload-success'>✔ {len(st.session_state['uploaded_files'])} dataset(s) uploaded successfully</div>"
        if "uploaded_files" in st.session_state and st.session_state["uploaded_files"]
        else "<div class='upload-fail'>✖ No datasets uploaded</div>",
        unsafe_allow_html=True
     )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Auckland Council AQI Forecast Dashboard • GDDA713 • © 2026"
    )


def render_footer():

 st.markdown("""
    <div class="custom-footer">
        🌿 Auckland Council AQI Forecast Dashboard • GDDA713 • © 2026
    </div>
    """, unsafe_allow_html=True)