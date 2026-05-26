import streamlit as st

from modules.cleaning import render_cleaning
from modules.visualisation import render_visualisation
from ui_components import load_full_ui, render_footer


st.set_page_config(
    page_title=" Visualisation & Cleaning ",
    page_icon="📊",
    layout="wide"
)

st.write("")
load_full_ui()
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
    max-width: 95% !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: #f8fbff;
    border: 1px solid #dbeafe;
    padding: 8px;
    border-radius: 16px;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0px 20px;
    background: white;
    border-radius: 12px;
    border: 1px solid #dbe3f0;
    color: #07184a;
    font-weight: 900;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #eaf3ff, #f8fbff) !important;
    border: 1px solid #74a7ff !important;
    color: #0433d9 !important;
}
</style>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["📊 Data Insights", "🧹 Data Cleaning"])

with tab1:
    render_visualisation()
  

with tab2:
    render_cleaning()

render_footer()
