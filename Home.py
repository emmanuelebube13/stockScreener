import streamlit as st

st.set_page_config(
    page_title="QuantPortal",
    layout="wide",
    page_icon=""
)

st.title(" QuantPortal: Unified Analytics")
st.write("Welcome to your unified financial dashboard. Select a module from the sidebar.")

col1, col2 = st.columns(2)

with col1:
    st.info("###  Pro Terminal")
    st.write("Real-time market monitoring, news, and technical analysis.")
    st.page_link("pages/1_Pro_Terminal.py", label="Launch Terminal", icon="📈")

with col2:
    st.error("### Institutional Quant")
    st.write("MRM-compliant backtesting, factor scoring, and risk management.")
    st.page_link("pages/2_Institutional_Quant.py", label="Launch Quant Lab", icon="🏦")