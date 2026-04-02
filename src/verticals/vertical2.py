import streamlit as st


def render_vertical2() -> None:
    st.title("Vertical 2 — Data Analytics")
    st.caption("Tactical insights and proprietary metrics from event data and reports")
    if st.button("Volver a Home", key="vertical2_back_home"):
        st.session_state.active_vertical = "home"
        st.rerun()
    st.info("Próximamente: esta vertical se encuentra en preparación.")
