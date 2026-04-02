import streamlit as st


def render_home() -> str | None:
    st.header("Football Analytics AI")
    st.subheader("Selecciona una vertical para continuar")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Vertical 1 — Computer Vision")
        st.markdown("Tracking and tactical metrics from broadcast video")
        if st.button("Entrar a Computer Vision", use_container_width=True, key="go_vertical1"):
            return "vertical1"

    with col2:
        st.markdown("### Vertical 2 — Data Analytics")
        st.markdown("Tactical insights and proprietary metrics from event data and reports")
        if st.button("Entrar a Data Analytics", use_container_width=True, key="go_vertical2"):
            return "vertical2"

    return None
