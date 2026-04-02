import streamlit as st
from src.verticals.home import render_home
from src.verticals.vertical1 import render_vertical1
from src.verticals.vertical2 import render_vertical2

ROUTE_HOME = "home"
ROUTE_VERTICAL1 = "vertical1"
ROUTE_VERTICAL2 = "vertical2"

if "active_vertical" not in st.session_state:
    st.session_state.active_vertical = ROUTE_HOME

route = st.session_state.active_vertical

if route == ROUTE_VERTICAL1:
    render_vertical1()
elif route == ROUTE_VERTICAL2:
    st.set_page_config(page_title="Soccer Analytics Platform", layout="wide", initial_sidebar_state="expanded")
    render_vertical2()
else:
    st.set_page_config(page_title="Soccer Analytics Platform", layout="wide", initial_sidebar_state="expanded")
    selected_route = render_home()
    if selected_route:
        st.session_state.active_vertical = selected_route
        st.rerun()
