from __future__ import annotations

import streamlit as st

from src.verticals.home import render_home
from src.verticals.vertical1 import render_vertical1
from src.verticals.vertical2 import render_vertical2

ROUTE_HOME = "home"
ROUTE_VERTICAL1 = "vertical1"
ROUTE_VERTICAL2 = "vertical2"


def _stop_execution() -> None:
    stop_fn = getattr(st, "stop", None)
    if callable(stop_fn):
        stop_fn()


def _configure_page() -> None:
    set_page_config = getattr(st, "set_page_config", None)
    if callable(set_page_config):
        set_page_config(
            page_title="Soccer Analytics Platform",
            layout="wide",
            initial_sidebar_state="expanded",
        )


def main() -> None:
    _configure_page()

    if "active_vertical" not in st.session_state:
        st.session_state.active_vertical = ROUTE_HOME

    route = st.session_state.active_vertical
    if route == ROUTE_VERTICAL1:
        render_vertical1()
        _stop_execution()
        return

    if route == ROUTE_VERTICAL2:
        render_vertical2()
        _stop_execution()
        return

    selected_route = render_home()
    if selected_route:
        st.session_state.active_vertical = selected_route
        st.rerun()
    _stop_execution()


if __name__ == "__main__":
    main()
