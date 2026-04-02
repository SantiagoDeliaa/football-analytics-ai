import streamlit as st


def _read_nav_query_param() -> str | None:
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        value = query_params.get("nav")
        if isinstance(value, list):
            return value[0] if value else None
        return value
    if hasattr(st, "experimental_get_query_params"):
        value = st.experimental_get_query_params().get("nav")
        if isinstance(value, list):
            return value[0] if value else None
        return value
    return None


def _clear_nav_query_param() -> None:
    query_params = getattr(st, "query_params", None)
    if query_params is not None:
        if "nav" in query_params:
            del query_params["nav"]
        return
    if hasattr(st, "experimental_set_query_params"):
        st.experimental_set_query_params()


def _render_navigation_card(title: str, description: str, icon: str, route: str, variant: str) -> None:
    st.markdown(
        f"""
        <a class="tip-card tip-card-{variant}" href="?nav={route}">
            <div class="tip-card-icon">{icon}</div>
            <div class="tip-card-title">{title}</div>
            <div class="tip-card-description">{description}</div>
            <div class="tip-card-cta">Open Module</div>
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> str | None:
    selected_route = _read_nav_query_param()
    if selected_route in {"vertical1", "vertical2"}:
        _clear_nav_query_param()
        return selected_route

    st.markdown(
        """
        <style>
            .tip-shell {
                max-width: 1080px;
                margin: 0 auto;
                padding-top: 3.2rem;
                padding-bottom: 1.2rem;
            }
            .tip-hero {
                text-align: center;
                margin-bottom: 2.2rem;
            }
            .tip-title {
                font-size: clamp(2rem, 4.2vw, 3.4rem);
                line-height: 1.08;
                font-weight: 700;
                letter-spacing: 0.01em;
                color: #e8edf4;
                margin: 0;
            }
            .tip-accent {
                color: #1dd790;
                font-weight: 800;
                text-shadow: 0 0 24px rgba(29, 215, 144, 0.35);
            }
            .tip-wordmark {
                margin-top: 0.8rem;
                font-size: 0.95rem;
                letter-spacing: 0.34em;
                color: #7ee3bb;
                text-transform: uppercase;
                font-weight: 700;
            }
            .tip-card {
                position: relative;
                display: block;
                text-decoration: none;
                min-height: 320px;
                padding: 1.6rem;
                border-radius: 22px;
                border: 1px solid rgba(132, 151, 178, 0.22);
                box-shadow: 0 16px 38px rgba(6, 10, 16, 0.34);
                overflow: hidden;
                transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
                margin-bottom: 0.8rem;
            }
            .tip-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 42px rgba(6, 10, 16, 0.52);
                border-color: rgba(126, 227, 187, 0.42);
            }
            .tip-card::after {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    radial-gradient(circle at 14% 16%, rgba(255, 255, 255, 0.14), transparent 32%),
                    radial-gradient(circle at 84% 82%, rgba(255, 255, 255, 0.09), transparent 38%);
                pointer-events: none;
            }
            .tip-card-cv {
                background:
                    linear-gradient(140deg, rgba(8, 16, 33, 0.92), rgba(10, 40, 54, 0.88)),
                    repeating-linear-gradient(0deg, rgba(85, 145, 175, 0.10) 0 1px, transparent 1px 20px),
                    repeating-linear-gradient(90deg, rgba(85, 145, 175, 0.10) 0 1px, transparent 1px 20px);
            }
            .tip-card-da {
                background:
                    linear-gradient(145deg, rgba(20, 18, 37, 0.92), rgba(15, 45, 38, 0.88)),
                    radial-gradient(circle at 24% 18%, rgba(29, 215, 144, 0.18), transparent 48%),
                    radial-gradient(circle at 78% 86%, rgba(75, 131, 255, 0.16), transparent 46%);
            }
            .tip-card-icon {
                font-size: 2.35rem;
                line-height: 1;
                margin-bottom: 1.1rem;
                opacity: 0.95;
            }
            .tip-card-title {
                font-size: 1.48rem;
                color: #f3f6fb;
                font-weight: 700;
                margin-bottom: 0.55rem;
            }
            .tip-card-description {
                font-size: 0.98rem;
                line-height: 1.45;
                color: #c1cfdf;
                max-width: 34ch;
                margin-bottom: 1.45rem;
            }
            .tip-card-cta {
                display: inline-block;
                font-size: 0.78rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                color: #b9f4dc;
                border: 1px solid rgba(185, 244, 220, 0.35);
                border-radius: 999px;
                padding: 0.38rem 0.78rem;
                background: rgba(7, 15, 20, 0.36);
            }
            @media (max-width: 980px) {
                .tip-shell {
                    padding-top: 2.4rem;
                }
                .tip-card {
                    min-height: 280px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tip-shell">
            <div class="tip-hero">
                <h1 class="tip-title">Tactical Intelligence <span class="tip-accent">Platform</span></h1>
                <div class="tip-wordmark"><span class="tip-accent">TIP</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        _render_navigation_card(
            title="Computer Vision",
            description="Tracking and tactical metrics from broadcast video",
            icon="📹",
            route="vertical1",
            variant="cv",
        )

    with col2:
        _render_navigation_card(
            title="Data Analytics",
            description="Tactical insights and proprietary metrics from event data and reports",
            icon="📊",
            route="vertical2",
            variant="da",
        )

    return None
