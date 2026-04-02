import streamlit as st


def _render_home_card(title: str, description: str, route: str, key: str) -> str | None:
    label = f"{title}\n{description}"
    if st.button(label, key=key, use_container_width=True):
        return route
    return None


def render_home() -> str | None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 18% 15%, rgba(36, 88, 62, 0.16), transparent 36%),
                    radial-gradient(circle at 82% 88%, rgba(24, 79, 58, 0.14), transparent 35%),
                    linear-gradient(180deg, #0a1016 0%, #0b131b 45%, #0a1119 100%);
            }
            .stApp > div.block-container {
                max-width: 1120px;
                padding-top: 3.9rem;
                padding-bottom: 2.2rem;
                position: relative;
            }
            .stApp > div.block-container::before {
                content: "";
                position: absolute;
                inset: 1rem 0.8rem auto 0.8rem;
                height: 76vh;
                border-radius: 30px;
                border: 1px solid rgba(126, 227, 187, 0.09);
                background:
                    radial-gradient(circle at 50% 50%, transparent 0 62px, rgba(126, 227, 187, 0.08) 62px 64px, transparent 64px),
                    linear-gradient(90deg, transparent 5%, rgba(126, 227, 187, 0.04) 6%, transparent 7%),
                    linear-gradient(0deg, transparent 34%, rgba(126, 227, 187, 0.04) 35%, transparent 36%);
                pointer-events: none;
                z-index: 0;
            }
            .tip-shell {
                max-width: 1040px;
                margin: 0 auto;
                padding-top: 0.6rem;
                padding-bottom: 1.8rem;
                position: relative;
                z-index: 2;
            }
            .tip-hero {
                text-align: center;
                margin-bottom: 3.2rem;
            }
            .tip-title {
                font-size: clamp(2.2rem, 4.8vw, 3.9rem);
                line-height: 1.05;
                font-weight: 800;
                letter-spacing: 0.01em;
                color: #f2f6fb;
                margin: 0;
                text-wrap: balance;
            }
            .tip-accent {
                color: #1dd790;
                font-weight: 800;
                text-shadow: 0 0 18px rgba(29, 215, 144, 0.22);
            }
            .tip-wordmark {
                margin-top: 0.9rem;
                font-size: 0.9rem;
                letter-spacing: 0.44em;
                color: #86f0c4;
                text-transform: uppercase;
                font-weight: 700;
            }
            div[data-testid="column"] {
                position: relative;
                z-index: 2;
            }
            div[data-testid="column"] .stButton > button {
                position: relative;
                min-height: 328px;
                padding: 1.8rem 1.4rem 2.7rem 1.4rem;
                border-radius: 24px;
                border: 1px solid rgba(140, 160, 188, 0.24);
                box-shadow: 0 16px 36px rgba(5, 9, 15, 0.45);
                color: #f3f7fd;
                font-size: 1.08rem;
                font-weight: 700;
                line-height: 1.55;
                white-space: pre-line;
                text-align: center;
                transition: transform 260ms ease-out, box-shadow 280ms ease-out, border-color 240ms ease-out;
                overflow: hidden;
                backdrop-filter: blur(2px);
            }
            div[data-testid="column"] .stButton > button::before {
                content: "";
                position: absolute;
                inset: 0;
                background:
                    radial-gradient(circle at 14% 16%, rgba(255, 255, 255, 0.12), transparent 32%),
                    radial-gradient(circle at 86% 84%, rgba(255, 255, 255, 0.08), transparent 38%);
                pointer-events: none;
            }
            div[data-testid="column"] .stButton > button::after {
                content: "OPEN MODULE";
                position: absolute;
                bottom: 1.1rem;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.16em;
                color: #c5f7e1;
                border: 1px solid rgba(197, 247, 225, 0.38);
                border-radius: 999px;
                padding: 0.32rem 0.82rem;
                background: rgba(8, 16, 22, 0.42);
            }
            div[data-testid="column"] .stButton > button:hover {
                transform: translateY(-4px) scale(1.01);
                border-color: rgba(126, 227, 187, 0.46);
                box-shadow: 0 20px 44px rgba(5, 9, 15, 0.62), 0 0 0 1px rgba(126, 227, 187, 0.16);
            }
            div[data-testid="column"]:nth-of-type(1) .stButton > button {
                background:
                    linear-gradient(140deg, rgba(7, 17, 32, 0.95), rgba(10, 38, 55, 0.88)),
                    repeating-linear-gradient(0deg, rgba(96, 152, 187, 0.1) 0 1px, transparent 1px 20px),
                    repeating-linear-gradient(90deg, rgba(96, 152, 187, 0.1) 0 1px, transparent 1px 20px);
            }
            div[data-testid="column"]:nth-of-type(2) .stButton > button {
                background:
                    linear-gradient(145deg, rgba(21, 19, 40, 0.95), rgba(13, 42, 39, 0.9)),
                    radial-gradient(circle at 25% 20%, rgba(29, 215, 144, 0.16), transparent 44%),
                    radial-gradient(circle at 78% 84%, rgba(75, 131, 255, 0.14), transparent 42%);
            }
            @media (max-width: 980px) {
                .tip-shell {
                    padding-top: 0.5rem;
                }
                .tip-hero {
                    margin-bottom: 1.9rem;
                }
                div[data-testid="column"] .stButton > button {
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
                <h1 class="tip-title">Tactical Intelligence Platform</h1>
                <div class="tip-wordmark">(<span class="tip-accent">TIP</span>)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        selected_route = _render_home_card(
            title="Computer Vision",
            description="Tracking and tactical metrics from broadcast video",
            route="vertical1",
            key="go_vertical1_card",
        )
        if selected_route:
            return selected_route

    with col2:
        selected_route = _render_home_card(
            title="Data Analytics",
            description="Tactical insights and proprietary metrics from event data and reports",
            route="vertical2",
            key="go_vertical2_card",
        )
        if selected_route:
            return selected_route

    return None
