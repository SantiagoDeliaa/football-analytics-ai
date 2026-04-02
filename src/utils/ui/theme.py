import streamlit as st


def apply_premium_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-0: #0b0d11;
            --bg-1: #12151b;
            --bg-2: #171c24;
            --bg-3: #1f2631;
            --text-0: #f2f5f7;
            --text-1: #c9d0d8;
            --text-2: #8f9baa;
            --accent: #4f8cff;
            --accent-2: #6aa7ff;
            --success: #17b26a;
            --warning: #f79009;
            --danger: #f04438;
            --border: #2a3342;
            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 18px;
            --space-xs: 0.35rem;
            --space-sm: 0.55rem;
            --space-md: 0.9rem;
            --space-lg: 1.2rem;
            --space-xl: 1.8rem;
        }
        .stApp {
            background: radial-gradient(circle at 20% 0%, #141922 0%, var(--bg-0) 48%);
            color: var(--text-0);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #10141b 0%, #0d1016 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stCheckbox label {
            color: var(--text-1);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: var(--text-0);
            letter-spacing: 0.01em;
        }
        [data-testid="stSidebar"] input[type="radio"] {
            accent-color: var(--accent) !important;
        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        .st-emotion-cache-1r6slb0,
        .st-emotion-cache-z5fcl4 {
            padding-top: 1.35rem;
        }
        button[data-testid="stSidebarCollapsedControl"] {
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: rgba(18, 24, 33, 0.85);
        }
        button[data-testid="stSidebarCollapsedControl"]:hover {
            border-color: #54719b;
            background: rgba(32, 44, 61, 0.9);
        }
        .platform-header {
            padding: 1.15rem 1.35rem 1.1rem 1.35rem;
            border-radius: var(--radius-lg);
            background: linear-gradient(115deg, #111a28 0%, #0f131a 55%, #162236 100%);
            border: 1px solid var(--border);
            margin-bottom: 1.05rem;
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.36);
        }
        .platform-kicker {
            color: #7fb4ff;
            text-transform: uppercase;
            font-size: 0.67rem;
            letter-spacing: 0.12em;
            font-weight: 650;
            margin-bottom: 0.42rem;
        }
        .platform-title {
            font-size: 2.25rem;
            font-weight: 750;
            line-height: 1.12;
            color: var(--text-0);
            margin-bottom: 0;
        }
        .platform-subtitle {
            font-size: 0.95rem;
            color: var(--text-1);
            margin-bottom: 0;
        }
        .section-title {
            font-size: 0.98rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--text-2);
            margin: 0.45rem 0 0.5rem 0;
            font-weight: 600;
        }
        .status-card {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: rgba(22, 27, 36, 0.8);
            padding: 0.8rem 0.95rem;
            margin: 0.35rem 0 0.75rem 0;
        }
        .status-title {
            color: var(--text-1);
            font-size: 0.83rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.2rem;
        }
        .status-value {
            color: var(--text-0);
            font-size: 1.02rem;
            font-weight: 600;
        }
        div[data-testid="stTabs"] button {
            border-radius: 12px;
            padding: 0.62rem 0.95rem;
            background: rgba(20, 25, 34, 0.65);
            border: 1px solid var(--border);
            color: var(--text-1);
            font-weight: 600;
            font-size: 0.93rem;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: linear-gradient(180deg, #1b2432 0%, #141b25 100%);
            color: var(--text-0);
            border: 1px solid #3a4c66;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid #3b4f6d;
            background: linear-gradient(180deg, #2a3a51 0%, #1f2d40 100%);
            color: #ecf2ff;
            font-weight: 600;
        }
        .stButton > button:hover {
            border-color: #54719b;
            background: linear-gradient(180deg, #314763 0%, #24354b 100%);
            color: #ffffff;
        }
        .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid var(--border);
            background: #161c25;
            color: var(--text-1);
        }
        [data-testid="stMetric"] {
            background: rgba(20, 24, 32, 0.85);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.5rem 0.7rem;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            overflow: hidden;
        }
        [data-testid="stAlert"] {
            border-radius: var(--radius-md);
            border: 1px solid var(--border);
        }
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] > div,
        [data-testid="stFileUploader"] section {
            background: #131a23 !important;
            border-color: var(--border) !important;
            color: var(--text-0) !important;
            border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            min-height: 88px;
        }
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {
            color: #c6d3e8 !important;
        }
        [data-testid="stFileUploaderDropzone"] small {
            color: #8fa5c3 !important;
        }
        [data-testid="stFileUploader"] button {
            border-radius: 10px !important;
            border: 1px solid #3c5578 !important;
            background: linear-gradient(180deg, #3f5f8b 0%, #2c4568 100%) !important;
            color: #f4f8ff !important;
            font-weight: 650 !important;
            padding: 0.4rem 0.95rem !important;
        }
        [data-testid="stFileUploader"] button:hover {
            border-color: #5f82b4 !important;
            background: linear-gradient(180deg, #4a6f9f 0%, #35537a 100%) !important;
            color: #ffffff !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            background: rgba(17, 21, 29, 0.6);
        }
        hr {
            border-color: rgba(120, 137, 164, 0.28) !important;
        }
        .platform-footer {
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 0.85rem 1rem;
            background: linear-gradient(180deg, rgba(17, 22, 31, 0.75) 0%, rgba(12, 16, 23, 0.9) 100%);
            color: var(--text-1);
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
        }
        .platform-footer strong {
            color: var(--text-0);
            font-weight: 650;
        }
        .platform-footer .footer-pill {
            border: 1px solid rgba(127, 180, 255, 0.3);
            border-radius: 999px;
            padding: 0.2rem 0.62rem;
            font-size: 0.78rem;
            color: #cfe3ff;
            background: rgba(33, 56, 86, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="platform-header">
            <div class="platform-kicker">AI-Driven Match Intelligence Suite</div>
            <div class="platform-title">Football Tactical Intelligence Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def render_status_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">{title}</div>
            <div class="status-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_dark_theme(fig) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f131a",
        plot_bgcolor="#141b24",
        font=dict(color="#d6deea"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=24, r=24, t=56, b=36),
        hoverlabel=dict(bgcolor="#0b0d11", bordercolor="#2a3342"),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(120, 137, 164, 0.18)",
        zeroline=False,
        linecolor="rgba(120, 137, 164, 0.3)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(120, 137, 164, 0.18)",
        zeroline=False,
        linecolor="rgba(120, 137, 164, 0.3)",
    )
