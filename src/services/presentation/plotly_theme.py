from __future__ import annotations


def apply_plotly_dark_theme(fig) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font=dict(color="#e5e7eb", family="Inter, sans-serif"),
        legend=dict(
            bgcolor="rgba(15, 19, 26, 0.85)",
            bordercolor="#1f2937",
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=20, r=20, t=48, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.12)", zeroline=False)
