"""Shared Plotly styling for publication-quality figures."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

COLORS = [
    "#2166AC",
    "#D6604D",
    "#1A9641",
    "#762A83",
    "#F4A582",
    "#4DAC26",
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
]

GRID = "#CCCCCC"
TEXT = "#1A1A1A"
BG = "#FFFFFF"
PANEL = "#F7F7F7"

pio.templates.default = "simple_white"


def color_for_index(index: int) -> str:
    return COLORS[index % len(COLORS)]


def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def base_layout(title: str, *, height: int | None = 520) -> dict:
    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Arial, sans-serif", color=TEXT, size=14),
        title=dict(
            text=title,
            font=dict(size=20, color=TEXT, family="Arial, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#AAAAAA",
            borderwidth=1,
            font=dict(size=11, color=TEXT),
            orientation="h",
            yanchor="top",
            y=-0.38,
            xanchor="center",
            x=0.5,
        ),
        xaxis=dict(
            gridcolor=GRID,
            gridwidth=0.8,
            linecolor="#333333",
            linewidth=1.5,
            tickfont=dict(size=12, color=TEXT),
            title_font=dict(size=14, color=TEXT),
            mirror=True,
            showline=True,
            ticks="outside",
            ticklen=5,
        ),
        yaxis=dict(
            gridcolor=GRID,
            gridwidth=0.8,
            linecolor="#333333",
            linewidth=1.5,
            tickfont=dict(size=12, color=TEXT),
            title_font=dict(size=14, color=TEXT),
            mirror=True,
            showline=True,
            ticks="outside",
            ticklen=5,
        ),
        margin=dict(l=80, r=60, t=90, b=130),
        autosize=True,
    )
    if height is not None:
        layout["height"] = height
    return layout


def apply_layout(
    fig: go.Figure,
    title: str,
    x_title: str,
    y_title: str,
    *,
    height: int | None = 520,
    barmode: str | None = None,
) -> go.Figure:
    layout = base_layout(title, height=height)
    layout["xaxis_title"] = x_title
    layout["yaxis_title"] = y_title
    if barmode:
        layout["barmode"] = barmode
    fig.update_layout(**layout)
    return fig
