"""GBSA / MMPBSA energy analysis from AMBER-style CSV output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from analysis.plot_style import apply_layout, color_for_index, hex_to_rgba


@dataclass
class GBSADataset:
    name: str
    filepath: str


@dataclass
class GBSAAnalysisResult:
    label: str
    plot_type: str
    figure: go.Figure
    summary: str = ""


GBSA_PLOT_TYPES = {
    "timeseries": "Binding Free Energy Time Series",
    "components": "Energy Components",
    "convergence": "Convergence",
    "distribution": "Distribution",
    "vdw_vs_eel": "VdW vs Electrostatics",
}

SECTION_HEADERS = (
    "GENERALIZED BORN:",
    "Complex Energy Terms",
    "Receptor Energy Terms",
    "Ligand Energy Terms",
    "Delta Energy Terms",
)

COMPONENTS = ["VDWAALS", "EEL", "EGB", "ESURF", "GGAS", "GSOLV"]
COMP_LABELS = ["ΔVdW", "ΔEEL", "ΔEGB", "ΔESURF", "ΔGGAS", "ΔGSOLV"]


def parse_gbsa(filepath: str) -> dict[str, pd.DataFrame]:
    sections: dict[str, pd.DataFrame] = {}
    current_section: str | None = None
    rows: list[list[float]] = []
    cols: list[str] | None = None

    with open(filepath, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    for raw in lines:
        line = raw.strip()
        if not line:
            if current_section and rows and cols:
                sections[current_section] = pd.DataFrame(rows, columns=cols)
            rows, cols, current_section = [], None, None
            continue

        if line in SECTION_HEADERS:
            current_section = line.rstrip(":")
            continue

        if line.startswith("Frame #"):
            cols = line.split(",")
            continue

        if current_section and cols:
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass

    if current_section and rows and cols:
        sections[current_section] = pd.DataFrame(rows, columns=cols)

    if "Delta Energy Terms" not in sections:
        raise ValueError(f"No 'Delta Energy Terms' section found in {Path(filepath).name}")

    return sections


def _dataset_stats(delta: pd.DataFrame, window: int = 10) -> dict:
    frames = delta["Frame #"].values
    total = delta["TOTAL"].values
    mean = float(total.mean())
    std = float(total.std())
    rolling = pd.Series(total).rolling(window, center=True).mean().values
    cumavg = pd.Series(total).expanding().mean().values
    return {
        "frames": frames,
        "total": total,
        "mean": mean,
        "std": std,
        "rolling": rolling,
        "cumavg": cumavg,
        "delta": delta,
    }


def build_timeseries_figure(datasets: list[tuple[str, dict]], *, window: int = 10) -> go.Figure:
    fig = go.Figure()
    for index, (label, stats) in enumerate(datasets):
        color = color_for_index(index)
        frames, total = stats["frames"], stats["total"]
        mean, std, rolling = stats["mean"], stats["std"], stats["rolling"]

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([frames, frames[::-1]]),
                y=np.concatenate([np.full_like(frames, mean + std), np.full_like(frames, mean - std)[::-1]]),
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.10),
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{label} ±1 SD ({std:.2f})",
                hoverinfo="skip",
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=total,
                mode="lines",
                name=f"{label} per-frame ΔG",
                line=dict(color=color, width=1.2),
                opacity=0.45,
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Frame: %{{x}}<br>ΔG: %{{y:.2f}} kcal/mol<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=rolling,
                mode="lines",
                name=f"{label} {window}-frame mean",
                line=dict(color=color, width=2.5),
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Frame: %{{x}}<br>Mean: %{{y:.2f}} kcal/mol<extra></extra>",
            )
        )
        fig.add_hline(
            y=mean,
            line=dict(color=color, width=1.8, dash="dash"),
            annotation_text=f"<b>{label} Mean = {mean:.2f}</b>",
            annotation_font=dict(color=color, size=11),
        )

    apply_layout(fig, "Binding Free Energy over Trajectory", "Simulation Frame", "ΔG<sub>binding</sub> (kcal/mol)", height=560)
    return fig


def build_components_figure(datasets: list[tuple[str, dict]], *, grouped: bool = True) -> go.Figure:
    fig = go.Figure()
    for index, (label, stats) in enumerate(datasets):
        color = color_for_index(index)
        delta = stats["delta"]
        means = [delta[c].mean() for c in COMPONENTS]
        stds = [delta[c].std() for c in COMPONENTS]
        fig.add_trace(
            go.Bar(
                x=COMP_LABELS,
                y=means,
                error_y=dict(type="data", array=stds, visible=True, color="#333333", thickness=1.5, width=5),
                marker=dict(color=color, opacity=0.85, line=dict(color="#333333", width=1.2)),
                name=label,
                hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Mean: %{{y:.2f}} kcal/mol<extra></extra>",
            )
        )

    fig.add_hline(y=0, line=dict(color="#333333", width=1.2))
    apply_layout(
        fig,
        "Mean Energy Components (ΔDelta)",
        "Energy Component",
        "ΔΔG (kcal/mol)",
        height=520,
        barmode="group" if grouped else "overlay",
    )
    return fig


def build_convergence_figure(datasets: list[tuple[str, dict]]) -> go.Figure:
    fig = go.Figure()
    for index, (label, stats) in enumerate(datasets):
        color = color_for_index(index)
        frames, cumavg = stats["frames"], stats["cumavg"]
        mean, std = stats["mean"], stats["std"]

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([frames, frames[::-1]]),
                y=np.concatenate([cumavg + std, (cumavg - std)[::-1]]),
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.10),
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{label} ±1 SD",
                hoverinfo="skip",
                legendgroup=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frames,
                y=cumavg,
                mode="lines",
                name=f"{label} cumulative mean",
                line=dict(color=color, width=2.5),
                legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>Frame: %{{x}}<br>Cumulative: %{{y:.2f}} kcal/mol<extra></extra>",
            )
        )
        fig.add_hline(
            y=mean,
            line=dict(color=color, width=1.8, dash="dash"),
            annotation_text=f"<b>{label} final = {mean:.2f}</b>",
            annotation_font=dict(color=color, size=11),
        )

    apply_layout(fig, "Convergence of Binding Free Energy", "Simulation Frame", "Cumulative Mean ΔG (kcal/mol)", height=560)
    return fig


def build_distribution_figure(datasets: list[tuple[str, dict]]) -> go.Figure:
    fig = go.Figure()
    for index, (label, stats) in enumerate(datasets):
        color = color_for_index(index)
        total, mean = stats["total"], stats["mean"]

        fig.add_trace(
            go.Histogram(
                x=total,
                nbinsx=20,
                marker=dict(color=color, opacity=0.45, line=dict(color="#FFFFFF", width=0.8)),
                name=f"{label} distribution",
                hovertemplate=f"<b>{label}</b><br>ΔG: %{{x:.1f}} kcal/mol<br>Count: %{{y}}<extra></extra>",
            )
        )
        kde_x = np.linspace(total.min() - 5, total.max() + 5, 300)
        kde_y = gaussian_kde(total)(kde_x)
        bin_width = (total.max() - total.min()) / 20 if total.max() != total.min() else 1.0
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y * len(total) * bin_width,
                mode="lines",
                name=f"{label} KDE",
                line=dict(color=color, width=2.5),
            )
        )
        fig.add_vline(
            x=mean,
            line=dict(color=color, width=2, dash="dash"),
            annotation_text=f"<b>{label} Mean {mean:.2f}</b>",
            annotation_font=dict(color=color, size=11),
            annotation_position="top right",
        )

    apply_layout(
        fig,
        "Distribution of Binding Free Energies",
        "ΔG<sub>binding</sub> (kcal/mol)",
        "Count",
        height=520,
    )
    fig.update_layout(barmode="overlay", bargap=0.05)
    return fig


def build_vdw_vs_eel_figure(datasets: list[tuple[str, dict]]) -> go.Figure:
    fig = go.Figure()
    for index, (label, stats) in enumerate(datasets):
        color = color_for_index(index)
        delta, frames = stats["delta"], stats["frames"]
        fig.add_trace(
            go.Scatter(
                x=delta["VDWAALS"].values,
                y=delta["EEL"].values,
                mode="markers",
                marker=dict(color=color, size=9, opacity=0.75, line=dict(color="#FFFFFF", width=0.5)),
                name=label,
                text=[str(int(f)) for f in frames],
                hovertemplate=(
                    f"<b>{label} — Frame %{{text}}</b><br>"
                    "ΔVdW: %{x:.2f} kcal/mol<br>"
                    "ΔEEL: %{y:.2f} kcal/mol<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=0, line=dict(color="#AAAAAA", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#AAAAAA", width=1, dash="dot"))
    apply_layout(fig, "VdW vs Electrostatic Contribution", "ΔVdW (kcal/mol)", "ΔEEL (kcal/mol)", height=540)
    return fig


def _single_dataset_figure(plot_type: str, label: str, stats: dict, *, window: int = 10) -> go.Figure:
    builders = {
        "timeseries": lambda: build_timeseries_figure([(label, stats)], window=window),
        "components": lambda: build_components_figure([(label, stats)], grouped=False),
        "convergence": lambda: build_convergence_figure([(label, stats)]),
        "distribution": lambda: build_distribution_figure([(label, stats)]),
        "vdw_vs_eel": lambda: build_vdw_vs_eel_figure([(label, stats)]),
    }
    fig = builders[plot_type]()
    base_title = GBSA_PLOT_TYPES[plot_type]
    fig.update_layout(title=dict(text=f"{label} — {base_title}", x=0.5, xanchor="center"))
    return fig


def _summary_line(label: str, stats: dict) -> str:
    return f"{label:<16} Mean ΔG: {stats['mean']:>8.2f}  ± SD: {stats['std']:>8.2f} kcal/mol"


def run_gbsa_analysis(
    datasets: list[GBSADataset],
    plot_types: list[str],
    *,
    combined: bool = True,
    rolling_window: int = 10,
    progress: Callable[[str], None] | None = None,
) -> list[GBSAAnalysisResult]:
    if not datasets:
        raise ValueError("Add at least one GBSA CSV file.")
    if not plot_types:
        raise ValueError("Select at least one GBSA plot type.")

    log = progress or (lambda _msg: None)
    parsed: list[tuple[str, dict]] = []

    for dataset in datasets:
        log(f"Parsing {dataset.name}...")
        sections = parse_gbsa(dataset.filepath)
        stats = _dataset_stats(sections["Delta Energy Terms"], window=rolling_window)
        parsed.append((dataset.name, stats))

    summaries = [_summary_line(label, stats) for label, stats in parsed]
    summary_text = "\n".join(summaries)
    results: list[GBSAAnalysisResult] = []

    builders = {
        "timeseries": lambda data: build_timeseries_figure(data, window=rolling_window),
        "components": lambda data: build_components_figure(data, grouped=len(data) > 1),
        "convergence": build_convergence_figure,
        "distribution": build_distribution_figure,
        "vdw_vs_eel": build_vdw_vs_eel_figure,
    }

    for plot_type in plot_types:
        if combined:
            fig = builders[plot_type](parsed)
            results.append(
                GBSAAnalysisResult(
                    label=f"Combined — {GBSA_PLOT_TYPES[plot_type]}",
                    plot_type=plot_type,
                    figure=fig,
                    summary=summary_text,
                )
            )
        else:
            for label, stats in parsed:
                fig = _single_dataset_figure(plot_type, label, stats, window=rolling_window)
                results.append(
                    GBSAAnalysisResult(
                        label=f"{label} — {GBSA_PLOT_TYPES[plot_type]}",
                        plot_type=plot_type,
                        figure=fig,
                        summary=_summary_line(label, stats),
                    )
                )

    return results
