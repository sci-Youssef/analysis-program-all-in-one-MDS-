"""Molecular dynamics trajectory analysis (RMSD, RMSF, RoG, H-bonds, SASA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import MDAnalysis as mda
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.topology.guessers import guess_types

from analysis.plot_style import apply_layout, color_for_index


@dataclass
class TrajectoryDataset:
    name: str
    topology: str
    trajectory: str
    sasa_file: str | None = None


@dataclass
class MDAnalysisResult:
    label: str
    plot_type: str
    figure: go.Figure
    summary: str = ""


MD_PLOT_TYPES = {
    "rmsd": "RMSD",
    "rmsf": "RMSF",
    "rog": "Radius of Gyration",
    "hbonds": "Hydrogen Bonds",
    "sasa": "SASA",
}


def _prepare_universe(topology: str, trajectory: str) -> mda.Universe:
    uni = mda.Universe(topology, trajectory)
    guessed = guess_types(uni.atoms.names)
    uni.add_TopologyAttr("elements", guessed)
    return uni


def _compute_rmsd(uni: mda.Universe, select: str = "name CA") -> tuple[np.ndarray, np.ndarray]:
    result = rms.RMSD(uni, uni, select=select, ref_frame=0).run()
    time_ns = result.results.rmsd[:, 1] / 1000.0
    values = result.results.rmsd[:, 2]
    return time_ns, values


def _compute_rmsf(topology: str, trajectory: str, select: str = "protein and name CA") -> tuple[np.ndarray, np.ndarray]:
    uni = _prepare_universe(topology, trajectory)
    align.AlignTraj(uni, uni, select=select, in_memory=True).run()
    atoms = uni.select_atoms(select)
    rmsf_result = rms.RMSF(atoms).run()
    return atoms.resids.astype(float), rmsf_result.results.rmsf


def _compute_rog(uni: mda.Universe, select: str = "protein") -> tuple[np.ndarray, np.ndarray]:
    atoms = uni.select_atoms(select)
    values = [atoms.radius_of_gyration() for _ in uni.trajectory]
    time_ns = np.linspace(0, uni.trajectory.totaltime / 1000.0, len(values))
    return time_ns, np.asarray(values)


def _compute_hbonds(uni: mda.Universe) -> tuple[np.ndarray, np.ndarray]:
    hb_run = HydrogenBondAnalysis(
        universe=uni,
        donors_sel="name S* or name N* or name O* or name F*",
        hydrogens_sel="name H*",
        acceptors_sel="name S* or name N* or name O* or name F*",
        between=["protein", "protein"],
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=160,
        update_selections=True,
    ).run()

    df = pd.DataFrame(
        hb_run.results.hbonds,
        columns=["frame", "donor_index", "hydrogen_index", "acceptor_index", "DA_distance", "DHA_angle"],
    )
    counts = df.groupby("frame").count().donor_index
    empty = pd.DataFrame({"donor_index": [0] * len(uni.trajectory)})
    full = pd.concat([empty, counts], axis=1).sum(axis=1)
    time_ns = full.index.to_numpy() * (uni.trajectory.dt / 1000.0)
    return time_ns, full.values


def _load_sasa_xvg(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.transpose(np.loadtxt(path, comments=["@", "#"]))
    time_ns = data[0] / 1000.0
    area = data[1] * 100.0
    return time_ns, area


def _single_trace_figure(
    plot_type: str,
    dataset: TrajectoryDataset,
    color: str,
    *,
    rmsd_select: str = "name CA",
    progress: Callable[[str], None] | None = None,
) -> tuple[go.Figure, str]:
    log = progress or (lambda _msg: None)
    log(f"Analyzing {dataset.name} — {MD_PLOT_TYPES[plot_type]}...")

    fig = go.Figure()
    summary = ""

    if plot_type == "rmsd":
        uni = _prepare_universe(dataset.topology, dataset.trajectory)
        x, y = _compute_rmsd(uni, rmsd_select)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=dataset.name,
                line=dict(color=color, width=2),
                hovertemplate="Time: %{x:.2f} ns<br>RMSD: %{y:.2f} Å<extra></extra>",
            )
        )
        apply_layout(fig, f"{dataset.name} — RMSD", "Time (ns)", "RMSD (Å)")
        summary = f"{dataset.name} RMSD mean: {y.mean():.2f} Å"

    elif plot_type == "rmsf":
        x, y = _compute_rmsf(dataset.topology, dataset.trajectory)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=dataset.name,
                line=dict(color=color, width=1.8),
                hovertemplate="Residue: %{x}<br>RMSF: %{y:.2f} Å<extra></extra>",
            )
        )
        apply_layout(fig, f"{dataset.name} — RMSF", "Residue ID", "RMSF (Å)")
        summary = f"{dataset.name} RMSF max: {y.max():.2f} Å"

    elif plot_type == "rog":
        uni = _prepare_universe(dataset.topology, dataset.trajectory)
        x, y = _compute_rog(uni)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=dataset.name,
                line=dict(color=color, width=2),
                hovertemplate="Time: %{x:.2f} ns<br>RoG: %{y:.2f} Å<extra></extra>",
            )
        )
        apply_layout(fig, f"{dataset.name} — Radius of Gyration", "Time (ns)", "RoG (Å)")
        summary = f"{dataset.name} RoG mean: {y.mean():.2f} Å"

    elif plot_type == "hbonds":
        uni = _prepare_universe(dataset.topology, dataset.trajectory)
        x, y = _compute_hbonds(uni)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=dataset.name,
                line=dict(color=color, width=2),
                hovertemplate="Time: %{x:.2f} ns<br>H-bonds: %{y}<extra></extra>",
            )
        )
        apply_layout(fig, f"{dataset.name} — Protein H-Bonds", "Time (ns)", "Count")
        summary = f"{dataset.name} H-bonds mean: {y.mean():.1f}"

    elif plot_type == "sasa":
        if not dataset.sasa_file or not Path(dataset.sasa_file).exists():
            raise FileNotFoundError(
                f"SASA file not found for {dataset.name}. Provide a GROMACS .xvg file or skip SASA."
            )
        x, y = _load_sasa_xvg(dataset.sasa_file)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=dataset.name,
                line=dict(color=color, width=2),
                hovertemplate="Time: %{x:.2f} ns<br>SASA: %{y:.1f} Å²<extra></extra>",
            )
        )
        apply_layout(fig, f"{dataset.name} — SASA", "Time (ns)", "Area (Å²)")
        summary = f"{dataset.name} SASA mean: {y.mean():.1f} Å²"

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    return fig, summary


def _combined_figure(
    plot_type: str,
    datasets: list[TrajectoryDataset],
    *,
    rmsd_select: str = "name CA",
    progress: Callable[[str], None] | None = None,
) -> tuple[go.Figure, list[str]]:
    titles = {
        "rmsd": ("Protein RMSD Comparison", "Time (ns)", "RMSD (Å)"),
        "rmsf": ("Protein RMSF Comparison", "Residue ID", "RMSF (Å)"),
        "rog": ("Radius of Gyration Comparison", "Time (ns)", "RoG (Å)"),
        "hbonds": ("Protein H-Bonds Comparison", "Time (ns)", "Count"),
        "sasa": ("SASA Comparison", "Time (ns)", "Area (Å²)"),
    }
    title, x_label, y_label = titles[plot_type]
    fig = go.Figure()
    summaries: list[str] = []

    for index, dataset in enumerate(datasets):
        color = color_for_index(index)
        single, summary = _single_trace_figure(
            plot_type,
            dataset,
            color,
            rmsd_select=rmsd_select,
            progress=progress,
        )
        for trace in single.data:
            fig.add_trace(trace)
        summaries.append(summary)

    apply_layout(fig, title, x_label, y_label, height=560)
    return fig, summaries


def run_md_analysis(
    datasets: list[TrajectoryDataset],
    plot_types: list[str],
    *,
    combined: bool = True,
    rmsd_select: str = "name CA",
    progress: Callable[[str], None] | None = None,
) -> list[MDAnalysisResult]:
    if not datasets:
        raise ValueError("Add at least one trajectory dataset.")
    if not plot_types:
        raise ValueError("Select at least one analysis type.")

    results: list[MDAnalysisResult] = []

    for plot_type in plot_types:
        if combined:
            fig, summaries = _combined_figure(
                plot_type,
                datasets,
                rmsd_select=rmsd_select,
                progress=progress,
            )
            results.append(
                MDAnalysisResult(
                    label=f"Combined — {MD_PLOT_TYPES[plot_type]}",
                    plot_type=plot_type,
                    figure=fig,
                    summary="\n".join(summaries),
                )
            )
        else:
            for index, dataset in enumerate(datasets):
                color = color_for_index(index)
                fig, summary = _single_trace_figure(
                    plot_type,
                    dataset,
                    color,
                    rmsd_select=rmsd_select,
                    progress=progress,
                )
                results.append(
                    MDAnalysisResult(
                        label=f"{dataset.name} — {MD_PLOT_TYPES[plot_type]}",
                        plot_type=plot_type,
                        figure=fig,
                        summary=summary,
                    )
                )

    return results
