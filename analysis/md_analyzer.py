"""Molecular dynamics trajectory analysis (RMSD, RMSF, RoG, H-bonds, SASA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import MDAnalysis as mda
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# ── Multi-chain RMSF helpers ──────────────────────────────────────────────────

def _detect_chains_by_resid_reset(universe: mda.Universe):
    """
    Split protein CA atoms into chains by detecting resid resets
    (e.g. ...,154,155,1,2,...). Returns a list of
    (label, "atom_range", (start_index, end_index)) tuples, where the
    indices are positions within the CA-only AtomGroup.
    """
    ca = universe.select_atoms("protein and name CA")
    resids = ca.resids

    boundaries = [0]
    for j in range(1, len(resids)):
        if resids[j] < resids[j - 1]:
            boundaries.append(j)
    boundaries.append(len(resids))

    chain_groups = []
    for idx in range(len(boundaries) - 1):
        start, end = boundaries[idx], boundaries[idx + 1]
        chain_groups.append((f"chain{idx + 1}", "atom_range", (start, end)))
    return chain_groups


def _chain_selection_string(universe: mda.Universe, kind: str, key) -> str:
    """Build an MDAnalysis selection string for one chain's CA atoms."""
    if kind == "atom_range":
        ca = universe.select_atoms("protein and name CA")
        start, end = key
        indices = ca.indices[start:end]
        return "index " + " ".join(map(str, indices))
    elif kind == "segid":
        return f"protein and segid {key} and name CA"
    elif kind == "chainID":
        return f"protein and chainID {key} and name CA"
    else:
        return "protein and name CA"


def _compute_rmsf_multichain(
    topology: str,
    trajectory: str,
    *,
    progress: Callable[[str], None] | None = None,
):
    """
    Compute RMSF per-chain (each chain aligned on itself) and return
    concatenated residue positions, RMSF values, and chain boundary info.

    Returns (combined_x, combined_y, chain_boundaries, chain_labels, n_chains).
    """
    log = progress or (lambda _msg: None)
    u_probe = _prepare_universe(topology, trajectory)
    chain_groups = _detect_chains_by_resid_reset(u_probe)
    log(f"  Detected {len(chain_groups)} chain(s): {[c[0] for c in chain_groups]}")

    combined_x: list[int] = []
    combined_y: list[float] = []
    chain_boundaries: list[int] = []
    running_offset = 0

    for label, kind, key in chain_groups:
        u_chain = _prepare_universe(topology, trajectory)
        sel_str = _chain_selection_string(u_chain, kind, key)

        align.AlignTraj(u_chain, u_chain, select=sel_str, in_memory=True).run()
        c_alphas = u_chain.select_atoms(sel_str)
        rmsf_result = rms.RMSF(c_alphas).run()

        n_res = len(c_alphas)
        chain_boundaries.append(running_offset)
        combined_x.extend(range(running_offset + 1, running_offset + n_res + 1))
        combined_y.extend(rmsf_result.results.rmsf.tolist())
        running_offset += n_res
        log(f"    {label}: {n_res} residues, mean RMSF = {np.mean(rmsf_result.results.rmsf):.2f} Å")

    chain_labels = [c[0] for c in chain_groups]
    return np.asarray(combined_x), np.asarray(combined_y), chain_boundaries, chain_labels, len(chain_groups)


def build_multichain_rmsf_figure(
    datasets: list[TrajectoryDataset],
    *,
    progress: Callable[[str], None] | None = None,
    use_subplots: bool = True,
) -> go.Figure:
    """
    Build a multi-chain-aware RMSF figure.

    If *use_subplots* is True, each system gets its own subplot with
    per-chain boundary markers drawn within that subplot (recommended
    when systems have different chain-length splits).

    If *use_subplots* is False, all systems are overlaid on a single plot
    with dotted vertical lines marking chain boundaries.
    """
    log = progress or (lambda _msg: None)

    if use_subplots:
        fig = make_subplots(
            rows=len(datasets), cols=1,
            shared_xaxes=False,
            subplot_titles=[d.name for d in datasets],
            vertical_spacing=0.12,
        )
    else:
        fig = go.Figure()

    for i, dataset in enumerate(datasets):
        color = color_for_index(i)
        log(f"Multi-chain RMSF: {dataset.name} ...")
        cx, cy, boundaries, chain_labels, n_chains = _compute_rmsf_multichain(
            dataset.topology, dataset.trajectory, progress=log,
        )

        if use_subplots:
            row = i + 1
            fig.add_trace(
                go.Scatter(
                    x=cx, y=cy,
                    name=dataset.name,
                    line=dict(color=color, width=1.8),
                    showlegend=False,
                    hovertemplate="Position: %{x}<br>RMSF: %{y:.2f} Å<extra></extra>",
                ),
                row=row, col=1,
            )
            y_max = float(np.max(cy)) * 1.1
            for b in boundaries[1:]:
                fig.add_shape(
                    type="line",
                    x0=b + 0.5, x1=b + 0.5,
                    y0=0, y1=y_max,
                    line=dict(color=color, dash="dot", width=1.5),
                    opacity=0.6,
                    row=row, col=1,
                )
            fig.update_xaxes(title_text="Residue position (chain 1 then chain 2)", row=row, col=1)
            fig.update_yaxes(title_text="RMSF (Å)", row=row, col=1)
        else:
            fig.add_trace(
                go.Scatter(
                    x=cx, y=cy,
                    name=dataset.name,
                    line=dict(color=color, width=1.8),
                    hovertemplate=f"<b>{dataset.name}</b><br>Position: %{{x}}<br>RMSF: %{{y:.2f}} Å<extra></extra>",
                )
            )
            for b in boundaries[1:]:
                fig.add_shape(
                    type="line",
                    x0=b + 0.5, x1=b + 0.5,
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color="gray", dash="dot", width=1),
                    opacity=0.5,
                )

    if use_subplots:
        fig.update_layout(
            title={"text": "Per-Chain Cα RMSF (one subplot per complex)", "x": 0.5, "xanchor": "center"},
            font_family="Arial, sans-serif",
            font_size=16,
            showlegend=False,
            height=350 * len(datasets),
        )
    else:
        apply_layout(
            fig,
            "Per-Chain Cα RMSF (unified axis)",
            "Residue position (chain 1 then chain 2)",
            "RMSF (Å)",
            height=560,
        )
    return fig


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
    rmsf_mode: str = "single",
    rmsf_subplots: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[MDAnalysisResult]:
    """
    Run MD analysis.

    Parameters
    ----------
    rmsf_mode : str
        ``"single"`` (default) — standard whole-complex RMSF.
        ``"multichain"`` — per-chain alignment & concatenated axis, avoids
        inter-chain wobble inflating fluctuation values.
    rmsf_subplots : bool
        When ``rmsf_mode="multichain"``, whether to use one subplot per
        system (True) or overlay all on a single plot (False).
    """
    if not datasets:
        raise ValueError("Add at least one trajectory dataset.")
    if not plot_types:
        raise ValueError("Select at least one analysis type.")

    results: list[MDAnalysisResult] = []

    # ── Handle multi-chain RMSF separately ────────────────────────────────
    plot_types_without_rmsf = [pt for pt in plot_types if pt != "rmsf"]
    if "rmsf" in plot_types and rmsf_mode == "multichain":
        rmsf_fig = build_multichain_rmsf_figure(
            datasets, progress=progress, use_subplots=rmsf_subplots,
        )
        results.append(
            MDAnalysisResult(
                label="Combined — Per-Chain RMSF (multi-chain)",
                plot_type="rmsf",
                figure=rmsf_fig,
                summary="Multi‑chain RMSF: per-chain alignment, concatenated residue axis.",
            )
        )

    for plot_type in plot_types_without_rmsf:
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
