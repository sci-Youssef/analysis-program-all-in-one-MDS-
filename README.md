# MDBindAnalyzer

> Automated MM-GBSA/PBSA binding free energy analysis and MD trajectory visualization for one or more simulation datasets.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**MDBindAnalyzer** is a reproducible, extensible Python pipeline for analyzing molecular dynamics simulations with a focus on binding free energy estimation. Built for computational biochemists and structural biologists, it streamlines the workflow from raw MMPBSA/GBSA output to fully annotated, publication-ready figures.

## Features

- **Multi-dataset support** — compare one or more simulation datasets in a single run
- **MM-GBSA/PBSA parsing** — robust CSV parser for multi-section MMPBSA output files
- **Energy decomposition** — per-component analysis (ΔVdW, ΔEEL, ΔEGB, ΔESURF, ΔGGAS, ΔGSOLV, ΔG_total)
- **Statistical reporting** — mean, standard deviation, rolling average, and cumulative convergence
- **Interactive visualizations** — publication-quality HTML figures via Plotly
- **Modular design** — easily extendable to additional MD analysis tools

## Use Case

Designed to accelerate the post-processing step in structure-based drug discovery and protein–ligand binding studies using AMBER's MMPBSA.py or gmx_MMPBSA.