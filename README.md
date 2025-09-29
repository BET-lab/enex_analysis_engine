# ENEX Analysis Engine

Exergy-focused thermodynamic modeling for building energy systems

## Overview

**ENEX Analysis Engine** provides component models for electric boilers, gas boilers, and heat pumps, along with utilities to compute energy, entropy, and exergy balances consistently across a system. It targets teaching, research, and prototyping for building energy systems where second-law (exergy) accounting matters

### Key capabilities

- Component models for typical heating technologies: electric boiler, gas boiler, heat pump  
- Built-in bookkeeping of energy, entropy, and exergy balances for each component  
- Utilities for common unit conversions to keep calculations consistent  
- Simple, composable API suitable for notebooks and scripts

## Project structure

Key modules

- `constant.py` — conversion constants and helper functions for temperature, length, energy, and power  
- `enex_engine.py` — component models and thermodynamic balance routines for electric boiler, gas boiler, and heat pump  

> The project uses a standard `src` layout with the Python package located under `src/enex_analysis` and is managed with a `pyproject.toml` and a `uv.lock` for reproducible environments

## Installation

### Option A — Work on the project locally (recommended for contributors)

```bash
# 1) Install uv (Windows PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2) Clone the repository
git clone https://github.com/BET-lab/enex_analysis_engine.git
cd enex_analysis_engine

# 3) Create and sync the virtual environment from pyproject + uv.lock
uv sync
