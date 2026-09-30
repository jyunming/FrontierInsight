---
title: The code project and run data
sources: [core/code_project.py, core/engine.py]
updated: 2026-09-30
---
# The code project and run data

## `code/` is a standalone project

Each quest's `code/` folder can be copied out and run without FI:

- `README.md`, `requirements.txt` (the exact versions that were installed), `run.py` (runs the simulation for every setting, then the analysis, with only the Python standard library) and `study.json` (the frozen grid).
- It has its own git history: every change (a run, a [[Refine]], an extension) is one commit and one `CHANGELOG.md` entry, kept across a [[Doing a step again|run from a step]].
- Before the paper is written FI checks, once per code version, that the project runs in a clean environment (`needs/CODE_PROJECT_CHECK.json`). This check only warns, and runs only with the venv sandbox and without background jobs.
- A file you edited by hand is never overwritten. With `pauses.review: ask` FI stops and asks; otherwise it logs a warning.

The web download zip leaves out `run_output/`.

## `data/results/` keeps the run's data

After an accepted run FI copies the data tables the run wrote (`.csv`, `.json`, `.npy`, `.parquet`, `.h5`, ... up to 200 MB each, never files that look like secrets) into `data/results/`, and for two-script quests the raw per-trial results into `data/results/raw/`. These are outputs, not input data. The layout redraw in [[Refine]] draws from them.

Related: [[Engine-callable simulations]], [[Scoring criteria]].
