---
title: The code project and run data
sources: [core/code_project.py, core/changelog.py, core/engine.py]
updated: 2026-09-30
---
# The code project and run data

## `code/` is a standalone project

Each quest's `code/` folder can be copied out and run without FI:

- `README.md`, `requirements.txt` (the exact versions that were installed), `run.py` (runs the simulation for every setting, then the analysis, with only the Python standard library) and `study.json` (the frozen grid).
- It has its own git history: every change (a run, a [[refine|Refine]], an extension) is one commit and one `CHANGELOG.md` entry, kept across a [[doing-a-step-again|run from a step]].
- Each `CHANGELOG.md` entry names its kind of change, decided by the step that made it, never by the model: **Added** (the first code, a number a refine asked for), **Changed** (a redesign, an improve round, an earlier version put back, an edit FI did not make), **Fixed** (a crash fix, a repair by the check against known answers, a number the review found wrong), **Tidied** (only files FI's own runs never read changed: `README.md`, `requirements.txt`, `run.py`, `study.json`, `METHODS.md`, `tests/test_oracles.py`).
- Each entry also says **whether the results changed**: the [[scoring-criteria|checks of correctness]] before and after, and whether the study's results changed, by name only (never their values). The line says "not measured yet" until a run of that code finishes with results, then it is filled in from `.fi/criteria_history.jsonl` (which keeps one hash per result for each full run). A run that measured several changes at once says so; a change undone before any run (an improve round given up, a loop cut short) says it was not measured (`core/changelog.py`).
- Before the paper is written FI checks, once per code version, that the project runs in a clean environment (`needs/CODE_PROJECT_CHECK.json`). This check only warns, and runs only with the venv sandbox and without background jobs.
- A project file FI keeps up to date (`README.md`, `requirements.txt`, `run.py`, `study.json`) that you edited by hand is never overwritten: with `pauses.review: ask` FI stops and asks; otherwise it keeps your version and logs a warning. The scripts themselves are rewritten when FI repairs or extends them.

The web download zip leaves out `run_output/`.

## `data/results/` keeps the run's data

After an accepted run FI copies the data tables the run wrote (`.csv`, `.json`, `.npy`, `.parquet`, `.h5`, ... up to 200 MB each, never files that look like secrets) into `data/results/`, and for two-script quests the raw per-trial results into `data/results/raw/`. These are outputs, not input data. The layout redraw in [[refine|Refine]] draws from them.

Related: [[engine-callable-simulations|Engine-callable simulations]], [[scoring-criteria|Scoring criteria]].
