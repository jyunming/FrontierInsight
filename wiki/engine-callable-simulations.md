---
title: Engine-callable simulations
sources: [core/engine.py, core/trial_runner.py, core/oracle_check.py, core/evidence.py]
updated: 2026-09-30
---
# Engine-callable simulations

Every quest that runs code gets **two scripts** by default (`execution.split_analysis: auto`):

- `code/simulate.py` defines `run_cell(cell)` (a deterministic model) or `run_trial(cell, trial_id, seed)` (a random one). FI calls it itself, for every setting and every trial.
- The analysis script turns the raw results into the paper's numbers.

Because FI calls the simulation, it can also run each oracle's own case through it and compare with the known answer. This is what lets a result reach *independently validated* on the evidence level (how strongly a result is backed). With `split_analysis: false` there is one script, oracle values come from the script itself, and the result never reaches that level. See [[How FI judges correctness]].

A one-script study whose script has no random source is treated as deterministic: FI runs it once and reports no interval, instead of repeating identical runs.

## Equation labels

The plan's [[The model behind the numbers]] lists the equations that produce the numbers (E1, E2, ...). Each of them must be marked in the simulation code with a comment or docstring such as `# E1` or `# E1-E3`.

- At the end of the code step FI asks once for missing labels and keeps the answer only if the code itself did not change.
- Before the oracle check, a missing label is a warning and holds the result below independently validated. Under `rigor_profile: research` the quest pauses and asks for it.
- FI checks that the label is there, not that the mathematics is right.

Related: [[The code project and run data]], [[Scoring criteria]].
