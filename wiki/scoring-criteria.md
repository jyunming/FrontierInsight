---
title: Scoring criteria
sources: [core/criteria.py, core/engine.py, core/plan.py, agents/plan_criteria.md]
updated: 2026-09-30
---
# Scoring criteria

The plan section **"How we will judge whether the code got better"** lists 2 to 5 checks of the code's correctness, fixed before the first full run. They are not the study's headline result; they answer "is this version of the simulation more correct than the last one?".

## What a criterion can measure

Each criterion has a name, one source and a direction (lower, higher, or towards a target), with its own tolerance:

- **An oracle** (a check with a known answer, see [[How FI judges correctness]]): by default the error |value − expected|, or the value itself.
- **A per-trial quantity** from the trials FI ran. One built-in measure is how independent the trials are (0.5 means independent); it needs at least 256 trials in settings of 32 or more.

A criterion may not take its number from the script's own result file, may not define its own case, and may not be about a headline metric: the code cannot grade itself.

## When it happens

1. At the plan step, if the plan has no usable criterion, FI does one literature search and asks the model once more.
2. Still none: with `pauses.plan: ask` (always the case under the research profile) the quest stops at the plan; otherwise it warns and goes on.
3. The criteria are frozen and hashed with the protocol; changing them is an amendment.
4. After **every** run FI computes them and appends a row to `.fi/criteria_history.jsonl` (run number, protocol version and hash, code commit, each value and whether it is met), and writes a `[criteria]` line to `run.log`.

Only a quest with two scripts (FI runs the simulation, see [[Engine-callable simulations]]) counts oracle values; one the script measured itself is shown but not counted. Nothing is decided from the history yet.

## Interfaces

The criteria appear in `plan.md` in all three interfaces. The history file has no viewer on the web page or in VS Code; read it from the quest folder.

Related: [[The code project and run data]], [[Explore then confirm]].
