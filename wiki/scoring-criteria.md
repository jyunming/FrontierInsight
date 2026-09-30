---
title: Scoring criteria
sources: [core/criteria.py, core/engine.py, core/plan.py, agents/plan_criteria.md]
updated: 2026-09-30
---
# Scoring criteria

The plan section **"How we will judge whether the code got better"** lists 2 to 5 checks of the code's correctness, frozen with the protocol before the first full run (with `engine.phased: true`, when exploration ends; see [[Explore then confirm]]). They are not the study's headline result; they answer "is this version of the simulation more correct than the last one?".

## What a criterion can measure

Each criterion has a name, one source and a direction (lower, higher, or towards a target), with its own tolerance:

- **An oracle** (a check with a known answer, see [[How FI judges correctness]]): by default the error |value − expected|, or the value itself.
- **A per-trial quantity** from the trials FI ran. One built-in measure is how independent the trials are: how fast the uncertainty of the average shrinks as trials are added, as a power of the number of trials (0.5, the square-root rule, means the trials are independent; less means they repeat or depend on each other); it needs at least 256 trials in settings of 32 or more.

A criterion may not take its number from the script's own result file, may not define its own case, and may not be about a headline metric: the code cannot grade itself.

## When it happens

1. At the plan step, if the plan has no criterion FI can measure itself, FI does one literature search and asks the model once more. (A one-script quest skips this: FI runs none of its checks itself, so it only warns that its criteria are shown but none counts.)
2. Still none: with `pauses.plan: ask` (always the case under the research profile) the quest stops at the plan; otherwise it warns and goes on.
3. The criteria are frozen and hashed with the protocol; changing them is an amendment.
4. After **every** run FI computes them and appends a row to `.fi/criteria_history.jsonl` (run number, protocol version and hash, code commit, each value and whether it is met), and writes a `[criteria]` line to `run.log`.

Only a value FI measured itself counts: an oracle with a `case`, run through a simulation FI can call (one with a `run_trial` or `run_cell` function, see [[Engine-callable simulations]]); one the script measured itself is shown but not counted. Nothing is decided from the history yet.

## Interfaces

The criteria appear in `plan.md` in all three interfaces. The history file has no viewer on the web page or in VS Code; read it from the quest folder.

Related: [[The code project and run data]], [[Explore then confirm]].
