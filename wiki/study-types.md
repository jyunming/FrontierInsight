---
title: Study types
sources: [core/optimisation_plan.py, core/optimise.py, core/optimum_check.py, core/engine.py]
updated: 2026-09-30
---
# Study types

Every plan says which of two kinds of study it is (`study_type` in the plan's design block):

- **Measurement** (`measure`): how a result changes over settings chosen in advance. The protocol lists those settings as a grid, and every setting is run the same number of times. This is what FI runs end to end.
- **Find the best design** (`find_best_design`): the design that makes one quantity as low or as high as possible within limits, compared with a baseline design. Any design that carries an `optimisation` block is treated as this kind, whatever it says.

## What a best-design plan must say

The plan's "What is being optimised" section lists the objective (a quantity to minimise or maximise), the design variables with their ranges, what stays fixed, the constraints (`quantity <= limit`), a baseline with its source, the numerical settings for the search and the finer settings the best designs are checked again at, the evaluation budget, the search method (when none is given FI picks one: try every combination when the variables take few values and they all fit the budget; with a coarse scan, search the whole range first and then locally from the best point found; otherwise search locally inside the ranges from several starting points) and how much better a design must be to count as better (by default: more than the numerical error of the two designs). See [[scoring-criteria|Scoring criteria]] for the separate question of whether the code itself got better.

## How the type is decided

1. When the question could be either, the clarify step asks one question: measure, find the best design, or let the plan decide ([[pauses|Pauses]]).
2. At the plan step FI sets the type to find-the-best-design when you answered so, or when the topic plainly asks for a best design.
3. The engine runs the search itself (`core/optimise.py`, `core/optimise_search.py`): it evaluates the baseline first, then the optional coarse scan, then tries one design after another within `starts × per_start` evaluations and `execution.timeout_s`, calling the simulation's `run_cell` (or `run_trial`) in a process of its own. FI alone writes `raw/optimisation_ledger.jsonl` and `results/best_design.json`; infeasible and failed designs are never chosen. Then FI checks the best design (`core/optimum_check.py`): the best design of each starting point and the baseline are evaluated again at the finer check settings, the best design is nudged, and the starting points are compared; `needs/OPTIMUM_CHECK.json` holds one verdict (for example "the improvement over the baseline disappears at finer settings") and a sentence per check. A failed check never stops the quest; it keeps the evidence level lower ([[how-fi-judges-correctness|How FI judges correctness]]). A best-design plan that cannot start (no block or budget, one script, trials on a cluster) stops before anything is written or run, on every resume; FI never quietly runs a search as a sweep.

4. The paper gets a section *Best design found*, written by FI from those two records, not by the writer: the baseline against the best design at each numerical setting, the improvement with its numerical error, the search, each check in plain words, any target and the limits. `results/best_design.md` is the same text; the CLI, the web quest page and the VS Code chat show the best design in one line with its files.
5. Not satisfied? Refine with "push further", a target ("at most 46 K", "6 K better than the baseline") or "100 more evaluations" (`core/optimise_refine.py`): the search goes on from the best design so far and is checked again, and the paper says whether the target was reached. Changing the objective, a limit or a range is a new study and is refused with the reason.

`engine.clarify_overrides.study_type` in the YAML answers it in advance when the agent answers the setup questions itself; when you are asked them, it only fills in the suggested answer (and only if that question is asked), and with `pauses.clarify: off` it is not used.

## Turning a best-design plan into a measurement

- CLI: `fi --config q.yaml --resume <id> --revise-plan "make this a measurement over ..."`
- Web: the Plan box on the quest page.
- VS Code: `@fi /plan <id> <change>`.

Related: [[explore-then-confirm|Explore then confirm]], [[how-fi-judges-correctness|How FI judges correctness]], [[literature-screening-record|Literature screening record]].
