---
title: Study types
sources: [core/optimisation_plan.py, core/engine.py]
updated: 2026-09-30
---
# Study types

Every plan says which of two kinds of study it is (`study_type` in the plan's design block):

- **Measurement** (`measure`): how a result changes over settings chosen in advance. The protocol lists those settings as a grid, and every setting is run the same number of times. This is what FI runs end to end.
- **Find the best design** (`find_best_design`): the design that makes one quantity as low or as high as possible within limits, compared with a baseline design. Any design that carries an `optimisation` block is treated as this kind, whatever it says.

## What a best-design plan must say

The plan's "What is being optimised" section lists the objective (a quantity to minimise or maximise), the design variables with their ranges, what stays fixed, the constraints (`quantity <= limit`), a baseline with its source, the numerical settings for the search and for the finer check levels, the evaluation budget, the search method (FI picks exhaustive, global-then-local or bounded local from the budget when none is given) and how much better a design must be to count as better (by default: more than the numerical error of the two designs). See [[Scoring criteria]] for the separate question of whether the code itself got better.

## How the type is decided

1. When the question could be either, the clarify step asks one question: measure, find the best design, or let the plan decide ([[Pauses]]).
2. At the plan step FI sets the type to find-the-best-design when you answered so, or when the topic plainly asks for a best design.
3. At the design step a quest that runs code with a best-design plan **stops before anything is written or run**, on every resume. The search runner is not built yet, so FI never quietly runs a search as a sweep.

`engine.clarify_overrides.study_type` in the YAML sets the answer in advance.

## Turning a best-design plan into a measurement

- CLI: `fi --config q.yaml --resume <id> --revise-plan "make this a measurement over ..."`
- Web: the Plan box on the quest page.
- VS Code: `@fi /plan <id> <change>`.

Related: [[Explore then confirm]], [[How FI judges correctness]].
