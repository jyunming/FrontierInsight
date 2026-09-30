---
title: The model behind the numbers
sources: [core/plan.py, core/oracle_check.py, core/engine.py]
updated: 2026-09-30
---
# The model behind the numbers

For a quest that runs code, the plan states the model that produces the numbers before any code is written. It is frozen with the protocol.

- a one-sentence **summary**, the **assumptions**, and **where it holds**;
- the **equations** E1, E2, ...: each with its formula, its role (`generates`: the simulation computes the data with it; `analyses`: used on the results) and its source (a retrieved paper `[n]`, or a derivation).

`plan.md` shows it under *The model behind the numbers*, next to *The sources this quest found* (numbered). Each oracle follows with its kind, expected value and where that value comes from ([[oracle-provenance|Oracle provenance]]). The plan's *Checks already made* list names what is missing: no model, no summary, no assumptions, no equations, an equation with no role or no checkable source, or an oracle of an unknown kind.

The section is only shown from the design block and never read back; a rewrite of the plan shows it again from the block. Parts a model wrote under other names are read and shown: `description` for the summary, `assumes` for the assumptions, `governing_equations` for the equations, and equations written as lines (`E1: ...`) or as a mapping of E1 to its formula. Under `rigor_profile: research` an empty model is one of the things FI asks the plan step once to fill in before the plan stop.

Each `generates` equation must be labelled in the simulation code (`# E1`); see [[engine-callable-simulations|Engine-callable simulations]].

## Using it

- CLI: read or edit `outputs/<quest>/plan.md`; `fi --config q.yaml --resume <id> --revise-plan "<change>"`. After the freeze a change is an amendment: `--approve-amendment <id> --approve-as <you>`.
- Web: the Plan panel on the quest page (edit, or ask for a change); the Approve button for an amendment.
- VS Code: `@fi /plan <id>` opens it, `@fi /plan <id> <change>` rewrites it, `@fi /approve-amendment <id>`.

Related: [[how-fi-judges-correctness|How FI judges correctness]], [[scoring-criteria|Scoring criteria]].
