---
title: Oracle provenance
sources: [core/oracle_check.py, core/engine.py]
updated: 2026-09-30
---
# Oracle provenance

Where each oracle's expected value comes from. A check against a number the model made up is not a check, so each oracle's `reference` must be one of:

- **a derivation**: `derivation: <steps>` with real steps (at least 20 characters that state a relation, with `=`, `<`, `≈`, `→`, ...). "From Butcher 2008", "see the handbook" or "well-known value" do not count;
- **a source this quest retrieved**: its number `[n]` (also `[1, 3]` or `[1-3]`), its DOI or its title (five words or more);
- **an equation of the [[The model behind the numbers]]** (`E1`) whose own source counts;
- for a second implementation: a statement of what code it does not share with the simulation. This is required but not verified.

Anything else, such as a paper recalled from memory, is a gap. It shows in the plan's *Checks already made*, as a `run.log` warning, and it holds the result below *independently validated* ([[How FI judges correctness]]). Under `rigor_profile: research` the quest stops at the plan on every resume until the source is fixed.

The check runs when the plan is written or resumed, and again right before the protocol is frozen. The frozen record keeps the sources as they were numbered at that moment.

Why this exists: a real plan expected an error of 1.637e-08 for a Runge–Kutta step whose true error was 3.33e-07, with no source. The simulation was right and the check was wrong.
