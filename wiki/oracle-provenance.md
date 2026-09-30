---
title: Oracle provenance
sources: [core/oracle_check.py, core/engine.py, core/accepted_checks.py]
updated: 2026-09-30
---
# Oracle provenance

Where each oracle's expected value comes from. A check against a number the model made up is not a check, so each oracle's `reference` must be one of:

- **a derivation**: `derivation: ...` with at least one equation (`=`, or a relation such as `≈` or `<`) that shows how the value follows (`integral of S over the domain = 1, because S is normalised`; a value true by definition is written the same way, `sum S = 1 by definition`). At least 20 characters (12 for a fact stated "by definition" or "by construction") and a relation (`=`, `<`, `≈`, `→`, ...). "From Butcher 2008", "see the handbook" or "well-known value" do not count, and neither does a plain sentence with no equation;
- **a source this quest retrieved**: its number `[n]` (also `[1, 3]` or `[1-3]`), its DOI or its full title (a title of five words or more);
- **an equation of the [[model-behind-the-numbers|The model behind the numbers]]** (`E1`) whose own source counts;
- for a second implementation: a statement of what code it does not share with the simulation. This is required but not verified.

What a model writes in other shapes is read, not lost: a reference under `citation` or `justification`, under `source` or `basis` only when it looks like one (`[n]`, a DOI, a derivation, `E1`), or inside the check's text (`(derivation: ...)`); a kind under `type`, or in words such as "normalization" (an invariant) or "horizontal symmetry" (a symmetry). The plan keeps what was written; only what the engine reads is mapped, so a frozen protocol hashes as before. One check that cannot be read is left out of the plan with a note naming it, not every check.

Anything else, such as a paper recalled from memory, is a gap. It shows in the plan's *Checks already made*, as a `run.log` warning, and it holds the result below *independently validated* ([[how-fi-judges-correctness|How FI judges correctness]]).

## Under `rigor_profile: research`

1. Before stopping, FI asks the plan step **once** per quest to fill in exactly what is missing (each check's `reference` and `kind`, and an empty model). A rewrite that changes anything else, such as a check's numbers, is put back.
2. If it is still missing, the quest stops at the plan on every resume, and the stop offers three ways on:
   - **let FI fill it in**: `--revise-plan "fill in where each check's expected value comes from"` (web Plan box, `@fi /plan <id> ...`). Every rewrite is told what FI reads in the block and what is missing, and afterwards the CLI says check by check what is still missing;
   - **change it yourself**: edit the `reference` in `plan.md`, or say what to change in words;
   - **go on as it is**: `--accept-checks <id> --approve-as <you>` (web *Go on as it is* button, `@fi /accept-checks <id>`). A name is required. The checks still run and are still judged. Each is marked "source not confirmed" in the evidence, which keeps the gap; the choice is in the audit trace and the frozen protocol's approval line; the paper is told to say so. It covers only the checks the stop named.
3. A later stop says which plan version arrived since, which checks are now fine and what each remaining one lacks.

`@fi /resume <id> --revise-plan "..."` rewrites the plan like `@fi /plan`. It used to be read as a plain resume, which ran the quest into the same stop.

The check runs when the plan is written or resumed, and again right before the protocol is frozen. The frozen record keeps the sources as they were numbered at that moment.

Why this exists: a real plan expected an error of 1.637e-08 for a Runge–Kutta step whose true error was 3.33e-07, with no source. The simulation was right and the check was wrong. A later quest (a source-optimisation study) showed the other side: its checks' values were derivable (a normalised source sums to 1, a symmetric one has a zero centroid), but its references and model were lost to shapes the plan did not read, and it had no way on but editing `plan.md` by hand.
