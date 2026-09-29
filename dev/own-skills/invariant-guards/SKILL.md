---
name: invariant-guards
description: Build checks on conserved or bounded quantities into a simulation script, so a wrong integrator or a silent blow-up fails the run instead of producing a plausible curve. Use when writing any time-stepping, ODE, PDE, particle, population or Monte Carlo code, and whenever a quantity such as mass, energy, momentum, probability, positivity or a symmetry should stay fixed or within bounds. The script reports each check as data the engine can read.
---
# Invariant guards

A run that finishes with no error is not evidence the physics is right. A wrong scheme often still returns numbers in a sensible range. Put the checks inside the script, next to the loop.

## Steps

1. **List the invariants before coding.** For this system, write each quantity that must stay fixed or bounded: total mass or population, energy, momentum, probability summing to 1, positivity of a density, a symmetry. Mark each one:
   - *exact*: the model conserves it exactly (drift should be near round-off);
   - *approximate*: the scheme conserves it only to its order (drift should shrink as the step shrinks).
2. **Measure drift.** Every N steps compute `drift = abs(I(t) - I(0)) / max(abs(I(0)), tiny)`. Keep the largest drift over the whole run, not just the last step.
3. **Set the tolerance from the scheme, and say why.** Exact: a few times machine epsilon times the number of steps. Approximate: the scheme's order times the step size, for example a second-order method with step 1e-3 allows about 1e-6. Write the reason next to the number. Never pick a tolerance because the run happens to pass.
4. **Fail loudly on NaN, Inf, or a violated bound** (negative density, probability above 1). Stop and report. Do not clip the value to hide it.
5. **Report the checks as data.** In the final `RESULT_JSON` line include `"invariants": [{"name": ..., "kind": "exact|approximate", "drift": ..., "tol": ..., "ok": true}]` so the engine can read them.

## Mistakes to avoid

- Checking an invariant only at the end. Drift is often small at first and large later.
- Computing the invariant from a variable the code rescales every step, so it cannot disagree with itself.
- A tolerance loose enough to pass a first-order method when the claim is a higher order.
- Reporting `ok: true` without a printed drift value.

## When NOT to use

Not for a system with no conserved or bounded quantity, or for a purely statistical analysis of existing data. Where an invariant is only approximate, do not treat a small non-zero drift as an error; compare it with the scheme's order.
