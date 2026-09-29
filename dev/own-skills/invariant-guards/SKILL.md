---
name: invariant-guards
description: Build checks on conserved or bounded quantities into a simulation script, so a wrong integrator or a silent blow-up fails the run instead of producing a plausible curve. Use when writing any time-stepping, ODE, PDE, particle, population or Monte Carlo code, and whenever a quantity such as mass, energy, momentum, probability, positivity or a symmetry should stay fixed or within bounds. Each check is printed with its drift and tolerance.
---
# Invariant guards

A run that finishes with no error is not evidence the physics is right. A wrong scheme often still returns numbers in a sensible range. Put the checks inside the script, next to the loop.

## Steps

1. **List the invariants before coding.** For this system, write each quantity that must stay fixed or bounded: total mass or population, energy, momentum, probability summing to 1, positivity of a density, a symmetry. Mark each one:
   - *exact*: the scheme conserves it by construction (a conservative finite-volume update, the total of a stochastic SIR). Drift should be near round-off.
   - *approximate*: the model conserves it but the scheme only to its order (energy under Euler or Runge-Kutta). Drift is truncation error and shrinks with the step.
2. **Measure drift.** Every N steps compute `drift = abs(I(t) - I(0)) / scale`, where `scale` is a size of the quantity itself (the initial value, or for one that starts at zero such as net momentum, the sum of the absolute parts). Keep the largest drift over the whole run, not just the last step.
3. **Set the tolerance from the scheme, and say why.** Exact: a few times machine epsilon times the number of steps. Approximate: run at step h and at h/2 and require the drift to fall by about 2^p for a method of order p, or take a constant measured on a coarse run times h^p. Write the reason next to the number. Never pick a tolerance because the run happens to pass.
4. **Fail loudly on NaN, Inf, or a violated bound** (negative density, probability above 1). Stop and report. Do not clip the value to hide it.
5. **Print each check** as `name, kind, drift, tolerance, ok` so the run log shows it. In a split experiment, where the trial function returns only numbers, return the drift as a numeric key (for example `energy_drift`) and do the tolerance test where the results are combined.

## Mistakes to avoid

- Checking an invariant only at the end. Drift is often small at first and large later.
- Computing the invariant from a variable the code rescales every step, so it cannot disagree with itself.
- A tolerance loose enough to pass a first-order method when the claim is a higher order.
- Saying a check passed without a printed drift value.

## When NOT to use

Not for a system with no conserved or bounded quantity, or for a purely statistical analysis of existing data. Where an invariant is only approximate, do not treat a small non-zero drift as an error; compare it with the scheme's order.
