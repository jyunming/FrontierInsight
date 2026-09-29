---
name: dimensional-consistency
description: Write the units next to every formula and check that both sides agree and that anything inside exp, log or a trigonometric function has no units. Use when translating an equation into code, mixing quantities from different sources, or when a result differs from the literature by a suspicious factor such as 2, pi, 1000 or a power of ten.
---
# Dimensional consistency

This catches a wrong formula before any number is computed. Unit conversion and propagation belong to `uncertainty-and-units`.

## Steps

1. **Annotate.** Put the unit in a comment beside every constant and every line of the formula, for example `# m/s`. A quantity with no written unit has not been checked.
2. **Check both sides.** Reduce each equation's two sides to base units. If they differ, the formula or a constant is wrong.
3. **Check the arguments.** Whatever goes into `exp`, `log`, `sin`, `cos` or `tanh` must have no units. If it does, a scale is missing.
4. **Check the limits.** Form a ratio with no units (kT/E, length over wavelength) and send it to zero or infinity; confirm the formula gives the answer the physics gives there.
5. **Chase a suspect factor.** A result off by 2, 4, pi, 1000 or a power of ten is usually a unit, a definition (radius against diameter, hertz against angular frequency) or a missing constant. Find which before touching the numbers.
6. **Convert once, at the edges.** Read inputs into one unit system, compute in it, convert only when writing outputs.

## When NOT to use

Not for purely dimensionless models, or data analysis where every column already carries a stated unit and no formula is being written.
