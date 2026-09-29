---
name: reference-cross-check
description: Compare a fast, vectorised or approximate implementation against a slow, obviously correct reference on small inputs, and prove the comparison can fail. Use when you write an optimised, vectorised, parallel, analytic-approximation or library-replacing version of a calculation, or when a result could be wrong in a way the fast code cannot reveal about itself.
---
# Reference cross-check

Fast code is easy to get subtly wrong (a transposed index, a dropped factor, a changed summation order). A second, simpler implementation of the same quantity catches that.

## Steps

1. **Write the reference first.** The plainest version that is clearly correct: explicit loops, a dense matrix, a textbook formula, or a standard library routine. It only has to run on small inputs.
2. **Prefer the strongest reference available:** a closed-form answer; an independent algorithm; an established library function; the same algorithm written differently (weakest, and say so).
3. **Compare on small random inputs.** Draw several inputs from the physically valid range, run both versions, and compare with a relative tolerance, never `==`. Take the tolerance from the arithmetic (machine epsilon times the problem size, times a condition number where one applies) and write why.
4. **Prove the check can fail.** In the same script, also compare the reference against a deliberately perturbed fast version (one constant times 1.01) and require that comparison to fail. A check that has never failed is not known to work. Print `detects_perturbation: true` or `false`.
5. **Print the outcome:** which reference was used, how independent it is, the largest difference, the tolerance, and `ok`. In a split experiment, where the trial function returns only numbers, return the largest difference as a numeric key and print the rest where the results are combined.

## Mistakes to avoid

- A reference that calls the code it is checking, or shares its helper functions.
- Comparing on one input, or only where both versions are trivially equal (zeros, identity).
- Loosening the tolerance until it passes.
- Calling a same-algorithm rewrite an independent check.

## When NOT to use

Not when the calculation is already the simplest possible form and no second version exists. Then check against a known analytic case or an invariant instead, and say that no reference was available.
