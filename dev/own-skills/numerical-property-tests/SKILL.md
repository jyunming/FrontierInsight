---
name: numerical-property-tests
description: Test scientific code with generated inputs instead of one hand-picked case, using Hypothesis to check properties that must hold whatever the input. Use when a function should respect a symmetry, a scaling law, a limit, monotonicity, linear superposition or a round trip, and when a bug could hide at one special input such as zero, a boundary or a nearly singular case.
---
# Numerical property tests

One worked example shows the code works for that example. A property says what must be true for every valid input, and a generator searches for a case that breaks it.

## Requirements

`pip install hypothesis` (numpy is already there). A `@given` function can be called directly from the experiment script, no pytest needed.

## Steps

1. **Generate only valid physical inputs.** Bound every value to a range that means something: `floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)`.
2. **Pick the properties that apply:**
   - *symmetry*: reordering or relabelling the input leaves the output unchanged;
   - *scaling*: rescaling a quantity rescales the result by the factor the units say;
   - *limit*: at a known limit (zero coupling, infinite time) the result equals the known value;
   - *monotonic*: more of a cause never gives less of the effect, where the physics says so;
   - *superposition*: for a linear problem, the response to a sum is the sum of the responses;
   - *round trip*: a transform followed by its inverse returns the input.
3. **Compare with `rtol` and `atol`**, never `==`, and write why the tolerance is what it is.
4. **Test the edges on purpose** with `@example(...)`: zero, the smallest and largest valid values, nearly equal values, a nearly singular matrix.
5. **Keep it cheap and repeatable.** `@settings(max_examples=50, deadline=None, derandomize=True, database=None)` keeps the check inside the run's time budget, tests the same inputs every run, and leaves no `.hypothesis` folder behind.
6. **Print the smallest failing case.** Hypothesis shrinks a failure to a minimal input; show it so it can be reproduced.

## Mistakes to avoid

- Checking against a value computed by the same code.
- Unbounded floats, which mostly produce overflow and NaN.
- A property that is only approximately true, tested with an exact comparison.

## When NOT to use

Not for code that only reads data and plots it, or when each example needs an expensive run. There, choose two or three cases by hand.

The kinds of property follow the common catalogue in property-based testing (for example the Trail of Bits `property-based-testing` skill, CC BY-SA 4.0); nothing is copied from it.
