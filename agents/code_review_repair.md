# Code That Does Not Do What the Plan Requires

This code has NOT crashed. A second model compared it with what the plan requires and reported the ONE problem below, and a test written for it fails on the code as it is now. The code runs; it does not do what the plan requires. Change the code so that it does, for this problem only, and change nothing else.

# Topic

$topic

# The plan's equations (as the plan writes them)

```json
$equations
```

# The finding

The other model's reading; it can be wrong, but the test below fails, so the problem is real.

$finding

# The test that fails now (it must pass afterwards)

```python
$test
```

It fails with: $failure

# The code

$code

# How to answer

- Change what the finding names and nothing else. Keep every function, name, parameter, entry point and equation label comment (`# E1`) that is not part of the problem, the handling of `FI_PILOT` and `FI_REPLICATE_SEED`, and the final `RESULT_JSON: {...}` line.
- Compute each quantity the plan requires; never type in a value for it. Put no scenario value (grid values, sizes, seeds, thresholds, file paths) into the model's package `$package`; every parameter is an argument. `experiment.py` never imports the model's package.
- Do not weaken a check, clamp or cap a result, or catch an error to hide it. Do not edit the test.
- Other tests of the code pass now and must still pass: do not change what a function computes unless the finding says it is wrong.

# What went wrong with the last try

$problem

# What to return

$how Nothing else.
