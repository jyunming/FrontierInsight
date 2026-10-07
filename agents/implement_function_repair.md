You are repairing ONE function of the model package `$package` in an automated research pipeline.

Only this function is shown, with the equation it must compute and what is wrong with it. Do not change its signature (same parameter names, same order). Do not add scenario values (grid values, sizes, seeds, file paths) to it.

# The equation this function computes

$equation_block

# The function as it is now

```python
$function_block
```

# What is wrong with it

$problem

# Functions of the package this one depends on (signatures)

$related_block

# What to return

Reply with ONE fenced Python block holding the corrected function (and, if it needs them, import lines and small helper functions). Keep the equation's id in a comment on the line above the `def`. Compute the equation exactly as it is written above. Nothing else.
