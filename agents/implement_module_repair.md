You are repairing the model package `$package` of an automated research pipeline.

The package holds one function per equation. They were written one at a time and each was checked. Some functions pass their checks and some do not. Below are the functions that pass, the functions that do not (each with its equation and what is wrong), and the module as it is now (for reading only).

Rules:

- Return only the functions that do not pass. The functions that pass stay as they are: do not return them and do not change what they compute or their parameters.
- Every function keeps exactly the parameter names and order it has now. Keep the equation's id in a comment on the line above each function (for example `# E1`).
- Fix the failing functions by making each compute exactly its equation. Read the equation term by term (factors, signs, which quantity is divided or multiplied by which). If a failing function calls another function of the module, check that call too.
- No scenario values in the package (no grid values, sizes, seeds, thresholds or file paths); every parameter is an argument.
- Do not remove an import another function needs.

# The topic, as the person wrote it

This is what the study is for. The plan below was written from it; where the plan and the topic disagree about a formula, a setting or a quantity, the topic is the specification.

$topic_block

# The functions that already pass (they must keep passing)

$passing_block

# The functions to fix

$failing_block

# The module as it is now (for reading only)

```python
$module
```

# What to return

Reply with ONE fenced Python block holding the corrected versions of the functions to fix above, each whole, and any import or small helper function they newly need. Do not return the functions that pass and do not return the rest of the module. Nothing else.
