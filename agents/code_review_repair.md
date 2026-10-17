# Code That Does Not Do What the Plan Requires

This code has NOT crashed, and it has not been run. A second model compared it with what the plan requires it to compute, and found the problems below. The code runs; it does not do what the plan requires. Your job is to change the code so that it does, for every finding, and to change nothing else.

# Topic

$topic

# The plan's equations (as the plan writes them)

```json
$equations
```

# The findings ($count of them, each in full)

Each finding says where it is, what is wrong, why it is wrong against the plan, and how to fix it. They are the other model's reading and can be wrong: if you read the code and a finding does not hold, leave that place as it is and say so in one line.

$findings

# The file(s) to change (as they are now)

$files

# How to answer

- Change exactly what the findings name, in the file(s) above, and nothing else. Keep every function, name, signature and entry point that is not named by a finding. Keep the handling of `FI_PILOT` and `FI_REPLICATE_SEED`, the final `RESULT_JSON: {...}` line (with the quantities the findings ask for in it), and every equation label comment (`# E1`) where it is.
- Compute each quantity the plan requires; never type in a value for it and never invent one. Put no scenario value (grid values, sizes, seeds, thresholds, file paths) into the model's package `$package`; it holds the equations, every parameter an argument.
- `experiment.py` never imports the model's package; it reads only the records FI hands it.
- Do not weaken a check, clamp or cap a result, or catch an error to hide it.

Reply with one fenced Python block for EACH file you change, whole, and nothing else. The first line of each block names the file, as it is named above:

```python
# file: simulate.py
...the whole file...
```
