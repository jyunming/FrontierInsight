# Compute a Known Answer Numerically, From the Definitions

A research quest checks its simulation against a **known answer** on one small case. Two workings of that answer disagree and could not be settled by reasoning. FI now decides by computing the number itself, numerically, from the definitions: not from anyone's formula for the answer. You write the small script that does it. You are not told either working or what the simulation produced.

# Topic

$topic

# The model behind the numbers (its definitions)

$model

# The check

- Name: `$name`
- What it checks: $check
- The case it runs on: $case
- The number it compares: `$measure`

# What to do

Write ONE short, self-contained Python script that computes this number on this case **by brute numerical means from the definitions above**: sum, integrate or enumerate the defining expression directly (a fine grid, many samples, an explicit sum over every term), never a simplified closed form you derived for the answer. Use only the Python standard library, numpy and scipy. Do not import, read or run anything else (no file, no other script, no simulation code). Make the numerical error small compared with the check's own scale (refine until the result stops changing) and say how small.

The script's LAST line of output must be:

```
$line {"value": <the number>, "how": "<the method and its numerical error, in one sentence>"}
```

Reply with the script in one fenced Python block and nothing else.
