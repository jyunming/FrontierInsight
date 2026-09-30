# Correctness Criteria

A research quest has written its plan, and the plan says nothing about how to judge whether the quest's code got better. The code will later be changed and a change kept only if it made nothing worse, so the quest needs two to five **checks of correctness**, fixed now, that FI can compute itself after every run of the code.

# Topic

$topic

# The protocol of the plan (what the experiment is held to)

```json
$protocol
```

# What the field uses to judge correctness for this kind of study

The sources this quest already found, and what one more search for how such studies are checked turned up:

$found

# What to return

Return ONE JSON object and nothing else:

{"criteria": [{"name": "<short name>", "what": "<one sentence a scientist in the field would read>", <one of the three sources below>, "direction": "<lower | higher | target>", "target": <a number: for lower the most it may be, for higher the least, for target the value aimed at; may be left out for lower or higher>, "tolerance": <a number of 0 or more: how close to the target counts as met, and how much a later version may change before it counts as worse>}]}

Each criterion takes its number from exactly one of:

- `"oracle": "<the name of a check in the protocol's oracles>"`, with `"use": "error"` (how far the check's measured value lands from the value it expects, the default) or `"use": "value"` (the measured value itself, such as an observed convergence order);
- `"case": {<the settings of one run>}, "measure": "<a number the simulation's run_trial or run_cell already returns for this study>"`: FI runs the simulation on that case itself;
- `"trials": "<a number every trial returns>"`: FI reads its own record of the trials and measures how fast that number's standard error shrinks as trials are added (0.5 when trials are independent; use `"direction": "target", "target": 0.5`).

Rules:

- Only checks of correctness: an error against a known answer, an observed order of convergence against the claimed one, the drift of a conserved quantity or an invariant, the agreement of two independent implementations, a standard error that shrinks as 1/sqrt(n). NEVER the study's own finding: a criterion on a number in the protocol's `metrics` or its `precision` is refused, because judging the code by its result would reward bending the code towards the result.
- FI computes each number itself; a number taken from the script's own printed results is refused.
- Give each criterion its own `tolerance`, from the method's known error or the noise of the measurement, not one percentage for all.
- If nothing FI can compute would say whether this code is right (the study runs no simulation, or no check above applies), return `{"criteria": []}` rather than inventing one.
