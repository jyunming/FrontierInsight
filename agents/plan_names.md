You wrote the plan for a simulation study. Its checks, labels and method use the names below, but the plan does not list them among the numbers the study reports. Say what each one is. Do not change the plan.

How the study measures (from your plan):
$method

The settings the study changes:
$settings

The numbers the study reports:
$results

The names to explain, and where the plan uses each:
$names

The equations of the plan's model: $equations

For each name, give one answer:
- "result": a number the simulation returns for every run, and the study reports.
- "reference": a theory or exact value. Give "formula", written with the grid settings above, the plan's fixed settings and thresholds, and numbers only (a factor every run goes through together is not usable in it); FI computes it itself, and it must be, in FI's words: $formula_language. Or give "equation", the id of one of the model's equations above that gives it. FI does not check an equation against the name.
- "diagnostic": a number the code returns or computes only to check itself; not reported as a finding.
- "derived": not returned by a run; computed afterwards from the reported numbers (a width, a threshold, a fitted slope, a count of settings). Give "inputs", the reported numbers it is computed from.
For "result", "reference" and "derived" also give "what" (what it is, in a few words).
$again
Answer with one JSON object and nothing else:
{"names": {"<name>": {"answer": "<one of the four>", "what": "<words>", "formula": "<for a reference>", "equation": "<for a reference>", "inputs": ["<reported number>"]}}}
Give every name above, once, and leave out the keys that do not apply.
