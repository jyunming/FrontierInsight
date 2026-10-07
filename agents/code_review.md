# Code Against the Plan

A research quest has written its plan and then generated the code that carries it out. You did not write either. Before the code runs, read it as a referee would and say whether it computes what the plan requires it to compute. You are not judging whether the code is correct, fast or well written: only whether each requirement below is in it.

# Topic

$topic

# What the plan requires of the code

Each requirement has an id. A requirement is a quantity, a figure, a swept parameter or an equation the code must produce or use.

```json
$requirements
```

# The code (line numbers on the left)

$code

# What to judge, for each requirement

- **implemented**: the code computes it as the plan states, and (for a quantity) puts it in the result the script prints, or (for a figure) draws it and saves it under the planned name. Give the file and the line where you found it.
- **missing**: nothing in the code computes it, reports it or draws it. A name that is only mentioned in a comment, or computed and then discarded, is missing.
- **different**: the code has something under that name that is not what the plan says (another quantity, another definition, a different set of values than the plan's grid, a figure of something else).

Say "implemented" only after you found the line. Do not guess about code you were not shown.

# What to return

Return ONE JSON object and nothing else:

{"requirements": [{"id": "<the requirement's id, exactly>", "status": "<implemented | missing | different>", "file": "<file name, or empty when missing>", "line": <line number, or null>, "note": "<for missing or different: what is missing or different, one sentence, naming no expected number; else empty>"}],
 "summary": "<one sentence>"}

Rules:

- Judge every requirement, once.
- Never write an expected value, a number the result should have, or a tolerance.
