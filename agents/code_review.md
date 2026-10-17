# Code Against the Plan

A research quest has written its plan and then generated the code that carries it out. You did not write either. Before the code runs, read it as a referee would. Your job is to FIND PROBLEMS, not to agree or disagree: say whether it computes what the plan requires, and whether it computes it the way the plan defines it. Every problem goes under `findings` with where it is, what is wrong, why it is wrong and how to fix it. Everything you checked and found no problem with goes under `checked`, one line each saying what you looked at, so that a reply with no problem is never taken for one that did not look.

# Topic

$topic

# What the plan requires of the code

Each requirement has an id. A requirement is a quantity, a figure, a swept parameter or an equation the code must produce or use.

```json
$requirements
```

# What the plan says about how it is computed

The plan's model (its equations as written), its method, the settings it varies, the statistics it names, and the cases of its checks of correctness. It holds no expected value, and you must not write one.

```json
$plan
```

# The code (line numbers on the left)

$code

$previous

# What to judge, for each requirement

- **implemented**: the code computes it as the plan states, and (for a quantity) puts it in the result the script prints, or (for a figure) draws it and saves it under the planned name. Give the file and the line where you found it.
- **missing**: nothing in the code computes it, reports it or draws it. A name that is only mentioned in a comment, or computed and then discarded, is missing.
- **different**: the code has something under that name that is not what the plan says (another quantity, another definition, a different set of values than the plan's grid, a figure of something else).

Say "implemented" only after you found the line. Do not guess about code you were not shown.

# What to judge about the logic

Read the code against the plan for these, and put each fault you find under `findings`:

1. **Equations**: each function marked `# E<n>` computes its equation as the plan writes it: the same normalisation (which length, area, time or amount it is divided or multiplied by), the same units, the same factors and signs.
2. **Statistics**: each statistic is the one the plan defines, not a neighbour of it: a spread and not a multiple of it, a mean and not a sum, the same estimand, the interval method the plan names, the pooling the plan names.
3. **Settings**: a quantity the plan says depends on a setting really uses that setting, and is not rescaled or normalised so that the dependence cancels out.
4. **Steps**: every step the plan's model or method names is in the code (a stage, a term, a boundary or initial condition, a correction); none is skipped, replaced by a shortcut, or computed and then not used.
5. **Constants**: nothing the code reports is a constant: no returned quantity is a literal, or the same whatever the settings, unless the plan says it is. For such a finding give the name of the returned quantity in `quantity` and set `constant` to true.
6. **The checks' cases**: the code can be run on each check's case (its settings) and returns what that check's measure uses, and the case is the plan's, not another.

A finding says WHERE (file:line, such as `simulate.py:42`), WHAT is wrong, WHY it is wrong against the plan, and HOW to fix it. Name the term, the factor, the operation or the step. Do not write a value: never say what a number should be, never give an expected value or a tolerance; a fix is an operation ("divide by the cell area, not by the length"), not a result.

# What to return

Return ONE JSON object and nothing else:

{"requirements": [{"id": "<the requirement's id, exactly>", "status": "<implemented | missing | different>", "file": "<file name, or empty when missing>", "line": <line number, or null>, "note": "<for missing or different: what is missing or different, one sentence, naming no expected number; else empty>"}],
 "findings": [{"where": "<file:line>", "what": "<what is wrong, one sentence>", "why": "<why it is wrong against the plan, one sentence>", "fix": "<how to fix it, one sentence, no value>", "severity": "<high | medium | low>", "quantity": "<the returned quantity it concerns, or empty>", "constant": <true when the quantity is a constant, else false>}],
 "checked": ["<a requirement or a point of the logic you checked and found no problem with, and what you looked at, one line>"],
 "summary": "<one sentence>"}

Rules:

- Judge every requirement, once.
- `high` means the quantity the study reports would be wrong because of it; `low` is a matter of care that does not change a number.
- Do not answer "looks fine". A reply with no `findings`, no `checked`, and no requirement that says where it was found is no review and is asked again.
- Never write an expected value, a number the result should have, or a tolerance.
