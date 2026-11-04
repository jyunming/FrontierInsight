# Code Against the Plan

A research quest has written its plan and then generated the code that carries it out. You did not write either. Before the code runs, read it as a referee would. Your job is to FIND PROBLEMS, not to agree or disagree: say whether it computes what the plan requires, and whether it computes it the way the plan defines it. Every problem goes under `findings` with where it is, what is wrong, why it is wrong and how to fix it. Everything you checked and found no problem with goes under `checked`, one line each saying what you looked at, so that a reply with no problem is never taken for one that did not look. Below, in this order: what to judge and what to return, the requirements, the plan, the methodology audit's proposals (they are not the plan), the code, and what FI saw when it ran the code.

# What to judge, for each requirement

- **implemented**: the code computes it as the plan states, and (for a quantity) puts it in the result the script prints, or (for a figure) draws it and saves it under the planned name. Give the file and the line where you found it.
- **missing**: nothing in the code computes it, reports it or draws it. A name that is only mentioned in a comment, or computed and then discarded, is missing.
- **different**: the code has something under that name that is not what the plan says (another quantity, another definition, a different set of values than the plan's grid, a figure of something else).

Say "implemented" only after you found the line. Do not guess about code you were not shown.

**The plan's frozen settings are its definition of those settings.** The `grid` and `in_each_run` sections hold the values the study runs at, and FI gives them to the code as they are. If the plan's prose (a metric's estimand, the method, an assumption) describes the same settings another way, for example as a derivation, a rounding or another number, the values in `grid` / `in_each_run` are what the plan defines: code that takes them as they are is **implemented**, not "different". Report such a disagreement once, as a finding with `plan_conflict` true (it is not a fault of the code). Code that does NOT use the frozen values as given (a value written into the code, another list, a rescaled one) is still "different", and so is any other departure from the plan, such as a different statistic.

# What to judge about the logic

Read the code against the plan for these, and put each fault you find under `findings`:

1. **Equations**: each function marked `# E<n>` computes its equation as the plan writes it: the same normalisation (which length, area, time or amount it is divided or multiplied by), the same units, the same factors and signs.
2. **Statistics**: each statistic is the one the plan defines, not a neighbour of it: a spread and not a multiple of it, a mean and not a sum, the same estimand, the interval method the plan names, the pooling the plan names.
3. **Settings**: a quantity the plan says depends on a setting really uses that setting, and is not rescaled or normalised so that the dependence cancels out.
4. **Steps**: every step the plan's model or method names is in the code (a stage, a term, a boundary or initial condition, a correction); none is skipped, replaced by a shortcut, or computed and then not used.
5. **Constants**: nothing the code reports is a constant: no returned quantity is a literal, or the same whatever the settings, unless the plan says it is. For such a finding give the name of the returned quantity in `quantity` and set `constant` to true.
6. **The checks' cases**: the code can be run on each check's case (its settings) and returns what that check's measure uses, and the case is the plan's, not another.

Some lines are FI's, not the study's: any lines between a `# >>> FI` line and its `# <<< FI` line, and the files FI writes whole (`experiment.py`, `run.py`, `fi_*.py`, the `fi_core` folder). Do not report a finding about them; read only the study's own functions. FI sets aside a finding whose place is in its lines.

A finding says WHERE (file:line, such as `simulate.py:42`), WHAT is wrong, WHY it is wrong against the plan, and HOW to fix it. Name the term, the factor, the operation or the step. Do not write a value: never say what a number should be, never give an expected value or a tolerance; a fix is an operation ("divide by the cell area, not by the length"), not a result.

# What to return

Return ONE JSON object and nothing else:

{"requirements": [{"id": "<the requirement's id, exactly>", "status": "<implemented | missing | different>", "file": "<file name, or empty when missing>", "line": <line number, or null>, "note": "<for missing or different: what is missing or different, one sentence, naming no expected number; else empty>", "changes_result": <true when a missing or different requirement changes a number or figure the study reports, else false>}],
 "findings": [{"where": "<file:line>", "what": "<what is wrong, one sentence>", "why": "<why it is wrong against the plan, one sentence>", "fix": "<how to fix it, one sentence, no value>", "severity": "<high | medium | low>", "quantity": "<the returned quantity, figure, setting or requirement it concerns, as the plan names it, or empty>", "constant": <true when the quantity is a constant, else false>, "changes_result": <true when the fault can change a number or a figure the study reports, false when it changes nothing reported>, "plan_conflict": <true only when the fault is that the PLAN contradicts itself, else false>}],
 "checked": ["<a requirement or a point of the logic you checked and found no problem with, and what you looked at, one line>"],
 "summary": "<one sentence>"}

Rules:

- Judge every requirement, once.
- List EVERY finding you have, with no limit on how many, and put the ones that change a reported number or figure first, then the others by severity. Do not leave a finding out because the list is long.
- One finding per fault. Do not list the same fault twice in other words, and do not list a fault under `findings` and again as a different requirement: put it once, with the quantity it concerns in `quantity`.
- `changes_result` is true only when, if the fault were left as it is, a number or figure the study reports would be wrong or missing. A dead constant, a missing optional output, a weak internal check or a style point is false.
- `high` means the quantity the study reports would be wrong because of it; `low` is a matter of care that does not change a number.
- `plan_conflict`: when two parts of the plan define one thing differently (for example a metric's definition and the method's text say different things), the code cannot satisfy both. Report that as ONE finding with `plan_conflict` true, say in `why` which two parts disagree and what each says, and give no fix for the code: it is not a fault of the code.
- Do not answer "looks fine". A reply with no `findings`, no `checked`, and no requirement that says where it was found is no review and is asked again.
- Never write an expected value, a number the result should have, or a tolerance.

# Topic (for context only)

$topic

# The requirements: what the plan requires of the code

Each requirement has an id. A requirement is a quantity, a figure, a swept parameter or an equation the code must produce or use.

```json
$requirements
```

# The plan: what it says about how it is computed (most important sections first)

The plan's model (its equations as written), its method, the settings it varies, the statistics it names, and the cases of its checks of correctness. It holds no expected value, and you must not write one.

```json
$plan
```

# The methodology audit's proposals (these are NOT the plan)

$open_items

# The code (line numbers on the left): read it against the plan above

$code

# What the code gave when FI ran it

FI ran the simulation itself on a few of the plan's settings (a few trials each) before this reading. Read the code with these results: a value that cannot be right for what the plan defines, a setting where trials fail, or a quantity that does not change where the plan says it must, points to a problem in the code. They are a small case, not the study's results.

$small_run

$previous
