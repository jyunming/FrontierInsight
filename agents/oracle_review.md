# Second Opinion on the Checks Against Known Answers

A research quest has written its plan. Before anything runs, the plan's **checks against known answers** (oracles) are the only thing that can show its simulation is right: FI runs the simulation on each check's small case, computes the check's number from what the simulation returns (`measure`, a returned name or a formula of them), and compares it with the value the plan expects. You did not write this plan. Read it as a referee would, and say whether these checks would catch a wrong simulation.

# Topic

$topic

# The model behind the numbers (what the simulation computes, and its equations)

```json
$model
```

# The checks the plan declares

```json
$oracles
```

Each kind has one numeric form: an `invariant`, a `symmetry` or a `second_implementation` is measured as the worst violation and expects 0; a `special_case`, a `published_value` or a `convergence_rate` is measured as the quantity itself and expects its known value.

# What to judge, for each check

1. **Appropriate**: does it test the model and the code that computes it, or something trivial (a number the code sets itself, an identity that holds whatever the code does, the input read back)?
2. **Discriminating**: name one plausible bug in the simulation (a wrong coefficient, a wrong sign, a first-order step where a fourth-order one is claimed, a missing term of an equation, a wrong boundary condition) and say whether this check, at its case and tolerance, would fail because of it.
3. **Its number is well defined**: its units and representation (a ratio or a violation, a per cent or a fraction, a sum or a mean), whether its form fits its kind, whether an expected value compared at a finite step is the value at that step and not the limit as the step goes to 0, and whether the tolerance is reachable by a correct method at that case and no tighter than the precision the expected value is written to.
4. **Its case isolates what its measure claims**: does the `case` produce a number that is the quantity the `measure` and the `check` text describe, and nothing else? Name what is wrong when it does not: a statistic computed over a region where the stated condition does not hold everywhere, a case that mixes two regimes, a measure computed from a different quantity than the expected value describes. Say "yes" only when you traced it; if you cannot tell, leave `isolates` empty.
5. **Better or additional**: a better check, or one more, when there is one; otherwise leave it empty.

Then say which equations of the model (their ids) the checks together test, and which none of them tests.

# What to judge, for each equation of the model

The plan's own model lists equations (`equations` above). The code will implement them exactly as written, and the checks' expected values are worked out from them, so an equation that is wrong in the plan makes everything after it wrong while every check still agrees. For each equation, say whether it is the **standard definition** of the quantity it names (including its normalisation: which length, area, time or amount it is divided or multiplied by, and which sign and units), or a correct derivation from the source it cites. If it is not, write the correct equation and give one line saying why. Say "no" only when you can state the correct equation. Say "yes" only when you checked it against the standard definition (or the derivation). If you cannot tell, leave `standard` empty: FI then records that nobody has verified this equation, which is better than a "yes" that was not checked.

# What to return

Return ONE JSON object and nothing else:

{"checks": [{"name": "<the check's name, exactly as the plan writes it>", "appropriate": "<yes | no>", "why": "<one sentence>", "discriminating": "<yes | no>", "bug_it_would_catch": "<the plausible bug you tried, and whether this check fails on it>", "well_defined": "<yes | no>", "definition_note": "<one sentence: what is unclear or wrong about its number, or empty>", "isolates": "<yes | no | empty when you cannot tell>", "isolation_note": "<when no: what the case mixes or leaves out, in one sentence; else empty>", "better": "<a better or additional check, in one sentence, or empty>"}],
 "equations_tested": ["<ids>"], "equations_not_tested": ["<ids of equations no check tests>"],
 "equations": [{"id": "<the equation's id, exactly as the plan writes it>", "standard": "<yes | no | empty when you cannot tell>", "correct": "<the correct equation, when standard is no; else empty>", "why": "<one line: what is different, or empty>"}],
 "add": [<zero to two new checks written exactly as the plan writes one: {"name", "kind", "check", "expected", "expected_formula", "tolerance", "tolerance_mode", "reference", "case", "measure"}>],
 "summary": "<one or two plain sentences for the person who will read the plan>"}

Rules:

- Judge only the checks and the equations: not the hypothesis, the grid or the statistics.
- Do not judge whether a check's `reference` names a source: FI checks that separately. Judge whether the expected value it gives is the right number for this check.
- Say "no" only for a reason you can state in the sentence beside it; a check that is fine gets "yes" and an empty note.
- A check you propose must be one FI can run: a small, fast `case` of the simulation, a `measure` computed from what it returns, a numeric `expected` worked out (and its `reference` saying how, as `derivation: <the steps>`), and a tolerance the method can reach at that case.
