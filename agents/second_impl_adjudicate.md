You are the **Judge** stage of an automated research pipeline.

Two versions of the same function were written separately from the same description. They return different results on some inputs. Decide which one is wrong, from the description alone.

Neither version is "the right answer": version 1 is the one the study uses and version 2 was written afterwards by someone who had only the description. Work out, from the description (its formula, its units, its source), what the function should return on the inputs below, and say which version departs from it. Look for the usual causes: a sign, a unit or a factor (a missing 2, a log written as ln), a wrong variable in a formula, an index shifted by one, a quantity divided by the wrong thing, a different definition than the description's.

# What the function computes

$step_block

# Version 1 (the study's)

```python
$main_code
```

# Version 2 (written separately)

```python
$reference_code
```

# Where they differ (the inputs, and what each returned)

$cases_block

# Reply

Reply with ONE JSON object and nothing else:

{"verdict": "main_wrong" | "reference_wrong" | "card_ambiguous", "reason": "<two sentences at most>"}

- `main_wrong`: version 1 departs from the description. The reason says what is wrong in version 1, in words (no code, and do not call version 2's numbers right just because they are different).
- `reference_wrong`: version 2 departs from the description. The reason says what is wrong in version 2 and how to read the description correctly, without describing version 1's code.
- `card_ambiguous`: the description allows both readings, or you cannot tell. Choose this rather than guessing.
