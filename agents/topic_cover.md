# Is It In The Plan?

A research topic asks for some things. A plan was written for it. For each item below, say whether the plan contains it. You are not judging whether the plan is good, only whether each item is there.

# The topic

$topic

# The items to look for

```json
$items
```

# What the plan has

```json
$plan
```

`settings_swept` are the settings the experiment varies, each with its values. `metrics` are the quantities it reports. `equations` are the formulas of the model the code will implement. `searched_for` are the design variables of the plan's search for the best design (the values FI's search changes), empty when the plan has no search.

# What to decide, for each item

- A `factor` is **covered** when one of the `settings_swept` is that factor under another name or a part of one (a setting that carries it, such as a kind plus its parameters), so that varying that setting varies the factor; when the item gives `levels` or a `range`, the setting's values must also include them (a level may be coded by a number or given by an abbreviation, such as a method by its initials; say which value stands for it in `where`; a value that is a different level than the one stated is not that level, even when the setting has as many values as the topic states levels). It is **missing** when no setting varies it, or its values leave a stated level or part of the range out, even when the plan's prose mentions it.
- A `quantity` is **covered** when a metric reports it (under another name or as part of one). When the topic asks how well that quantity is estimated, fitted or recovered, a metric of its estimate (the estimate's error, bias, spread or how often its interval holds the true value) reports it too. Otherwise **missing**.
- A `formula` is **covered** when an equation states the same relation, up to notation: other symbol names, a rearrangement, an equivalent form. It is **replaced** when no equation states it but the plan says in an equation's `source` or in the assumptions that it uses another formula in its place, and why. Otherwise **missing**. A formula that looks like it but differs in a term, a factor or an argument is not the same: say **missing**.

- A `factor` may carry `searched_by_fi`: levels of it that are a design the plan's search finds (checked as a `search` item of its own). They are not values a setting must hold: judge the factor on its `levels` only.
- A `search` (a design the topic asks the study to search for) is **covered** only when one of `searched_for` is that design, or a part of it, under another name. A setting of `settings_swept`, even one whose value is worded as already optimised, is a design given in advance: say **missing**.
- An item may carry `plan_says_it_is_covered_by`: the plan names that setting, metric or equation as covering it. Say **covered** only if that element really represents the item (the setting is the factor itself, or carries it; the equation implements the formula up to notation). Values or an id that merely fit are not enough: say **missing** and give the reason.

# What to return

Return ONE JSON object and nothing else:

{"items": [{"id": "<the item's id>", "verdict": "<covered | replaced | missing>", "where": "<the setting, metric or equation that has it, or empty>", "why": "<one sentence>"}]}

# Rules

- One entry for every item. Say `missing` only for a reason you can state; if the plan has it, say where.
- Do not suggest how to fix the plan. Do not write any number.
