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

`settings_swept` are the settings the experiment varies, each with its values. `metrics` are the quantities it reports. `equations` are the formulas of the model the code will implement.

# What to decide, for each item

- A `factor` is **covered** when one of the `settings_swept` is that factor under another name or a part of one (a setting that carries it, such as a kind plus its parameters), so that varying that setting varies the factor. It is **missing** when no setting varies it, even when the plan's prose mentions it.
- A `quantity` is **covered** when a metric reports it (under another name or as part of one). Otherwise **missing**.
- A `formula` is **covered** when an equation states the same relation, up to notation: other symbol names, a rearrangement, an equivalent form. It is **replaced** when no equation states it but the plan says in an equation's `source` or in the assumptions that it uses another formula in its place, and why. Otherwise **missing**. A formula that looks like it but differs in a term, a factor or an argument is not the same: say **missing**.

# What to return

Return ONE JSON object and nothing else:

{"items": [{"id": "<the item's id>", "verdict": "<covered | replaced | missing>", "where": "<the setting, metric or equation that has it, or empty>", "why": "<one sentence>"}]}

# Rules

- One entry for every item. Say `missing` only for a reason you can state; if the plan has it, say where.
- Do not suggest how to fix the plan. Do not write any number.
