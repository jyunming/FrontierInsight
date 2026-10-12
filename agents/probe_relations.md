You read a research plan and list the RELATIONS it states between runs of its simulation. A relation says how a result
must move (or not move) when one setting changes. FI will run the simulation itself and test each relation with
statistics. You never give a number the simulation should produce; if you write one, it is thrown away.

# The plan

$plan_text

# What the simulation can vary (settings, each with the values it takes)

$grid_settings

# What the simulation reports (metrics)

$metrics_list

# What to extract

Only relations that the plan itself states, or that are general facts of the method the plan uses (for example: a smaller
step size gives a smaller numerical error; averaging more samples shrinks the spread). Do not invent relations to fill the
list. At most 8, the most important first. Every relation needs a short quote from the plan (or a few words naming the
method) in `from_plan`. Use only setting names from the list above and only metric ids from the list above; a name that is
not there is dropped.

Kinds (use exactly these words):

- `sensitivity`: the metric changes when `setting` changes.
- `invariance`: the metric does not depend on `setting`.
- `monotone`: the metric rises or falls as `setting` grows; give `direction` as `up` or `down`.
- `equal_when`: two cells give the same metric (a null case, a symmetry, a limit that reduces to another case); give `a`
  and `b`, each a mapping from setting name to a value inside that setting's range.
- `scaling`: multiplying `setting` by `factor` multiplies the metric by `factor` to the power `exponent`.
- `convergence`: the error shrinks as the knob `setting` is refined; give `order` (the plan's stated order, a positive
  number), `refine` (`decrease` when a smaller value is finer, such as a step size; `increase` when a larger value is
  finer, such as a count) and optionally `of` (`spread` when the quantity that shrinks is the seed-to-seed spread, as in
  Monte Carlo; otherwise leave it out).

`strength` is `core` when the plan's conclusions rest on the relation, otherwise `supporting`. `ledger_id` is optional: the
id of a simplification the plan lists, when the relation tests that simplification.

# Reply

JSON only, no other text:

{"relations": [
  {"id": "R1", "kind": "sensitivity", "setting": "<setting name>", "metric": "<metric id>", "strength": "core",
   "from_plan": "<short quote>"},
  {"id": "R2", "kind": "convergence", "setting": "<knob name>", "metric": "<metric id>", "order": 2, "refine": "decrease",
   "strength": "supporting", "from_plan": "<short quote>"}
]}

If the plan states none, reply {"relations": []}.
