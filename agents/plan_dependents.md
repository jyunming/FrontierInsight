You wrote the plan for a simulation study. Its list of dependent variables has entries that are not exactly one of the numbers the study reports. Say what each one is, so that every number the study reports is declared before the plan is frozen. Do not change the plan otherwise.

How the study measures (from your plan):
$method

The settings the study changes:
$settings

The numbers the study reports (the last ones are worked out afterwards from them):
$results

The entries to explain:
$names

For each entry, give one answer:
- "result": the entry is only another word for one or more of the reported numbers above. Give "inputs", the reported numbers it is (their names exactly as listed above; one is enough).
- "setting": the entry is only one of the settings listed above. Give "setting", its name exactly as listed.
- "derived": the entry is a number the study must report that none of the reported numbers above is, worked out afterwards from them (for example a spread or a correlation across runs). Give "id", a new name for it in code (letters, digits and underscores, not a name listed above), "inputs", the reported numbers it is worked out from, and "what", what it is in a few words. It is added to the plan as a new derived result.
- "not_reported": the reported numbers above do not carry the entry, it is not a setting, and the study does not report it.

Uncertainties: if your plan says the study reports an uncertainty of one of the reported numbers (a standard error, a spread or an interval half-width) that is not itself one of the reported numbers above, list it under "uncertainties", keyed by the reported number it belongs to, with "what", the uncertainty in a few words, and the answer "derived" (with "id" and "inputs" as above: it becomes its own derived result) or "not_reported". Leave out an interval that is worked out from the repeated runs of a setting, and leave "uncertainties" empty when the plan names none.
$again
Answer with one JSON object and nothing else:
{"names": {"<D id>": {"answer": "<result, setting, derived or not_reported>", "inputs": ["<reported number>"], "setting": "<for a setting>", "id": "<for derived>", "what": "<for derived>"}}, "uncertainties": {"<reported number>": {"what": "<the uncertainty>", "answer": "<derived or not_reported>", "id": "<for derived>", "inputs": ["<reported number>"]}}}
Give every entry above, once, and leave out the keys that do not apply.
