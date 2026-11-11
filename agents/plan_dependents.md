You wrote the plan for a simulation study. Its list of dependent variables has entries that are not exactly one of the numbers the study reports. Say what each one is. Do not change the plan.

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
- "not_reported": the reported numbers above do not carry the entry and it is not a setting; nothing is reported under it.
$again
Answer with one JSON object and nothing else:
{"names": {"<D id>": {"answer": "<result, setting or not_reported>", "inputs": ["<reported number>"], "setting": "<for a setting>"}}}
Give every entry above, once, and leave out the keys that do not apply.
