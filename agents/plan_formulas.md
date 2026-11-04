You wrote the plan for a simulation study. It works out the quantities below with a formula, and FI, which works out each formula itself, cannot work these out as written. For each one, give a formula FI can work out, or say that the study's own code computes it. Do not change the rest of the plan.

The quantities, each with its formula and why FI cannot work it out:
$formulas

How FI works out a formula: at each setting of the study, from that setting's mean of each reported number, the setting's own values, the plan's fixed values and the other quantities; a formula that names none of the per-setting numbers is worked out once. A formula may name: $known. It is written with $formula_language.

A quantity worked out over the trials of a setting or over the settings (the largest value over the settings, a spread over the trials, a fitted slope) is not a formula of this kind: answer "function" and give "inputs", the reported numbers it is computed from; the study's code then computes it.

How the study measures (from your plan):
$method
$again
Answer with one JSON object and nothing else, one entry for each quantity above:
{"formulas": {"<quantity>": {"formula": "<a formula FI can work out>"}}}
or, for a quantity the study's code computes:
{"formulas": {"<quantity>": {"function": true, "inputs": ["<reported number>"]}}}
