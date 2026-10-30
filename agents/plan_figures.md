You wrote the plan for a simulation study. It names the figures below only by file name. FI draws each figure itself, from the study's numbers, so FI needs to know what each one shows. Do not change the rest of the plan.

The figures, each with what the plan says about it:
$figures

What a figure may show (the numbers the study reports and the quantities worked out from them):
$shows

A figure's x axis (`x`, required) and what it draws as separate lines or groups (`series`, a different setting, or null) are settings of the study that take more than one value. They may be: $settings
A figure has at most $max_panels panel(s).

How the study measures (from your plan):
$method
$again
Answer with one JSON object and nothing else, one entry for each figure above, under its file name exactly as written:
{"figures": {"<file name>": {"shows": ["<number>"], "x": "<setting>", "series": "<setting or null>", "panels": 1}}}
A figure that cannot be drawn from the numbers above (it needs something the study does not report) is answered:
{"figures": {"<file name>": {"drop": true, "why": "<one sentence>"}}}
