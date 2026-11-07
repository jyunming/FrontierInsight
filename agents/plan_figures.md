You wrote the plan for a simulation study. It names the figures below only by file name. FI draws each figure itself, from the study's numbers, so FI needs to know what each one shows. Do not change the rest of the plan.

What FI's record of the trials holds, which a figure is drawn from (each reported number below, not a quantity worked out from them, has the value of every trial, not only a summary):
$record
Answer every figure with what it shows. FI itself decides whether a figure can be drawn, from the numbers below: it does not take a figure out on your word while the study reports numbers to draw.

The figures, each with what the plan says about it:
$figures

What a figure may show (the numbers the study reports and the quantities worked out from them):
$shows

A figure's x axis (`x`, required) and what it draws as separate lines or groups (`series`, a different setting, or null) are settings of the study that take more than one value. They may be: $settings
A figure may instead use as `x` one of the numbers every trial returns ($per_trial), when everything it `shows` is such a number too: it then draws each trial's pair of values (a scatter of one number against another, a point for each trial of a setting).
A figure has at most $max_panels panel(s).
For each figure also say how its two axes are scaled, `x_scale` and `y_scale`, each "linear" or "log". Answer "log" for an axis when the numbers drawn along it are all above zero and run over several powers of ten (for example an error that falls from 1e-4 to 1e-10 as the steps grow, or settings 20, 200 and 2000), so that a reader can tell the small ones apart. Answer "linear" otherwise, in particular when the numbers can be zero or negative, are differences, proportions or correlations, or stay within a factor of ten or so. The panels of a figure share these two answers, so answer "linear" when they would not all want "log".

How the study measures (from your plan):
$method
$again
Answer with one JSON object and nothing else, one entry for each figure above, under its file name exactly as written:
{"figures": {"<file name>": {"shows": ["<number>"], "x": "<setting>", "series": "<setting or null>", "panels": 1, "x_scale": "linear", "y_scale": "log"}}}
Only when the study reports no number to draw and varies no setting, answer a figure:
{"figures": {"<file name>": {"drop": true, "why": "<one sentence>"}}}
