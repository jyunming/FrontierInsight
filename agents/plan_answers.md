You wrote the plan for a simulation study. The topic asks the study to report the quantities below. FI answers each in the paper from the study's numbers, so FI needs to know which result of the study answers each one. Do not change the rest of the plan.

The quantities the topic asks for (a quantity in square brackets is already mapped by your plan; keep that map and only say where it is answered):
$quantities

The numbers the study reports or works out (`metric:<id>`):
$shows

The equations of the model (`equation:<id>`): $equations

The settings of the study and their values:
$settings

`by` is `metric:<id>` or `equation:<id>` from the lists above: the one result that is the answer. If no number the study reports or works out IS the quantity (not a number it is computed from, and not a neighbouring one), answer `"by": "none"` rather than naming the nearest one. Where the plan already maps a quantity to a result, keep that map, except when a number worked out from that very result (listed with "worked out from") IS the quantity, such as the value of a setting at which the result crosses a level: answer that number's id instead. `at` is needed only when the topic asks for the quantity at ONE value of a setting (not across the settings): give that setting and its value, as `{"<setting>": <value>}`; otherwise leave it out or null.

How the study measures (from your plan):
$method
$again
Answer with one JSON object and nothing else, one entry for each quantity above, under its id exactly as written:
{"answers": {"<id>": {"by": "metric:<id>", "at": {"<setting>": <value>}}}}
