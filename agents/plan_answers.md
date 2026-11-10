You wrote the plan for a simulation study. The topic asks the study to report the quantities below. FI answers each in the paper from the study's numbers, so FI needs to know which result of the study answers each one. Do not change the rest of the plan.

The quantities the topic asks for (a quantity in square brackets is already mapped by your plan; keep that map and only say where it is answered):
$quantities

The numbers the study reports or works out (`metric:<id>`):
$shows

The equations of the model (`equation:<id>`): $equations

The settings of the study and their values:
$settings

`by` is `metric:<id>` or `equation:<id>` from the lists above: the one result that is the answer. If no number the study reports or works out IS the quantity (not a number it is computed from, and not a neighbouring one), answer `"by": "none"` rather than naming the nearest one. Where the plan already maps a quantity to a result, keep that map, except when a number worked out from that very result (listed with "worked out from") IS the quantity, such as the value of a setting at which the result crosses a level: answer that number's id. FI keeps the plan's map and lists your answer beside it in the quality report. A result in another unit than the quantity (see `unit`) does not change the plan's map: FI states both units and converts nothing.

`unit` is the physical unit of the quantity as the topic asks for it, `by_unit` the physical unit of the number `by` names, and, for a quantity the plan maps (in square brackets above), `plan_unit` the physical unit of the result the plan maps it to. Write both as a short symbol, in the same notation everywhere: a length in metres `m`, a time in seconds `s`, an angle in degrees `deg`, and `1` for a proportion, a ratio, a count, an order or any other pure number. A result in another unit is not the quantity: a length and the relative error of that length are not the same, and neither are a rate and its standard error. Leave `by_unit` out when `by` is `none`, and `plan_unit` out when the plan maps nothing. `at` is needed only when the topic asks for the quantity at ONE value of a setting (not across the settings): give that setting and its value, as `{"<setting>": <value>}`; otherwise leave it out or null.

`statistic` says what of that result the quantity asks for, one of: $statistics. `value` (the default) is the result itself, its mean in each setting. `bias` is for a quantity that asks how far the estimates lie from the truth: FI works out the mean of the result minus the true value your plan declares for it (`known_value.true_value`), so use it only when the result has one. `sd` is for a quantity that asks how much the estimates scatter: FI works out their standard deviation over the trials. A result that is already the bias or the spread itself (a number the study computed under that name) is `value`.

How the study measures (from your plan):
$method
$again
Answer with one JSON object and nothing else, one entry for each quantity above, under its id exactly as written:
{"answers": {"<id>": {"unit": "<unit of the quantity>", "by": "metric:<id>", "by_unit": "<unit of that result>", "plan_unit": "<unit of the result the plan maps it to>", "statistic": "value", "at": {"<setting>": <value>}}}}
