A figure function draws only through the calls of `fig`, and returns nothing. FI creates the figure, writes every axis label, the legend and the limits, then saves and closes it, so the function never calls `savefig`, `show`, `close`, `figure` or `subplots`, and never sets a title, label, legend, limit or scale (FI overwrites them). It draws only values read from `table`. `panel` is the 0-based place in `fig.axes`.

The calls (draw the data of a panel before its band or curve):
- `fig.line(panel, series, x, y, what, yerr=None, step=False, yerr_meaning="other", form="mean")`: a line of one series.
- `fig.points(panel, series, x, y, what, yerr=None, yerr_meaning="other", form="mean")`: markers only.
- `fig.bars(panel, series, x, y, what, yerr=None, yerr_meaning="other", form="mean")`: bars.
- `fig.band(panel, series, x, lo, hi, what, meaning)`: a shaded interval around a series already drawn, at its x values; `meaning` is `ci95`, `sd`, `range` or `other`.
- `fig.ref(panel, ref_id, axis="y")`: a reference line at a value the plan holds, named by its plan id.
- `fig.note(panel, x, y, what, form="mean", key=None)`: words with one `{result id}` where FI writes the number it reads from `table` (`"mean growth: {growth}"`); the words hold no digits; `form` is `mean`, `n`, `low` or `high`.
- `fig.curve(panel, label, xs, ys, source="computed in the figure")`: an analytic overlay (a theory curve, a slope, a fit); it is marked with its source.
- `fig.image(panel, z, x=None, y=None, what="other")`: a heat map of `z` (rows of values).

Names:
- `series`, `x` and `what` are ids of the plan (its results, settings and compared values), never words you make up. `what` is the result `y` shows; use `what="other"` for a quantity the plan does not declare (FI then states nothing about its value).
- `x` is the list of x values; to put another plan setting or result on x, give `x=(id, values)`.
- FI compares `y` with the table's mean of `what` at each x (a setting); declare `form=` (`sd`, `se`, `min`, `max`, `range`, `max_abs`) when `y` is not the mean, or FI states no value from that series.
- Give `yerr` (half-widths) with `yerr_meaning` whenever the plan's interval or spread is known.

Use `fig.raw(panel)` (the real axes) only for a drawing none of these calls can make. FI then states nothing about that panel, so use it as little as possible.

Never type text into a figure: no title, no axis label, no legend, no number written as a string. FI writes them from the plan and from the table.
