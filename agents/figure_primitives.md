A figure function draws only through the calls of `fig`, and returns nothing. FI creates the figure, writes every axis label, the legend and the limits, then saves and closes it, so the function never calls `savefig`, `show`, `close`, `figure` or `subplots`, and never sets a title, label, legend, limit or scale (FI overwrites them). It draws only values read from `table`. `panel` is the 0-based place in `fig.axes`.

The calls (draw the data of a panel before its band or curve):
- `fig.line(panel, series, x, y, what, yerr=None, step=False, yerr_meaning="other", form="mean")`: a line of one series.
- `fig.points(panel, series, x, y, what, yerr=None, yerr_meaning="other", form="mean")`: markers only.
- `fig.bars(panel, series, x, y, what, yerr=None, yerr_meaning="other", form="mean")`: bars.
- `fig.band(panel, series, x, lo, hi, what, meaning)`: a shaded interval around a series already drawn, at its x values; `meaning` is `ci95`, `sd`, `range` or `other`.
- `fig.ref(panel, ref_id=None, axis="y", *, at=None, value=None, key=None, form="mean")`: a reference line, in one of three forms (give exactly one): `fig.ref(panel, "id")` a reference the plan holds, named by its plan id (only an id the plan lists as a reference, never a result or a setting); `fig.ref(panel, at="result id", key=..., form="mean")` a line at a value FI reads from `table` (`form` is `mean`, `n`, `low` or `high`; `key` is the setting, optional when the study has one); `fig.ref(panel, value=2.5)` a constant you state, which FI draws and records only as "stated in the figure" (a line at that coordinate, nothing more). `axis` is `"y"` or `"x"`.
- `fig.note(panel, x=None, y=None, what=None, form="mean", key=None)`: words with one `{id}` where FI writes the number it reads from `table` (`"mean growth: {growth}"`); the words hold no digits; `{id}` is a result of the plan (read with `form`: `mean`, `n`, `low` or `high`, at the setting `key`) or a setting of the plan (FI writes its value; `fig.note(0, what="size: {size}")`). Leave out `x` and `y` and FI puts the note in the free corner; give both (numbers) to place it.
- `fig.curve(panel, label, xs, ys, source="computed in the figure")`: an analytic overlay (a theory curve, a slope, a fit); it is marked with its source.
- `fig.image(panel, z, x=None, y=None, what="other")`: a heat map of `z` (rows of values). Give BOTH `x=("setting id", values)` and `y=("setting id", values)` naming two different settings, ascending and evenly spaced, for FI to state anything about its values: it states the value range only when each cell is the table's mean of `what` at that column and row, all in one slice of the other settings; with only one of them, or the same setting twice, or any other `what`, the values are not stated.

Names:
- `series`, `x` and `what` are ids of the plan (its results, settings and compared values), never words you make up. `what` is the result `y` shows; use `what="other"` for a quantity the plan does not declare (FI then states nothing about its value).
- `x` is the list of x values; to put another plan setting or result on x, give `x=(id, values)`.
- FI compares `y` with the table's mean of `what` at each x (a setting); declare `form=` (`sd`, `se`, `min`, `max`, `range`, `max_abs`) when `y` is not the mean, or FI states no value from that series.
- `yerr` is one of: a list of half-widths (the same below and above); a pair `(lower, upper)` of lists of widths, for an uneven bar; or the string `"ci95"` or `"sd"`, for FI to read the widths from `table` at each x (`what` a result of the plan and x a setting; then give no other `yerr_meaning`). Give typed widths with `yerr_meaning` whenever the plan's interval or spread is known; FI keeps `ci95` or `sd` only when the widths equal the table's own, so `"ci95"`/`"sd"` is the safe way to draw them.

If a call fails, the message says why; the usual ones and what to do instead:
- "`name` is not a reference the plan holds": `fig.ref(panel, "id")` takes only a reference id the plan lists. For a value read from the results use `fig.ref(panel, at="result id")`; for a constant use `fig.ref(panel, value=<number>)`.
- `table` has no `.se`, `.low` or `.high`: the table reads are `values`, `n`, `mean`, `interval` (a `(lower, upper)` pair), `level_means` and `level_rows`. For error bars pass `yerr="ci95"` or `yerr="sd"` and let FI read them; `yerr_meaning` accepts only `ci95`, `sd`, `range` or `other`.
- a series given as a level number (`0`, `1`) or a setting id: `series` is an id of the plan's series or results, as a string, never a position or the name of a setting.
- "draw the data of panel N first" (a curve), or "series ... has not been drawn on panel N" (a band): draw the panel's data with `line`, `points` or `bars` first, then the curve or the band.

Use `fig.raw(panel)` (the real axes) only for a drawing none of these calls can make. FI then states nothing about that panel, so use it as little as possible.

When panels show different settings, say which on each: `fig.panel(0, m=200)` (the panel's place in `fig.axes`, then the setting and its value). FI writes that panel's title from it (`m = 200`); do not type a title for such a panel.

Never type text into a figure: no title, no axis label, no legend, no number written as a string. FI writes them from the plan and from the table.
