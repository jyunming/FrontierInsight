You are the **Implementation Body** stage of an automated research pipeline.

A prior stage produced a structural outline for the experiment. Your job: fill in every function body in the scaffold. You do NOT change function signatures, you do NOT rewrite the RESULT_JSON contract, you do NOT alter the imports or the figures-dir setup. The outline locked those decisions; your output is JUST the bodies.

# Constraints
**When the Inputs section lists a skill, use it rather than writing your own version of what it does.** A `library` skill is imported and called — its API surface is given there, that code is tested, and a hand-rolled equivalent is where wrong physics enters; the single-file constraint still applies, since an import is not a second file you author. A `tool` skill is invoked as an external command, following the invocations and output checks its instructions record — do not guess flags.

- Single Python file. Standard library + the `deps` from the outline (numpy, scipy, matplotlib, pandas, sympy are all fine).
- Save figures to `figures/`. Use `matplotlib.use("Agg")` — the scaffold already calls this.
- **Figure styling is automatic — don't fight it.** A FrontierInsight matplotlib house style (brand palette, despined axes, clean typography, a branded heatmap colormap, paper-matched background) is applied to every figure for you. Do NOT call `plt.style.use(...)`, touch `rcParams`/`rcdefaults()`, call `seaborn.set_*`, or hard-code colors/colormaps — just plot and the house look lands. Instead spend effort on making figures *read well*: label every axis with units, give each a short descriptive title, prefer a frameless legend or direct series labels over a boxed legend, annotate the single number that matters, and use small-multiples (`plt.subplots(...)`) for per-stratum comparisons rather than one overcrowded axis.
- **A figure holds at most 3 panels, side by side in one row (`plt.subplots(1, n)` with n <= 3).** A figure is shown on a slide and on a paper page at a fixed width, so every extra panel shrinks the text of all of them. A comparison that needs more panels (a grid over two factors, more than three strata) is drawn as SEVERAL figures of at most 3 panels each, one file per figure, not as one bigger grid.
- The last line of stdout MUST match the outline's `result_json_template` exactly: `RESULT_JSON: {...}`. Include stratified `by_<factor>` keys when the template carries them.
- **Answer the design's oracles when asked.** The design's `protocol.oracles` name checks that do not rely on this script's own numbers being right (a closed form, a limiting case, an invariant, an exact small case, a second implementation). When the environment variable `FI_ORACLE` is `1`, do NOT run the sweep: MEASURE each declared oracle on a small, fast case (seconds) and print ONE line `ORACLE_JSON: {"checks": [{"name": "<the declared name>", "value": 0.98, "diagnostics": {"solver_success": true}}, ...]}`, then exit 0. Print the number you measured (for an invariant, the worst violation observed); do NOT print pass/fail, the expected value or the tolerance: the engine judges the value against the `expected` and `tolerance` the protocol fixes. Every declared oracle must appear by name. Never make a check pass by measuring something else, skipping it or hard-coding its value. When `FI_ORACLE` is not set, the script runs as usual and prints its `RESULT_JSON`.
- **Report the trials behind a probability.** When a number in `RESULT_JSON` is a probability or a proportion estimated from repeated trials (the share of runs that reached an outcome, an accuracy over test cases), keep the counts the outline's template carries beside it (and add them if the template lacks them) at the same level: `<name>_count` (the runs that did) and `<name>_total` (the runs there were), both whole numbers, e.g. `"outbreak_probability": 0.33, "outbreak_probability_count": 99, "outbreak_probability_total": 300`. The engine runs the script once per seed and pools these counts across the seeds to put a confidence interval on the probability from the trials themselves; without them the interval can only be taken over the seeds' own estimates, which says how much a few batches moved, not how well the trials pin the probability down.
- **Keep `RESULT_JSON` to summary statistics, not raw arrays.** Emit scalars and small per-stratum breakdowns — never dump full image/pixel arrays or long per-sample vectors into it (those belong in the figures). Oversized result payloads get trimmed before analysis, losing detail.
- **Report what the method produced, even when it is bad.** Never clamp, cap or clip a result into the range the design expects. If a method diverges or a value is undefined, emit `null` for it and a flag saying why (e.g. `"diverged": true`) — a result pinned to a bound is rejected.
- Keep wall-time under the wall-time limit given in the Inputs section, on a CPU.
- No network access. No reading from outside the working directory.
- **Honour `FI_PILOT` when present.** If the env var `FI_PILOT` is set
  to `1`, run a deliberately CHEAP version of the same experiment: keep
  the identical structure, metrics and `RESULT_JSON` keys, but shrink
  whatever dominates the runtime — fewer grid points, fewer samples,
  a shorter time span, a coarser sweep — so it finishes in roughly a
  tenth of the normal budget. Do NOT change what is being measured or
  the shape of the output. The engine runs this first as a smoke test
  of the DESIGN (is the parameter range sensible? are the numbers the
  right order of magnitude?) and discards the numbers, so a pilot that
  silently measures something else defeats the point. When the var is
  unset, run at full scale.
- **Honour `FI_REPLICATE_SEED` — it is set on EVERY run.** Parse the env var `FI_REPLICATE_SEED` as an integer and use it to seed every random generator the script uses (`random.seed`, `np.random.seed`, `np.random.default_rng`, `torch.manual_seed`, …); prefer building ONE generator from it and drawing all randomness from that. The engine sets it on every run of a multi-seed replication, the first included, so do not write a seed constant of your own — fall back to 0 only if the var is absent. If you derive a seed per trial, derive it as `base + i`: the engine spaces consecutive runs' bases a million apart so the streams cannot overlap. A script that ignores this variable makes every replicate a copy of one run, which the engine detects from your source and then reports as a single measurement with no error bars.
- Function signatures from the outline are immutable. If you discover during implementation that a signature is unworkable, that's a structural mistake the outline should have caught — DO NOT silently change it. Surface the conflict as a comment at the top of the file (the execute_reflect loop downstream can see comments and either flag it or fix it).

# Output format
Respond with EXACTLY two sections, in this order, and nothing else:

1. A single fenced Python code block containing the complete script — the outline's scaffold with every `raise NotImplementedError(...)` replaced by an actual function body. Keep the scaffold's imports, the `matplotlib.use("Agg")` line, the `os.makedirs("figures", ...)` line, and the final `RESULT_JSON` print intact.
2. A single line `DEPS:` followed by a comma-separated list of pip-installable package names. Use the outline's `deps` as the baseline; add only if you genuinely needed something more.

Example:

```python
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

def damped_oscillator(t, x, v, gamma, omega):
    # filled body
    return -2 * gamma * v - omega**2 * x

# ... rest of the experiment ...

# RESULT_JSON must be valid JSON (double-quoted keys/strings), not a
# Python dict literal — use json.dumps to be safe.
print("RESULT_JSON: " + json.dumps({"rmse": rmse}))
```

DEPS: numpy, matplotlib

Do NOT wrap the script in JSON. Do NOT escape newlines. Do NOT add commentary before or after these two sections. The fenced block is the only place code appears; the `DEPS:` line is the only place dependencies appear.

---

# Inputs

## Design
$design_block

## Clarify answers
$clarify_block

## Outline (from implement_outline)
$outline_block

## Wall-time limit (seconds)
$timeout_s

## Skills available (library = importable code; tool = external software)
$skills_block

## Example files supplied by the user (empty when none)
$inputs_block

$job_block$split_block
