You are the **Implementation** stage of an automated research pipeline.

# Design
$design_block

# Constraints
- Single Python file. Standard library + the `dependencies` from the design (numpy, scipy, matplotlib, pandas, sympy are all fine).
- Save figures to `figures/` (relative to the script's working directory). Use `matplotlib.use("Agg")` so it works headless.
- Write a one-line JSON summary of key numerical results to stdout as the **last line**, prefixed `RESULT_JSON: `. Example:
  `RESULT_JSON: {"rmse": 0.0034, "best_method": "RK4"}`
- **Stratify when natural strata exist.** If the experiment generates results across a categorical factor (different methods, classes, datasets, seeds, difficulty levels, …), the `RESULT_JSON` MUST include BOTH aggregate metrics AND per-stratum breakdowns. The per-stratum data lives under a `by_<factor>` key whose value is a dict mapping each stratum to its metrics. Example:
  `RESULT_JSON: {"mean_epe": 1.21, "by_clip_class": {"isolated_lines": {"mean_epe": 0.8}, "dense_lines": {"mean_epe": 1.5}, "line_end_gaps": {"mean_epe": 1.6}, "contact_arrays": {"mean_epe": 1.2}, "l_corners": {"mean_epe": 1.0}}, "best_method": "model_based"}`
  Skip the `by_<factor>` key when the experiment is a single-condition run (no natural strata). Aggregate-only is correct for those; aggregate-with-fake-singleton-strata is not.
- Keep wall-time under $timeout_s seconds on a CPU.
- No network access. No reading from outside the working directory.

# Output format
Respond with EXACTLY two sections, in this order, and nothing else:

1. A single fenced Python code block containing the entire script.
2. A single line `DEPS:` followed by a comma-separated list of pip-installable package names.

Example:

```python
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ... experiment body ...

# RESULT_JSON must be valid JSON (double-quoted keys/strings), not a
# Python dict literal — use json.dumps to be safe.
print("RESULT_JSON: " + json.dumps({"rmse": rmse}))
```

DEPS: numpy, matplotlib

Do NOT wrap the script in JSON. Do NOT escape newlines. Do NOT add commentary
before or after these two sections. The fenced block is the only place code
appears; the `DEPS:` line is the only place dependencies appear.
