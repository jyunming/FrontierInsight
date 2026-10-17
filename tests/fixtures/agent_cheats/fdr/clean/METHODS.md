# Methods: where each equation is computed

Written by Frontier Insight from the plan's model and the equation labels in the code (`# E1`). It follows the code each time the code changes.

**The model:** m independent unit-variance normal test statistics, a fixed number shifted by 2.5, one-sided p-values, thresholded by the uncorrected, Bonferroni or BH rule at 0.05.

| Equation | What it says | Where the code computes it |
|---|---|---|
| E1 | p = 1 - Phi(z) | `fdr_bonferroni_vs_bh/model.py`, `pvalue_one_sided()` (line 8) |

## How the files fit together

- `fdr_bonferroni_vs_bh/`: the model's equations, one function each; no scenario values.
- `simulate.py`: one setting of the study, computed with the package (the function Frontier Insight calls).
- `experiment.py`: reads the results of every setting, computes the numbers and draws the figures.
- `tests/test_oracles.py`: the plan's checks against known answers, as unit tests.
- `run.py`: runs every setting, then the analysis (`python run.py`).
