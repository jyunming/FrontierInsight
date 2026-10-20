# Methods: where each equation is computed

Written by Frontier Insight from the plan's model and the equation labels in the code (`# E1`). It follows the code each time the code changes.

**The model:** First-order decay c=c0 exp(-kt) with Temperature-law k(T), additive Gaussian noise, NLS fit of k per temperature, then weighted linear regression of ln k on centred 1/T.

| Equation | What it says | Where the code computes it |
|---|---|---|
| E1 | c(t) = c0 * exp(-k * t) | `decay_model/model.py`, `concentration()` (line 8) |
| E2 | k = A * exp(-Ea / (R * T)) | `decay_model/model.py`, `rate_constant()` (line 14) |

## How the files fit together

- `decay_model/`: the model's equations, one function each; no scenario values.
- `simulate.py`: one setting of the study, computed with the package (the function Frontier Insight calls).
- `experiment.py`: reads the results of every setting, computes the numbers and draws the figures.
- `tests/test_oracles.py`: the plan's checks against known answers, as unit tests.
- `run.py`: runs every setting, then the analysis (`python run.py`).
