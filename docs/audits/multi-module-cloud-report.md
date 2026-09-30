# Multi-module "tool project" as the default code layout: cloud session report

- **hostname:** `vm`
- **pwd:** `/home/user/FrontierInsight`
- **branch:** `feat/multi-module-default`, from `origin/main` at `8338c30`
- **Python:** 3.11.15, fresh `.venv` (pip/setuptools/wheel upgraded, `pip install -r requirements.txt`, `pip install -e .`)
- **status:** IN PROGRESS: code and the new tests are in; docs and the long regression runs follow (this file is
  updated at the end).

## Design summary

A quest that runs a simulation (the two-script path, `execution.split_analysis` on) now gets, by default, its `code/`
laid out as a small research tool:

| Part | Written by | What it is |
|---|---|---|
| `code/<package>/__init__.py`, `code/<package>/model.py` | the model | the model's equations only (no grid values, sizes, seeds, thresholds, paths), each function labelled `# E1` |
| `code/simulate.py` | the model | unchanged contract: `run_cell` / `run_trial` / `oracle`, now the scenario around the package (`from <package> import model`) |
| `code/experiment.py` | the model | the analysis, unchanged |
| `code/tests/test_oracles.py` | FI | the plan's checks with a number to agree with, as unit tests; runs the simulation on each check's case the way the oracle gate does (same seed for `run_trial`), compares with the plan's expected value and tolerance; runs under pytest or plain `python` |
| `code/METHODS.md` | FI | each `generates` equation of the plan's model and the function that carries its label (file, function, line) |
| `code/run.py` | FI (existing, `core/code_project.py`) | the sweep over every setting and the one command that runs it all (the CLI entry) |

The package name comes from the quest title (`package_name`), is kept in `.fi/code_layout.json` so a rename does not
move it, and never shadows the stdlib or the scripts beside it.

**The engine's entry stays stable.** FI still imports and calls `code/simulate.py` through the same
`run_cell`/`run_trial`/`oracle` contract: `core/trial_runner.py`'s harness already puts `simulate.py`'s folder on
`sys.path`, so the package imports work in the trial runner, the oracle gate and `run.py` without change. One fix was
needed so results stay right: the trial record's key (`trial_runner._run_key`) now includes the package's files, so a
changed equation re-runs the trials instead of reusing ones computed with the old equations. The label check
(`Engine._simulation_sources`) reads the package too, so a label in `model.py` counts.

**Cost cap, at plan time.** `code_layout.estimate` works out, from the plan's protocol, what the layout adds over two
scripts: 4 files, lines of code (the model's part: 20 + 5 per `generates` equation; FI's part: tests 45 + 4 per
check, METHODS 12 + 1 per equation), and requests to the model (at most one extra per code-writing pass, i.e.
`1 + engine.max_iterations`, made only when a reply leaves the package out). `plan.md` gets a section *How the code will
be laid out* saying this in plain words, and run.log a matching line. New keys under `execution`:

- `code_package: true` (default): the layout is on; `false` keeps two scripts.
- `code_package_max_extra_lines: 400`, `code_package_max_extra_calls: 3`: over either, the quest keeps two scripts and
  plan.md and run.log say so, with the numbers and the key to raise.

**Check after implement.** Before anything runs (next to the equation-label check in `_node_execute`),
`Engine._check_code_layout` checks: the package exists and holds a function, `simulate.py` imports it, a unit test
exists, `METHODS.md` exists, every `generates` equation is labelled on a function of the package. A missing part is a
plain warning in run.log; under `rigor_profile: research` the quest stops through the existing contract pause
(`kind: split`, `contract_stage`; no new pause name), and a resume reads the folder again.

**Unchanged paths.** `execution.split_analysis: false`, the no-simulation path, a survey and `--analyze` have no layout
decision (`_code_layout` returns `None`), no plan.md section, no prompt change and no check. A reply that does not hold
the package is asked for once more; if it still does not, the quest keeps two scripts this time and says so.

## Files changed

- `core/code_layout.py` (new): decision, estimate, plan.md lines, prompt block, reply parsing, package writing, equation
  map, METHODS.md and unit-test generation, the layout check.
- `core/engine.py`: small hooks only: `_code_layout`, `_plan_code_layout`, `_code_package_reply`, `_package_in_use`,
  `_check_code_layout` (new methods, grouped before `_check_equation_labels`); one line each in `_node_plan` (render),
  `_node_implement` (reply, write package), `_refresh_code_project`, `_simulation_sources`, `_node_execute`,
  `_split_block`, and the pause log label.
- `core/config.py`: `execution.code_package`, `code_package_max_extra_lines`, `code_package_max_extra_calls`.
- `core/plan.py`: `render(..., code_layout=...)`.
- `core/trial_runner.py`: `_run_key` includes the package.
- `core/code_project.py`: `refresh(..., extra_files=..., readme_files=...)`; files in subfolders; README lists them.
- `tests/test_code_layout.py` (new).

## Tests run

(to be filled in)

## What is unverified

(to be filled in)
