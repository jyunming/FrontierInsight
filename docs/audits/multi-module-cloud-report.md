# Multi-module "tool project" as the default code layout: cloud session report

- **hostname:** `vm`
- **pwd:** `/home/user/FrontierInsight`
- **branch:** `feat/multi-module-default`, from `origin/main` at `8338c30`
- **Python:** 3.11.15, fresh `.venv` (pip/setuptools/wheel upgraded, `pip install -r requirements.txt`, `pip install -e .`)
- **status:** drafted in the cloud session (this machine), then completed locally on Windows: a design review, fixes
  from it and from an external review, and the local test runs (see *Completed locally* below). The PR is opened from
  `feat/multi-module-default-local`, which supersedes `feat/multi-module-default`.

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
move it, and never shadows the stdlib, an installed or common library, or the scripts beside it. A package the reply
wrote under a name of its own, and that simulate.py imports, is kept under that name.

**The engine's entry stays stable.** FI still imports and calls `code/simulate.py` through the same
`run_cell`/`run_trial`/`oracle` contract: `core/trial_runner.py`'s harness already puts `simulate.py`'s folder on
`sys.path`, so the package imports work in the trial runner, the oracle gate and `run.py` without change. One fix was
needed so results stay right: the trial record's key (`trial_runner._run_key`) now includes the package's files, so a
changed equation re-runs the trials instead of reusing ones computed with the old equations. The label check
(`Engine._simulation_sources`) reads the package too, so a label in `model.py` counts.

**Cost cap, at plan time.** `code_layout.estimate` works out, from the plan's protocol, what the layout adds over two
scripts: 4 files and lines of code (the model's part: 20 + 5 per `generates` equation; FI's part: tests 45 + 4 per
check, METHODS 12 + 1 per equation). The extra requests to the model are a budget for the whole quest, not a
prediction: a reply that leaves the package out is asked for again while the budget lasts (counted in
`.fi/code_layout.json`), then the quest keeps the code it has and says so. (The cloud draft estimated requests as
`1 + engine.max_iterations`, which tied the default cap to an unrelated knob: raising `max_iterations` to 3 turned the
layout off.) `plan.md` gets a section *How the code will be laid out* saying this in plain words, and run.log a
matching line. New keys under `execution`:

- `code_package: true` (default): the layout is on; `false` keeps two scripts.
- `code_package_max_extra_lines: 400`: over it, the quest keeps two scripts and plan.md and run.log say so, with the
  numbers and the key to raise.
- `code_package_max_extra_calls: 3`: at most this many extra requests in the whole quest.

**Check after implement.** Before anything runs (next to the equation-label check in `_node_execute`),
`Engine._check_code_layout` checks: the package exists and holds a function, `simulate.py` imports it, a unit test
exists (when the plan has a check with a number), `METHODS.md` exists, every `generates` equation is labelled on a
function of the package, the package makes no random numbers the per-trial seed cannot reach, and `experiment.py`
does not import it. A missing part is a plain warning in run.log; under `rigor_profile: research` the quest stops
(pause kind `code_layout`, with its own to-do card; the cloud draft reused the `split` card, whose advice was wrong
for this stop), and a resume reads the folder again. The check runs again after an oracle repair that changed the
simulation or the package.

**Unchanged paths.** `execution.split_analysis: false`, the no-simulation path, a survey and `--analyze` have no layout
decision (`_code_layout` returns `None`), no plan.md section, no prompt change and no check. Neither has a quest whose
code was written before it decided a layout (begun before this existed, then resumed or refined), and an extension of
code kept as two scripts is not asked to restructure it.

**Wherever simulate.py is rewritten, the package goes with it** (added locally). A refine that extends the study is
shown the package and told to give it back unchanged unless the new number needs an equation changed; a package file
given back with names missing is not written. The repair of a crashed simulation and the repair for a failed oracle
check are shown the package and may return `package_files` (a fix in the package alone keeps simulate.py); the oracle
gate's undo of a repair also undoes its package change; a missing label is asked for in `model.py`; a later pass that
writes simulate.py again without the package is asked again, and otherwise says the earlier package is kept.
`requirements.txt` reads the package's imports; the older contract's raw-file reuse (`split_run.simulation_sha`) and
the attempt record's code hashes know the package.

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
- `tests/test_research_acceptance.py`: its fake model now answers the package request (adds `<package>/model.py` and
  the import in `simulate.py`), as a real model would under the new default; the gates it checks are unchanged.
- `tests/test_run_manifest.py`: `test_the_older_contract_under_research_is_sent_back_for_the_trial_contract` sets
  `execution.code_package: false`: it tests the older-contract send-back, which the layout stop would otherwise
  pre-empt under research.
- Docs: `docs/USAGE.md`, `docs/capabilities-reference.md`, `docs/rigor.md`, `dev/registry.md`.

## Tests run

All with `--basetemp ./.pytest_tmp/<unique>`; no test calls a real model.

- Failing first: with `core/code_layout.py` absent, `tests/test_code_layout.py` failed at collection; with the module
  but before the engine/config hooks, 8 failed and 7 passed.
- `tests/test_code_layout.py`: **15 passed**.
- Required suites (`test_engine_smoke.py`, `test_self_correction_e2e.py`, `test_research_acceptance.py`,
  `test_web_e2e.py`): first run **34 passed, 14 failed**. All 14 failures were in `test_research_acceptance.py`, which
  now stopped at the layout check because its fake reply had no package. After the fake was updated,
  `test_research_acceptance.py` alone: **22 passed**. The other three files (26 passed in the first run) do not depend on
  that change. Together: **48 passed, 0 failed**.
- Related suites (`test_code_layout`, `test_engine_callable_simulation`, `test_split_analysis`, `test_oracle_gate`,
  `test_run_manifest`, `test_split_run`, `test_engine_measured_oracle`, `test_quest_data_saved`,
  `test_replicate_seed_repair`, `test_evidence`, `test_criteria`): **382 passed, 1 failed**. The 1 failure was the
  `test_run_manifest` research test above; after the fix it passed alone (**1 passed**).
- `test_docs_claims`, `test_config*`, `test_plan*`, `test_code_project*`, `test_trial_runner*`: **251 passed**.
- Every other test file (260 files, `-n 3`): **5659 passed, 220 skipped, 7 failed**. The 7 are environment failures that
  fail identically on `origin/main` in this VM (checked in a separate worktree): `test_execution.py::
  test_lock_pins_a_requested_package_the_venv_inherited`, `test_bootstrap.py::test_this_interpreter_is_not_itself_a_venv`,
  `test_office_pdf.py::test_a_real_deck_exports_one_pdf_page_per_slide`, three `test_pptx_slides.py` LibreOffice tests
  (LibreOffice exports no PDF here), `test_pdf_text.py::test_a_scanned_page_is_read_by_ocr`.

## What is unverified

- No run against a real model: whether a real model writes a package that keeps the scenario out of `model.py` is only
  asked for in the prompt, not checked (the check verifies the package exists, is used by `simulate.py`, and labels
  every equation on a function).
- The line estimate is a rough formula, not calibrated against real quests; the default caps (400 lines, 3 requests)
  are a first choice.
- The cluster path (`background_jobs`) with a package was not run: the job-array tasks load `simulate.py` through the
  same harness, so the package should import, but no test covers it.
- The Docker sandbox with a package was not run.
- The research acceptance, split and related suites were not rerun as one combined run after the last test fixes;
  each fixed file was rerun on its own.
- The concurrent improve-loop branch was not merged in; the engine changes are kept to new methods plus one-line hooks
  so a merge should be local.
