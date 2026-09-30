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

## Completed locally (Windows)

The cloud draft was reviewed and completed on the maintainer's Windows machine (Python 3.11, `pytest -n 4`,
`--basetemp ./.pytest_tmp/<unique>`), first on `origin/main` at `a9ab141`, then rebased on `03ebd79` (after the
optimisation runner and the oracle-form change landed; the conflicts were in `_split_block`, the reflect note for
simulate.py and `attempt_records._generated_project_files`, each resolved by keeping both sides). Rebased once more on `f8d2b2e`, after the improve loop landed: its editable files, snapshots and undo
now include the package's modules (by path in code/), so the loop can change an equation where the equations are.

**Design review (before any external review).** A refine that extends the study, the repair of a crashed simulation and
the repair for a failed oracle check were all blind to `model.py` (they showed and rewrote simulate.py only), so an
extension could overwrite `model.py` with a partial file and an oracle failure caused by an equation could not be fixed
there; a missing label was asked for in simulate.py although the layout check reads it in the package; a plan with no
numeric check made the layout check report "no unit tests" (a stop under research for a file FI chose not to write);
`requirements.txt` missed what the package imports. All fixed.

**External reviews** (separate `claude -p` sessions, written reports, five passes: one full review, then targeted
re-checks of each round of fixes until no P2+ remained). Fixed from them: the layout stop now has its own to-do card
(`code_layout`; the `split` card's advice was wrong for it); a quest begun before this existed is left as it was on
resume or refine; the request limit is a quest-wide budget, not `1 + max_iterations`; a later pass that rewrites
simulate.py without the package is asked again and otherwise says the old package is kept; a package the reply named
itself (and imports) is kept; simulate.py importing a package that was not written is said plainly; repairs may fix the
package alone, and the oracle gate's undo restores it; the package is checked for randomness the per-trial seed cannot
reach (module-level generators, unseeded generators outside the `rng = rng or default_rng()` fallback) and experiment.py
for importing it; the older contract's raw-file reuse and the attempt record's code hashes know the package; package
names never hide a library; generated tests take any check name; a `<pkg>/simulate.py` block is not taken for the
script.

**Tests run locally:**

- First run of the cloud code plus the first local fixes, every test file touching `code_project`, `trial_runner`,
  `run_manifest`, `split_analysis`, `rerun_from`, refine/extend, `execute_reflect`, the oracle gate or labels (52 files,
  including `test_self_correction_e2e.py` and `test_research_acceptance.py`) plus `test_engine_smoke.py` and
  `test_web_e2e.py`, `-n 8`: **1556 passed, 6 failed**. All 6 were environment failures under load: 4 timed out on
  `~/.frontier-insight/pip-install.lock` (shared with other sessions on this machine) and 2 on an asyncio timeout;
  none of them passes through the new code, and all passed in the run below.
- After all review fixes (at `3034c93`), the same selection widened to `split_run`, `attempt_records`, `todo` and
  `code_layout` (60 files) plus `test_engine_smoke.py` and `test_web_e2e.py`, `-n 4`: **1739 passed, 1 skipped,
  0 failed** (71 min).
- After the rebase onto the optimisation runner (`03ebd79`), the selection widened again to the optimisation and
  oracle-form tests (69 files) plus `test_engine_smoke.py` and `test_web_e2e.py`, `-n 4`: **2183 passed, 5 skipped,
  0 failed** (66 min).
- After the rebase onto the improve loop (`f8d2b2e`) and the improve-loop fixes up to `fbdd8fb`, the same selection
  plus every test touching `improve` (76 files) and the two e2e files, `-n 4`: **2391 passed, 5 skipped, 0 failed**
  (35 min).
- The later commits change only the improve loop's handling of its own copies (links, a moved `.fi`) and add tests:
  every test file touching `improve` or `code_layout` (13 files) plus `test_engine_smoke.py`,
  `test_self_correction_e2e.py` and `test_research_acceptance.py`, `-n 4`: **730 passed, 1 skipped, 0 failed**; and
  `tests/test_improve.py` with `tests/test_code_layout.py` at the final commit: **73 passed**.

**The improve loop and the package** (added after the rebase onto `f8d2b2e`). The loop that changes the simulation one
step at a time saw only the top-level scripts; it now sees and may edit the package's modules (by their path in
`code/`), and its kept copies, undo and resume include them. External reviews of that (four more targeted passes)
found and had fixed: a kept copy was cleared and read through a link or junction inside it or one level up (which could
delete files outside the quest), a copy saved before the package existed deleted the package on restore, a moved `.fi`
went unnoticed by the tamper guard, and a `.fi` link loop crashed the quest instead of stopping the loop. A `.fi` a
person keeps elsewhere through a link is used as it is.

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
- Added locally: known gaps left on purpose. The protocol drift check (`protocol_check.check`, including `rng_reuse`)
  still reads only simulate.py and experiment.py, so a grid value hard-coded in `model.py` is not caught (the prompt
  forbids scenario values there; feeding the package in risks false matches on coefficient lists with no repair target).
  An extension's partial package file is rejected, not merged (the run then fails on the missing function and is
  repaired, one repair spent). Nothing checks for figures or `RESULT_JSON` inside the package. Class attributes are not
  read by the randomness check, and a few contrived one-line shapes pass its fallback exemption. A package name is
  checked against libraries installed for FI itself, not those installed only in the quest's environment.
- Added locally: the execute_reflect path where the repair fixes only the package, and its ledger entry, are covered by
  code reading and the oracle-path test, not by a test of their own.
- Added locally: no quest was run with a real model, locally either; the Windows runs above use the fake-model tests.
- Added locally: the improve loop's handling of links was probed on Windows with junctions only (this machine cannot
  make symlinks) and the Docker case was reasoned from the code, not run. After the guard detects a moved `.fi`, the
  loop stops, but the rest of the quest still writes to `.fi` wherever it now leads (that predates this branch; whether
  such a quest should stop entirely is a decision left open). The engine-level path where the abort cannot write its
  copies back is covered by code reading and review probes, not by a test of its own.
