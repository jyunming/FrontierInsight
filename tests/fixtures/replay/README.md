# Replay fixtures

Each folder here is a small copy of the real files of a quest that failed in a validation run, kept as evidence for a
test that runs FI's gate logic on them (no model call: a model is stubbed where one is needed).

These files are the quests' own and may contain field words. That is acceptable ONLY here: this folder is evidence, not
FI's code, and nothing under `core/` or `agents/` may take a word from it.

| folder | what failed | test |
| --- | --- | --- |
| `redesign_sweep_unread/` | a redesign's settings to vary could not be read and the quest paused without asking the plan's model | `tests/test_replay_redesign_parts.py` |
| `final_code_unread/` | the code version that produced the results was never read by the different-model reviewer | `tests/test_replay_final_code_read.py` |
| `undefined_trials/` | a metric had fewer per-trial values than the manifest's successful trials | `tests/test_replay_undefined_trials.py` |
| `run_count_lost/` | a redesign wrote no run count, so one trial per setting ran where the plan said 2500, the analysis script hard-coded 1000, and a paper with no results was delivered | `tests/test_replay_run_count_lost.py` |
| `quantity_other_than_planned/` | the code computed a quantity as something other than the plan defined (a second model said so and the repairs did not change it) and the paper reported it under the plan's name | `tests/test_replay_quantity_other_than_planned.py` |
| `expected_moved/` | a rewrite of the plan about one check moved another check's expected value away from the plan's own formula, and the right result was reported as failing it | `tests/test_replay_expected_moved.py` |
