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
| `plan_part_misplaced/` | a plan's search block sat beside `protocol:`, its one design variable's `kind` was a sentence for an array, the ask listed all four parts as missing, the model rewrote the same block, and the quest paused asking a person to try another model | `tests/test_replay_misplaced_part.py` |
| `pilot_undefined_metric/` | a protocol metric was not a number in any of 52000 trials of a two-hour study, found only after the whole run, with the repairs spent | `tests/test_replay_pilot_undefined.py` |
| `pilot_stuck_proportions/` | the pilot read 28 trials and found nothing wrong, then the full run gave proportions that were exactly 0 or 1 in every trial and nothing could be measured | `tests/test_replay_pilot_stuck.py` |
| `expected_moved/` | a rewrite of the plan about one check moved another check's expected value away from the plan's own formula, and the right result was reported as failing it | `tests/test_replay_expected_moved.py` |
| `plan_constant_overwritten/` | the run-time setting guard took a constant the plan defines (a reference value of 80, with a fallback in the code) for the swept setting spelt another way, and its repair set the reference to the swept setting | `tests/test_replay_plan_constant_guard.py` |
| `code_findings_wide/` | a code reader's 35 findings over six readings under a cap of twelve: the same issue came back reworded, eight result-changing faults were still open when the code ran | `tests/test_replay_code_findings.py` |
| `code_findings_conflict/` | a finding the reader got wrong (every value it said was missing was there), seven findings that told one fault, and a plan that defined a metric two ways, sent to the code | `tests/test_replay_code_findings.py` |
| `code_findings_small/` | a package of four small functions with a real finding a failing test shows; its scripted tests and repairs live in `scripted/` | `tests/test_replay_code_findings.py` |
| `unnamed_list_false_match/` | two quests stopped with no paper because a list not named after a protocol axis (a loop over four output positions; the values of another axis) shared values with it | `tests/test_replay_unnamed_list.py` |
| `random_term_exact/` | an equation whose formula stated a distribution (`epsilon ~ N(0,1)`) with its flag beside the example was tested as a deterministic one, a correct random function could never pass, and the log gave no reason | `tests/test_replay_random_term.py` |
