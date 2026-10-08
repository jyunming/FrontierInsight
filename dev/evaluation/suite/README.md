# Evaluation suite

One fixed set of topics, run on every FI change with the same settings, so FI's quality can be improved from numbers:
what share of runs end in a trustworthy paper, what the failures have in common, and whether the last change helped.
The harness is general. The topics, the models and the key file live in a suite folder outside the repository.

## The suite folder

```
suite.yaml            groups (a group with acceptance_only: true runs only with --acceptance), topics, rounds_dir, env_file
config_template.yaml  the settings every topic gets (the provider block is REPLACED; other keys are merged)
topics/<name>.yaml    an ordinary FI config per topic
rounds/<round>/       one folder per round: round.manifest.json, <topic>/{config.yaml,out/,stdout.log}, round.json, round.md, grades.json
rounds/trend.md       one row per round
rounds/pareto.md      failures of all rounds by cause class
```

## One round

```
python -m dev.evaluation.suite run --suite S/suite.yaml --group dev --round r7 --fi <frozen FI worktree> --max-parallel 3
python -m dev.evaluation.suite status --suite S/suite.yaml --round r7
python -m dev.evaluation.suite wait   --suite S/suite.yaml --round r7
python -m dev.evaluation.suite collect --suite S/suite.yaml --round r7        # round.json, round.md, trend.md, pareto.md
python -m dev.evaluation.suite grade-template --suite S/suite.yaml --round r7  # <topic>/GRADER_PROMPT.md per delivered paper
python -m dev.evaluation.suite grade-import --suite S/suite.yaml --round r7 --topic t --report r7/t/GRADE.md [--stop correct|wrong]
python -m dev.evaluation.suite collect --suite S/suite.yaml --round r7        # again, now with the grades
```

`--fi` is a worktree you do not edit during the round; its commit is recorded. A topic runs as `launch.py --config ...` in
its own detached process (it survives the command), with its own Axon store. The key file is read into the process
environment only. `run` starts a detached driver that keeps up to N topics running; `drive` re-attaches if it died. Only one driver runs a round at a time (it holds `driver.lock` in the round folder): a second `drive` says which process has it and stops; a lock left by a driver that is gone is taken over.

## What the files mean

* Outcome per topic: paper, honest stop, pause, crash. Trustworthy paper: graded G1, G2, G3 pass and the run's own record
  agrees. Fake paper: fails G1, or its record contradicts it (not executed, or a checks/code/run/analysis problem not
  fixed). False stop: an honest stop a grader marked wrong (`STOP: wrong`) or whose cause a grader set to `fi_bug`.
* The done criteria (trustworthy >= 70%, fake = 0, false stops <= 10%) read **met** or **not met** only when the data decide
  them whatever the ungraded papers and unjudged stops turn out to be; else **not yet known**.
* The taxonomy and its rules are documented at the top of `taxonomy.py`. A run no rule fits is `unclassified`; `fi_bug` is
  set only by a grader (`grade-import --class fi_bug --stage S`), never by a rule.
* `grades.json` holds the grader's G1/G2/G3, mean, per-dimension scores and stop verdict; it is the only grade input.

`adopt --from seed.yaml --collect` brings runs that already exist in as rounds (see `adopt.py` for the seed format).
