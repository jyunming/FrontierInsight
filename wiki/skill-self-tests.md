---
title: Skill self-tests
sources: [core/skills/scaffold.py, core/skills/known_checks.py, core/skills/removal.py, scripts/import_scientist_skills.py]
updated: 2026-09-30
---
# Skill self-tests

A skill is a packaged piece of know-how (instructions, sometimes scripts) FI can give the model. Before a skill is used, its self-test must pass.

## Generated self-tests

A skill imported without a test of its own gets a generated `selftest.py`. It runs each bundled script with `--help`. A skill with no scripts would then only prove its files exist, so:

- a starter-list skill imported by `scripts/import_scientist_skills.py`, with its library declared, also gets **checks with a known answer from that library**: for example SciPy's gamma(5) = 24, astropy's speed of light = 299 792 458 m/s, or a Kaplan–Meier estimate of 2/3. A broken or missing install then fails the test;
- each check is plain, readable code in the test file, never a string run with `exec`, and it fails unless it explicitly succeeds;
- `--import-skill` adds no such checks. A test you wrote yourself is never touched. Re-running the import script refreshes a generated test, and the skill then has to be approved again.

## The starter list

79 skills: 75 from 11 public skill collections, and 4 written for FI (`dev/own-skills/`) on simulation correctness: invariant guards, a cross-check against a slow obvious reference, property tests, and dimensional consistency. `numerical-verification` (MIT licence) is included too. They connect to [[how-fi-judges-correctness|How FI judges correctness]]: they help the model write code that can pass its oracles honestly.

## Removing a skill

- CLI: `python launch.py --remove-skill <name>`. A skill in FI's own folder is deleted, and its approval is withdrawn first. A skill another tool installed is hidden, not deleted; `--restore-skill <name>` shows it again. A skill from a pip package gets the `pip uninstall` command to run.
- Web: the Remove button on the skills page. There is no restore button.
- VS Code: `@fi /remove-skill <name>`. There is no restore command.

`--revoke-skill` is different: it only withdraws the approval.
