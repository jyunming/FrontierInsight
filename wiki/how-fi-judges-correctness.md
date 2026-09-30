---
title: How FI judges correctness
sources: [core/oracle_check.py, core/engine.py, core/trial_runner.py, core/evidence.py, core/config.py]
updated: 2026-09-30
---
# How FI judges correctness

Most of FI's checks compare the paper with what the script printed. That cannot catch a simulation that uses the wrong model or a wrong parameter: it still prints plausible numbers. So the plan also declares **oracles**: checks with a known answer that do not rely on the script's own numbers being right.

## What an oracle is

One of six kinds: a special or limiting case with a known answer, an invariant (such as a conserved quantity), a symmetry or scaling law, a convergence rate, a published value, or a second independent implementation. Each oracle in the plan has:

- a `check` (what is compared with what, on which small case),
- a numeric `expected` value and `tolerance` (absolute, or relative with `tolerance_mode: relative`). Without them FI cannot judge the check,
- a `reference`: where the expected value comes from (see [[Oracle provenance]]),
- optionally a `case` (the settings of one small run) and a `measure` (the number that run returns), so FI can run the check itself,
- optionally an `order`: the claimed order of accuracy.

## Who does what

- The **plan** supplies the oracle, its expected value, tolerance and source. You can edit it in `plan.md`.
- The **simulation** supplies only the measured value.
- **FI** supplies the verdict: |value − expected| ≤ tolerance (or ≤ tolerance × |expected| with `tolerance_mode: relative`), with both numbers taken from the plan. A pass/fail, expected value or tolerance the script prints about itself is ignored.

## When and how the checks run

Before the pilot and the main run:

1. For a simulation FI can call (one with a `run_trial` or `run_cell` function, which the default two scripts give; see [[Engine-callable simulations]]) FI runs the simulation on each oracle's `case` in its own process, with the environment a real trial gets (never `FI_ORACLE`), so the simulation cannot tell it is being checked. That value is recorded as measured by FI.
2. An oracle without a case, and every oracle in a one-script quest (`execution.split_analysis: false`), is answered by the script itself (its `oracle()` function, or a run with `FI_ORACLE=1`). That value is recorded as measured by the script.
3. A missing value, a value outside the tolerance, a crash or a timeout is a problem. With `engine.oracle_check: block` (the default, and always under the research profile) the quest stops before the main run. With `warn` it logs and continues; with `off` there is no check.

The result is in `needs/ORACLE_CHECK.json`.

## Repairs never bend the check

A failing check asks the model to repair the script, up to `engine.oracle_repair_attempts` times (default 2). It is told to find out whether the simulation, the measurement or the check is wrong, and never to loosen, skip or hard-code a check. If it concludes the *check* is wrong (a miscalculated expected value, or a tolerance smaller than the method's own error), it may only **propose** new numbers. Its code is set aside and the script on disk is kept as it is. With `engine.oracle_check: block` the quest then stops with the proposal: before the protocol is frozen, you accept it by editing the oracle in `plan.md` or asking for the change with `--revise-plan`; after the freeze the stop explains how to ask for an amendment instead (and a quest under the research profile cannot change the check inside that quest at all). With `warn` the run goes on with the failure recorded. Nothing changes without you.

## What counts as independent evidence

The evidence level says how strongly a result is backed. A result reaches **independently validated** only when all of these hold:

- the check ran and passed, judged by FI,
- every declared oracle passed with a value FI measured itself (a check that judged no value is not evidence, and a value the script reported about itself never counts),
- every expected value has a checkable source ([[Oracle provenance]]),
- every equation that produces the numbers is labelled in the code ([[Engine-callable simulations]]).

For a random simulation a case is one trial, so only checks one trial shows exactly can be run by FI. A probability or a mean stays the script's own answer and keeps the quest below this level.

## Oracles FI added itself

If the plan had no usable oracle, FI rewrites the plan to add one. Such oracles are listed in `.fi/oracles_added.json`. With `pauses.plan: ask` the quest stops at the plan again so you can read them. If nobody was shown them, the frozen protocol records that nobody approved them; FI never records them as your approval.

## Limits

FI cannot tell a strong check from a weak one. Expected values come from the plan, written by the same model. FI checks that a source is named, not that the derivation is right.

Related: [[The model behind the numbers]], [[Scoring criteria]], [[Skill self-tests]].
