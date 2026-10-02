---
title: How FI judges correctness
sources: [core/oracle_check.py, core/oracle_card.py, core/oracle_triage.py, core/todo.py, core/engine.py, core/trial_runner.py, core/evidence.py, core/acceptance.py, core/config.py]
updated: 2026-10-02
---
# How FI judges correctness

Most of FI's checks compare the paper with what the script printed. That cannot catch a simulation that uses the wrong model or a wrong parameter: it still prints plausible numbers. So the plan also declares **oracles**: checks with a known answer that do not rely on the script's own numbers being right.

## What an oracle is

One of six kinds: a special or limiting case with a known answer, an invariant (such as a conserved quantity), a symmetry or scaling law, a convergence rate, a published value, or a second independent implementation. Each oracle in the plan has:

- a `check` (what is compared with what, on which small case),
- a numeric `expected` value and `tolerance` (absolute, or relative with `tolerance_mode: relative`). Without them FI cannot judge the check,
- a `reference`: where the expected value comes from (see [[oracle-provenance|Oracle provenance]]),
- optionally a `case` (the settings of one small run) and a `measure` (the number that run returns), so FI can run the check itself,
- optionally an `order`: the claimed order of accuracy.

## Who does what

- The **plan** supplies the oracle, its expected value, tolerance and source. You can edit it in `plan.md`.
- The **simulation** supplies only the measured value.
- **FI** supplies the verdict: |value − expected| ≤ tolerance (or ≤ tolerance × |expected| with `tolerance_mode: relative`), with both numbers taken from the plan. A pass/fail, expected value or tolerance the script prints about itself is ignored.

## When and how the checks run

Before the pilot and the main run:

1. For a simulation FI can call (one with a `run_trial` or `run_cell` function, which the default two scripts give; see [[engine-callable-simulations|Engine-callable simulations]]) FI runs the simulation on each oracle's `case` in its own process, with the environment a real trial gets (never `FI_ORACLE`), so the simulation cannot tell it is being checked. That value is recorded as measured by FI.
2. An oracle without a case, and every oracle in a one-script quest (`execution.split_analysis: false`), is answered by the script itself (its `oracle()` function, or a run with `FI_ORACLE=1`). That value is recorded as measured by the script.
3. A missing value, a value outside the tolerance, a crash or a timeout is a problem. With `engine.oracle_check: block` (the default, and always under the research profile) the quest stops before the main run. With `warn` it logs and continues; with `off` there is no check.

The result is in `needs/ORACLE_CHECK.json`.

## Repairs never bend the check

A failing check asks the model to repair the script, up to `engine.oracle_repair_attempts` times (default 2). It is told to find out whether the simulation, the measurement or the check is wrong, and never to loosen, skip or hard-code a check. If it concludes the *check* is wrong (a miscalculated expected value, or a tolerance smaller than the method's own error), it may only **propose** new numbers. Its code is set aside and the script on disk is kept as it is. With `engine.oracle_check: block` the quest then stops with the proposal: before the protocol is frozen, you accept it by editing the oracle in `plan.md` or asking for the change with `--revise-plan`; after the freeze the stop explains how to ask for an amendment instead (and a quest under the research profile cannot change the check inside that quest at all). With `warn` the run goes on with the failure recorded. Nothing changes without you.

## What you see when the checks stop the quest

Every screen calls an oracle a **known-answer check**. When the checks stop the quest, FI builds one card (`core/oracle_card.py`). The terminal, `NEXT_STEP.md`, the web quest page and the VS Code chat all show it, from one payload kept in `.fi/pause.json` and `.fi/todo.json`. For each check that did not pass, the card shows:

- what it checks, and its kind;
- the expected value and where it comes from;
- the measured value and who measured it (FI on the check's case, or the script itself);
- the tolerance, and the gap: how far off, how many times the tolerance, and the measured value as a multiple of the expected one;
- the case and the measure;
- the file and line where the script computes the number.

Then the card gives the most likely cause, using only what FI already has: a repair's proposal and its reason, the test run's size verdict, two checks under one name, or the error and line when nothing was measured. It also lists what FI tried and two or three ways on, each with its command. Resuming repairs the script again, up to `engine.oracle_repair_attempts` times (not offered when every failing check is disputed or set aside: that finding is kept); a check with no numbers is sent back to the plan instead. A change to the check is filled in as a request to work the expected value out again from its source; it never takes the measured value. The card reports the verdict and never changes it.

Before the first repair of a failing check, FI looks itself (`core/oracle_triage.py`). Another model works out the expected value again without seeing the measured one; if it lands near the measurement, the expected value is disputed and offered as a proposal (never for a violation expecting 0). A deterministic check with a step is run at half and a quarter of it and extrapolated to step 0; if it converges at the declared order (or 1.5 or more) to the expected value, the gap is the method's own error. A random check gets three more seeds; a spread larger than the tolerance, with a mean that agrees, is noise. A usual unit factor (×100, ×1000, ×2π, the reciprocal, a count) is named on the card. The same exception after a repair stops the repairs, and a timed-out run gets one retry at twice the time. A check the recompute, the smaller step or the seeds point to is not repaired for, and it still fails. Every finding is under `attempts[].triage` in `needs/ORACLE_CHECK.json` and on the card. A dispute survives a resume while the check is unchanged. (A check that names no source for its expected value stops earlier, at the plan, with its own "go on as it is": [[oracle-provenance|Oracle provenance]].)

### Going on although a check failed

When every problem the stop found is a check that was measured and failed, the card offers one more way on: **mark the check unconfirmed and go on** (`--accept-checks <quest> --approve-as <you>`, the quest page's button, `@fi /accept-checks`), before or after the freeze, under research too. It needs a name, and it repeats what it binds to: the check's expected value, tolerance, case and measure, the measured value, and the script with its version (a hash of the script the checks ran and the model's package in `code/`). On the next run of the gate the choice is honoured right after the checks are measured, before any plan change or repair could alter the code; if the check's numbers or the code changed, it no longer applies and the check is judged again (so a review that sends the quest back to change the code gets the check judged again too). It is not offered when nothing was measured: there is no failure to record, and the card says why. An exploration (`result_use: explore`, or unsaid, outside research) goes on by itself after FI's repairs, recorded as automatic and listed on the to-do card; research and decision quests stop. Either way the check stays failed: `needs/ORACLE_CHECK.json` says `went_on_failing` and who, the audit trace names them, the evidence keeps a gap below independently validated ("<name> chose to go on although the known-answer check '<check>' failed (…)"), and the paper is told to say so. This replaces the card's old advice to set `engine.oracle_check: warn`, which on a quest the interview wrote stopped it a second time to approve the changed setting.

## What counts as independent evidence

The evidence level says how strongly a result is backed. A result reaches **independently validated** only when all of these hold:

- the check ran and passed, judged by FI,
- every declared oracle passed with a value FI measured itself (a check that judged no value is not evidence, and a value the script reported about itself never counts),
- every expected value has a checkable source ([[oracle-provenance|Oracle provenance]]),
- every equation that produces the numbers is labelled in the code ([[engine-callable-simulations|Engine-callable simulations]]),
- under `rigor_profile: research`: a second model, shown by the record of the calls to differ from the one that wrote the checks, read them and gave a usable answer, and each invariant, symmetry or second-implementation check also passed at a setting the code never saw ([[oracle-provenance|Oracle provenance]]).

For a random simulation a case is one trial, so only checks one trial shows exactly can be run by FI. A probability or a mean stays the script's own answer and keeps the quest below this level.

The top level, **publication ready**, also needs a person: the review's accept is a model's opinion, so a result accepted with no person asked (`pauses.auto_accept_on_pass`, or `pauses.review: off`) is marked "not reviewed by a person" and stays one level below it. Before a person accepts they see what the result does not guarantee and its main gaps, and answer one question: have they reviewed the evidence record, and do they accept these claims and the limits listed ([[pauses|Pauses]]). Only an explicit "yes" reaches `publication_ready`; "I did not check" and "partly" (with its note) each leave a gap one level below it. The answer is recorded as a receipt bound to the paper and evidence record it was given for.

## Oracles FI added itself

If the plan had no usable oracle, FI rewrites the plan to add one. Such oracles are listed in `.fi/oracles_added.json`. With `pauses.plan: ask` the quest stops at the plan again so you can read them. If nobody was shown them, the frozen protocol records that nobody approved them; FI never records them as your approval.

## Limits

FI cannot tell a strong check from a weak one. Expected values come from the plan; under research a second, different model reads them, but it does not work them out again, and outside research nothing requires another model. FI checks that a source is named, not that the derivation is right. A special case, a published value and a convergence rate are never run at a hidden setting.

Related: [[model-behind-the-numbers|The model behind the numbers]], [[scoring-criteria|Scoring criteria]], [[skill-self-tests|Skill self-tests]].
