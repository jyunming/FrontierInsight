# How FI checks its own experiments

An experiment that runs and prints numbers is not yet evidence. FI writes down what the experiment must do **before** it runs, holds the run to that, and says at the end how much of the result was actually checked. This page explains those pieces in the order they happen.

```
plan.md  ->  protocol frozen  ->  run  ->  checks against the protocol  ->  evidence level
```

You do not have to use any of this to get a paper. The defaults already run every check and tell you plainly what was and was not shown; `rigor_profile: research` (last section) makes the checks stop the quest instead of only reporting.

## 1. The plan and its protocol

The design block can also carry a **`protocol`**: the grid the experiment sweeps (each parameter with every value), the runs per setting, the thresholds, how randomness is seeded, how uncertainty is estimated and what would count as support. The topic's own numbers (a set in braces, a count of runs) have to appear in it, and `plan.md` says under *Checks already made* which ones do not. When the experiment's script is written it is compared with the protocol: a script that sweeps other values than the plan fixed (say `R0 = [0.8, 1.2, 2.0, 3.0]` for a topic that asked for `{0.9, 1.5, 3.0}`) is sent back for up to two repairs, and if it still differs the quest stops with what differs (`engine.protocol_check: block`, the default; `warn` only records it, `off` does not look). Edit the script or the plan, then resume: a script that now agrees with the plan is used as it is.

The protocol also carries the **precision** the claim needs (`precision.target_half_width`), and the runs follow from it: `plan.md` says how many trials a probability near 0.5 needs for that width (about 1068 for ±0.03, 385 for ±0.05) against the trials the protocol plans, and the analysis is told, after the run, which probabilities did not reach the target. A script that rebuilds its random generator from the seed alone inside a function it calls for every setting (so the settings share their random numbers) is treated like any other difference from the protocol's independent-streams policy: sent back, then the quest stops. `plan.md` also says when a parameter the design makes a claim about (convergence, scaling, a threshold) has fewer than five values.

## 2. Oracles: checks the script cannot pass by declaring itself right

The protocol also declares **oracles**: checks that do not rely on the script's own numbers being right (a closed form the simulation must reproduce, a limiting case with a known answer, an invariant every run must satisfy, a small case whose exact answer is known, a second implementation). Each oracle carries the numbers it is judged by (a numeric `expected` and `tolerance`, optionally `tolerance_mode: relative`, and the `reference` the expected value comes from). Before the pilot and the main run, the script is run once with `FI_ORACLE=1`, in which it *measures* each check and prints its `value` in an `ORACLE_JSON` line, and **the engine judges**: it compares the value with the protocol's expected value and tolerance itself, and ignores any pass/fail, expected value or tolerance the script prints (a script that decides its own verdict can always pass; it is still a problem when the script reports a check as failed). An oracle with no numbers to be judged by is not run: before the freeze the plan is rewritten to give it some, after it only an amendment can. A protocol with no oracle first has the plan rewritten with one, a check that is missing, has no finite value or is outside its tolerance is sent back for up to two repairs, and if it still does not pass the quest stops before its main sweep (`engine.oracle_check: block`, the default; `warn` records it, `off` does not look). A person who edits the protocol in `plan.md` while the quest is stopped is heard at the next resume.

## 3. The run's own warnings

A run that exits 0 and prints its results can still have been told by its own numerics that something is wrong (an overflow, an invalid value or a division by zero in NumPy, a solver that did not converge, an argument that had no effect, a NaN or an infinity in a result). Those warnings are read from the run and sent back through the same repairs as a failed run; if they remain when the repairs are spent the quest stops (`engine.numeric_warnings: block`, the default; `warn` records them, `off` does not look), and a resume runs the script again if you changed it, or accepts the run with its warnings on record if you did not.

## 4. The protocol is frozen, and changes go through you

**The protocol is frozen before the first full run.** Until then the protocol in `plan.md` is a draft (you may edit it, and the engine may add an oracle to it); right before the first full run it is frozen into `needs/FROZEN_PROTOCOL.json` with its SHA-256, who approved it (you, when the plan was held for you with `pauses.plan: ask`, otherwise the engine on its own) and when. From then on every gate, and the evidence level, reads that record and never the mutable design, so a redesign after the review that leaves the protocol out cannot make the gates see nothing, and editing the protocol in `plan.md` after the freeze changes nothing (the log says the two differ). A redesign that wants a different protocol asks for an *amendment*: the quest stops (`needs/PROTOCOL_AMENDMENT_PENDING.json` says what changes and why) until a person approves it as an act of its own: `python launch.py --approve-amendment <quest_id> --approve-as <you>`, the Approve button on the quest page, or `@fi /approve-amendment <quest_id>` in VSCode. Resuming without approving keeps the frozen protocol. An approved amendment writes `needs/PROTOCOL_AMENDMENT_<n>.json`, freezes a new version and starts a new run; if the results had already been seen, the old run (raw outcomes, code, paper, evidence) is archived under `archive/run_<n>/` first, the amendment is recorded as not pre-specified, the paper is told to say so, and the evidence level names it as a gap of `publication_ready`.

## 5. What the simulation says it did

**What the simulation says it did is compared with the protocol after the run.** The static protocol check reads the source, so a constant that is never used, a loop that runs 30 times under `NUM_RUNS = 300` or a grid a helper overrides pass it. In the two-script layout `simulate.py` also writes `run_manifest.json` into its raw-data folder (the grid it really swept, the trials it attempted and completed in every setting, the failed trials, the thresholds it used, `"schema": "fi.run-manifest/v1"`), and after the first run FI compares it with the frozen protocol: a missing setting, another run count, a threshold that differs, or a failure that was dropped instead of listed (or that the protocol does not say how to treat: `failure_policy`) sends `simulate.py` back once (`engine.run_manifest_repair_attempts`) and then stops the quest with what differs (`engine.run_manifest_check: block`, the default; `warn` records it, `off` does not look). The other seeds' manifests are checked too and a difference is recorded. A stochastic quest that runs as one script writes no manifest, which the evidence level names as a gap (a deterministic one needs none). `execution.split_failure: block` makes a reply without both scripts, or scripts that break the contract (the analysis imports the simulation, the simulation prints `RESULT_JSON` or draws figures), stop the quest instead of running it as one script. Every check is in `needs/RUN_MANIFEST_CHECK.json`.

## 6. What each number estimates

**Each headline number says what it estimates.** The protocol declares `metrics` (a *metric spec* per number: `id` = the name the script uses in `RESULT_JSON`, `estimand`, `kind` proportion or mean, `unit`, `cluster` for observations that come in clusters, `paired` when trial *i* of every setting used the same random numbers, and the `family` a multiplicity correction covers), shown in `plan.md` where you can edit it and frozen with the rest of the protocol. The engine picks the estimator from the spec and not from the names in `RESULT_JSON`: Wilson for pooled counts, a bootstrap for a mean, a cluster bootstrap over whole clusters, and for a contrast between two settings a two-proportion test, a bootstrap with a permutation p-value, or a sign-flip permutation of the per-pair differences; the p-values of a family are Holm-adjusted together, and `analyze` is told to quote those and never to judge significance from overlapping intervals. A paired design over clusters, or a design whose data the script did not give (`<name>_values`, `<name>_clusters`), is reported as unsupported. A metric with counts or values and no spec keeps the old guess, is listed as undeclared, and the evidence level does not call the statistics adequate while there is one.

## 7. The evidence level

A finished quest says how far its result was checked, in six ordered levels. Each needs the one before it, and each says what it guarantees and what it does not. The record is `needs/EVIDENCE.json`; the run summary, the quest page's banner (hover a level for its guarantee and blind spots) and the `[FI] evidence:` line in VSCode show it, with the sentence that stands between the quest and the next level.

| Level | What it guarantees | What it does not |
|---|---|---|
| `executed` | The experiment ran to the end and printed results. | That the results are right, or that the run did what the plan said. |
| `internally_reconciled` | The number, statistics and provenance audits found the paper faithful to what the script printed. | That the script is right. |
| `protocol_runtime_matched` | The protocol was frozen and is intact, the script held to it, the run's manifest says it did what the protocol fixed, no numeric warning was accepted. | The manifest is the script's own statement. |
| `independently_validated` | The engine judged the script's oracle measurements against the protocol's expected values and tolerances. | The expected values come from the plan (the same model). |
| `statistically_adequate` | Every headline metric has a declared estimator matched to its data, contrasts have engine-computed p-values, precision targets were reached. | The estimators' assumptions are declared, not tested. |
| `publication_ready` | The review accepted the paper with no must-fix finding, and the protocol was not amended after the results were seen. | The review is an opinion; it does not re-run anything. |

A check that was turned off is a gap, not a pass. A record written by an older version (`internally_consistent`, `validated_against_oracle`) is shown under the current names as `internally_reconciled`, because it lacked the runtime, engine-judged and statistical checks the newer levels stand for.

## 8. The research profile

`rigor_profile: research` (default `default`; the `--new` / `/interview` / `@fi /new` interview asks for it as an advanced question and recommends `research` for a simulation study; it is chosen when the quest is created, so `--update` does not offer it) turns on together what a study needs before its result can be trusted:

- the plan is held for you to read before the protocol is frozen (`pauses.plan: ask`);
- the simulation and its analysis stay in two scripts, and a reply without both stops the quest instead of running as one script (`execution.split_analysis: true`, `split_failure: block`);
- the protocol, oracle, numeric-warning and run-manifest checks stop the quest and cannot be turned off (`engine.*_check: block`);
- the cross-check verification and the review panel (methodologist, statistician, reproducibility, devil's advocate) are on.

If the same config sets one of these to the opposite, loading is refused and the key is named: the profile is a guarantee, and a later line of YAML must not take it apart quietly. A review panel of your own is kept.

## 9. Reading and changing the plan

On the web, the quest page has a **Plan** panel: read it, *Edit the plan* and save, or type a change and press *Rewrite the plan*, then *Resume*. In VSCode, `@fi /plan <quest_id>` opens the file beside the chat, `@fi /plan <quest_id> <what to change>` rewrites it, and `@fi /resume <quest_id>` runs it. A rewrite whose design block cannot be read is refused and leaves the file as it was; a file you saved by hand with an unreadable block stops the quest again with the reason instead of being guessed at. Every version is kept in `.fi/plan_versions/`, listed in `needs/PLAN_HISTORY.json` with who wrote it (the model, your request, or you), and the first entry of `needs/DESIGN_HISTORY.json` names the hash of the plan it came from.
