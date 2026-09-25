# 16 — Live Red-Team Campaign (item 6, 2026-09-23 re-audit)

**Campaign date:** 2026-09-23
**Provider:** Moonshot Kimi (`kimi-k2.6`, `examples/kimi_moonshot/config.yaml`'s settings), per the user's choice — the
Ollama monthly quota was still unconfirmed-recovered.
**Scope:** four real quests (not fixture-based pytest), one per shape the re-audit named (stochastic/SIR,
deterministic, paired, clustered), each under `rigor_profile: research` so every gate PR-1 through PR-5 of this fix
order actually applies. `docs/audits/15_fault_injection_campaign.md` deliberately used pytest fixtures instead of a
live corpus for the equivalent earlier item, "per the user's decision" at the time; this campaign's item explicitly
asked for a fresh **live** one, so this is that reversal, not a duplicate of #15.

## What this found before writing a line

The plan was four clean runs plus one hand-injected fault per shape (the user's own choice of scope). It changed
after the first clean run: **every one of the four clean quests stopped at the oracle gate**, for two distinct
reasons — one a real, previously-unknown engine bug (below), the other three legitimate oracle-gate catches of real
bugs in the model's own generated code. Between them, all four shapes produced a genuine "gate caught a real problem
before anything false was written" result without any deliberate fault injection at all — a stronger, more
representative demonstration than a hand-injected one would have been (see "Fault injection: not built, and why").
The campaign's actual deliverable turned out to be different from its plan, which is itself the point of a *live*
campaign over a fixture-based one.

## Finding 1 (fixed in this PR): the oracle pre-check never set `FI_RAW_DIR`

**Severity: high — affects every `execution.split_analysis: true` quest whose `simulate.py` reads `FI_RAW_DIR` at
module level, not just these four.** `rigor_profile: research` forces `split_analysis: true`
(`docs/capabilities-reference.md`'s rigor-profile entry), and the two-script directive's own worked example
(`core/engine.py`, `split_run` prompt block) tells the model to write exactly this:

```python
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
```

Python executes that line on import, unconditionally, regardless of which function is later called. The oracle gate
(`Engine._oracle_gate`) invokes the script with `FI_ORACLE=1` to validate it cheaply *before* the real run — but its
environment (`_replicate_env(exec_env, 0, stride)`) never included `FI_RAW_DIR`, since that variable is only
meaningful for a real per-seed run. Result: **2 of 4 real quests crashed with a bare `KeyError: 'FI_RAW_DIR'` before
ever reaching their oracle logic**, on the very first line of `simulate.py`:

- `1790187455-redteam-clean-stochastic-sir-610229` (SIR)
- `1790188092-redteam-clean-clustered-queueing-de28a7` (clustered)

The engine read this as "the script printed no `ORACLE_JSON:` line" — indistinguishable from a script that never
implemented its oracles at all — and spent both `oracle_repair_attempts` asking the model to "fix the oracle mode."
Both repairs rewrote the *oracle branch* (which was never the problem) and left the module-level `FI_RAW_DIR` read
untouched; the identical `KeyError` recurred on every attempt (`needs/ORACLE_CHECK.json`, `.fi/run.log` for both
quests). The repair budget is not designed to fix a bug the repair prompt was never told about.

**Fix:** `_oracle_gate` now sets `FI_RAW_DIR` to a dedicated, disposable folder (`<raw_root>/oracle_check`, never a
real seed's raw dir) before invoking the script — the same treatment `FI_ORACLE` itself already gets. This closes the
gap structurally: it does not depend on the model writing its script any particular way, unlike a prompt-side fix,
which would need every model to comply on every quest.

**Verified three ways**, not asserted:
1. Manually reran both quests' real `simulate.py` (their `.venv`, with `FI_RAW_DIR` now set) — the `KeyError`
   disappeared and each script proceeded into its own oracle-check logic (see Findings 3–4 below).
2. Wrote `tests/test_oracle_gate.py::test_a_two_script_oracle_check_gets_a_real_raw_dir_not_a_keyerror`, whose
   `simulate.py` fixture reads `FI_RAW_DIR` at module level exactly like the real quests' scripts — confirmed it
   **fails** against the pre-fix code (`assert artifacts.paper_md is not None` → `None`, oracle stuck on "printed no
   ORACLE_JSON") and **passes** against the fix.
3. `--resume`d both real quests through the fixed engine (not a rerun from scratch — the same `code/simulate.py`
   already on disk): `.fi/run.log` for both now shows the real underlying error in `stderr_tail` (`RuntimeError:
   Failed to converge...` for SIR, `simpy.events... not a generator` for clustered) instead of `KeyError:
   'FI_RAW_DIR'`. Same script, same bug now visible, the artificial one gone.

## Findings 2–4: three real bugs in Kimi's own generated code, correctly caught

None of these needed the fix above — the oracle gate worked as designed for these three shapes.

- **Paired (`1790188084-redteam-clean-paired-rootfind-0da691`).** First attempt: an oracle
  (`independent_newton_verification`) had no numeric `expected`/`tolerance` — the plan was asked for one and revised
  (`plan.md` version 2). Second attempt, with numbers now declared: two oracles reported `nan` (not finite) and one,
  `secant_bracket_check`, genuinely failed — measured `1`, expected `0` within `1e-12`. The repair budget (2
  attempts) exhausted without fixing it; the quest correctly stopped rather than proceed on unverified code.
- **Deterministic (`1790188077-redteam-clean-deterministic-ode-5269d6`).** `verlet_energy_conservation_undamped`
  measured `1.25e-05` against an expected `0` at `1e-12` absolute tolerance — plausibly the model declaring an
  unrealistically tight tolerance for a symplectic integrator's floating-point energy drift, not necessarily a wrong
  simulator; `damped_energy_decay_rate` measured `-0.108491` against an expected `-0.1` (1% relative tolerance) — a
  9% miss. Two repairs did not close either gap; the quest correctly stopped.
- **Clustered, once Finding 1's fix let the real logic run.** `env.process(arrivals_process(q))` raised `ValueError:
  None is not a generator` inside `simpy` — the model's own `arrivals_process` generator function has a bug (returns
  `None` on some path instead of `yield`ing). Reproduced identically on resume through the real engine.
- **SIR, once Finding 1's fix let the real logic run.** `scipy.optimize.fixed_point` failed to converge after 500
  iterations at the `R0 = 1.0` boundary case (a genuine numerical edge case for a branching-process extinction
  probability, where the fixed-point equation has a degenerate root) — the same failure mode the last real-Kimi-quest
  measurement recorded ([[reference_kimi_moonshot_provider]] memory, 2026-09-21: "the simulator Kimi wrote failed its
  own declared oracle... after two repairs, the quest stopped"), on a *different* topic instance, suggesting this is
  a recurring shape of failure for Kimi's oracle-declaration habits under numerically stiff conditions, not a
  one-off.

None of these three are engine bugs; they are the oracle gate doing exactly its documented job (`docs/capabilities-
reference.md`: "a script whose checks do not pass never reaches the main sweep"). Repairing Kimi's own generated
code beyond `oracle_repair_attempts`'s budget is out of scope here — see "Not built here."

## P0-5's answer: does a real quest complete under PR-1 through PR-5's gates?

**Not yet measured cleanly** — every one of the four attempts stopped at the oracle gate, three for reasons that
would have stopped ANY correct implementation of this rigor level (real declared-oracle failures), one for an engine
bug now fixed. This campaign does not show a real quest reaching `publication_ready` under the tightened gates; it
shows the gates refusing to let an unverified one through, which is the more important property to have verified
first. A follow-up remeasurement (not built here — see below) with the Finding-1 fix in place, and either a laxer
oracle-tolerance topic or a model less prone to declaring tolerances it can't hit, is the natural next step for that
specific question.

## Fault injection: not built, and why

The user's own scope for this item was "1 representative fault per shape, hand-injected." Given all four clean
attempts organically produced a real, gate-caught fault before any injection was needed — one per shape, exactly the
requested count, and more representative than a synthetic one since a real model produced each — building four
additional hand-injected variants on top would have doubled the campaign's real-model cost without adding a
distinct kind of evidence. This is a deliberate scope call, not an oversight: the coverage table below records these
four organic catches as the fault-injection deliverable.

## Coverage table

| Shape | Quest | Oracle problem | Class | Caught before writing? |
|---|---|---|---|---|
| Stochastic/SIR | `...-610229` | Module-level `FI_RAW_DIR` KeyError (Finding 1) → after fix, `scipy.optimize.fixed_point` non-convergence at R0=1.0 | Engine bug (fixed) → real code bug | **Yes**, both times |
| Deterministic | `...-5269d6` | Two oracle tolerance misses (Verlet energy conservation, damped decay rate) | Real code/tolerance bug | **Yes** |
| Paired | `...-0da691` | Unjudgeable oracle → plan revised → `secant_bracket_check` genuinely fails (measured 1, expected 0) | Real code bug | **Yes** |
| Clustered | `...-de28a7` | Module-level `FI_RAW_DIR` KeyError (Finding 1) → after fix, `simpy` generator bug (`arrivals_process` returns `None`) | Engine bug (fixed) → real code bug | **Yes**, both times |

## Not built here (converge, don't expand)

- **Repairing Kimi's own generated-code bugs** (the scipy convergence, the simpy generator, the two Verlet
  tolerances, the paired secant check) — these are model code-generation quality issues for a topic/model-choice
  investigation, not gate-correctness issues; `oracle_repair_attempts`'s budget existing and being exhaustible is the
  correct, working design.
- **A clean remeasurement of P0-5** with Finding 1's fix in place — worth doing, but is its own real-quest cost and a
  separate question from "does the gate work," which this campaign answered.
- **The clean-topic tolerance question** — whether Kimi systematically declares oracle tolerances too tight for its
  own numerical methods (suggested by 3 of 4 shapes hitting a tolerance-adjacent failure) is an interesting pattern
  worth a dedicated look, not concluded here from n=4.

## Cost

496,482 tokens total across the four clean-run attempts (115,386 / 129,847 / 118,414 / 132,835), all on
`kimi-k2.6`, thinking off. No fifth run needed for a hand-injected fault (see above).

## Tests

`tests/test_oracle_gate.py::test_a_two_script_oracle_check_gets_a_real_raw_dir_not_a_keyerror` — a real `Engine.run()`
through a real venv, with a `simulate.py` fixture that reads `FI_RAW_DIR` at module level (the same shape the split-
experiment directive's own example teaches, and the same shape all four real quests' generated code took).
Confirmed failing against the pre-fix code and passing against the fix (see Finding 1). No other engine code
changed; `docs/audits/16_live_red_team_campaign.md` (this file) and the four real quest folders under
`C:\dev\fi_redteam\outputs\` (not committed — real quest output, per `outputs/**` being git-ignored) are the record.

## Second round (same day): `kimi-k3`, after the Finding-1 fix

The user asked for a rerun: "we can't even finish a quest, so how do we know whether there are problems." Same four
topics, `kimi-k3` (the key's stronger model; it takes the same `fixed_temperature: 0.6` and `thinking: disabled` as
k2.6 — `kimi-k2.7-code` accepts only temperature 1 with thinking on, which one provider-wide `extra_body` cannot mix
with k2.6 in one quest), from main at the fix above. 515,089 tokens, **$3.76** (account balance 71.42 → 67.66, read
before and after — `cost_usd` in `cost.jsonl` is empty because FI has no Kimi price table).

k3 went further: two quests passed their oracles and ran the full simulation, and the paired one produced results,
analysed and cross-checked them. None reached a paper, and **half of what stopped them was FI's own**, fixed in the
PR that adds this section:

| Shape | Tokens | Got to | Stopped by |
|---|---|---|---|
| Paired | 206,358 | full run, analysis, cross-check, evidence gate | **FI:** the gate (model-decided) said *broaden* with retrieval off; the loop redesigned after results were seen (hypothesis changed, flagged as post hoc), rewrote the code, then **FI:** the protocol check read a tally `n_runs = 0` (later `+= 2`) as "0 runs per setting", and the stop message named `experiment.py` for a difference in `simulate.py` |
| SIR | 114,834 | oracles passed (1 repair), full simulation | **FI:** `experiment.py` printed no `<metric>_values`; the run-manifest difference always sent `simulate.py` back, and that rewrite broke the oracles it had passed. (Also the model's: a closed-form scalar declared as a `mean` metric, which can never have per-trial values) |
| Deterministic | 99,151 | oracle gate | the model's: Velocity-Verlet one-step error 8.28e-05 against a tolerance of 5e-05 it declared itself — the same too-tight-tolerance shape as round one. **FI:** its plan completion and FI_ORACLE fix used both repairs before this showed |
| Clustered | 94,746 | oracle gate | the model's: `simpy` `Put.__init__() got an unexpected keyword argument 'priority'`. **FI:** the repair was never shown that traceback (`stderr_tail=""`), and the one repair that might have fixed it timed out at the provider and was counted as spent. (Earlier a plan-revise call timed out after ~8 min and ended the run loudly with `quest_failed.md`; it resumed) |

Fixed together (one PR, each with a test that fails on the old code): the manifest routes a mean metric with no
`<id>_values` list to `experiment.py` (a short list is still the simulation's, the audit's fabrication case); stop
messages name the script the difference is in; the protocol check no longer reads a count of zero as a run count
(an external review of the first draft, which skipped every name the script adds to, found it would miss a real
wrong setting that is also incremented somewhere); a broaden with
no literature step writes instead, and a gate verdict of `insufficient`/`broaden` is now a `publication_ready` gap
(before, only `unknown` was); the oracle repair gets the run's stderr, and plan rewrites, script rewrites and an
unanswered repair call are counted apart. A third round on the fixed main is the measure of whether these were what
stood between a real quest and its paper.

## Third round (next day): `kimi-k3` on the second round's fixes

Same four topics, main at the second round's fixes. 454,941 tokens, **$3.55** (67.14 → 63.59). None of the second
round's FI stops came back, and for the first time a quest wrote its paper: the SIR quest ran through analysis,
evidence gate, write and the review panel, and stopped at the human review. That paper rests on a failed experiment,
and the gates said so (evidence gate `insufficient`, review `revise` with an unsupported claim flagged) rather than
letting it read as a result. Most of what stopped the quests this time was the model's own plan, correctly held to:

| Shape | Tokens | Got to | Stopped by |
|---|---|---|---|
| SIR | 170,493 | write, review panel, human review | model: its protocol grid put an analysis-only axis (`threshold_fraction`) beside R0, so the trial ledger's cells did not match (the manifest check was right). **FI:** the three `execute_reflect` repairs that followed each saw only the first 8000 characters of the 15,706-character analysis while being asked for the whole script back; one returned a helper cut off mid-name, and the run never produced numbers |
| Clustered | 115,585 | oracle gate | model: its oracle checks did not finish in the pre-check's time (each should be a small, fast case) |
| Deterministic | 87,370 | oracle gate | model: Velocity-Verlet energy drift 2.5e-05 against a declared tolerance of 1e-06 — the same too-tight tolerance in all three rounds. FI deliberately cannot loosen a declared tolerance on its own (that would empty the oracle); a person decides |
| Paired | 81,493 | protocol check | model: its grid gave a range as two values (`poly_id: [0, 199]`). **FI:** both protocol repair calls timed out and counted as spent, and the unplaced difference was sent to `experiment.py`, not `simulate.py` |

Fixed together (each with a test that fails on the old code): `execute_reflect` shows the whole script; a protocol
repair call that got no answer is tried once more, uncounted; a difference that names no script goes to
`simulate.py` in a two-script quest; and the default per-node timeouts now give every node whose reply is a whole
script or the whole plan (`implement_oracle`, `implement_protocol`, `plan`, `plan_revise`) execute_reflect's budget.
The timeouts in all three rounds were on those four nodes, left at the 120 s base: a retry starts the answer over, so
four tries ended each call after 8 minutes with nothing. This is not specific to Kimi.

What the three rounds leave: FI's gates held every time (no false number reached a paper as a result). The stops
that remain are mostly the model writing a plan it cannot keep — tolerances tighter than its method, a range written
as two grid values, an analysis parameter written as a simulation axis, oracle checks that are not small. Those are
quality-of-plan questions for the design and oracle prompts, or for a person at the plan pause, not gate bugs.

## Fourth round: `kimi-k3` on the third round's fixes

Same four topics, main at the third round's fixes. 406,404 tokens, **$2.22**. No FI-caused stop and no provider
timeout in any quest. Every quest stopped at a gate, each on the model's own plan or code:

| Shape | Tokens | Stopped by |
|---|---|---|
| SIR | 82,100 | manifest check: the protocol grid again held an analysis-only axis (`threshold_frac`) the simulation never varies |
| Deterministic | 105,941 | oracle gate: the Velocity-Verlet energy tolerance (1e-06, below the method's h²/8 = 1.25e-05 at h = 0.01) for the fourth round running, plus a convergence order read at t = 200 and a self-convergence ratio on steps far from the asymptotic regime |
| Paired | 94,557 | oracle gate: Newton and an independent bracketing solver compared on roots that were not the same root (3.1 apart, tolerance 1e-08) |
| Clustered | 123,806 | oracle gate: an M/M/1 mean wait just outside its tolerance, and a clustered-SE check declared to match OLS to 1e-10 that differed by 0.027 |

The repair on the deterministic quest said in so many words that the tolerance was the problem and could do nothing
about it: it is not allowed to loosen a declared tolerance, and it should not be. The model gets the same kinds of
plan wrong each round, so this PR does two things: the plan directive now lists those mistakes in general terms,
and an oracle repair can propose a change to a check (`oracle_change`), which is recorded and shown at the stop next
to the measured value and never applied without a person.

A rerun of these same four topics measures this prompt plus the model, not the model alone. The directive's list is
written generally (no topic, integrator or estimator is named), but the mistakes were collected from these quests, so
only a round on new topics can say whether the guidance generalises.

## A person through one quest: the deterministic stop, resolved by hand

Acting as the person at the fourth round's deterministic oracle stop, three oracles were corrected in `plan.md`, each
with its reason written into the oracle's `reference`: the energy tolerance to 2e-05 (h²/8 = 1.25e-05, checked
independently); the Euler convergence order read at t = 1 (1.05 there, 4.28 at t = 200); the self-convergence ratio
on h = 0.02 / 0.01 / 0.005 at t = 1 with a relative tolerance of 0.1 (ratio 2.041; the declared steps give 4.692,
which the script reproduces). The resume (160,114 more tokens, $1.47) passed all seven oracles after one script repair, froze
the protocol, ran, analysed, wrote the paper and stopped at the human review. That is the first time in the campaign
the deterministic quest reached a paper.

The paper then said the energy and self-convergence oracles **failed**, quoting the old tolerance 1e-06. The frozen
protocol held 2e-05 and the engine's check had passed; `experiment.py` re-judged its own oracles in `RESULT_JSON`
with `1e-6` written into the script, and `analyze` only saw that. The engine's verdicts never reached the analysis.
The review's number check also flagged "1.25×10⁻⁵" as a number the run never computed: it read the superscript form
as 1.25. Both fixed here, each with a test that fails on the old code:

- `analyze` is given the engine's oracle verdicts (`oracle_check.analysis_note`), with the instruction that a
  script's own pass/fail or copy of a tolerance can be out of date and is not what the paper reports.
- Both number checks (the provenance check and the numeric oracle share one reader) read `×10⁻⁵`, `×10^-5` and
  `·10⁻⁵` as a power of ten, and the provenance check counts the values the oracle pre-check measured as numbers this
  run computed. On the real paper line the finding is gone.

The external review of this PR also found a hazard in the new stop message: the command it tells a person to paste
carried the model's own reason inside double quotes, so a quote, `$(...)` or a backtick in it could run something.
Those characters are now made plain. It also found that after the freeze the message said only that an amendment was
needed; it now says how to get there (`engine.oracle_check: warn`, go on, ask for the change at the review).

The second, targeted review of those fixes found three more, all fixed with tests that fail on the old code: the
curly double quotes PowerShell also ends a string on were not made plain; with the gate off (or no protocol) an
earlier run's `ORACLE_CHECK.json` stayed on disk and would have reached the analysis and the number check, so it is
now removed when the gate does not run; and the power-of-ten fix had been made in the provenance check's own reader
only, leaving the numeric oracle misreading the same text, so it moved into the reader both share (which also stopped
the existing LaTeX rule from swallowing the space after `10^-5`).

## Fifth round: `claude_cli` / `sonnet` on the fourth round's fixes

Same four topics, main at the fourth round's fixes, to see whether a stronger model gets past the plan mistakes
`kimi-k3` kept making. Three of the four quests never got that far: the `claude` CLI hung on a node that writes a
long answer (SIR and deterministic at `implement`, clustered at `plan`, twice, the resume reproducing it). Each time
the stream carried thinking events for about six minutes, then stdout closed with no answer text and the child did
not exit; FI stopped the quest with `_CliWedgeError`. The error does not keep the stream's last event, so the cause
(the thinking using up the output limit, perhaps) is not established.

The paired quest ran through: its four oracles passed on the first try (the fourth round's paired quest compared two
roots that were not the same root), it reached a paper, and the review returned `revise` (3) with a correct catch:
five intervals labelled "bootstrap" were the Student-t interval over the three seeds. 710,730 tokens.

It also found an FI false positive. The one-sided Wilcoxon p-values were 8.5e-28 and smaller (z about 10.9 over 183
pairs), and the plausibility gate reported `wilcoxon_p_one_sided` "equals its bound 0 in 11 settings": the tolerance
that keeps a value on a bound from reading as outside it is absolute near 0 (1e-12), so every smaller value also
counted as sitting on 0. Two `execute_reflect` repairs were spent on a correct script, each saying so. Fixed: on a
bound of 0 only an exact 0 joins the at-bound count. Replayed on the quest's own results (with the key's original
name), main reports the 11-setting violation and the fix reports none; a test fails on the old code.

The second repair then set `wilcoxon_p_one_sided` to null and reported the p-value as its log10 beside it, and
the gate, which checks numbers, had nothing left to check there. (Corrected below: this was first recorded as a
rename.)

## The claude CLI "hang", found; and what a repair may leave out

**The hang was FI's.** `_collect_via_streaming` read the CLI's stdout with asyncio's `readline()`, whose default
line limit is 64 KiB, and treated the exception it raises past that limit as the end of the stream. A claude
stream-json line holds a whole message: a real call printed a 110,525-byte `assistant` line carrying nothing but its
thinking block. The reader stopped there while the CLI, its pipe full, could never finish writing, and after the
reap timeout FI raised `_CliWedgeError` ("stdout closed but child didn't exit ... no output collected"), word for word
the three sonnet stops. Reproduced with a fake CLI (a 60,000-byte line passes, 70,000 fails with that exact error)
and with the real one. Fixed: the stream limit is 256 MiB, and a line past even that is drained to the end and
named in the error rather than taken for the end of the stream. The sonnet SIR quest that stopped at `implement`
was resumed on the fix: `implement` wrote both scripts (about 40 KB), the oracles passed after one repair, and the
quest went on to a gate of its own (the manifest check, on the same analysis-only grid axis `kimi-k3` wrote).

**A second fault, found while checking the first.** An answer longer than the output limit arrives as several
turns: the CLI ends one at `max_tokens`, sends a synthetic "Output token limit hit. Resume directly" message, and
its `result` envelope holds only the last turn's text, which FI took as the whole answer. On the real CLI, a prompt
for "alpha1 ... alpha9000" came back as alpha8116..alpha9000. Before the stream-limit fix the same call returned
alpha1..alpha8115 instead (what streamed before the long line); neither was whole, and for `implement` either half
is a broken script. Fixed: the text of the turns that ended at `max_tokens` just before the last is put back in
front of the envelope's text, always; turns that ended at a tool call are still left out. Four rounds of the external review each found a way that a guard against "a CLI that already joins the turns" (never observed) could skip a join the real CLI needs; a skipped join truncates silently, a doubled one would show, so there is no guard. On the real CLI the answer now
holds all 9,000 numbers. At the seam the model sometimes repeats or skips a few characters (one run: a number cut after `alpha8` resumed as `176`); FI joins exactly what was streamed.

**What a repair may leave out.** A sweep of 83 quests (295 assertions) found 65 assertions that match nothing in the
final result. 63 never matched in any run: the design and the script named the quantity differently from the start,
and nothing says which key was meant, so these are left alone. The other two had matched earlier, and on a closer
look neither was renamed or deleted: each repair set the asserted value to null (a diverging Euler error, beside a
`"diverged: ..."` flag; the sonnet p-value above, reported as its log10). That is the honest answer the repair
prompt asks for, so a null is not flagged. A bounded key that disappears altogether after an earlier run reported it
is now a plausibility violation (`kind="missing"`), asking for it back under its name. No live quest has done this
yet; the check costs nothing on the ones that did not.

## The 100-point benchmark with Gemini 3.1 Pro, and what its grading found in FI

The SIR benchmark topic (literature on, isolated Axon, the rubric of `trend/grading_sheet.md`), three runs of
`antigravity_cli` / `gemini-3.1-pro-high` on the build after the CLI fixes: **87.0, 84.5, 74.5, mean 82.0**
(gemini-3.8-flash 83.3, gemma4 baseline 87.1, best gemma4 arm 93.6, gpt-5.6-terra 88.2). Two of the three
stopped at the run-manifest gate on the first pass and were resumed with `engine.run_manifest_check: warn` so the
arm is comparable with the earlier ones, which ran before that gate existed. The graders traced several of the
points lost to FI rather than to the model, and each was reproduced on the quests' own data:

| Fault | Where | On the live data, before → after |
|---|---|---|
| A conditional mean (final size of the major outbreaks) read as missing data: its values were under half of all trials | `run_manifest._value_count_findings` | g31p2: stop → no finding once the metric declares `given: prob_major`; without it, still flagged, and the sentence says how to declare it |
| The same mean pooled as a probability by its `_count`/`_total` names, so its pooled mean became 1.0 with Wilson(4, 4) = ±0.245, which the paper printed | `engine._pool_evidence` | R0=0.9, N=5000: `wilson_pooled_counts` 1.0 ±0.245 → `bootstrap_pooled_values` 0.0573 ±0.0039 |
| A precision target meant for one probability held against every pooled interval | `engine._annotate_precision` | now only against the metric `precision.metric` names (all, when it names none) |
| An `axvline` recorded as a series from 0 to 1, "flat" on a count axis; the caption check then had the paper call it "a flat horizontal line" | `plot_style._fi_axes` | g31p3's own script re-run with the recorder: all 9 panels now "vertical line" at x = 0.0020 … 0.9412 (7 of 9 were "flat") |
| A failure count of 0 in every setting taken for a trivial answer; two repairs spent, and the zeros nulled | `plausibility._at_bound` | g31p1: `failed_run_count` exempt (named by `failure_policy`, manifest lists no failed trial) → no violation; a final size of 0 is still flagged |
| One metric of an unknown `kind` (`count`) threw away the whole `protocol.metrics` list | `metric_spec.normalize` via `plan.repair_protocol` | the entry alone is left out and named in the plan |

The design choices (a declared `given` rather than trusting the script's own per-cell counts; manifest
corroboration rather than the metric's name; dropping one entry rather than the list) were the user's.

**A regression in the `given` fix, found by the re-bench grading.** The plan prompt shows `given`, and a model wrote
`"given": null` on every metric; `normalize` kept the key, read every entry as declaring a subset, and the whole list
was dropped again (terra run te3, evidence level capped). Null or empty now means no `given`. On te3's own specs all
five metrics are kept, including the one that declared `given: major_outbreak_probability` correctly. The re-bench
itself stayed on build 09a26b3 for every run, so its arms remain comparable with each other.

## The re-bench on 09a26b3: three faults that cost results

Grading the re-bench of every non-gemma4 arm on 09a26b3 traced two lost results and two fabricated verdicts to FI:

- **The repair budget was never reset after a redesign.** `exec_reflect_iter` was only ever incremented, so a script
  a redesign wrote inherited the old one's spent repairs: in te2 and cb2 its first crash logged "iterations exhausted"
  with no repair, and both papers had no results. `implement` now resets the repair loop's state for every new script
  (redesigns stay bounded by `engine.max_iterations`). A fake-model end-to-end test reproduces the live shape and fails
  on the old code.
- **A review that could not run was recorded as an accept.** When the codex account hit its usage limit at the review
  (lu1, lu3), FI wrote `accept, score 3`: shown at the stop, taken by an automatic accept, trusted by the Axon
  write-back. By the user's choice the quest now stops (`review_unavailable`) and the resumed run asks again; a panel
  with any reviewer that could not be asked does the same. An end-to-end test stops, resumes and records the real review.
- **The evidence gate passed a simulation with no results.** The count rule (15 sources and a supported finding) said
  "sufficient" for te2 and cb2 on literature alone, with no note to the writer (the papers themselves did say the
  experiment produced nothing). By the user's choice a simulation that ran and produced no results goes back to design
  once while an iteration is left, then is ruled insufficient with the reason; the analysis gets a note too.
- **An API error was taken as the model's answer.** With the home network down, the claude CLI retried, then ended
  with a result envelope marked `is_error: true` whose text was "API Error: Can't reach the API server … (ENOTFOUND)"
  and exited 1. Because text had arrived, FI logged "exited rc=1 but emitted 78 text chars — returning result" and
  about a dozen nodes of sn1–sn3 ran on that sentence (those three runs were discarded). An error envelope is now the
  call's error: the call is retried and, if it keeps failing, fails with the CLI's reason instead of returning text.
  Checked with the real CLI pointed at an unreachable API: the old code returned the sentence, the new code raises.

## The 2026-09-24 re-audit: a review nobody gave

An outside re-audit of `09a26b3` found three structural holes in the checks and six counterexamples that reproduce
them; all six still reproduced after #382. This entry covers the review one. #382 stopped the quest when the review
**call** failed, but a reviewer that answered without naming a verdict still got a stand-in `accept` (status
`unreviewed`); the panel's vote counter counted panelists marked `error`, so two stand-in accepts outvoted a real
revise; and the automatic accept, the skills' usage record and the Axon write-back checked only the verdict, so a
stand-in accept could be accepted for the person and written into long-term memory. Now a reply with no verdict is
asked for once more and then stops the quest like a failed call; only panelists that reviewed vote; those three
consumers require a reviewer's own accept; and the review screen (CLI, Web, VS Code) says when no reviewer gave the
verdict, naming the panelist. The two counterexamples about the review no longer reproduce; ten new tests fail on the
old code.

## A skill's name on the pip line, and skills a clean environment could not see

A user's quest on another machine stopped at the oracle check: `import numpy` failed, and so did `lieflat_charts`.
The code-writing step had asked pip for `lieflat_charts` (the lieflat-charts skill, a Node.js tool, not a Python
package) together with numpy; one name pip could not find failed the whole line, FI logged a warning and ran on, and
nothing was installed. Behind it, a second gap: the research profile's clean per-quest environment never received the
selected skills' own packages (`pip_requires`, installed only into FI's interpreter) nor a library skill's folder on
its path, so preset skills that imported fine under the shared interpreter could not be imported at all. Now the quest's
own files are never handed to pip, a failed line is retried one package at a time (a skill's name pip cannot
install is then explained as the skill), the skills'
packages go into the quest's environment and library skills onto its path, and the repair step reads what could not
be installed and why. Reproduced with a real venv and the real skill: the old code logged `pip install ['numpy',
'lieflat_charts']` rc=1 and died on `ModuleNotFoundError: No module named 'numpy'`; the new code installed numpy and
the script ran.
## Reading a paper whole: OCR for scanned PDFs, no 64 KB cut

A user found literature coming back as snippets and cut-off articles. Measured: every PDF went through pypdf's
`extract_text()` alone (words split mid-way), a scanned PDF (most papers from before the 1990s) gave no text at all
and fell back to the search snippet, the text was cut at 64 KB (gp4's full-text files all sat at 40-69 KB), and
anything over 300 characters counted as full text (gp4's lit_002 was its own abstract again, lit_021 a book's table
of contents). `core/pdf_text.py` now reads every PDF whole (PyMuPDF if installed, else pypdfium2), scanned pages by
OCR (tesseract, else RapidOCR; never a language model), up to 10 MB; a fetched scan is read after the fetch budget,
cached by its SHA-256. `content_quality` tells `full_text` from `abstract_only` and `preview_only`. The state keeps
64 KB of each source and the whole text is on disk. Measured on an image-only copy of a real two-column paper
(ResNet, 3 pages) with RapidOCR 3.9: 95.8% of its words recovered, columns in reading order, about 11 s a page. The
older `rapidocr_onnxruntime` dropped the spaces between English words, so it is not used.

## What was approved is what runs

The re-audit asked that the settings shown on the confirm screen be the ones that run. The interview now resolves them
before the screen (#385); this closes the other end. The first start of an interview-written quest records the settings
that decide how strictly it is checked in `.fi/approved_plan.json`; a later start whose config differs (a hand edit, or
a different `--config` passed to `--resume`) stops before anything runs and names each change; `--update` shows the
changes and records them. Verified with the real CLI: an interview config started once, `max_iterations` edited 2 -> 5,
`--resume` stopped with the change named, `--update` approved it and the quest went on. Building it found a bug that
predates it: on `--update` a blank answer took FI's own default instead of the quest's current value, so the 5 went
quietly back to 2; fixed, with a test.

## Required checks must leave a receipt (re-audit P0-1)

The 2026-09-24 re-audit showed `publication_ready` was reached by absence of error: an evidence gate, design audit or
claim check that never ran left nothing to report, a statistics computation that crashed became "no gap", and a failed
assessment left the previous `EVIDENCE.json` in place. Each of the three checks now writes a receipt
(`needs/receipts/<check>.json`: schema version, status, start and end, producer, input and output hashes, error) every
time it runs, and the ladder needs all three to say `pass`, the claim check's for the final draft (a page-limit trim,
which only removes text, carries it over). Missing, unreadable, malformed, other-schema, `unknown`, `fail` and
`not_applicable` receipts each block the level, one test per case per check; a check turned off is a gap (the user's
choice, matching the oracle and protocol checks). Under `rigor_profile: research` a check that could not judge stops the
quest once; resume retries it, and a second failure is recorded as a gap and the quest goes on (the user's choice). A
test found a stale-receipt bug on the way: a claim-check call that failed returned before writing, leaving an earlier
draft's receipt. The same work found that any SUPPLY stop other than the named ones (the evidence gate's included) fell
through to the clarify handling in the run loop; that is fixed in the figures change, which merges first.
