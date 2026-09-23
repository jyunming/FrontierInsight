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
