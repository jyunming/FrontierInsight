# FI direction study: correctness, custom simulation tools, token efficiency, mooting

Research pass, 2026-09-04. Four parallel audits of the live codebase, findings
verified against the running system where claimed. Nothing here is implemented.

---

## The convergence

The three capability asks — accuracy, custom simulation tooling, token
efficiency — are not three projects. They meet at one fact:

**FI regenerates every simulation from scratch, every quest.**

`outputs/1780518844-low-k1-duv-aerial-profile-codex-4e0954/code/experiment.py`
is 447 lines that hand-derive a scalar thin-mask Abbe/Hopkins partially-coherent
imaging model: `sample_binary_mask_fourier`, `sample_dipole_source`,
`synthesize_aerial_image`, `extract_image_metrics`. Imports: `numpy`,
`matplotlib`. A repo-wide grep for hopkins/abbe/aerial_image outside `outputs/`
finds only prose comments — there is no shared physics code anywhere in FI.

Meanwhile `C:\dev\optical_simulation_push` holds **ambit**: a tested library
doing exactly that physics, at exactly that working point (λ=193 nm, NA=1.35),
with a clean API — `WorkPoint`, `Source`, `Mask2D.aerial()`, `dose_to_size`,
`cd_at_threshold`, `through_pitch`, GDS input, five range metrics. Its test
suite passes (15 tests, run 2026-09-04). It exists in four independently
agent-validated builds, each with a `FIXES.md`.

And `demo_low_k1_source_ambit_{claude,codex,copilot}.yaml` in the FI repo root
run quests on that same physics — with the topic string pleading *"Keep the
simulation code simple and compact (fewer than 8 helper functions total)."*
That instruction is a hand-brake on a problem a library import would delete.

One change addresses all three asks:

| Ask | What a trusted kernel does |
|---|---|
| Accuracy | Physics comes from tested code, not from a fresh LLM derivation each run |
| Custom tools | The kernel registry *is* the custom-tool capability |
| Tokens | `implement` stops emitting 450 lines of re-derived physics per quest |

Everything below is ordered against that.

---

## 1. Research accuracy and correctness

### Current state

Real machinery exists. `core/stats.py` computes CIs, Cohen's d, and Bonferroni
corrections deterministically and feeds them to `analyze`, so the model
describes correct statistics rather than inventing them. `claim_check`
(`engine.py:4554-4644`) labels each paper claim `experiment` / `citation` /
`unsupported` and its evidence block does include raw `result_json`. The
must-flag router (`engine.py:1298`) genuinely forces a revise pass.

### The central gap

**Nothing deterministically verifies that a number in `paper.md` equals the
number the simulation produced.**

`analyze` (`engine.py:3933-4093`) is the single translation point from
`result_json` to prose findings, and nothing downstream re-derives it. The two
mechanisms that touch both sides are (a) `claim_check` — one LLM call judging
semantically whether a number "appears" in a JSON blob rendered as text, with no
arithmetic comparison anywhere in the codebase, and which degrades silently to a
no-op on parse failure (`engine.py:4596-4603`); and (b) the `reproducibility`
persona, which asks the right question ("are headline numbers consistent with
RESULT_JSON?") but carries no must-flag power and is off unless named.

Every check traces back to an LLM reading text. There is no code-level oracle.

For silently-wrong physics the picture is worse. `degenerate_run_guard`
(`engine.py:7256-7293`) fires only when *every* numeric leaf is ≤1e-12. A unit
error, a factor-of-two, or a sign flip that still produces plausible magnitudes
passes execute, execute_reflect, analyze, claim_check, cross_check (literature
agreement only — it never touches `result_json`), evidence_gate (runs before the
paper exists), and review. Zero gates catch it.

### Gaps, ranked

1. **No numeric oracle between `result_json` and `paper.md`.**
2. **Methodology MUST-FLAG checks are off by default.** `review_panel` is
   `Field(default_factory=list)` (`config.py:581`). The circular-evaluation /
   single-point-eval / weak-baseline / pseudo-units checklist lives only in
   `review_persona_methodologist.md:10-30`, which loads only when the panel is
   non-empty. **All 10 hand-written configs in your repo root omit
   `review_panel`** — so every one of those quests ran with the enforcement
   surface dormant. The interview path sets a 3-persona panel
   (`interview.py:787`); hand-written YAML does not.
3. **The "non-bypassable" must-flag is bypassable.** `engine.py:1298` gates on
   `state["iteration"] < engine.max_iterations`; `max_iterations: 0` disables
   it silently. Four of your configs set `max_iterations: 1`.
4. **`degenerate_run_guard` only catches exact zeros.**
5. **`/critique` — the one genuinely adversarial fresh-eyes pass — is manual and
   post-publication** (`core/critique.py`). It gates nothing.

### Recommendations

**A1. Set `review_panel: [methodologist, statistician, devil_advocate]` in your
configs.** One line, no code change, activates machinery you already built. Cost
is ~3× the review node only. Do this today.

**A2. Build a deterministic numeric oracle.** A non-LLM pass that extracts
numbers from `paper.md`, matches them against `result_json` within tolerance,
and emits an `unverified_number` must-flag. This is arithmetic, not judgment —
the highest-value correctness work available, precisely because it is the one
check that does not route back through an LLM.

**A3. Widen `degenerate_run_guard` into a physics-plausibility gate.** Range and
unit assertions declared in the design spec (`k1` in [0.25, 0.5], CD > 0, NILS
in a sane band), checked in code. A kernel library (§2) makes these assertions
natural because the kernel knows its own valid domain.

**A4. Make `max_iterations: 0` refuse to disable must-flag** — or warn loudly at
config validation. A safety mechanism silently switched off by an unrelated
budget knob is not a safety mechanism.

---

## 2. Custom tools and simulation design

### Current state

- **Execution** (`core/execution.py`): `VenvExecutor` (default) builds a
  per-quest venv and runs the script as a plain child process — the docstring
  says "no sandbox... suitable for personal/trusted-topic use" (`:62-66`).
  `DockerExecutor` (`:299-432`) is opt-in with `network_disabled=True`.
- **Dependencies**: `design.md:45` emits a free-text `"dependencies"` list of
  pip names — per-run LLM output, not a curated allowlist.
- **The prompts forbid what you want.** `implement.md:7` and
  `implement_body.md:15` both say: "Single Python file. Standard library + the
  `dependencies` from the design (numpy, scipy, matplotlib, pandas, sympy are
  all fine)." No domain package is ever named, so none is ever used.
- **`tools/`** contains only `tectonic.exe`. There is no plug-in registry.
- **`ExecutionConfig`** (`config.py:700-704`) has `sandbox`, `timeout_s`,
  `python_version`, `docker_image`. No tool path, no package allowlist.

### The precedent already in the codebase

`core/datasets/base.py:48-76` defines a `DatasetAdapter` ABC with one method,
registered by short name in `ADAPTER_REGISTRY`, opted into via
`engine.dataset_adapters: ["worldbank"]`, with a best-effort no-raise contract.
This register-by-name + YAML-opt-in + defensive-contract pattern is exactly the
right shape for a simulation-tool registry. It needs re-targeting from "fetch
rows" to "run or design a simulation", not redesigning.

### Recommendations

**B1. `core/tools/` mirroring `core/datasets/`.** A `SimulationKernel` ABC, a
`KERNEL_REGISTRY`, and an `engine.simulation_kernels: ["ambit"]` YAML knob.
Follow the adapter pattern deliberately — it is proven in this codebase and
reviewers already understand it.

**B2. Teach the prompts the kernel exists.** `design.md` and `implement*.md`
must receive the kernel's API surface (signatures + docstrings, not source) so
`implement_outline` calls `Mask2D.aerial()` instead of re-deriving it. This is
the change that actually saves the tokens; the registry without the prompt
change buys nothing.

**B3. Extend `Executor` to install local/git packages.** `install()` currently
handles `pip install <name>`. ambit, SEMulator and gds2sem are not on PyPI, so
this is a hard prerequisite for B1.

**B4. Seed the registry with ambit.** It is tested, validated four ways, and
matches the physics of the quests you actually run. `gds2sem` (`SEM_physics.py`)
and `SEMulator` (calibrated, with `checkpoints/` and `lut_cache/`) are the next
two candidates.

**B5. Let the kernel carry its own assertions.** A kernel that declares its
valid domain gives A3 its teeth, and gives `execute_reflect` something better
than a traceback to reason about.

---

## 3. Token efficiency

### Current state — better than expected in one place, worse in another

**FI does measure itself.** `_log_chat_cost` (`engine.py:5555-5619`) writes one
row per call to `.fi/cost.jsonl` with node, model, usage and estimated USD;
`_write_cost_summary` aggregates `by_node`/`by_model` into `cost.summary.json`,
which the web UI renders. That is real instrumentation.

**But the numbers are estimates on the transports you actually use.** Real usage
comes back only from HTTP-direct providers (`provider.py:2021-2031`). For
`claude_cli` and `vscode_extension` — the two paths `docs/PROVIDERS.md`
recommends as primary — the provider returns no usage, so FI falls back to a
~4-chars-per-token heuristic (`provider.py:2052-2069`). Good enough to rank
nodes against each other; not good enough for dollar accounting.

**`core/summarizer.py` is not what its name suggests.** It is a standalone
folder-digest utility behind `--summarize <folder>`, disconnected from the quest
pipeline. There is no in-quest context compressor. Prompt assembly is otherwise
disciplined — per-node named slots, no full-transcript blob.

### Gaps, ranked

1. **Literature is re-rendered in five nodes, uncapped.**
   `_format_lit_from_state` (`engine.py:6175-6204`) iterates *all* of
   `state["literature"]` with no document-count cap, and literature accumulates
   across `broaden_lit` re-entries by dedup-merge (`engine.py:2105-2124`).
   Ideate, design, analyze, cross_check and write each re-render it
   independently. A 20-doc corpus at 4000 chars/doc is ~20K tokens, paid five
   times per pass.
2. **No prompt caching anywhere.** `grep -ic cache core/provider.py` → **0**.
   Every call is one flat user message (`engine.py:5008`); no system/user split,
   no `cache_control`. Worse, the templates defeat prefix caching *by
   construction*: `write.md` has `$persona_block` on line 1, `$topic` on line 6,
   `$title` on line 9 — variables before the ~4K tokens of static instruction.
   Even providers with automatic prefix caching get nothing.
3. **Ensemble moderator re-sends all candidates verbatim**
   (`ensemble.py:189-192`), so an ensembled node costs (N+1)× with the moderator
   often the single most expensive call.
4. **Review panel multiplies full context per persona** — each gets the same
   `paper_md[:16000]` plus design/analysis/claims (`engine.py:4699-4711`).
5. **Static template bloat**: `write.md` 15.7K chars, `clarify.md` 12.0K,
   126K chars across 38 files — resent uncached on every call.

### Recommendations

**C1. Reorder every template: static instructions first, variables last.** This
is a mechanical edit with no behavior change that unlocks automatic prefix
caching on OpenAI-compatible transports immediately, and is the precondition for
explicit caching. Cheapest real win available.

**C2. Cache one literature block per quest pass.** Build it once, keyed on
query + corpus hash, reuse across the five consumers. Cap document count, not
just per-document chars.

**C3. Add explicit `cache_control` on an HTTP-direct Anthropic path.** Larger
job — no such provider exists today — but combined with C1 it turns the static
prompt bulk into a cache read.

**C4. Trim `clarify.md` and `write.md`.** Both contain inline commentary aimed
at prompt engineers rather than the model.

**C5. Fix the measurement before optimizing further.** The 4-chars/token
estimate is fine for ranking but will mislead any before/after comparison you
run on these changes. At minimum, surface `estimated_rows` prominently so you
know which numbers are real.

**Note the tension with §1:** A1 (3-persona panel) and A2 both *cost* tokens.
That is the right trade — but it means the token work should land first, so the
correctness work spends a leaner budget.

---

## 4. Mooting

### What it is, mechanically

Seats (Claude Code, Codex, Copilot, Antigravity) are woken concurrently against
a shared event cursor (`supervisor.py:387-482`); they post, propose, vote, and
ask via an MCP server over SQLite. There is no `mooting_decide` tool — absent,
not disabled (`mcp_server.py:8-16`) — and `Store.decide()` refuses a non-human
caller (`store.py:1671-1675`). Measured turn latency: 279 s default effort,
31.8 s at `--effort low`.

**It is drivable programmatically**, which matters for any integration
question: an HTTP+SSE server with `POST /api/topics/{slug}/run`
(`server.py:369-410`), and direct Python import following the
`default_supervisor()` pattern (`server.py:89-103`).

### Does it improve token efficiency?

**No — and it isn't meant to.** Seats think concurrently, so a round costs
wall-clock ≈ the slowest seat, but total inference ≈ N seats. What changes is
whose meter runs: no API keys, every seat on a subscription already paid for.
That is cost-shifting onto flat-rate quota, not token reduction. Worth having
for what it is; it will not make FI leaner.

### Does it guide to better results?

Unmeasured, by your own account. The honest position is the one already written
down: no measurement shows four rival seats catch more than one good agent and
an attentive person.

### The structural mismatch

FI's five human gates all funnel through `_pause_for_human`
(`engine.py:2429-2476`): clarify (`:1332`), papers (`:2160-2193`), data
(`:3018`), generic supply (`:2478-2509`), and review (`:4869-4936`).

Three of those — papers, data, supply — are file-presence checks. A council has
nothing to argue about there. That leaves clarify and review.

And a council cannot *close* either one. Mooting's whole design is that only a
human decides; FI's gate likewise needs `human_review_answer.json` or
`--resume --accept`. Wiring them together yields two human decision points
chained by a script. Auto-forwarding the council's tally to satisfy FI's gate is
trivial to build and worth nothing — it launders an unattended decision through
two systems that both advertise human sign-off, weakening both claims at once.

### Recommendations

**D1. Do not integrate mooting into the unattended pipeline.** The premise
mismatch is fundamental, not incidental.

**D2. If you want it, build one narrow thing: `fi review --council`.** Interactive
only, never in `--fleet`. Attaches the finished `paper.md` to a mooting topic,
drives it by direct Python import, prints the council's recommendation, and
leaves you to type `--accept`. Seat `cwd` must be a neutral directory, not
`quest_root` — mooting's own `Seat.cwd` docstring (`drivers/base.py:116-126`)
warns that pointing a seat at a real project leaks its `CLAUDE.md` into the
council. Note this would be CLI-only and so sits outside FI's three-interface
parity rule until extended.

**D3. The genuinely valuable link is the other direction: use FI to measure
mooting.** Your positioning doc says the measurement is cheap and the project is
the instrument. FI is the *better* instrument — it already runs seeded quests,
has a `/critique` adversarial pass, and produces artifacts that can be scored.
The experiment: N quests with known-seeded defects, scored three ways —
single-reviewer, FI's 3-persona panel, and a mooting council — on defects caught
and false positives raised. That produces the evidence the "better outcomes"
claim is waiting on, and it simultaneously validates FI's own review panel,
which nothing currently measures either.

---

## Suggested order

1. **A1** — add `review_panel` to your configs. One line, today.
2. **C1** — reorder templates, static-first. Mechanical, no behavior change.
3. **B3 + B1 + B4** — executor local-install, kernel registry, seed with ambit.
4. **B2** — teach `design`/`implement` the kernel API. This is where the token
   savings and the physics-correctness gain actually land.
5. **A2** — the deterministic numeric oracle.
6. **C2** — literature block caching.
7. **D3** — the measurement experiment, once 1-6 give it a stable baseline.

A1 and C1 are afternoon-sized and independent of everything else.

---

## What was not determined

- How often `broaden_lit` re-entry actually fires, which sets whether the
  literature-accumulation waste is the common case or an edge case.
- Real-world split of CLI vs HTTP-direct transport across quests, which sets how
  much of the cost accounting is estimated rather than measured.
- Whether SEMulator and gds2sem have test suites of ambit's quality. Only
  ambit's suite was run.
- `_parse_json_lenient` failure-mode robustness — how often malformed LLM JSON
  silently degrades a gate to its defaults.
