# LLM Call Efficiency Audit (Unit 02)

**Date:** 2026-05-15
**Scope:** Per-quest LLM call profile, prompt sizes, merge/batch opportunities,
retry strategy, and no-simulation savings.
**Baseline:** `docs/USAGE.md:28-46` — quest envelope is 8-18 calls
(floor 7-8, ceiling 18+, worst-case ~24 with full retry/panel loops).

---

## Findings

### 1. Call topology and where the cost actually lands

`core/engine.py:328-399` (`_build_graph`) wires the eleven LLM-firing nodes
into a single LangGraph state machine. The static, always-on calls are:
`ideate` (1), `literature` (1, retrieval not LLM — but the source-router
in `knowledge.asearch` *does* fire a chat call when the catalog is
enabled — see `core/engine.py:642`), `design` (1), `implement` (1),
`analyze` (1), `cross_check` (≥1), `write` (1), `review` (1). That is a
hard floor of **7 LLM-billed calls** when the source-router is on,
matching the table in `docs/USAGE.md:28-46`.

The variable calls — and the reason the ceiling is so high — sit in
five "loop" nodes:

| Node | Worst-case calls | Trigger | File:line |
|---|---|---|---|
| `clarify` | 0–1 | `engine.clarify_mode != "off"` | `core/engine.py:468-499` |
| `ideate_reflect` | 0–1 | `engine.ideate_reflect=True` | `core/engine.py:656-695` |
| `design` retry | up to +2 | analyze-reroute (`next_step ∈ {re_experiment, broaden_lit}`) OR review verdict=revise; capped by `engine.max_iterations=2` | `core/engine.py:443-455, 457-464` |
| `implement` retry | up to +2 | same loops as design | (same edges) |
| `execute_reflect` | 0–3 | `rc!=0` AND iterations < `exec_reflect_max_iterations=3` | `core/engine.py:1207-1308, 1229` |
| `cross_check` | 1 per finding, capped at 10 | one LLM call per `key_findings` entry | `core/engine.py:1347, 1376` |
| `review_panel` | N + 1 | `engine.review_panel` non-empty → N persona calls + 1 moderator | `core/engine.py:1488-1531` |
| `write` second pass | 0–1 | review verdict=revise | (same review loop) |

Real-world ceiling I can construct from this code: clarify=1, ideate=1,
ideate_reflect=1, source-router pings ≈3, literature-synthesis=0 (Axon
only), design×3=3, implement×3=3, execute_reflect×3=3, analyze=1,
cross_check×10=10, write×2=2, review_panel(4)=4+1=5 → **33 calls**.
The "24+ worst-case" cited in the brief is conservative; in pathological
configurations (large panel, full cross-check budget) we approach 33.

### 2. Prompt sizes — the three obvious targets

Word counts on `agents/*.md` (close proxy for token count; ÷0.75 → tokens):

| Prompt | Lines | Words | ~Tokens | Note |
|---|---|---|---|---|
| `clarify.md` | 119 | 940 | ~1250 | **Largest.** Long inline routing-rule commentary for `simulatability`. |
| `write.md` | 80 | 842 | ~1120 | Heavy "honesty constraints" + survey-shape recognition + length rules. |
| `critique.md` | 84 | 742 | ~990 | One-shot adversarial review; not on hot path. |
| `portfolio.md` | 73 | 716 | ~955 | One-shot all-time synthesis; not on hot path. |
| `summarize.md` | 94 | 615 | ~820 | One-shot; not on hot path. |
| `digest.md` | 56 | 564 | ~750 | One-shot. |
| `data_load.md` | 74 | 575 | ~770 | No-sim path only. |
| `proposal.md` | 66 | 536 | ~715 | Pre-quest, optional. |
| `review.md` | 49 | 405 | ~540 | **Fires once or N+1 times in panel mode** — small but volume-multiplied. |
| `execute_reflect.md` | 56 | 282 | ~375 | |
| `analyze.md` | 53 | 243 | ~325 | |
| `review_moderate.md` | 33 | 237 | ~315 | |
| `implement.md` | 40 | 224 | ~300 | |
| `cross_check.md` | 33 | 215 | ~285 | **Fires 1-10×** — volume multiplier. |
| `ideate_reflect.md` | 35 | 192 | ~255 | |
| `design.md` | 35 | 140 | ~190 | |
| `ideate.md` | 29 | 126 | ~170 | |

The three single-pass largest are `clarify.md` (940), `write.md` (842),
`critique.md` (742). Of these, only the first two run inside the
standard quest pipeline; `critique.md` is the `/critique` slash command
(one-shot, called by the user explicitly).

### 3. The cross_check call multiplier is the worst per-call value

`core/engine.py:1331-1398` shows `cross_check` doing N+1 things in a loop:
(a) one Axon retrieval per finding, (b) one classification LLM call per
finding. The cap at line 1347 (`findings[:10]`) means up to **10 LLM
calls for a single quest step** that produces nothing but
`supporting/conflicting/neutral` labels per candidate. The prompt
(`agents/cross_check.md`, 215 words) is short, so each call is cheap —
but the dollar cost is `10 × ~285 tokens prompt + ~150 tokens response
≈ 4.4k tokens` just for cross-checking, with the request volume making
provider rate-limits the actual bottleneck (not token count).

### 4. ideate_reflect is structurally a 2-stage chain-of-thought, not a
true peer-review

`core/engine.py:634-695` calls `ideate` to brainstorm 3-5 ideas + pick
one, then *immediately* calls `ideate_reflect` to critique its own
pick. There is no third party. The reflection prompt
(`agents/ideate_reflect.md:15-23`) explicitly says "do NOT introduce a
new idea". This is a deliberate two-step that could collapse into one
prompt asking the model to brainstorm AND self-critique in a single
JSON object.

### 5. review_moderate is a synthesis of structured data, not a free reason

`core/engine.py:1517-1545` shows that after N parallel persona reviews,
the `_aggregate_panel_reviews()` helper produces the numeric
verdict/score/agreement *deterministically* (median, intersection,
union — `agents/review_moderate.md:9-16` documents the rules). The
moderator LLM call exists **only** to produce the prose `rationale`
field and (optionally) re-attribute suggestions. The numeric fields
from `agg` already win at lines 1536-1539. So one LLM call is being
spent on what amounts to a 1-2 sentence prose explainer.

### 6. execute_reflect's retry budget eats the model uniformly

`core/engine.py:1229, exec_reflect_max_iterations=3` (`config.py:102`).
A flaky import, a Windows DLL-load race, or a single typo all consume
the same expensive-model budget on retry. There is no model
de-escalation on the first retry. For Copilot's premium-request budget
this matters: 3 retries at `claude-sonnet-4-7` is exactly 3 premium
requests for what might be a one-line fix a `haiku-4-5` would handle.

### 7. no-simulation savings, confirmed

When `no_simulation_resolved=True` (`core/engine.py:421-423`),
`_route_after_design` sends the graph into
`auto_collect_data → wait_for_data → data_load → analyze` instead of
`implement → execute → execute_reflect → analyze`. This skips:

- `implement` (1-3 calls saved)
- `execute_reflect` (0-3 calls saved)
- `execute` itself isn't a LLM call

But it ADDS:

- `data_load` (1 LLM call — `agents/data_load.md`, ~770 tokens)
- `auto_collect_data` may fire a source-router ping (1 sub-call)

Net: **save 0 to 5 LLM calls** in no-simulation mode vs the simulation
floor/ceiling. The savings come almost entirely from skipping
`execute_reflect` retry storms, since `implement` only loops back via
the design-retry mechanism which is shared with no-sim.

### 8. Source-router is an invisible call multiplier

`core/engine.py:641-643, 703-705, 1352-1356` pass
`chat_fn=functools.partial(self._chat_messages, node="source_router")`
into `knowledge.asearch`. Each Axon retrieval can fire one extra LLM
call to choose sources from the catalog. The user-visible cost table in
`docs/USAGE.md` does NOT mention these — they're folded into "retrieval
uses Axon embeddings, not LLMs" at line 35. In a quest with `ideate +
literature + cross_check × 5`, that's **7 extra source-router calls**
on top of the 7-18 documented. Worth noting but the calls are typically
small (catalog selection, not synthesis).

---

## Recommendations

Ranked by impact (calls saved per quest × likely usage frequency).
Each tagged `[impact: S|M|L]` `[effort: S|M|L]`.

### R1 — Merge `ideate` + `ideate_reflect` into one prompt [impact: M] [effort: S]

**Concrete change:** Modify `core/engine.py:_node_ideate` (lines 634-695)
to fire a single LLM call. Combine `agents/ideate.md` and
`agents/ideate_reflect.md` into one prompt that returns:

```json
{
  "ideas": [...],
  "chosen": {...},
  "self_critique": {
    "strongest_objection": "...",
    "swap_to": "<title or empty>",
    "refined_rationale": "..."
  }
}
```

Then apply the `swap_to` logic (currently lines 667-688) on the parsed
JSON without a second roundtrip. **Saves 1 LLM call per quest** when
`engine.ideate_reflect=True` (the default per `config.py:116`).

**Tradeoff:** Less independence between the brainstorm and the
critique. A two-shot can in principle catch its own blindspot better
than a one-shot, but the existing prompt at `ideate_reflect.md:23`
already explicitly forbids introducing new ideas, so the "critique"
function is really just rationale-strengthening + within-list
re-ranking — fully achievable in one prompt.

### R2 — Batch all `cross_check` findings into one LLM call [impact: L] [effort: M]

**Concrete change:** Modify `core/engine.py:_node_cross_check`
(lines 1331-1415). Instead of looping 1-10 times, send one prompt
listing all findings + the per-finding candidate set, expecting a JSON
array indexed by finding. Update `agents/cross_check.md` to accept and
return multiple findings:

```json
{
  "findings": [
    {"finding": "...", "supporting": [...], "conflicting": [...], "neutral": [...], "summary": "..."},
    ...
  ]
}
```

The per-finding Axon retrievals stay parallelized (they're cheap and
already async). Only the LLM classification collapses.

**Saves up to 9 LLM calls per quest** (10 findings → 1 call instead of
10). On a default quest with 3 findings, saves 2 calls.

**Tradeoff:** Loses parallel-API throughput on the LLM dimension. Wall
clock might be marginally worse on a fast-API provider since 1 large
sequential call can be slower than 10 parallel small ones. On
premium-request budgets (Copilot) this saves 1-9 requests outright. On
token-billed APIs, prompt-token overhead per call is amortized — the
shared instructions only ship once. Net token savings ~15-20%.

### R3 — Drop the `review_moderate` LLM call, derive rationale from aggregator [impact: M] [effort: S]

**Concrete change:** Modify `core/engine.py:1517-1545`. The
deterministic aggregator (`_aggregate_panel_reviews`) already produces
the verdict, score, agreement, strengths, weaknesses, suggestions. The
moderator's only unique contribution is the prose `rationale` field.
Replace the LLM call with a templated rationale built from the
deterministic agg fields:

```python
review["rationale"] = (
    f"{agg['verdict'].title()} ({agg['agreement']}, "
    f"median score {agg['score']}). "
    f"Top weakness: {agg['weaknesses'][0] if agg['weaknesses'] else '(none)'}."
)
```

Or, alternatively, fold the moderator's job into each persona's
response (each persona returns its rationale; aggregator picks the
strongest persona's rationale by score).

**Saves 1 LLM call per quest** when `review_panel` is configured.

**Tradeoff:** Loses the LLM's narrative bridge between dissenting
personas. For users who want the "controversial split between
methodologist and statistician" prose, this is a real loss. Mitigation:
ship as `engine.review_moderator: bool` (default False once stable),
let users opt back in.

### R4 — Cheap-model first retry on `execute_reflect` [impact: M] [effort: S]

**Concrete change:** Document a `provider.node_models` recipe in
`docs/USAGE.md` for tiered execute-reflect:

```yaml
provider:
  node_models:
    execute_reflect: "claude-haiku-4-5"   # cheap first pass
    execute_reflect.escalate: "claude-sonnet-4-7"  # if iter >= 2
```

Then in `core/engine.py:_node_execute_reflect` (lines 1207-1308), pass
a qualified node name based on iteration:

```python
iters = state.get("exec_reflect_iter", 0)
node_name = "execute_reflect" if iters < 2 else "execute_reflect.escalate"
text = await self._chat(prompt, node=node_name)
```

The `_model_for_node` resolver at `core/engine.py:1590-1608` already
supports hierarchical keys via prefix-match — the change is one line in
`_node_execute_reflect`.

**Saves ~60% of execute-reflect cost** when the bug is simple (typical
import error, NaN propagation, off-by-one — haiku handles these). Falls
back to the expensive model on genuinely hard bugs.

**Tradeoff:** Two-tier model strategies are harder to debug — users
who see "weird patches" need to know which model produced them. Mitigate
by logging the model in the existing `[execute_reflect]` log line.

### R5 — Trim `clarify.md` and `write.md` by ~30% [impact: S] [effort: M]

**Concrete change for `agents/clarify.md` (940 words → ~650):**
Lines 21-76 contain ~250 words of inline routing-rule commentary
(`// NOTE — this slot is now METHODOLOGY-ONLY...`,
`// CRITICAL — the engine uses simulatability as the ROUTING SIGNAL`).
This commentary is for *developers reading the prompt*, not for the
LLM. Move it to a code comment in `core/engine.py` and replace each
inline block with a 1-line directive (`// pick "no" for cultural/
historical/qualitative topics`). The LLM doesn't need the engine's
internal routing logic to answer the question correctly.

**Concrete change for `agents/write.md` (842 words → ~580):**
The "Honesty constraints" section (lines 39-55) is excellent and stays.
The "Topic-shape recognition" section (lines 56-67) is good but
verbose; can compress to 3 bullets without losing the survey-vs-
experimental distinction. The example title block (lines 75-79) is 3
worked examples; one suffices.

**Saves ~600 tokens per quest** on `clarify` + `write` calls combined.
Small per-call, but adds up across fleet runs.

**Tradeoff:** Risk of regression on the "honest about broken results"
behavior — `write.md:39-55` is load-bearing for that quality. Suggest
behavior-equivalence test (run the same 3 fixtures pre- and
post-trim, compare paper outputs) before merging.

### R6 — Batch `review_panel` personas into one prompt (low-tier mode) [impact: M] [effort: M]

**Concrete change:** Add `engine.review_panel_mode: "parallel" | "batched"`
(default `"parallel"`, current behavior). When `"batched"`, fire one
LLM call that includes all persona prefixes and asks for N response
sections in one JSON. Modify `core/engine.py:1488-1516` to dispatch on
the new flag.

**Saves N−1 calls** when batched (e.g., 4-persona panel → 1 call
instead of 4 parallel). Combined with R3 (no moderator), saves
N calls total.

**Tradeoff:** Loses parallel wall-clock speedup that `asyncio.gather`
buys today. Loses the genuine "independent reviewer" property — a
single LLM session that sees all persona prefixes simultaneously will
self-correlate (the methodologist persona's response biases the
statistician's). Recommend `"parallel"` stays default; `"batched"` is
the tight-budget option for Copilot premium-request constraints.

### R7 — Add `provider.node_models` cookbook recipes to USAGE.md [impact: S] [effort: S]

**Concrete change:** Append a "Cost-tiered model recipes" section to
`docs/USAGE.md` after the cost table. Three recipes:

1. **Cheap-floor:** haiku for `clarify, ideate_reflect, cross_check,
   review_moderator, execute_reflect`; sonnet for `design, implement,
   analyze, write, review`.
2. **Premium-only-on-write:** haiku everywhere except `write` and
   `review` (the user-visible outputs).
3. **Adversarial-cross-family:** rotate provider per node so single-
   model blindspots are caught (e.g., codex for design, claude for
   review).

**Saves 0 calls but 30-60% in dollar/quota** depending on provider.
This is documentation, not code — and the routing already works
(`core/engine.py:1590-1608`, `core/config.py:50-74`).

### R8 — Skip `literature` LLM synthesis when Axon returns >0 docs [impact: S] [effort: S]

`core/engine.py:697-711` doesn't actually fire a synthesis LLM call —
it just packs `docs[].content[:2000]` into state. So the `literature`
"1 call" in `docs/USAGE.md:35` actually represents the source-router
call inside `knowledge.asearch` (per finding 8 above). The cost table
is misleading; consider correcting it OR add an explicit
literature-synthesis call (which would make the table accurate but cost
1 more call per quest — don't recommend).

**No code change needed.** Just clarify `docs/USAGE.md:35` to say "1
source-router call when knowledge.source_router is on; 0 otherwise".

### R9 — De-prioritize: token-level prompt caching [impact: L] [effort: L]

Anthropic + OpenAI both support prompt caching with a `cache_control`
breakpoint. For FI, the static prefix of each prompt (everything before
the per-quest `$topic`, `$design_block` etc) is cacheable — typically
~60% of the prompt by token count. Implementation requires touching
`core/providers/*.py` to emit the cache breakpoint and wiring the
config plumbing per provider. **Skip for now** — providers vary
(Copilot CLI has no public caching control, Ollama doesn't need it,
gemini's caching is incompatible). Revisit when one provider becomes
dominant in usage telemetry.

### R10 — Out-of-scope: cross-quest cache for `ideate` and `literature` [impact: L] [effort: L]

Two quests on the same topic re-run `ideate + literature` from scratch.
A topic-hash-keyed cache in `outputs/_cache/` could short-circuit the
first 2-3 calls on repeat work. **Skip** — adds non-trivial complexity
(invalidation rules, hash-stability across prompt edits) for a workflow
that isn't documented to be the common case.

---

## Ranked summary

| Rec | Impact | Effort | Calls saved/quest | Notes |
|---|---|---|---|---|
| R2 cross_check batching | L | M | 2-9 | Best ratio; touch one node + prompt |
| R1 ideate merge | M | S | 1 | Trivial; default-on path |
| R3 drop moderator | M | S | 1 | Only if review_panel used |
| R4 cheap retry tier | M | S | 0-2 (¢-weighted: 60% of retries) | Provider-quota relief |
| R7 USAGE cookbook | S | S | 0 (¢ savings) | Pure docs |
| R6 panel batching | M | M | N−1 (only if used) | Tradeoff: serial wall-clock |
| R5 prompt trims | S | M | 0 (token reduction) | Needs regression check |
| R8 USAGE clarification | S | S | 0 | Doc correctness |
| R9 prompt caching | L | L | 0 (¢ savings) | Defer — provider fragmentation |
| R10 cross-quest cache | L | L | 2-3 (when applicable) | Defer — invalidation hairy |

**Top three to ship first:** R2 (cross_check batching) + R1 (ideate
merge) + R4 (cheap-retry tier doc). Together they cut a default-config
quest from 8-10 LLM calls to **5-7**, without losing material quality —
the merges are within-LLM consolidations, not skips of distinct
reasoning steps.

---

## References

- `docs/USAGE.md:28-46` — current cost table (8-18 calls per quest).
- `core/engine.py:328-399` — `_build_graph`, the canonical call topology.
- `core/engine.py:634-695` — `_node_ideate` + `_node_ideate_reflect`, R1 target.
- `core/engine.py:1331-1415` — `_node_cross_check`, the per-finding loop and R2 target.
- `core/engine.py:1437-1558` — `_node_review` panel + moderator, R3/R6 targets.
- `core/engine.py:1207-1308` — `_node_execute_reflect`, R4 target.
- `core/engine.py:1590-1608` — `_model_for_node`, the resolver R4/R7 build on.
- `core/config.py:50-74` — `ProviderConfig.node_models`.
- `core/config.py:102` — `exec_reflect_max_iterations=3` default.
- `core/config.py:116` — `ideate_reflect=True` default.
- `core/config.py:130` — `review_panel: list[str]` default empty.
- `agents/clarify.md` (119 lines, ~1250 tokens) — R5 trim target.
- `agents/write.md` (80 lines, ~1120 tokens) — R5 trim target.
- `agents/ideate.md` + `agents/ideate_reflect.md` — R1 merge inputs.
- `agents/cross_check.md` — R2 modify target.
- `agents/review_moderate.md` — R3 delete target.
