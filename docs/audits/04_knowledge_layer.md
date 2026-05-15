# Audit Unit 04 — Knowledge layer + dataset adapters

Scope: `core/knowledge.py` (1 657 LOC), `core/datasets/` (754 LOC across
`__init__.py`, `base.py`, `worldbank.py`, `wikipedia.py`), and
`tests/test_knowledge.py` (48 tests, 1 115 LOC). Read on 2026-05-15.

> All LOC counts in this audit come from `wc -l`. An editor or
> reviewer using "last-line number" counting will see N+1 (e.g. the
> editor reports 1 658 for `knowledge.py`); both numbers refer to the
> same file.

## Findings

### F1 — `core/knowledge.py` is a *grab-bag* module, not "one thing"

The file's docstring (`core/knowledge.py:1-57`) is honest about it
carrying "two responsibilities" (retrieval + ingest), but in practice
the file contains **six** logically distinct concerns crammed into a
single 1 657-LOC translation unit:

1. External-source adapters — seven functions, `_arxiv_search` through
   `_google_scholar_search` (`core/knowledge.py:131-435`). Each is
   ~30-60 LOC of pure HTTP + parse, structurally identical.
2. Full-text fetch + PDF magic-bytes paywall handling
   (`core/knowledge.py:543-787`).
3. Local-paper loader (PDF / Markdown / .txt) (`core/knowledge.py:454-540`).
4. Source-catalog + LLM-routed source picker
   (`core/knowledge.py:790-985`) — a 12-entry catalog with prompt
   templating + JSON parsing.
5. Dedup keys + the actual router (`core/knowledge.py:988-1059`).
6. Structured-ingest helpers (spine, header, topic event) +
   `Knowledge` facade itself (`core/knowledge.py:1062-1657`).

The seam between (1)+(5) and (2)+(3) and (6) is sharp — there's no
shared state, no cross-references except `_route_external` invoking
`_SOURCE_REGISTRY`. The file reads like it grew by accretion across
three or four PRs (the docstring confirms: arxiv, then openalex, then
crossref, then catalog routing, then full-text fetch, then structured
ingest).

Re-importing this whole module (e.g., from `tests/test_knowledge.py`)
forces eager evaluation of all seven adapter top-level imports, the
`_SOURCE_CATALOG` literal, and `_AXON_AVAILABLE` probing — none of
which a test that monkeypatches a single adapter actually needs.

### F2 — Layering is clean upward; coupling is intra-module

`core/knowledge.py`'s only non-stdlib imports are `httpx`, `yaml`, and `.config`
(`core/knowledge.py:59-75`). It does NOT import `engine`, `provider`,
`execution`, or anything else from `core/`. Good — `Knowledge` is a
true facade and `engine.py` is the only consumer that knows about it
(`core/engine.py:163, 640, 700, 731, 1353, 1794`).

`core/datasets/` is similarly clean upward — `base.py` imports
`abc` + `dataclasses`; `worldbank.py` imports `.base`; `wikipedia.py`
imports `.base` AND `.worldbank` (for `_http_get_json_sync` and
`_query_keywords`, `core/datasets/wikipedia.py:39`). The
adapter-engine boundary lives in `engine.py:864-942` and imports
`.datasets` lazily inside the method (`core/engine.py:881`) — fine.

Net: no circular risk, no hidden upward calls into the engine. The
audit's coupling concern is internal to `knowledge.py` only.

### F3 — External sources are best-effort with NO rate-limit handling

All seven adapters log + return `[]` on any exception
(`core/knowledge.py:109-128`, the `_http_get_json` / `_http_get_text`
helpers swallow every `Exception`). That's the right default for a
"fallback" path: don't crash a quest because Semantic Scholar is
down. But the implementation has serious gaps:

- **No 429 / Retry-After awareness.** Semantic Scholar's free tier
  rate-limits aggressively without an API key, and the docstring
  (`core/knowledge.py:826-832`) admits this. Neither
  `_semantic_scholar_search` nor any other adapter inspects HTTP
  status — `r.raise_for_status()` lumps 429 in with 500, the
  exception fires, the adapter returns `[]`, and the agent has no
  signal that "try later" would have worked.
- **No backoff.** A rate-limited adapter is permanently dead for the
  duration of the process. (Even worse for `_fetch_indicators` in the
  WorldBank adapter — which DOES cache, but caches success only;
  `core/datasets/worldbank.py:135-152`.)
- **No timeout differentiation.** Every adapter uses `timeout_s=10`
  (except Google Scholar at 30, `core/knowledge.py:390`). A rate-
  limited Semantic Scholar response will reliably succeed in <1 s
  with a 429; a real timeout takes the full 10 s. There's no way for
  the router to learn "Semantic Scholar is 429ing on us, skip it next
  query."

Parallelism IS correct: `_route_external` uses
`asyncio.gather(*(run_one(n) for n in valid))` with
`return_exceptions=True` implicit (each `run_one` catches its own,
`core/knowledge.py:1027-1031`), so a slow adapter doesn't starve the
others — they finish in `max(t1, t2, ..., tn)` not the sum.

### F4 — Source-router calls the LLM once per `asearch()` invocation

When `source_routing="auto"` (the default,
`core/config.py:241`), every retrieval triggers
`_route_sources_with_llm` (`core/knowledge.py:916-985`). That's a
zero-shot prompt of ~1 300 chars (the catalog) plus the topic. With
~5 retrievals per quest (ideate, literature, analyze, paper-gen,
critique), this is **5 extra LLM round-trips per quest** that the
manual config-list path doesn't pay. No caching by `(topic, idea)`.
For tight retrieval loops this may add 10-20 s to a quest. Worth
either:

1. Memoizing the choice per quest by `(topic, chosen_idea hash)`, or
2. Making `auto` opt-in instead of default.

(The cost is small per quest, but it compounds in `--fleet` runs.)

### F5 — Full-text fetch budget logic is correct but hides resource leak

`_enrich_with_full_text` (`core/knowledge.py:685-787`) is one of the
more carefully-thought-out functions in the file. The author
explicitly chose `as_completed` over `wait_for(gather(...))` because
the latter discards completed results on timeout
(`core/knowledge.py:719-732`). There's even a regression test for
this (`tests/test_knowledge.py:633-666`).

But the caveat documented at `core/knowledge.py:727-732` is real and
material: `asyncio.to_thread` tasks "cannot truly be cancelled
mid-blocking-call." On a slow VPN with a 90-s budget and a 200-s
publisher response, the budget timer pops, the function returns, and
the host process still has N detached threads each in the middle of
a `httpx.Client.get(...)`. They'll complete eventually and their
results are dropped (the future was cancelled), but during that
window the process holds open sockets it can't reclaim. For
`--fleet` runs of 8 quests in parallel this could plausibly chew
file descriptors or fill the thread pool.

A cleaner pattern would be a custom `httpx.AsyncClient` so the entire
chain is cancellable. The current code is fine for v0 but should be
called out as a known limitation.

### F6 — `DatasetAdapter` contract is too thin in three concrete ways

`core/datasets/base.py` defines:

```python
class DatasetAdapter(ABC):
    name: str = "base"
    @abstractmethod
    async def search(self, query: str, *, top_k: int) -> list[DatasetRow]:
```

Three gaps:

1. **No per-call timeout.** The engine wraps each adapter in a try /
   except (`core/engine.py:898-906`) but cannot say "this adapter has
   3 seconds to respond, then bail." WorldBank's `_fetch_indicators`
   bakes in `timeout_s=8.0` (`core/datasets/worldbank.py:94, 134`)
   and Wikipedia's `_fetch_summary` likewise (`core/datasets/wikipedia.py:148`).
   Hard-coded per file — the engine cannot change it. Should be an
   adapter kwarg or a method on the base class.
2. **No multi-query support.** WorldBank's pattern is "fetch indicator
   catalog ONCE, then issue N data-point queries"
   (`core/datasets/worldbank.py:112-155, 321-323`). The catalog cache
   lives at module level (`_indicator_cache`) — a side channel that
   bypasses the `search()` contract entirely. A more honest interface
   would let the adapter declare a lifecycle: `prepare()` (run once
   per process), `search()` (per query), `teardown()` (optional). This
   matters more as adapters grow: OECD SDMX needs a dataflow catalog;
   Eurostat needs a code-list cache.
3. **No way to surface "I cannot answer this query."** Returning
   `[]` is ambiguous — it could mean "no results" or "I'm a Hofstede
   adapter and you asked about Bernstein-Vazirani." The engine can't
   distinguish, so it can't prioritize one adapter over another for a
   given topic. A `relevance_score(query) -> float` hook would let the
   engine skip irrelevant adapters before even calling search.

### F7 — Adapter discoverability has no canonical surface

`ADAPTER_REGISTRY` (`core/datasets/__init__.py:29-32`) is the source
of truth. The user finds out which adapters exist by:

- Reading the `__init__.py` (technical user only).
- Reading the `dataset_adapters` field comment in `core/config.py:176-181`
  which says *"Available names: ``\"worldbank\"``"* — incomplete; the
  worktree has shipped `wikipedia` but the comment still names only
  WorldBank. Drift bug.
- Reading `docs/USAGE.md:170` which says
  *"# structured-data + web-fetch adapters. Available: 'worldbank', 'wikipedia'"* —
  the docs are correct.
- Reading `docs/USAGE.md:347-351` — the prose still says "Available
  adapters: `worldbank`". Drift bug #2 in the same file.

No CLI command (`launch.py` has no `--list-adapters` flag, confirmed by
inspection at `launch.py:38ff`). No `core.datasets:cli` entry. No
README section. New users will discover adapters by accident.

Even worse: the seven *literature* sources in `_SOURCE_REGISTRY`
(`core/knowledge.py:438-446`) have a parallel discoverability problem.
The `_SOURCE_CATALOG` (`core/knowledge.py:799-898`) is gorgeous —
title, fields, access, when-to-use — but it's invisible to users
unless they read the source.

### F8 — Code duplication between WorldBank and Wikipedia adapters is minor but symptomatic

`wikipedia.py` imports two helpers FROM `worldbank.py`:
`_http_get_json_sync` and `_query_keywords`
(`core/datasets/wikipedia.py:39`). That makes Wikipedia structurally
dependent on WorldBank — if WorldBank were ever moved or renamed, the
Wikipedia adapter would break. The right place for `_http_get_*` and
`_query_keywords` is `base.py` or a sibling `_util.py`. The fact that
the author reached for the worldbank module to grab two utilities
shows the abstraction isn't quite right yet.

Beyond the two helpers, the two adapters share:

- A "user-agent string built from package metadata" pattern
  (`worldbank.py:105`, `wikipedia.py:45-49`) — duplicated.
- The "log warning + return []" error pattern is repeated 3 times in
  `worldbank.py` and 2 times in `wikipedia.py`. A decorator could
  consolidate.
- Both maintain a tiny per-call sync HTTP wrapper. Both could share
  the `_http_get_json` / `_http_get_text` helpers already living in
  `core/knowledge.py:109-128`. Two parallel hand-rolled HTTP utilities
  is one too many for a 700-LOC subsystem.

### F9 — Quest write-back is gated correctly but has no quality filter

`add_quest_artifacts` lays down 4 + N docs per accepted quest:
spine, body, summary, topic event, plus N external-ref spines
(`core/knowledge.py:1500-1657`). Engine gates on
`verdict == "accept"` (`core/engine.py:1726-1728`). Good.

But there's **no recency or duplicate-quest filter**:

- The same paper cited by 10 different quests would accumulate 10
  `fi_external_ref_spine` documents (with different
  `consumed_by_quests` lists). The text body has the same spine
  content but Axon, depending on its dedup behavior, may treat them
  as distinct. Over 100 quests this could create a 1000-row "spine
  corpus" where 70 % are near-duplicates.
- No write-back-time content quality gate beyond "the verdict says
  accept." A 3-out-of-5 accept is treated identically to a
  5-out-of-5. If the corpus eventually drives the LLM's retrieval
  context, low-confidence accepts will pollute future quests'
  literature retrieval.

After 100 accepted quests with ~5 external refs each, the corpus
will be roughly: 100 `fi_quest_paper` (full bodies) + 100
`fi_paper_spine` + 100 `fi_quest_summary` + 100 `fi_topic_event` +
500 `fi_external_ref_spine` ≈ 900 documents. Modest, but if
write-back-on-accept is loosened, this 10×s easily.

The structured-finding JSON envelope embedded inside `fi_quest_summary`
(`core/knowledge.py:1605-1610`) is good — it's interrogable at
retrieval time. But that's another reason to be conservative about
who gets to write into the corpus.

### F10 — The `_paper_short_id` dedup is title-prefix sensitive

`_paper_short_id` (`core/knowledge.py:1089-1099`) and
`_doc_dedup_key` (`core/knowledge.py:993-1006`) both fall through to
a normalized-title hash. Normalization strips non-word chars and
collapses whitespace but does NOT:

- Strip "The "/"A " prefixes.
- Lowercase Unicode-folded characters (e.g., "Schrödinger" vs
  "Schrodinger" produce different keys).
- Strip subtitle separators (": A Study Of…" vs "; A Study Of…").

Two crossref vs openalex hits for the same paper without a DOI on
either could survive dedup. Not common, but the comment claims this
function is the canonical cross-quest identifier so it's worth
hardening.

### F11 — `KnowledgeConfig.external_fallback` parsing is forgiving but underdocumented

`core/config.py:282-291` accepts `None`, `""`, `"none"`, a single
string, or a list. The behavior is sane. But the docstring at
`core/config.py:216-234` says "Override per-quest in YAML; set to `[]`
or 'none' to disable" — fine — yet the `Literal` for `source_routing`
is documented inline (`core/config.py:235-241`) but the
`external_fallback` list of supported names is documented only as
prose in the comment. Easy drift surface — if someone adds an adapter,
they need to update the comment, the catalog, the YAML schema (none),
the user docs, AND `_SOURCE_REGISTRY`. No single point of truth.

### F12 — Test coverage strong on knowledge, thinner on datasets

`tests/test_knowledge.py` has 48 tests covering disabled-path, fallback
fixturing for axon, arXiv mock, local-paper PDF/MD/txt, paywall
checking, structured-ingest helpers, write-back fallback chain, and
the catalog/source-router. Excellent — this is the most rigorously
tested module I've looked at in FrontierInsight.

`tests/test_dataset_adapters.py` (388 LOC, 17 tests) and
`tests/test_wikipedia_adapter.py` (319 LOC, 11 tests) are thinner.
They cover helper functions, country detection, single-adapter
happy path, registration. They do NOT cover:

- Two adapters running back-to-back via the `_run_dataset_adapters`
  iteration.
- Adapter timeout enforcement (because there's no timeout mechanism
  to test).
- Indicator-cache thundering herd (the `asyncio.Lock` is mentioned
  but not unit-tested).

## Recommendations

1. **[high impact / medium effort] Split `core/knowledge.py` into a
   package.** Move external adapters to `core/knowledge/sources/`,
   full-text fetch to `core/knowledge/fulltext.py`, catalog +
   LLM-router to `core/knowledge/router.py`, structured-ingest
   helpers to `core/knowledge/ingest.py`, and keep `Knowledge` itself
   in `core/knowledge/__init__.py` re-exporting public symbols. Sets
   a precedent: when a single file passes ~1 000 LOC, audit for
   responsibilities first. Tests already import via
   `core.knowledge` so the migration is mechanical. Re-imports
   become cheaper too.

2. **[high impact / low effort] Add a `--list-adapters` CLI flag and
   fix the doc drift.** One-line addition to `launch.py`: print
   `ADAPTER_REGISTRY.keys()` + adapter docstrings, exit. While there,
   fix the `dataset_adapters` field comment in `core/config.py:181`
   ("Available names: 'worldbank'") and `docs/USAGE.md:347-351` to
   include Wikipedia. Better: replace the prose with a generated
   block (or have the docstring read from `ADAPTER_REGISTRY` at
   import time). Single source of truth.

3. **[high impact / medium effort] Promote per-call timeout and
   lifecycle into the `DatasetAdapter` contract.** Add an optional
   `timeout_s` kwarg to `search()` and an optional `async prepare()`
   method. Engine passes `engine.dataset_adapter_timeout_s` (new
   config) to every `search()`. Adapters that want a global catalog
   (WorldBank, future OECD) override `prepare()` instead of using a
   module-level cache. This eliminates the module-level
   `_indicator_cache` side channel.

4. **[medium impact / medium effort] Add 429-aware retry + skip
   to the external-source adapters.** Standardize a helper that
   wraps `httpx.Client.get`, inspects status, sleeps `Retry-After`
   if ≤ a small budget (e.g., 3 s), retries once, then returns None.
   For the LLM-routed path, also let the router LEARN — pass a
   "recently-rate-limited sources" set to the next router call so it
   skips a 429-prone adapter for the rest of the quest.

5. **[medium impact / low effort] Cache the source-router decision
   per quest by `(topic, idea_hash)`.** A single dict on the
   `Knowledge` instance, evicted on engine teardown. Saves N-1 LLM
   calls per quest when `source_routing="auto"`. Two-line change.

6. **[medium impact / low effort] Pull `_http_get_json_sync`,
   `_http_get_text`, and `_query_keywords` out of `worldbank.py`
   into `core/datasets/_util.py`.** Removes Wikipedia's vestigial
   dependency on WorldBank's internals. Trivial refactor; passes
   straight-through.

7. **[medium impact / medium effort] Add a quest write-back quality
   filter.** Knob: `KnowledgeConfig.write_back_min_score: int = 4`
   (skip writes when verdict score < this). Default 0 preserves
   current behavior; users running large fleets can raise it. While
   there, dedup `fi_external_ref_spine` writes against the existing
   `paper_id` so we don't accumulate 10 copies of the Grenville 2015
   spine.

8. **[medium impact / low effort] Strengthen `_paper_short_id`
   title normalization.** Strip leading "The /A /An ", Unicode-fold
   via `unicodedata.normalize("NFKD", s)`, collapse all whitespace.
   ~5 LOC. Catches a real failure case (Schrödinger vs Schrodinger
   dedup).

9. **[low impact / low effort] Add a `DatasetAdapter.relevance_score(
   query) -> float | None` optional hook.** Adapters can declare "I'm
   probably useful for this query" or "I'm probably not." Engine
   short-circuits adapters returning a low score. Future Hofstede /
   Pew adapters benefit most — they're heavily domain-scoped.

10. **[low impact / low effort] Convert sync `httpx.Client` to
    `httpx.AsyncClient` in the external-source adapters.** The
    current `asyncio.to_thread` indirection has a real socket-leak
    risk (F5). With AsyncClient, cancellation is honored properly.
    The router stays mostly the same — `asyncio.gather` over async
    functions instead of `to_thread` wrappers.

11. **[low impact / low effort] Document the API-key story.** Today
    only CORE has a key (read from env at call time,
    `core/knowledge.py:351-354`). Future adapters (Hofstede CSV,
    Pew, premium S2) will need keys. The contract should formalize:
    where do keys live (env var only? config file?), what happens
    when missing (skip silently like CORE does today vs. warn at
    startup), and how does `--list-adapters` show which need a key.

12. **[low impact / medium effort] Outline OECD / Eurostat additions
    as a follow-up.** Both speak SDMX-JSON, share a dataflow-catalog
    pattern, would benefit hugely from F6's `prepare()` lifecycle,
    and don't need API keys. They could be implemented as a single
    `SDMXAdapter` base subclass + two thin wrappers (URLs differ).
    Estimated 300 LOC each. Worth one ticket; defer Hofstede + Pew
    because they need licensed CSVs and an out-of-band ingest path
    that doesn't fit the `search(query, top_k)` contract.

## References

- `core/knowledge.py:1-1657` — full module (six concerns wedged into
  one file: external adapters, paywall fetch, local-paper loader,
  source catalog + LLM-routed picker, dedup + router, structured
  ingest + `Knowledge` facade).
- `core/datasets/__init__.py:1-40` — `ADAPTER_REGISTRY` (the sole
  discoverability surface, currently `worldbank` + `wikipedia`).
- `core/datasets/base.py:1-75` — `DatasetAdapter` ABC, single
  `search(query, top_k) -> list[DatasetRow]` method.
- `core/datasets/worldbank.py:1-421` — heuristic country detection,
  module-level `_indicator_cache` + `asyncio.Lock`, parallel
  per-indicator gather, 5-year window default.
- `core/datasets/wikipedia.py:1-218` — opensearch → summary chain,
  imports `_http_get_json_sync` + `_query_keywords` from worldbank.py
  (F8 finding).
- `tests/test_knowledge.py:1-1115` — 40 tests, exemplary coverage of
  the knowledge module's branches.
- `core/config.py:166-194` — `dataset_adapters` + `dataset_adapter_top_k`
  YAML knobs (with the doc-drift bug in F7).
- `core/config.py:204-307` — `KnowledgeConfig` (Axon + external-fallback
  + paywall flags).
- `core/engine.py:864-942` — `_run_dataset_adapters`, the engine-side
  consumer of `ADAPTER_REGISTRY`.
- `core/engine.py:1724-1798` — write-back gate + invocation site for
  `add_quest_artifacts`.
- `docs/USAGE.md:170-359` — user-facing adapter docs (with the
  doc-drift bug in F7).
