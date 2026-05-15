# 08 — Documentation drift: concrete fix list

Date: 2026-05-15
Scope: stale documentation surfaces after PRs #58–#62. Output: a concrete edit
list a follow-up PR can apply mechanically.

---

## Findings

### Per-feature coverage matrix (re-stated, with current line references)

Each row shows which doc surface currently mentions the feature **substantively**
(not by passing string match). All line numbers refer to `main` at SHA `c5acf90`
(post PR #63).

| Feature (PR) | README.md | docs/USAGE.md | docs/capabilities.md | docs/INSTALL.md | vscode-frontier-insight/README.md |
|---|---|---|---|---|---|
| Pandoc/LaTeX pre-flight + `output.require_pdf` strict mode (#58) | ⚠ partial — L41 mentions optional `pandoc`/tectonic install, but the **strict-mode pre-flight + `require_pdf` field is absent** | ✓ L207, L210–237 dedicated section | ✓ L63 capability + L191 YAML field | ⚠ partial — covers tooling install but **no `require_pdf` pre-flight semantics** | ✗ |
| Simulatability decision (#59) | ✗ | ✓ L167 schema + L297–325 prose | ✓ L72 (single mega-row) | ✗ | ✗ |
| Auto-collect data (#60) | ✗ — no `data/auto_collected/`, no `auto_collect_data` mention | ✓ L168–169 schema + L326–375 prose | ✓ L72 (same mega-row) | ✗ | ⚠ no — `wait_for_data` / `auto_collect_data` flow absent |
| Dataset adapters / WorldBank (#61) | ✗ | ✓ L170–171 schema + L342–360 prose | ✓ L72 (same mega-row) | ✗ — WorldBank has zero network/HTTPS notes | ✗ |
| Wikipedia adapter (#62) | ✗ | ⚠ partial — listed as available name on L170 + L171 prose mentions `"wikipedia"`, but **no behaviour paragraph dedicated to it** | ⚠ partial — listed in passing (capability mega-row) | ✗ | ✗ |

**Empirical confirmation** of "absent":

- `README.md` matches for `no_simulation|auto_collect|dataset_adapters|wikipedia|simulatability|require_pdf` → **0 hits**.
- `docs/INSTALL.md` same regex → **0 hits**.
- `vscode-frontier-insight/README.md` same regex → **0 hits**.

So the matrix from the prior audit slightly over-credited README/INSTALL on #58 and #60: the docs mention `pandoc` and `tectonic` (which predate #58) but **not** the strict-mode pre-flight delivered in #58, and they mention output paths but not `data/auto_collected/` from #60. The actual coverage is closer to 20/20/20% for README/INSTALL/VSCode rather than 40/40/20%.

### Schema drift in `docs/USAGE.md` (YAML config schema block, L131–208)

Spot-checked the schema block against `core/config.py` (the source of truth):

| Field in `core/config.py` | In `USAGE.md` schema block? | Notes |
|---|---|---|
| `Config.extra_directives` (top-level free text) | ✗ Missing | Only mentioned by name in `architecture.md` L69. The field is wired into prompts but invisible in the user-facing schema reference. |
| `ProviderConfig.base_url` | ✗ Missing | Required for OpenAI-compatible custom endpoints (proxies, local OAI gateways). Present in `capabilities.md` L137. |
| `ProviderConfig.api_key_env` | ✗ Missing | Override the standard env-var name; used in air-gapped setups. Present in `capabilities.md` L138. |
| `ProviderConfig.extra` | ✗ Missing | Transport-specific bag (e.g., extra headers). Present in `capabilities.md` L139. |
| `EngineConfig.enable_analyze_reroute` | ✓ Present L161 | OK |
| `EngineConfig.exec_reflect_max_iterations` | ✓ Present L159 | OK |
| `EngineConfig.cross_check_per_finding_k` | ✓ Present L160 | OK |
| `ExecutionConfig.python_version` | ✓ Present L176 | OK |
| `KnowledgeConfig.seed_source_catalog` | ✓ Present L191 (visible nearby) | OK |
| `OutputConfig.require_pdf` | ✓ Present L207 + dedicated section | OK |

So **4 schema fields are missing** from the user-facing YAML reference in `USAGE.md`. All 4 live on `ProviderConfig` / top-level `Config`, not on the engine — but they all affect day-to-day user behavior (proxy setup, multi-env API keys, extra free-form steering text).

### Schema drift in `docs/capabilities.md` (config block, L128–194)

Same cross-check against `core/config.py`:

| Field in `core/config.py` | In capabilities.md config block? | Notes |
|---|---|---|
| `EngineConfig.no_simulation` | ✗ Missing from L151–161 block (only described inside the long capability-row at L72) | The YAML reference at L151 still lists the *pre-#57* engine fields. The post-#57/#60/#61 fields (`no_simulation`, `auto_collect_data`, `auto_collect_top_k`, `dataset_adapters`, `dataset_adapter_top_k`) are entirely absent from this canonical schema block. |
| `EngineConfig.auto_collect_data` | ✗ Missing | same |
| `EngineConfig.auto_collect_top_k` | ✗ Missing | same |
| `EngineConfig.dataset_adapters` | ✗ Missing | same |
| `EngineConfig.dataset_adapter_top_k` | ✗ Missing | same |
| `OutputConfig.require_pdf` | ✗ Missing from L190–194 block | The output block stops at `output_dir`; doesn't mention `require_pdf`. |

So `docs/capabilities.md` claims to be "every YAML field with defaults" (L400 of README references it as such), yet **6 engine/output fields shipped in PRs #57–#61 are missing** from its config-reference block. The behavioral description exists (in L72 as a 600-character compound row) but the user can't find the field by scanning the schema.

### Provider matrix in `docs/PROVIDERS.md`

Counted against `core/config.py:ProviderName` (13 entries: `codex, openai, gemini, ollama, vllm, claude_code, github_copilot_cli, github_copilot_vscode, codex_cli, claude_cli, copilot_cli, gemini_cli, vscode_extension`).

`PROVIDERS.md` matrix at L45–59 lists all **13/13** ✓. No drift here.

### Architecture diagram in `docs/architecture.md` (L5–62)

Comparing the DAG box at L19–26 to today's engine:

```
ideate ▸ literature ▸ design ▸ implement ▸ execute ▸ analyze ▸ write ▸ review
```

vs. shipped DAG (per `core/engine.py` and `capabilities.md` L9–27):

```
clarify → ideate → literature → design → implement → execute
       ↘ execute_reflect ↗
                              ↘ analyze → cross_check → write → review
```

**Missing from diagram (actual graph nodes)**: `clarify` (Phase I, pre-#57),
`execute_reflect` (Phase K), `cross_check` (Phase L), `auto_collect_data` (#60),
`wait_for_data` (#57), `data_load` (#57). `ideate_reflect` and `review_panel` are
in-node behaviors (the reflect call lives inside `_node_ideate`; the panel runs
inside `_node_review`) — they don't need separate boxes in the DAG, just a
"reflect" hint on the ideate node and a "panel fan-out" hint on review.

**Missing from supporting boxes**:

- The `knowledge.py` box (L33–38) says `router (7 srcs)` — count is right but adapter
  count is now also relevant: `core/datasets/` has WorldBank (#61) + Wikipedia (#62)
  and the diagram doesn't reflect dataset-adapter retrieval as a separate path.
- The `Backends` box (L43–53) misses `vscode_extension`, `claude_cli`, `codex_cli`,
  `copilot_cli`, `gemini_cli` — only HTTP-direct + 2 proxies are shown. This is
  pre-CLI-transport (pre-Phase H) staleness.
- `_run_dataset_adapters` (engine.py:864) — no helper-level mention anywhere in
  `architecture.md`, although the file's own Key Contracts section claims to
  enumerate engine helpers.

Grep confirms it: `auto_collect|wait_for_data|simulatability|data_load|worldbank|wikipedia` in `docs/architecture.md` → **0 matches**.

---

## Recommendations

Concrete TODO table for the follow-up PR. **Section/line refs are approximate
insertion points; the follow-up author should still verify before applying.**

Impact legend: **L**arge = first-time user makes wrong decisions without it;
**M**edium = power user has to read source to learn it; **S**mall = polish /
internal consistency. Effort legend: **S** = 1–3 sentences; **M** = a fenced
example + paragraph; **L** = restructure or substantial new section.

| # | File | Section (approx. line) | What to add | Impact | Effort |
|---|---|---|---|---|---|
| 1 | `README.md` | Requirements (L37–41) — append after L41 | New line clarifying network surfaces: *"The Wikipedia + WorldBank dataset adapters (PR #61, #62) use stdlib `urllib` against `api.worldbank.org` and `en.wikipedia.org` — no extra `pip install` needed, but the runtime needs outbound HTTPS to those hosts when `engine.dataset_adapters` is enabled."* (Original draft erroneously suggested `pip install httpx beautifulsoup4`; `httpx` is already a runtime dep and the adapters use neither.) | S | S |
| 2 | `README.md` | "What you'll have after one quest" (L45–61) | Add to the file-tree block, under `.fi/run.log`: `data/auto_collected/<rank>_<slug>.md       ← Axon hits (no-sim mode)` AND `data/auto_collected/<adapter>/<rank>_<slug>.md  ← dataset-adapter hits (no-sim mode)` — adapters write into a per-adapter subdirectory, Axon writes directly under `auto_collected/`. | M | S |
| 3 | `README.md` | "Common things you might want next" — insert new H3 between L376 ("Drop a paywalled PDF") and L383 ("Three example quests") | New section: **"Topics that need real data, not a Python script"** — 4-6 lines: `engine.no_simulation: true` for qualitative/social/archival research. Show YAML stub. Mention Axon + WorldBank + Wikipedia auto-collection in one bullet; mention `--resume` after dropping `data/`. Link to `USAGE.md#topics-that-need-real-data-not-simulation`. | L | M |
| 4 | `README.md` | "Common things you might want next" — insert new H3 near "Run many quests in parallel" (L237) | New section: **"Make the PDF compile a hard gate"** — 3 lines: `output.require_pdf: true` for CI/unattended; explain pre-flight saves ~15 min on doomed quests; link to `USAGE.md#outputrequire_pdf--strict-mode-pdf-enforcement`. | M | S |
| 5 | `README.md` | "Why Frontier Insight?" table (L26–31) | Add one cell to the existing "Hand-rolled LangGraph + OpenAI scripts" row: *"+ structured-data adapters (`worldbank`, `wikipedia`) that pull live evidence into `data/auto_collected/` for qualitative/archival topics, no Python experiment required."* Or add a new dedicated row for the no-sim path. | M | S |
| 6 | `docs/INSTALL.md` | After "System tools" table (L104–118) | New H3 **"Optional: dataset adapters"** — *"PR #61 + #62 add structured-data adapters that hit public APIs. Network reqs: HTTPS to `api.worldbank.org`, `en.wikipedia.org`. Implemented with stdlib `urllib` — no extra `pip install` needed. Enable per-quest via `engine.dataset_adapters: [worldbank, wikipedia]`."* | M | S |
| 7 | `docs/INSTALL.md` | After Path 3 or in Troubleshooting (L153) | New troubleshooting bullet: **"`output.require_pdf: true` aborts pre-flight"** — list the install recipe for each missing tool (pandoc, pdflatex, tectonic) referenced by the pre-flight error message. (Pre-flight implementation is `Engine._preflight_paper_pdf` in `core/engine.py`, NOT a separate `core/preflight.py` module — there is no such file.) | M | S |
| 8 | `docs/INSTALL.md` | Verifying the install (L141–151) | Add a "Verify the no-simulation path" sub-step: a `fi --config examples/no_sim_belgium_taiwan.yaml` line (if/when the example ships) OR a short YAML inline showing `engine.no_simulation: true` + `engine.dataset_adapters: [worldbank]`, with expected pause behavior. | S | M |
| 9 | `docs/INSTALL.md` | Path 2 — No-admin install (L42–67) | One-line addition under MiKTeX config: *"For unattended CI runs, set `output.require_pdf: true` in your YAML so a misconfigured MiKTeX fails fast pre-flight instead of after a 15-minute LLM burn."* | S | S |
| 10 | `vscode-frontier-insight/README.md` | Settings table (L70–74) | No change needed for the settings themselves — but add a new H3 **"Topics that need real data (no simulation)"** below "Resume a crashed quest" (L138–161). Two short paragraphs: (a) When `engine.no_simulation: true`, the chat panel shows a *"Quest paused for data — drop files into `data/`"* line and clean exit; (b) `auto_collect_data` lights `data/auto_collected/` with Axon + dataset-adapter hits BEFORE that pause, so many no-sim quests run end-to-end without manual data drops. | L | M |
| 11 | `vscode-frontier-insight/README.md` | "Cost & rate-limit reality" (L229–242) | Add one bullet to per-quest burn table: *"No-simulation quest (no design/implement/execute steps): ~6 premium requests"*. | S | S |
| 12 | `vscode-frontier-insight/README.md` | "Usage" section (after L101 modal flow list) | Add modal **"Real-world data?"** to the 6-question list: turn it into a 7th question — *"7. Real-world data? — simulation OK / needs real data (no-sim mode)."* This mirrors the underlying `engine.no_simulation` knob the interview should be setting. (Also requires extension code change; flag as docs-only for now and note "(extension change tracked separately)".) | M | S |
| 13 | `vscode-frontier-insight/README.md` | Limitations (L259–276) | Add one bullet: *"`engine.dataset_adapters` adapter calls go through the **engine's** httpx client, NOT vscode.lm — they DO need outbound HTTPS even when all LLM calls go through Copilot."* | M | S |
| 14 | `vscode-frontier-insight/README.md` | After "Per-node model routing" (L206–227) | Add **"PDF strict mode"** mini-section: paste a 3-line YAML fragment showing `output.require_pdf: true`, mention CI use, link to `USAGE.md` strict-mode docs. | S | S |
| 15 | `docs/USAGE.md` | YAML config schema (L131–208) — Provider block (L142–151) | Add the 3 `ProviderConfig` fields missing from `USAGE.md` (note: `docs/capabilities.md` already documents these — this row is USAGE-only): `base_url: null` (comment: "for OpenAI-compatible custom endpoints"), `api_key_env: null` (comment: "override default env-var name"), `extra: {}` (comment: "transport-specific bag — e.g., proxy headers"). | M | S |
| 16 | `docs/USAGE.md` | YAML config schema (L131–208) — top-level | Add `extra_directives: ""` after `output:` block (comment: "free-text steering prepended to every node's system prompt"). | M | S |
| 17 | `docs/USAGE.md` | After L171 (schema for `engine.dataset_adapters`) | Add a 3-line dedicated **Wikipedia adapter** sub-bullet mirroring the WorldBank one at L355–360. Cover the ACTUAL behavior (per `core/datasets/wikipedia.py`): compresses query to top-6 keywords → hits `opensearch` for candidate titles → fetches `/page/summary/<title>` for each → writes Markdown carrying `description`, `extract`, canonical `url`, and `wikipedia_type` in YAML front matter. (Do NOT describe infoboxes or revision-id storage — those are not in the implementation.) | M | S |
| 18 | `docs/USAGE.md` | YAML schema block — `output:` (L203–207) | Move the `require_pdf: false` line to be the FIRST output field instead of last (or add an inline `# strict mode — see section below` so first-time readers don't skip past it). Visibility nudge only. | S | S |
| 19 | `docs/capabilities.md` | Configuration block (L151–161) — engine sub-block | Add the 5 post-#57 fields with defaults + 1-line comments: `no_simulation: false`, `auto_collect_data: true`, `auto_collect_top_k: 5`, `dataset_adapters: []`, `dataset_adapter_top_k: 3`. Currently absent from this canonical schema reference. | L | S |
| 20 | `docs/capabilities.md` | Configuration block (L190–194) — output sub-block | Add `require_pdf: false` with one-line comment. Mirrors `core/config.py:OutputConfig.require_pdf`. | M | S |
| 21 | `docs/capabilities.md` | Provider block (L134–149) | Add `base_url: null`, `api_key_env: null`, `extra: {}` to mirror `ProviderConfig`. | S | S |
| 22 | `docs/capabilities.md` | Top of file or under Configuration block | Add `extra_directives: ""` to the top-level field list. | S | S |
| 23 | `docs/capabilities.md` | Capability inventory (L72 — the mega-row for no-simulation) | Split the 600-character mega-row into 3 smaller rows: (a) "No-simulation decision" (#59), (b) "Auto-collect via Axon" (#60), (c) "Dataset adapters" (#61 + #62). Improves scannability AND lets each row link to its own USAGE.md anchor. | M | M |
| 24 | `docs/architecture.md` | DAG box (L19–26) | Replace the 7-node string with the full 11-node DAG (matching `capabilities.md` L9–27). Add the conditional-edge labels: `execute_reflect → execute` retry, `cross_check → design` redesign, `review → design` revise. | L | M |
| 25 | `docs/architecture.md` | DAG box — new branch | Add the no-simulation branch under `design`: `→ auto_collect_data → wait_for_data → data_load → analyze`. Show as a parallel branch alternative to `implement → execute → execute_reflect`. | L | M |
| 26 | `docs/architecture.md` | Knowledge box (L29–39) | Add a sub-box / arrow for `core/datasets/` adapters (`worldbank`, `wikipedia`) writing into `data/auto_collected/<adapter>/`. Currently invisible in the architecture diagram. | M | M |
| 27 | `docs/architecture.md` | Backends box (L43–53) | Add the missing transports: `vscode_extension` (VSCode bridge), `claude_cli` / `codex_cli` / `copilot_cli` / `gemini_cli` (CLI exec). Pre-Phase-H content. Out-of-scope drift but a one-line fix while editing the file. | M | S |
| 28 | `docs/architecture.md` | Key contracts → QuestState (L73–87) | Add `no_simulation_resolved`, `auto_collected_count`, `data_load_*` (whatever names PR #57 + #60 added). The current QuestState doc enumerates `clarify_*`, `exec_reflect_*`, `cross_check`, `ideate_critique`, `review_panel` — but not the no-simulation fields. | M | S |
| 29 | `docs/architecture.md` | After "Concurrency model" (L141–168) | New short paragraph documenting `_run_dataset_adapters` (engine.py:864) as the helper that calls adapter modules in `core/datasets/`. Counterpart to the `cross_check` / `execute_reflect` helper mentions implicit elsewhere. | S | S |
| 30 | `docs/PROVIDERS.md` | No change | All 13 providers already listed. Schema is in sync. (Row in this table for completeness — explicitly no action.) | — | — |

### Decisions made for "is it worth adding?"

A handful of cells in the original matrix are correctly *empty* — adding them would
hurt rather than help:

- **Wikipedia adapter (#62) in `vscode-frontier-insight/README.md`**: Skip a
  dedicated section. It's a config knob (`engine.dataset_adapters`) with no
  user-visible UI surface in the chat extension; row 13 above covers the one
  network caveat that's chat-extension-specific (httpx vs `vscode.lm`).
- **Dataset adapters (#61) in `README.md`**: Skip a top-level feature highlight.
  Power-user knob, not first-impression material. Row 5 above is a one-cell
  table addition rather than a full section.
- **Pandoc/LaTeX pre-flight (#58) in `vscode-frontier-insight/README.md`**:
  Row 14 above adds a 3-line note; full section would over-weight a feature
  that's identical for chat-extension users and CLI users.
- **Simulatability decision (#59) in `docs/INSTALL.md`**: Skip — it's a *behavior*
  flag, not an install concern. Anything that happens at run-time per quest
  doesn't belong in INSTALL.

### Rough total

30 rows ≈ ~25 actual file edits (some rows in `architecture.md` cluster). Effort
mix: ~16 S, ~10 M, ~4 L. A single follow-up PR can ship all of it; the L-effort
rows (3, 10, 19, 24, 25) are the only ones likely to require >15 minutes each.

### Suggested PR sequencing

If 30-row PRs feel heavy, split into 3 surgical PRs:

1. **Schema sync** (rows 15–22, 28): pure additive — fields already work, docs
   catch up. Low risk, fast review.
2. **No-simulation user surface** (rows 1–5, 10–14, 23): user-facing prose for
   PRs #57/#59/#60/#61/#62. The bulk of "what does FI actually do" drift.
3. **Architecture diagram refresh** (rows 24–27, 29): the architecture.md
   redraw is its own substantive change and probably wants visual review.

---

## References

Files inspected (absolute paths, current SHA `c5acf90`):

- `C:\dev\FrontierInsight\README.md` (424 lines)
- `C:\dev\FrontierInsight\docs\INSTALL.md` (165 lines)
- `C:\dev\FrontierInsight\docs\USAGE.md` (434 lines)
- `C:\dev\FrontierInsight\docs\capabilities.md` (246 lines)
- `C:\dev\FrontierInsight\docs\PROVIDERS.md` (130 lines)
- `C:\dev\FrontierInsight\docs\architecture.md` (180 lines)
- `C:\dev\FrontierInsight\vscode-frontier-insight\README.md` (276 lines)
- `C:\dev\FrontierInsight\core\config.py` (345 lines — source of truth for schema + ProviderName enum)
- `C:\dev\FrontierInsight\core\engine.py` (lines 791, 864 — `_run_dataset_adapters`)

PRs whose docs surface is incomplete:

- #58 — pre-flight pandoc/LaTeX check + `output.require_pdf` strict mode
- #59 — transparent simulatability decision
- #60 — Phase D1 auto-collect via Axon
- #61 — Phase D2 dataset adapters (WorldBank)
- #62 — Phase D3 Wikipedia adapter

The PRs themselves landed clean tests + USAGE/capabilities entries; what's
missing is the up-stream user-facing prose (README, INSTALL, VSCode README)
and the up-to-date schema reference inside the docs that *claim* to be canonical.
