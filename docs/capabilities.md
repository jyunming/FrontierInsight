# Frontier Insight — capability reference

What this doc is: a feature index for people evaluating FI or looking
for a specific knob to twist. README is the elevator pitch; USAGE.md
is the YAML schema; this file is the breadth catalogue.

## The 14-node DAG

> Simplified happy path below — see ``core/engine.py:_build_graph``
> for the authoritative edge set. Conditional branches that aren't
> drawn: the design→implement vs design→auto_collect_data fork (no-
> simulation mode), and the ``broaden_lit`` arc from cross_check back
> to literature when analyze asks for a wider sweep.

```
START → clarify → ideate → literature ←─────┐ (broaden_lit)
                                │             │
                                ↓             │
                              design ─────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ↓ (computational)                       ↓ (no_simulation)
        implement ──→ execute ──┐         auto_collect_data
                                │                   ↓
                          execute_reflect ←─(retry) wait_for_data
                                │                   ↓
                                ↓ (proceed)     data_load
                              analyze ←──────────────┘
                                ↓
                          cross_check ──┐ (re_experiment → design)
                                │       │ (broaden_lit → literature)
                                ↓ (write)
                              write
                                ↓
                              review ──→ END (accept)
                                │
                                ↓ (revise → design)
```

Feedback loops in this diagram: `execute_reflect → execute` (retry),
`cross_check → design` or `cross_check → literature` (analyze re-
route), `review → design` (revise). The human-feedback gate, when
enabled, inserts a `human_feedback` node between review and END.

Feedback loops:

| Loop | Trigger | Bound by |
|---|---|---|
| `execute_reflect → execute` | Script crashed (rc != 0 or no `RESULT_JSON`) | `engine.exec_reflect_max_iterations` (default 3) |
| `cross_check → design` | `analyze.next_step == "re_experiment"` | `engine.max_iterations` (default 2) |
| `cross_check → literature` | `analyze.next_step == "broaden_lit"` — re-enters literature with the design's hypothesis folded into the query | `engine.max_iterations` (default 2) |
| `review → design` | `review.verdict == "revise"` | `engine.max_iterations` (default 2) |

## Capabilities

Grouped for skimmability. If you want to know whether FI does X, this
is the index.

### Engine + execution

- **Async LangGraph engine** with 14 nodes and 4 feedback loops; see `core/engine.py:_build_graph` for the actual edges.
- **Per-quest venv** — agent-generated Python is installed and run in isolation.
- **Docker sandbox** — `execution.sandbox: docker` runs the experiment subprocess with network disabled, mounted at `/work`.
- **Provider matrix** — direct HTTP, proxy, CLI exec, and VSCode-extension transports (see provider matrix below).
- **Per-node model routing** via `provider.node_models` (e.g., a cheap model for `clarify`, a strong one for `write`).
- **Pre-flight `clarify` node** — slot survey before `ideate`; off / auto / interactive modes.
- **Execute-repair loop** — agent reads traceback, patches code, retries (`engine.exec_reflect_max_iterations`).
- **Analyze-driven re-route** — `re_experiment` re-enters design; `broaden_lit` re-enters literature with the hypothesis folded in.
- **Cross-paper check** — per-finding literature search + supporting / conflicting / neutral classification.
- **Ideate self-reflection** — extra LLM call may swap the chosen idea (or `engine.ideate_tournament: true` for pairwise N-way comparison).
- **Reviewer panel** — N personas in parallel + moderator synthesis (`rigor_score` + `depth_score` axes).
- **Human-feedback gate** — `engine.human_feedback_gate: after_review` pauses with a snapshot for the user to accept / reject / refine. Snapshot at `<quest_root>/.fi/human_review.json`; CLI wired through `--interactive`. Default `off`.
- **Multi-model ensemble** — fan out a node's LLM call across multiple models in parallel and merge. Per-node `provider.node_ensemble` with `models` / `merge` (`tournament` | `synthesize` | `vote`) / `moderator`. Interview surfaces four preset profiles (`off` / `cross_check_only` / `ideate_and_check` / `full`, cost multipliers 1.0× → 2.5×). `analyze.merge: synthesize` is rejected at config load.
- **No-simulation decision** — three-tier (`engine.no_simulation` → clarify `simulatability` → legacy `empirical_vs_theoretical`); routes through `auto_collect_data → wait_for_data → data_load → analyze`.
- **Auto-collect before pause** — `auto_collect_data` queries Axon with `topic + hypothesis` and drops up to `engine.auto_collect_top_k` (default 5) hits into `data/auto_collected/` before pausing for the user.
- **Dataset adapters** — `engine.dataset_adapters` opts into structured external lookups (`worldbank`, `wikipedia`). Writes evidence into `auto_collected/<adapter>/`.

### Knowledge layer

- **Axon-backed retrieval + write-back** — literature retrieval and cross-quest memory.
- **Independent Axon vs external caps** — `knowledge.top_k` (default 8, RAG precision) is independent of `knowledge.external_top_k` (default 20, web breadth). The literature node passes both; ideate/cross_check seeds respect the smaller per-call cap.
- **Audience-aware citations** — `output.audience: "external" | "internal"`. External papers drop FI-internal cross-quest entries from References; internal papers keep everything.
- **Axon sidecar auto-launch** — every CLI / `--serve` boots `python -m axon.api` on `127.0.0.1:8000` if not already listening. Skip with `--no-axon-sidecar` or `FI_NO_AXON_SIDECAR=1`.
- **Multi-source literature router** — arXiv / OpenAlex / Crossref / Semantic Scholar / PubMed / CORE / Google Scholar in parallel, DOI-dedup.
- **Topic-aware LLM source routing** — agent picks venues from a 12-entry catalog.
- **Local paper feed** — `knowledge.local_papers` pins paywalled PDFs / MDs to the retrieval head.
- **Pause-for-user-papers gate** — `knowledge.pause_for_user_papers: true` pauses when retrieval is abstract-only; writes `needs/<slug>.json` stubs + `inputs/papers/README.md`. User drops PDFs, runs `fi --resume`; literature node walks them via pypdf and appends as `source=user_supplied`. Default `false`.
- **Opportunistic full-text fetch** — host-network publisher PDFs with login-wall rejection.
- **Structured ingest** — title-searchable spine docs + citation-header'd body + topic rollups.

### Outputs

- **Paper PDF** via pandoc + LaTeX — real venue-flavored templates for `generic` / `neurips` / `iclr` / `ieee_access` two-column / `nature_mi`. Preprocessor lifts `# H1` into YAML `title:`, lifts `## Abstract` into YAML `abstract:`, shifts heading levels by −1, dedupes `"Title. Title."` patterns, sets PDF metadata via `\hypersetup`.
- **Non-scientific paper formats** — `essay` / `report` / `policy_brief` / `whitepaper`, each with a full LaTeX template + per-format writer persona.
- **Slides** — `slides.md` always; `slides.html` / `slides.pdf` (Marp CLI) and `slides.pptx` (pandoc) when available. `poster` via beamerposter; `speech` via single LLM call.
- **Cite-by-content references** — prior-work block carries author / year / title / venue / DOI so References renders proper citations. Writer prompt forbids fabricating fields or embedding fenced code in the body.

### Cost + observability

- **Engine cost instrumentation** — `core/provider.py:MODEL_PRICING` + `estimate_cost_usd(...)`. `Engine._chat` appends `{ts, node, model, usage, cost_usd}` to `<quest_root>/.fi/cost.jsonl`. Char-based token estimate (~4 chars/token) when the transport returns no `usage`; row carries `usage.estimated: true`. Quest finalization writes `<quest_root>/.fi/cost.summary.json` with totals + per-node + per-model breakdowns.

### Interviews + frontends

- **Unified interactive interview** — CLI `--new`, VSCode `@fi /new`, web `/interview` all drive `core/interview.py` (snapshot in `core/interview_schema.json`). 6 Tier-1, ~7 Tier-2 auto-derived, ~6 Tier-3 advanced. Pinned values land in `engine.clarify_overrides`.
- **Mid-quest interview re-entry** — `--update <id>` / `@fi /update <id>` / `POST /api/interview/update/{id}` re-open the interview with editable fields. `STAGE_INVALIDATION` decides which nodes re-run. `no_simulation` is hard-locked mid-quest.
- **Status GUI** — FastAPI server at `web/static/index.html`: fleet list, SSE log stream, clarify panel, paper preview, panel-review cards. Tools dropdown with ARIA semantics (`aria-haspopup` / `aria-expanded`) + Escape-to-close + first-menuitem auto-focus on open.
- **Web UI quest launching** — `--serve` spawns `python launch.py --config <yaml>` via `web/quest_launcher.py` (capped at `--max-concurrent`). Detail page auto-refreshes every 3 s. Non-loopback bind logs a WARNING.
- **Web UI "everything" pass** — `/tools/{proposal,critique,digest,portfolio,summarize,analyze,fleet,ingest}`, file browser, zip download, resume button, clarify-resume panel, labels, cost chart, paper iterations, exec edit (gated by `FI_WEB_ALLOW_EXEC_EDIT=1`), trash bin at `/trash`, `/compare?a=&b=`, `/settings`, `/about`. Self-hosted Markdown renderer (`md_lite.js`).
- **Bridge-aware vscode_extension across all serve forms** — `/interview`, `/update/<id>`, and every `/tools/*` form surface `vscode_extension` at the top of the provider picker when the dashboard was launched from a VSCode terminal (`FI_VSCODE_BRIDGE_PORT` inherited). Without a bridge, the entry stays hidden — same source of truth as the existing submit-time 400 guard so the UI never offers a path that the server would refuse. Model dropdowns offer "Other (type your own)" with a text-input fallback on every page.
- **VSCode extension** — sanctioned `vscode.lm.*` integration with `@fi /new`, `/start`, `/fleet`, `/resume`, `/summarize`, `/proposal`, `/critique`, `/digest`, `/portfolio`, `/analyze`, `/update`.
- **Accessibility** — `prefers-reduced-motion` honoured on `marketing/index.html` (terminal fade-in + pulse-dot disabled when the OS preference is set).

### CLI tools

- **`--serve`** — local web UI; honours `FI_VSCODE_BRIDGE_PORT` env var for routing LLM calls through Copilot when launched from a VSCode terminal. Refuses `provider=vscode_extension` quests when no bridge is wired (400 at submit).
- **`--resume <quest_id>`** — re-enter the LangGraph from the last checkpointed node when a prior run died mid-pipeline.
- **`--fleet <yaml> ...`** — bounded concurrency, ref-counted proxies, `--memory-cap-mb`, optional `viztracer --profile`.
- **`--summarize <folder>`** — walks mixed content, classifies each file, calls LLM once, writes `outputs/<summary_id>/summary.md`. Ingested into Axon.
- **`--digest --days N`** — weekly PM digest with WeekDiff (✅ promoted / 🆕 new / ⚠️ still-in-progress / 🛑 stalled / ❓ dropped) vs prior digest. Ingests as `fi_digest`.
- **`--portfolio`** — all-time cross-quest synthesis: topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps. Ingests as `fi_portfolio`.
- **`--critique <id>`** — adversarial second-pass review (`--critique-provider <name>` to use a different model family). Writes `outputs/<id>/critique.md`. Ingests as `fi_critique`.
- **`--proposal "<topic>"`** — pre-quest planning doc + companion YAML under `outputs/_drafts/`. Companion YAML auto-pins the proposal into `knowledge.local_papers`. Ingests as `fi_proposal`.
- **`--analyze <data-path> --analyze-topic "..."`** — no-simulation quest on pre-staged data. Routes through `auto_collect_data → wait_for_data → data_load → analyze → write → review`. ~6 premium requests.
- **`--ingest <paths>`** — one-shot Axon ingest, no quest. Zero LLM calls.
- **`--install-tectonic`** — drops a self-bootstrapping LaTeX binary (~70 MB) into `tools/`. SHA-256 verified.
- **`--install-tectonic-from <path>`** — airgapped variant: install from a locally-staged archive or extracted binary. No network call. ELF / Mach-O / PE header sanity check.
- **Resumable quests** — SQLite-checkpointed state at `<quest_root>/.fi/state.sqlite`.

## Provider matrix

| Provider | Transport | Auth | ToS standing |
|---|---|---|---|
| `openai` | HTTP direct | `OPENAI_API_KEY` env | ✅ Sanctioned |
| `codex` | HTTP direct | `OPENAI_API_KEY` env | ✅ Sanctioned |
| `gemini` | HTTP direct | `GEMINI_API_KEY` env | ✅ Sanctioned |
| `ollama`, `vllm` | HTTP direct (local) | none | ✅ Self-hosted |
| `claude_cli` | CLI exec | `claude login` (Claude Pro/Max OAuth) | ✅ Sanctioned |
| `codex_cli` | CLI exec | `codex login` (ChatGPT Plus/Pro OAuth) | ✅ Sanctioned |
| `copilot_cli` | CLI exec | `gh auth login` (Copilot subscription) | ⚠️ Agentic — replies conversationally; use `vscode_extension` for Copilot instead. |
| `gemini_cli` | CLI exec | `gemini` OAuth / Google AI Studio key | ✅ Sanctioned |
| **`vscode_extension`** | **VSCode bridge** | **VSCode Copilot Chat sign-in** | **✅ Sanctioned via `vscode.lm`** |
| `claude_code` | HTTP via proxy | `claude login` + spawned wrapper | ⚠️ Third-party wrapper |
| `github_copilot_cli` | HTTP via proxy | `gh auth login` + spawned `copilot-api` | ⚠️ Against ToS spirit (use `copilot_cli` instead) |
| `github_copilot_vscode` | HTTP via proxy | VSCode Copilot extension + spawned `copilot-api` | ⚠️ Against ToS spirit (use `vscode_extension` instead) |

`github_copilot_*` providers AND `copilot_cli` emit a one-time warning
at engine init. Set `FI_SUPPRESS_PROXY_WARN=1` to silence (use at your
own risk).

## Knowledge layer — three-layer retrieval

```
asearch(query)
  ├─ 1. Pinned local_papers (always first)
  ├─ 2. Axon (your long-term store; falls through on empty)
  └─ 3. External router: agent picks 1–5 sources from the catalog
        ├─ openalex / arxiv / crossref / semantic_scholar / pubmed / core / google_scholar
        └─ (optional) full-text fetch via host network
```

After an **accepted** quest, the write-back lays down a structured
bundle so Axon stays title-searchable and topic-linkable:

| kind | Purpose |
|---|---|
| `fi_paper_spine` | One tight chunk per paper — title / authors / DOI / abstract / key claims. Title queries hit this directly. |
| `fi_quest_paper` | Full paper body with a 1-line `[Title · Year · Venue · DOI]` citation header on every chunk. |
| `fi_quest_summary` | Structured-findings JSON (hypothesis, key_findings, result_json, verdict, score, model). |
| `fi_topic_event` | One pointer per accepted quest keyed by topic slug — enables "what do we know about X" rollups. |
| `fi_external_ref_spine` × N | Curated card-catalog entries for cited papers. Only refs from *accepted* quests persist. |

## Configuration

See [USAGE.md](USAGE.md) for the full annotated YAML schema. Quick
landmarks:

- `provider.*` — provider name, model, per-node model routing, ensemble.
- `engine.*` — clarify mode, max_iterations, review_panel, ensemble, no_simulation, human_feedback_gate.
- `execution.*` — venv vs docker, timeout, python version.
- `knowledge.*` — Axon config, top_k (RAG) + external_top_k (web), local_papers, pause_for_user_papers, full-text fetch.
- `output.*` — kinds, paper_format, audience, output_dir, require_pdf.

## Example quests

| Example | Provider used | Topic |
|---|---|---|
| [`integrator_bakeoff`](../examples/integrator_bakeoff/config.yaml) | (any) | Three numerical integrators on a damped harmonic oscillator (RK4 / Velocity-Verlet / forward Euler). |
| [`euv_mor_shot_noise`](../examples/euv_mor_shot_noise/config.yaml) | `ollama` → cloud-routed reasoning models | Theoretical LER floor imposed by Poisson photon shot noise in metal-oxide EUV resists at production doses (10–60 mJ/cm²). |
| [`bernstein_vazirani_noise`](../examples/bernstein_vazirani_noise/config.yaml) | `claude_cli` (reuses `claude login` OAuth) | Bernstein-Vazirani algorithm under per-gate depolarizing noise. |

## Demo scripts

| Script | Shows |
|---|---|
| `scripts/demo_structured_ingest.py` | The 4-doc + N-ref-spine bundle written to Axon after an accepted quest. |
| `scripts/demo_paywall_fetch.py` | arXiv success vs SPIE login-wall rejection — opportunistic full-text fetch. |
| `scripts/demo_local_papers.py` | Pinned-to-head behavior of `knowledge.local_papers`. |
| `scripts/probe_cli_providers.py` | Live-verify `claude_cli` / `codex_cli` / `copilot_cli` / `gemini_cli`. |
| `scripts/probe_copilot_vscode.py` | Live-verify `github_copilot_vscode` (prefer `vscode_extension`). |

## Tests

Pytest-based, Windows-native Python 3.11+, fake LLMs everywhere
(`monkeypatch.setattr` on `LLMClient.chat`) — zero real API spend in
CI. A few tests skip gracefully when their external tool isn't on
PATH (Docker daemon, Marp CLI, pdflatex).

## Architecture

See [`architecture.md`](architecture.md) for the layered diagram and
contracts (`Config`, `QuestState`, `QuestArtifacts`, `Executor`,
generator protocol).
