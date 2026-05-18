# Frontier Insight — capability reference

Detailed inventory of what Frontier Insight does end-to-end. The
front-page [`README.md`](../README.md) is the first-time-user guide;
this file is the deep reference.

## The 11-node DAG

```
START → clarify → ideate → literature → design → implement → execute
                                                       ↓
                                                execute_reflect ──┐
                                                       │           │
                                                  (retry)│  (proceed)
                                                       ↓           ↓
                                                    execute      analyze
                                                                   ↓
                                                              cross_check
                                                       ┌── (redesign) │ (write)
                                                       │              ↓
                                                      design        write
                                                                      ↓
                                                                    review → END
                                                                      │ (revise)
                                                                      ↓
                                                                    design
```

Three feedback loops:

| Loop | Trigger | Bound by |
|---|---|---|
| `execute_reflect → execute` | Script crashed (rc != 0 or no `RESULT_JSON`) | `engine.exec_reflect_max_iterations` (default 3) |
| `cross_check → design` | `analyze.next_step == "re_experiment"` | `engine.max_iterations` (default 2) |
| `cross_check → literature` | `analyze.next_step == "broaden_lit"` — re-enters the literature node so a second retrieval fetches fresh evidence (with the design's hypothesis folded into the query). Returns merge into the existing literature list with DOI / URL / content-prefix dedup, so design sees the accumulated corpus. | `engine.max_iterations` (default 2) |
| `review → design` | `review.verdict == "revise"` | `engine.max_iterations` (default 2) |

## Capability inventory

| Capability | Status |
|---|---|
| Async LangGraph engine: 11-node DAG with three feedback loops | ✅ |
| Per-quest venv; agent-generated Python is installed and run in isolation | ✅ |
| Docker sandbox via `execution.sandbox: docker` (network disabled, mounted at `/work`) | ✅ |
| Provider matrix — direct HTTP, proxy, CLI exec, and VSCode-extension transports | ✅ |
| Per-node model routing — different model per node via `provider.node_models` | ✅ |
| Pre-flight `clarify` node — 7-slot survey before `ideate`, off / auto / interactive modes | ✅ |
| Self-correction — execute-repair loop — agent reads traceback, patches code, retries | ✅ |
| Self-correction — analyze-driven re-route — `re_experiment` re-enters design; `broaden_lit` re-enters the literature node for a fresh retrieval (with design's hypothesis added to the query), then design picks up the accumulated, deduped literature corpus. | ✅ |
| Cross-paper check — per-finding literature search + supporting/conflicting/neutral classification | ✅ |
| Ideate self-reflection — extra LLM call may swap chosen idea | ✅ |
| Reviewer panel — N personas in parallel + moderator synthesis (`rigor_score` + `depth_score` axes) | ✅ |
| Multi-model ensemble — fan out a node's LLM call across multiple models in parallel and merge their answers. Per-node configuration under `provider.node_ensemble`: each node lists `models` (the per-call model overrides), `merge` (`tournament` = moderator picks one verbatim winner; `synthesize` = moderator merges with `⚠ Disagreement:` flags; `vote` = pure majority tally), and optional `moderator`. Supported on `ideate` / `analyze` / `cross_check` engine nodes plus the `--proposal` / `--critique` standalone tools via `--proposal-ensemble` / `--critique-ensemble` CLI flags. The interview surfaces four preset profiles (`off` / `cross_check_only` / `ideate_and_check` / `full`) with rough cost multipliers (1.0× → 2.5×) so the user picks intent, not raw models. Engine paths fall back to the single-call path on `EnsembleError` (every fan-out call failed); moderator transport failures fall back to first-survivor with a note inside the merger. Per-fan-out cost rows + per-moderator metadata rows land in `<quest_root>/.fi/cost.jsonl` for the cost chart. The `analyze` node refuses `merge: synthesize` at config load because its downstream parser expects JSON. | ✅ |
| Axon-backed knowledge layer: literature retrieval + cross-quest memory write-back | ✅ |
| Audience-aware citations — `output.audience: "external" \| "internal"` (default external). External papers drop FI-internal cross-quest entries (kind=fi_critique / fi_digest / fi_portfolio / fi_proposal / fi_summary / fi_source_catalog) from the References section so a journal / open-web reader doesn't see citations they can't look up. Internal papers (project reports, memos, onboarding docs) keep everything. Real external sources (arxiv / openalex / etc.) and `fi_local_paper` entries you ingested yourself are always kept. | ✅ |
| Axon sidecar auto-launch — every CLI / `--serve` invocation idempotently boots `python -m axon.api` on `127.0.0.1:8000` if it's not already listening, so the embedding model + vector indexes stay warm across quests instead of paying a 5-15 s cold-init per quest. Skip with `--no-axon-sidecar` or `FI_NO_AXON_SIDECAR=1`. Web UI surfaces status at `GET /api/axon/status`; VSCode probes on activate and offers a one-click "Start in terminal" if the sidecar is down, plus `@fi /axon-status` for explicit checks. | ✅ |
| Multi-source literature router when Axon is empty: arXiv / OpenAlex / Crossref / Semantic Scholar / PubMed / CORE / Google Scholar in parallel, DOI-dedup | ✅ |
| Topic-aware LLM source routing — agent picks which venues to query from a 12-entry catalog | ✅ |
| Local paper feed — drop paywalled PDFs / MD into `knowledge.local_papers`, pinned to retrieval head | ✅ |
| Opportunistic full-text fetch — host-network publisher PDFs with login-wall rejection | ✅ |
| Structured ingest — title-searchable spine docs + citation-header'd body + topic rollups | ✅ |
| Status GUI — FastAPI server with a vanilla-JS single-page frontend (`web/static/index.html`): fleet list, SSE log stream, clarify panel, paper preview, panel-review cards | ✅ |
| VSCode extension — sanctioned `vscode.lm.*` integration with `@fi /new`, `/start`, `/fleet`, `/resume`, `/summarize`. The `/new` interview asks 8 questions: topic, title, output kinds, **paper format** (all 9 venues — generic/neurips/iclr/ieee_access/nature_mi/essay/report/policy_brief/whitepaper, maps to `output.paper_format`), **research approach** (computational vs. observational, maps to `engine.no_simulation`), clarify mode, reviewer panel, knowledge layer. Format and approach slots match the clarify agent's `paper_venue` + `simulatability` slots so users can lock both upfront without a clarify call. | ✅ |
| Folder summarizer — `python launch.py --summarize <folder>` and `@fi /summarize`; auto-detects content kind (literature / code / study / execution / mixed); always ingests input + summary into Axon | ✅ |
| No-admin LaTeX install — `python launch.py --install-tectonic` drops a self-bootstrapping LaTeX binary into `tools/` for corporate environments where MiKTeX install is blocked | ✅ |
| Resumable quests — `python launch.py --resume <quest_id>` / `@fi /resume` re-enter the LangGraph from the last checkpointed node when a prior run died mid-pipeline | ✅ |
| Paper PDF via pandoc + LaTeX — real venue-flavored templates for all five scientific formats (`generic`, `neurips`, `iclr`, `ieee_access` two-column, `nature_mi`). The PDF preprocessor lifts the first `# H1` into a YAML `title:` field so pandoc populates `\title{}` (no longer landing the paper title as a numbered section heading), lifts `## Abstract` into a YAML `abstract:` field rendered inside `\begin{abstract}…\end{abstract}` (no longer numbered "1 Abstract"), enables `lists_without_preceding_blankline` so bullet/numbered lists immediately after a paragraph still render as lists, shifts heading levels by –1 so the visible sections start at "1 Introduction" not "0.1 Introduction", and dedupes `"Title. Title."` reference-line patterns the writer inherits from prior-work excerpts. PDF metadata (`pdftitle`, `pdfauthor`) gets populated via `\hypersetup` so reference managers pick up the title. | ✅ |
| Slides (when `output.kinds` includes `slides`) — `slides.md` always, plus `slides.html`/`slides.pdf` (when `marp` CLI is on PATH) and `slides.pptx` (when pandoc is on PATH); poster via `beamerposter`; speech script via single LLM call | ✅ |
| SQLite-checkpointed state for resumability (`<quest_root>/.fi/state.sqlite`) | ✅ |
| `--fleet` runner with bounded concurrency, ref-counted proxies, `--memory-cap-mb`, optional `viztracer --profile` | ✅ |
| Folder summarizer — `@fi /summarize <folder>` / `python launch.py --summarize` walks mixed content (papers, code, study notes, logs), classifies each file, calls LLM once, writes `outputs/<summary_id>/summary.md`; full input set ingested into Axon. Caps prompt size by file count + total content chars so a 31K-file folder doesn't blow the token budget. | ✅ |
| Weekly PM digest — `@fi /digest [N days]` / `python launch.py --digest --days N` walks `outputs/` for quests touched in window, classifies each by LangGraph terminal-node state, computes a deterministic WeekDiff vs the most-recent prior digest (✅ promoted / 🆕 new / ⚠️ still-in-progress / 🛑 stalled / ❓ dropped), and asks the LLM to produce a markdown report under `outputs/_digests/<YYYY-Www>.md`. Ingests into Axon as `fi_digest` so future quests retrieve prior-week context. | ✅ |
| Portfolio synthesis — `@fi /portfolio` / `python launch.py --portfolio` walks every quest under `outputs/` (no time window), feeds an LLM the structured corpus + deterministic stats (provider breakdown, completion cadence), and produces `outputs/_portfolio/<YYYY-MM-DD>.md` with topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, and prioritized next-quest suggestions. Ingests as `fi_portfolio`. | ✅ |
| Adversarial critique — `@fi /critique <quest_id>` / `python launch.py --critique <id> --critique-provider <name>` reads a finished quest's paper.md + experiment code + in-quest review, then asks the LLM (ideally a different provider) to surface what the in-quest reviewer missed. Writes `outputs/<quest_id>/critique.md` with Verdict / Methodology / Statistics / Reproducibility / Alternative explanations / Recommended follow-ups. Ingests as `fi_critique`. | ✅ |
| Pre-quest proposal — `@fi /proposal <topic>` / `python launch.py --proposal "<topic>"` asks the LLM for a 1-page planning doc (TL;DR, background, hypothesis, plan, success criteria, risks, scope limits, recommended next step) BEFORE committing compute. Writes `outputs/_drafts/<id>-proposal.md` + companion `outputs/_drafts/<id>.yaml` ready for `--config`. The companion YAML auto-pins the proposal MD into `knowledge.local_papers` so the quest that consumes it sees the planning doc verbatim at retrieval time (not just via probabilistic Axon hits). Ingests as `fi_proposal`. | ✅ |
| Pre-staged analyze — `@fi /analyze <data-path> <topic>` / `python launch.py --analyze <path> --analyze-topic "<topic>"`. The inverse of `/proposal`: the user already has a dataset and wants FI to write a paper analyzing it. Files under `<data-path>` are copied into the new quest's `data/` directory (recursive, symlinks skipped, common noise filtered). Engine runs in no-simulation mode and routes `auto_collect_data` (passthrough) → `wait_for_data` (passthrough since data is already there) → `data_load → analyze → cross_check → write → review`. Cost: ~6 premium requests, no `ideate`/`literature`/`design`/`implement`/`execute`. | ✅ |
| Non-scientific paper formats — `output.paper_format` accepts `essay` / `report` / `policy_brief` / `whitepaper` alongside the five scientific venues. Each non-scientific format ships a full LaTeX template (no stubs) and the `write` node auto-swaps the default IMRAD voice for the format's natural persona (essayist / consulting analyst / policy analyst / industry analyst) via `Engine._resolve_write_persona`. The `paper_venue` clarify slot widens to cover both buckets and defaults to `essay` for non-simulatable topics. | ✅ |
| Cite-by-content references — the prior-work block fed to the writer carries the full citation handle (author(s), year, title, venue, DOI / arXiv id / URL) drawn from the retrieval layer's metadata, so the `## References` section in the produced paper renders proper citations (`Smith & Lee (2021). Title. Journal. DOI: 10.x/y.`) instead of bare titles. The writer prompt explicitly forbids fabricating fields the prior-work block didn't supply and forbids fenced code blocks in the body — reproducibility points at the bundled `experiment.py` instead. | ✅ |
| Unified interactive interview across every frontend — `python launch.py --new` (CLI), `@fi /new` (VSCode chat), and `GET /interview` (`--serve` web UI) all drive the SAME Python interview defined in `core/interview.py` (with `core/interview_schema.json` as the JSON snapshot consumed by TS + HTMX). CLI and the web UI ask 14 questions; VSCode asks 12 (skips the provider + model picker because the extension pins `provider.name = "vscode_extension"` and captures `provider.model` automatically from `vscode.lm.selectChatModels()`). The shared 12-question core covers topic / title / output bundle / paper format / research approach / **study depth** / **comparative baseline** (topic-tuned LLM-suggested default via `agents/clarify_preflight.md`) / **success metric** (topic-tuned) / **budget** (topic-tuned) / clarify mode / reviewer panel / Axon. The two CLI/web-only extras are **provider** + **model** (curated per-provider list with an "Other (type your own)" escape hatch). Answers land in `engine.clarify_overrides` so the engine's clarify node short-circuits in auto mode (no wasted LLM call). | ✅ |
| Mid-quest interview re-entry — `python launch.py --update <quest_id>` (CLI), `@fi /update <id>` (VSCode, opens an integrated terminal that runs the CLI command), `POST /api/interview/update/{quest_id}` (web UI) re-open the interview pre-filled with the quest's current YAML, limited to research-shaping fields (topic / title / provider / model / no_simulation stay locked). A stage-invalidation matrix in `core/interview_update.py:STAGE_INVALIDATION` decides which LangGraph nodes need to re-run; affected `QuestState` keys are soft-cleared from the latest checkpoint, the YAML is rewritten (with a `config.yaml.before-update` backup), then the quest resumes via the existing `--resume` path. Flipping `no_simulation` mid-quest is hard-refused; deleting `<quest_root>/.fi/state.sqlite` is documented as the workaround for a guaranteed full re-run. | ✅ |
| Web UI activation — `python launch.py --serve` is now a proper quest-management surface, not a passive dashboard. The new visual design uses Inter sans-serif, a single deep-blue accent, and JetBrains Mono for code, with a `prefers-color-scheme: dark` media query for dark-mode users. Dashboard `/` lists every quest and has a prominent `+ New Quest` CTA. `/interview` runs the same 14-question schema as the CLI; a "Launch the quest immediately after submit" checkbox (default ON) triggers `POST /api/interview/submit?launch=true`, which spawns `python launch.py --config <yaml>` as a subprocess via `web/quest_launcher.py:QuestLauncher` (capped at `--max-concurrent`, with `FI_PRESEED_QUEST_ID` passed through so the redirect URL is stable). After submit the user lands on `/quest/<quest_id>` — a dedicated detail page that auto-refreshes every 3 s, shows the current LangGraph node, log tail, figures grid, paper preview, and a cancel button (sends SIGTERM / CTRL_BREAK_EVENT through the subprocess pool). Binding the server to a non-loopback address logs a loud WARNING because the launch endpoint is privileged. | ✅ |
| Webview "everything" pass — the dashboard now exposes every CLI / VSCode feature. Top-nav **Tools** dropdown leads to dedicated pages for `/tools/{proposal,critique,digest,portfolio,summarize,analyze,fleet,ingest}` (each form-driven, schema-fed from `web/tools_routes.py:TOOL_SPECS`; file-taking tools accept either multipart upload OR server-side path). `/quest/<id>` detail page now carries: SSE-streamed log tail (instead of 3 s polling), a file browser (read-only tree of `paper/figures/code/data` with per-file download links), a quest **zip download**, **resume button** for paused/crashed quests, **clarify-resume panel** for quests paused at the clarify node, **free-text tags** stored in `<quest_root>/.fi/labels.json`, **cost & runtime chart** (inline SVG, reads per-call rows from `<quest_root>/.fi/cost.jsonl` written by the engine's provider instrumentation), **paper iterations browser** for diffing across review iterations, and an **edit + re-execute experiment.py** flow gated behind `FI_WEB_ALLOW_EXEC_EDIT=1` env var. **Trash bin** at `/trash` — DELETE renames to `outputs/_trash/<id>-<ts>`; restore/purge UI included. **Compare** at `/compare?a=&b=` shows two quests side-by-side. **Settings** at `/settings` lists provider auth availability + offers a 1-click tectonic install button. **About** at `/about` is the platform-intro page. Dashboard ships **search / filter / sort** + a manual light/dark toggle (overrides `prefers-color-scheme`). Quest detail fires a **browser notification** when the quest transitions to a terminal state. Self-hosted minimal Markdown renderer (`md_lite.js`, ~150 LOC) — paper preview renders headings / lists / tables / code / blockquotes / images inline; toggle to view the raw `.md`. | ✅ |
| Engine cost instrumentation — `core/provider.py` exposes `MODEL_PRICING` (per-1k-token rates for the model variants FI supports) and `estimate_cost_usd(model, prompt_tokens, completion_tokens)`; `LLMClient` captures the response's `usage` block (when the upstream returns one) on `last_usage`; `Engine._chat` / `_chat_messages` append one JSON line per call to `<quest_root>/.fi/cost.jsonl` with `{ts, node, model, usage, cost_usd}`. CLI / vscode_bridge transports + older Ollama versions don't return structured usage — those calls fall back to a char-based estimator (~4 chars/token) and the resulting row carries `usage.estimated: true` so the chart can render it differently from server-reported usage. At quest finalization, the engine writes `<quest_root>/.fi/cost.summary.json` with totals + per-node + per-model breakdowns so the cost endpoint doesn't re-walk the raw log on every render; ensemble breadcrumb rows are skipped during aggregation so per-call rows aren't double-counted. The web UI's cost endpoint reads + returns these records for `/quest/<id>`. | ✅ |
| No-simulation / data-required decision — three-tier (logged at INFO as `[clarify] simulatability resolved: ... source=<yaml \| clarify_simulatability \| clarify_empirical_legacy \| default>`): (1) `engine.no_simulation: true` in YAML wins; (2) the clarify agent's `simulatability` slot (`default: yes \| no \| uncertain` + `reason`) — `no` triggers; (3) legacy fallback on `empirical_vs_theoretical: empirical`. | ✅ |
| Auto-collect before pause — when no-simulation is decided, `auto_collect_data` queries Axon with `topic + hypothesis` and writes the top `engine.auto_collect_top_k` (default 5) hits to `outputs/<id>/data/auto_collected/<rank>_<slug>.md` with YAML provenance front matter. When `knowledge.enabled: false` or Axon returns nothing, the node logs INFO and falls through. | ✅ |
| Dataset adapters — when `engine.dataset_adapters` lists registered names (`worldbank`, `wikipedia`), each runs an external lookup and writes evidence into `auto_collected/<adapter>/<rank>_<slug>.md`. WorldBank produces Markdown tables with country/indicator/year columns; Wikipedia writes per-article summaries with canonical URL + YAML front matter. If every adapter returns nothing, `wait_for_data` falls through to the manual pause (rc=0 with `data/README.md`). User drops more files (or the auto-collected files are enough) and re-runs `fi --resume <id>`. The `data_load` node walks the dir (auto + manual together), synthesizes a `result_json`, and the quest continues through `analyze → cross_check → write → review`. | ✅ |

## Provider matrix

| Provider | Transport | Auth | ToS standing |
|---|---|---|---|
| `openai` | HTTP direct | `OPENAI_API_KEY` env | ✅ Sanctioned |
| `codex` | HTTP direct | `OPENAI_API_KEY` env | ✅ Sanctioned |
| `gemini` | HTTP direct | `GEMINI_API_KEY` env | ✅ Sanctioned |
| `ollama`, `vllm` | HTTP direct (local) | none | ✅ Self-hosted |
| `claude_cli` | CLI exec | `claude login` (Claude Pro/Max OAuth) | ✅ Sanctioned |
| `codex_cli` | CLI exec | `codex login` (ChatGPT Plus/Pro OAuth) | ✅ Sanctioned |
| `copilot_cli` | CLI exec | `gh auth login` (Copilot subscription) | ⚠️ Agentic — replies conversationally to FI's prompts; use `vscode_extension` for Copilot instead. |
| `gemini_cli` | CLI exec | `gemini` OAuth / Google AI Studio key | ✅ Sanctioned |
| **`vscode_extension`** | **VSCode bridge** | **VSCode Copilot Chat sign-in** | **✅ Sanctioned via `vscode.lm`** |
| `claude_code` | HTTP via proxy | `claude login` + spawned wrapper | ⚠️ Third-party wrapper |
| `github_copilot_cli` | HTTP via proxy | `gh auth login` + spawned `copilot-api` | ⚠️ Against ToS spirit (use `copilot_cli` instead) |
| `github_copilot_vscode` | HTTP via proxy | VSCode Copilot extension + spawned `copilot-api` | ⚠️ Against ToS spirit (use `vscode_extension` instead) |

The two `github_copilot_*` providers AND `copilot_cli` emit a one-time
warning at engine init:

- `github_copilot_*` — risk warning about the third-party proxy and
  GitHub's abuse-detection systems.
- `copilot_cli` — broken-as-chat-backend warning (the CLI is agentic
  and replies conversationally instead of producing structured node
  output). Recommended replacement for Copilot users is
  `vscode_extension`.

Set `FI_SUPPRESS_PROXY_WARN=1` to silence either warning (use at your
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

## Configuration — every YAML field

```yaml
topic: |
  <required, free text — the research topic>

title: optional-short-slug

provider:
  name: vscode_extension          # see provider matrix above
  model: gpt-5                    # global default
  base_url: ...                   # only for HTTP-direct overrides
  api_key_env: ...                # env var name; defaults to provider's standard
  extra: {}                       # bag for transport-specific fields
  node_models:                       # Per-node override — flat dict[str, str]
    clarify:                  gpt-4o-mini
    ideate:                   claude-3-5-sonnet
    cross_check:              gpt-4o-mini
    write:                    claude-3-5-sonnet
    review_panel:             gpt-5            # default for any persona
    review_panel.statistician: claude-opus-4-7 # per-persona override (dotted key, flat value)
    review_moderator:         gpt-4o-mini

engine:
  framework: langgraph
  max_iterations: 2               # bounds the design-level revise/re-experiment loop
  review_loop: true
  clarify_mode: off               # off | auto | interactive
  ideate_reflect: true            # extra ideate self-critique pass
  exec_reflect_max_iterations: 3  # execute-repair loop bound
  cross_check_per_finding_k: 3    # per-finding literature hits
  enable_analyze_reroute: true    # analyze-driven re_experiment / broaden_lit
  review_panel: []                # personas: [methodologist, statistician, devil_advocate, reproducibility]
  no_simulation: false            # YAML hard-override: skip implement → execute and route through auto_collect_data → wait_for_data → data_load → analyze. See row "No-simulation / data-required decision" above.
  auto_collect_data: true         # Try Axon (and any registered dataset_adapters) BEFORE the user-data pause.
  auto_collect_top_k: 5           # Axon ``top_k`` used by auto_collect_data when knowledge.enabled.
  dataset_adapters: []            # Structured-data + web-fetch adapters. Available: "worldbank", "wikipedia".
  dataset_adapter_top_k: 3        # rows per adapter — each row is one external API hit.

execution:
  sandbox: venv                   # or docker
  timeout_s: 1800
  python_version: "3.11"
  docker_image: python:3.11-slim

knowledge:
  enabled: true
  axon_config:                    # inline AxonConfig — or pass a path to a YAML
    embedding: { provider: ollama, model: nomic-embed-text }
    llm:       { provider: ollama, model: qwen2.5-coder:32b }
  top_k: 8                          # Axon RAG retrieval cap (precision).
  external_top_k: 20                # External (arXiv / OpenAlex / Crossref / S2) cap (breadth) when Axon misses.
  write_back_quests: true
  write_back_only_on_accept: true

  external_fallback: [openalex, arxiv, crossref]
  source_routing: auto            # or manual
  seed_source_catalog: true

  local_papers:
    - ~/papers/grenville_2015_inpria_mor.pdf
    - ~/papers/hinsberg_meyers_2017_imaging.md

  try_fetch_full_text: false
  full_text_fetch_timeout_s: 15.0
  full_text_fetch_total_s: 90.0
  full_text_max_kb: 64

output:
  kinds: [paper_md, paper_pdf, slides, poster, speech]
  paper_format: generic           # scientific: generic|neurips|iclr|ieee_access|nature_mi; non-scientific prose: essay|report|policy_brief|whitepaper
  output_dir: ./outputs
  require_pdf: false              # Strict mode: when ``paper_pdf`` in kinds AND pandoc/LaTeX missing, abort pre-flight (saves ~15 min of LLM cost). Default keeps the graceful skip + ``paper_pdf_skipped.md`` diagnostic.

# Reserved free-text steering slot — declared in ``core/config.py``
# but NOT YET wired into any prompt template or ``Engine._chat`` path
# as of today. Parses and round-trips through the schema; setting it
# has no behavioural effect until a future PR threads it into the
# system prompts.
extra_directives: ""
```

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
| `scripts/probe_copilot_vscode.py` | Live-verify `github_copilot_vscode` (use only with the proxy provider; prefer `vscode_extension`). |

## Tests

The test suite is pytest-based, runs on Windows-native Python 3.11+,
and uses fake LLMs everywhere (`monkeypatch.setattr` on `LLMClient.chat`)
so there is zero real API spend in CI. A few tests skip gracefully
when their external tool isn't on PATH (Docker daemon for
`test_docker_executor.py`; Marp CLI for the slides render gate; pdflatex
for the paper-PDF gate). Suite organization:

- `test_config.py` — YAML schema, tilde expansion, validators.
- `test_execution.py` / `test_docker_executor.py` — venv + Docker executors.
- `test_engine_smoke.py` — full DAG end-to-end with fake LLM (the regression detector).
- `test_engine_helpers.py` — direct unit tests for `_parse_json_lenient`, `_extract_result_json`, `_strip_outer_fence`, `_slugify`, `_new_quest_id`, `_parse_implement_response`, `_format_lit`, plus the review/analyze/write prompt-shape pins.
- `test_engine_resume.py` — `--resume` checkpoint reuse, `Engine(resume_quest_id=...)`, copilot_cli agentic warning.
- `test_clarify.py`, `test_ideate_reflect.py`, `test_execute_reflect.py`, `test_cross_check.py`, `test_review_panel.py` — clarify, ideate-reflection, execute-repair, cross-paper-check, and reviewer-panel feature tests.
- `test_per_node_model_routing.py` — per-node model selection.
- `test_vscode_bridge.py` / `test_vscode_extension_typescript.py` — VSCode bridge protocol + provider integration (mock VSCode), plus TS compile + .vsix package gates.
- `test_knowledge.py` — source adapters, dedup, source-router LLM, local-paper load + pin, full-text fetch, structured-ingest helpers, external-ref spines.
- `test_knowledge_writeback.py` — accept-gated cross-quest memory bundle.
- `test_web_server.py` / `test_web_e2e.py` — status GUI.
- `test_self_correction_e2e.py` — end-to-end self-correction proofs.
- `test_launch.py` — `parse_args`, `_run_generators`, `_await_under_cap`, fleet counters, `--resume` traversal validation, `source_yaml_path` quest-dir copy.
- `test_paper_gen.py` / `test_slides_speech.py` / `test_poster.py` — generators.
- `test_provider.py` / `test_provider_cli.py` — HTTP-direct, CLI-exec, and proxy transports.
- `test_fleet.py` — two engines concurrent under one process.
- `test_platform.py` — `detect_system()`.

## Architecture

See [`architecture.md`](architecture.md) for the layered diagram and
contracts (`Config`, `QuestState`, `QuestArtifacts`, `Executor`,
generator protocol). See [`plan.md`](plan.md) for the phased history.
