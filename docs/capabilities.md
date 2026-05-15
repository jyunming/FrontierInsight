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
| `cross_check → design` | `analyze.next_step ∈ {re_experiment, broaden_lit}` | `engine.max_iterations` (default 2) |
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
| Self-correction — analyze-driven re-route — `re_experiment` / `broaden_lit` routes back to design | ✅ |
| Cross-paper check — per-finding literature search + supporting/conflicting/neutral classification | ✅ |
| Ideate self-reflection — extra LLM call may swap chosen idea | ✅ |
| Reviewer panel — N personas in parallel + moderator synthesis (`rigor_score` + `depth_score` axes) | ✅ |
| Axon-backed knowledge layer: literature retrieval + cross-quest memory write-back | ✅ |
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
| Paper PDF via pandoc + LaTeX (`generic` and `neurips` templates ship; others stub) | ✅ |
| Slides (when `output.kinds` includes `slides`) — `slides.md` always, plus `slides.html`/`slides.pdf` (when `marp` CLI is on PATH) and `slides.pptx` (when pandoc is on PATH); poster via `beamerposter`; speech script via single LLM call | ✅ |
| SQLite-checkpointed state for resumability (`<quest_root>/.fi/state.sqlite`) | ✅ |
| `--fleet` runner with bounded concurrency, ref-counted proxies, `--memory-cap-mb`, optional `viztracer --profile` | ✅ |
| Folder summarizer — `@fi /summarize <folder>` / `python launch.py --summarize` walks mixed content (papers, code, study notes, logs), classifies each file, calls LLM once, writes `outputs/<summary_id>/summary.md`; full input set ingested into Axon. Caps prompt size by file count + total content chars so a 31K-file folder doesn't blow the token budget. | ✅ |
| Weekly PM digest — `@fi /digest [N days]` / `python launch.py --digest --days N` walks `outputs/` for quests touched in window, classifies each by LangGraph terminal-node state, computes a deterministic WeekDiff vs the most-recent prior digest (✅ promoted / 🆕 new / ⚠️ still-in-progress / 🛑 stalled / ❓ dropped), and asks the LLM to produce a markdown report under `outputs/_digests/<YYYY-Www>.md`. Ingests into Axon as `fi_digest` so future quests retrieve prior-week context. | ✅ |
| Portfolio synthesis — `@fi /portfolio` / `python launch.py --portfolio` walks every quest under `outputs/` (no time window), feeds an LLM the structured corpus + deterministic stats (provider breakdown, completion cadence), and produces `outputs/_portfolio/<YYYY-MM-DD>.md` with topic clusters, near-duplicate detection, meta-paper candidates, coverage gaps, and prioritized next-quest suggestions. Ingests as `fi_portfolio`. | ✅ |
| Adversarial critique — `@fi /critique <quest_id>` / `python launch.py --critique <id> --critique-provider <name>` reads a finished quest's paper.md + experiment code + in-quest review, then asks the LLM (ideally a different provider) to surface what the in-quest reviewer missed. Writes `outputs/<quest_id>/critique.md` with Verdict / Methodology / Statistics / Reproducibility / Alternative explanations / Recommended follow-ups. Ingests as `fi_critique`. | ✅ |
| Pre-quest proposal — `@fi /proposal <topic>` / `python launch.py --proposal "<topic>"` asks the LLM for a 1-page planning doc (TL;DR, background, hypothesis, plan, success criteria, risks, scope limits, recommended next step) BEFORE committing compute. Writes `outputs/_drafts/<id>-proposal.md` + companion `outputs/_drafts/<id>.yaml` ready for `--config`. Ingests as `fi_proposal`. | ✅ |
| Non-scientific paper formats — `output.paper_format` accepts `essay` / `report` / `policy_brief` / `whitepaper` alongside the five scientific venues. Each non-scientific format ships a full LaTeX template (no stubs) and the `write` node auto-swaps the default IMRAD voice for the format's natural persona (essayist / consulting analyst / policy analyst / industry analyst) via `Engine._resolve_write_persona`. The `paper_venue` clarify slot widens to cover both buckets and defaults to `essay` for non-simulatable topics. | ✅ |
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
  top_k: 5
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
