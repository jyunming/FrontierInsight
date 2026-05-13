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
| Per-node model routing (Phase O) — different model per node via `provider.node_models` | ✅ |
| Pre-flight `clarify` node (Phase I) — 5-slot survey before `ideate`, off / auto / interactive modes | ✅ |
| Self-correction — execute-repair loop (Phase K) — agent reads traceback, patches code, retries | ✅ |
| Self-correction — analyze-driven re-route (Phase L) — `re_experiment` / `broaden_lit` routes back to design | ✅ |
| Cross-paper check (Phase L) — per-finding literature search + supporting/conflicting/neutral classification | ✅ |
| Ideate self-reflection (Phase M) — extra LLM call may swap chosen idea | ✅ |
| Reviewer panel (Phase N) — N personas in parallel + moderator synthesis | ✅ |
| Axon-backed knowledge layer: literature retrieval + cross-quest memory write-back | ✅ |
| Multi-source literature router when Axon is empty: arXiv / OpenAlex / Crossref / Semantic Scholar / PubMed / CORE / Google Scholar in parallel, DOI-dedup | ✅ |
| Topic-aware LLM source routing — agent picks which venues to query from a 12-entry catalog | ✅ |
| Local paper feed — drop paywalled PDFs / MD into `knowledge.local_papers`, pinned to retrieval head | ✅ |
| Opportunistic full-text fetch — host-network publisher PDFs with login-wall rejection | ✅ |
| Structured ingest — title-searchable spine docs + citation-header'd body + topic rollups | ✅ |
| Status GUI (Phase J) — FastAPI + HTMX server: fleet list, SSE log stream, clarify panel, paper preview, panel-review cards | ✅ |
| VSCode extension (Phase P) — sanctioned `vscode.lm.*` Copilot integration, with `@fi /new`, `@fi /start`, `@fi /fleet`, `@fi /resume` | ✅ |
| Resumable quests — `python launch.py --resume <quest_id>` / `@fi /resume` re-enter the LangGraph from the last checkpointed node when a prior run died mid-pipeline | ✅ |
| Paper PDF via pandoc + LaTeX (`generic` and `neurips` templates ship; others stub) | ✅ |
| Slides (when `output.kinds` includes `slides`) — `slides.md` always, plus `slides.html`/`slides.pdf` (when `marp` CLI is on PATH) and `slides.pptx` (when pandoc is on PATH); poster via `beamerposter`; speech script via single LLM call | ✅ |
| SQLite-checkpointed state for resumability (`<quest_root>/.fi/state.sqlite`) | ✅ |
| `--fleet` runner with bounded concurrency, ref-counted proxies, `--memory-cap-mb`, optional `viztracer --profile` | ✅ |
| Folder summarizer — `@fi /summarize <folder>` / `python launch.py --summarize` walks mixed content (papers, code, study notes, logs), classifies each file, calls LLM once, writes `outputs/<summary_id>/summary.md`; full input set ingested into Axon. Caps prompt size by file count + total content chars so a 31K-file folder doesn't blow the token budget. | ✅ |
| Weekly PM digest — `@fi /digest [N days]` / `python launch.py --digest --days N` walks `outputs/` for quests touched in window, classifies each by LangGraph terminal-node state, computes a deterministic WeekDiff vs the most-recent prior digest (✅ promoted / 🆕 new / ⚠️ still-in-progress / 🛑 stalled / ❓ dropped), and asks the LLM to produce a markdown report under `outputs/_digests/<YYYY-Www>.md`. Ingests into Axon as `fi_digest` so future quests retrieve prior-week context. | ✅ |

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
  node_models:                    # Phase O — per-node override
    clarify:       gpt-4o-mini
    ideate:        claude-3-5-sonnet
    cross_check:   gpt-4o-mini
    write:         claude-3-5-sonnet
    review_panel:                 # default for any persona
      gpt-5
    review_panel.statistician:    # per-persona override
      claude-opus-4-7
    review_moderator: gpt-4o-mini

engine:
  framework: langgraph
  max_iterations: 2               # bounds the design-level revise/re-experiment loop
  review_loop: true
  clarify_mode: off               # off | auto | interactive
  ideate_reflect: true            # Phase M
  exec_reflect_max_iterations: 3  # Phase K bound
  cross_check_per_finding_k: 3    # Phase L per-finding hits
  enable_analyze_reroute: true    # Phase L
  review_panel: []                # Phase N personas: [methodologist, statistician, devil_advocate, reproducibility]

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
  paper_format: generic           # or neurips, iclr, ieee_access, nature_mi
  output_dir: ./outputs
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

**379 tests** collected on Windows-native Python 3.11.9 (`pytest --collect-only`); the full suite runs in ~9 minutes. A few tests skip
gracefully when their external tool isn't on PATH (Docker daemon for
`test_docker_executor.py`; Marp CLI for the slides render gate; etc.).
Suite organization:

- `test_config.py` — YAML schema, tilde expansion, validators.
- `test_execution.py` / `test_docker_executor.py` — venv + Docker executors.
- `test_engine_smoke.py` — full DAG end-to-end with fake LLM (the regression detector).
- `test_engine_helpers.py` — direct unit tests for `_parse_json_lenient`, `_extract_result_json`, `_strip_outer_fence`, `_slugify`, `_new_quest_id`, `_parse_implement_response`, `_format_lit`, plus the review/analyze/write prompt-shape pins.
- `test_engine_resume.py` — `--resume` checkpoint reuse, `Engine(resume_quest_id=...)`, copilot_cli agentic warning.
- `test_clarify.py`, `test_ideate_reflect.py`, `test_execute_reflect.py`, `test_cross_check.py`, `test_review_panel.py` — Phases I, M, K, L, N respectively.
- `test_per_node_model_routing.py` — Phase O.
- `test_vscode_bridge.py` / `test_vscode_extension_typescript.py` — Phase P (bridge protocol + provider integration; mock VSCode; plus TS compile + .vsix package gates).
- `test_knowledge.py` (50 tests) — source adapters, dedup, source-router LLM, local-paper load + pin, full-text fetch, structured-ingest helpers, external-ref spines.
- `test_knowledge_writeback.py` — accept-gated cross-quest memory bundle.
- `test_web_server.py` / `test_web_e2e.py` — Phase J GUI.
- `test_self_correction_e2e.py` — end-to-end Phase K + L proofs.
- `test_launch.py` — `parse_args`, `_run_generators`, `_await_under_cap`, fleet counters, `--resume` traversal validation, `source_yaml_path` quest-dir copy.
- `test_paper_gen.py` / `test_slides_speech.py` / `test_poster.py` — generators.
- `test_provider.py` / `test_provider_cli.py` — HTTP-direct, CLI-exec, and proxy transports.
- `test_fleet.py` — two engines concurrent under one process.
- `test_platform.py` — `detect_system()`.

## Architecture

See [`architecture.md`](architecture.md) for the layered diagram and
contracts (`Config`, `QuestState`, `QuestArtifacts`, `Executor`,
generator protocol). See [`plan.md`](plan.md) for the phased history.
