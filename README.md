# Frontier Insight

**End-to-end automated research pipeline. Windows + Linux native, async-first, knowledge-grounded.**

Frontier Insight (FI) takes a research topic and produces a finished IMRAD paper, slide deck, conference poster, and 10-minute talk script — running the experiment code itself in a per-quest sandbox along the way. The orchestrator is async LangGraph; the knowledge layer is [Axon](https://github.com/jyunming/Axon); experiment code runs in a per-quest venv (default) or a Docker container (opt-in).

---

## What works today

| Capability | Status |
|---|---|
| Async LangGraph engine: `ideate → literature → design → implement → execute → analyze → write → review` with a `revise` loop | ✅ |
| Per-quest venv; agent-generated Python is installed and run in isolation | ✅ |
| Docker sandbox via `execution.sandbox: docker` (network disabled, mounted at `/work`) | ✅ |
| Provider matrix: `codex` / `openai` / `gemini` / `ollama` / `vllm` direct; `claude_code` and `github_copilot_*` via local proxies; `claude_cli` / `codex_cli` via CLI exec (reuses CLI OAuth) | ✅ direct + proxy paths structural, `claude_cli` live-verified |
| Axon-backed knowledge layer: literature retrieval + cross-quest memory write-back | ✅ |
| **Multi-source literature router** when Axon is empty: arXiv / OpenAlex / Crossref / Semantic Scholar / PubMed / CORE / Google Scholar in parallel, DOI-dedup | ✅ |
| **Topic-aware LLM source routing** — agent picks which venues to query from a 12-entry catalog (extensible via Axon ingestion) | ✅ |
| **Local paper feed** — drop paywalled PDFs / MD into `knowledge.local_papers`, pinned to retrieval head + ingested permanently | ✅ |
| **Opportunistic full-text fetch** (`try_fetch_full_text`) — host-network publisher PDFs with login-wall rejection (Content-Type + `%PDF-` magic) | ✅ |
| **Structured ingest** — title-searchable spine docs + citation-header'd body + topic-event rollups + curated external-ref spines | ✅ |
| Paper PDF via pandoc + LaTeX (`generic` and `neurips` templates ship; others stub) | ✅ |
| Slides via Marp; poster via `beamerposter`; speech script via single LLM call | ✅ |
| SQLite-checkpointed state for resumability (`<quest_root>/.fi/state.sqlite`) | ✅ |
| `--fleet` runner with bounded concurrency, ref-counted proxies, `--memory-cap-mb`, optional `viztracer --profile` | ✅ |

**217 tests pass** on Windows-native Python 3.11.9, no WSL2, ~3 minutes (2 skipped — Docker daemon and Marp CLI not on PATH). See [`TEST_RESULTS.md`](TEST_RESULTS.md) for the live-research run logs.

---

## Quick start

```bash
# Python 3.10+
pip install -r requirements.txt

# (optional) Knowledge layer:
#   pip install -e <path-to-Axon-checkout>
# (optional) Paper PDF:  install pandoc + a TeX engine (MiKTeX on Windows, TinyTeX on Linux/macOS).
# (optional) Slides:     npm install -g @marp-team/marp-cli
# (optional) Docker sandbox: install Docker Desktop / dockerd.
# (optional) Provider proxies (only if you use them):
#   claude_code:           clone RichardAtCT/claude-code-openai-wrapper, `poetry install`,
#                          set FI_CLAUDE_CODE_WRAPPER_DIR, then `claude login`.
#   github_copilot_*:      `npx copilot-api@latest auth` (one-time).
# (optional) CLI providers (zero infra; reuses CLI OAuth):
#   claude_cli:            npm i -g @anthropic-ai/claude-code && claude login.
#   codex_cli:             npm i -g @openai/codex && codex login.

# Single quest
export OPENAI_API_KEY=sk-...
python launch.py --config examples/integrator_bakeoff/config.yaml

# Many quests in parallel
python launch.py --fleet quests/a.yaml quests/b.yaml quests/c.yaml \
                 --max-concurrent 4 --memory-cap-mb 4096
```

Artifacts land at `<output_dir>/<quest_id>/`: `paper.md`, `paper.pdf`, `figures/`, optional `slides.{md,html,pdf}` / `poster.{tex,pdf}` / `talk.md`, the run log at `.fi/run.log`, the LangGraph checkpoint at `.fi/state.sqlite`, and a `frontier_insight_summary.json` index.

---

## Architecture

```
launch.py → Config → Engine(LangGraph) → QuestArtifacts → generators
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
           provider  execution  knowledge
            (httpx)  (venv|docker) (Axon)
```

See [`docs/architecture.md`](docs/architecture.md) for the layered diagram, contracts (`Config`, `QuestState`, `QuestArtifacts`, `Executor`, generator protocol), and the concurrency model. See [`docs/plan.md`](docs/plan.md) for the phased history (A through H).

---

## Configuration

A minimal `config.yaml`:

```yaml
topic: |
  Compare three numerical integrators on a damped harmonic oscillator...

provider:
  name: codex          # or openai, gemini, ollama, vllm, claude_code,
                       # github_copilot_cli, claude_cli, codex_cli
  # model: gpt-5
  # base_url: ...      # override per provider
  # api_key_env: ...   # env var name for the key

engine:
  framework: langgraph
  max_iterations: 2
  review_loop: true

execution:
  sandbox: venv        # or docker
  timeout_s: 1800

knowledge:
  enabled: true
  axon_config:         # inline AxonConfig — or pass a path to a YAML
    embedding: { provider: ollama, model: nomic-embed-text }
    llm:       { provider: ollama, model: qwen2.5-coder:32b }
  top_k: 5
  write_back_quests: true
  write_back_only_on_accept: true   # only ingest quests with review verdict == "accept"

  # External literature router — fires when Axon returns empty.
  external_fallback: [openalex, arxiv, crossref]   # or "none" to disable
  source_routing: auto              # agent picks sources per-topic; "manual" = use list verbatim
  seed_source_catalog: true         # seed the 12-entry catalog into Axon as fi_source_catalog

  # Drop paywalled PDFs / notes here — pinned to retrieval head, ingested permanently.
  local_papers:
    - ~/papers/grenville_2015_inpria_mor.pdf
    - ~/papers/hinsberg_meyers_2017_imaging.md

  # Opportunistic full-text fetch via host network (VPN/Shibboleth/EZproxy).
  # Login walls are rejected by Content-Type + %PDF-magic two-factor check.
  try_fetch_full_text: false
  full_text_fetch_timeout_s: 15.0   # per-doc HTTP timeout
  full_text_fetch_total_s: 90.0     # total batch budget
  full_text_max_kb: 64              # cap extracted text per doc

output:
  kinds: [paper_md, paper_pdf, slides, poster, speech]
  paper_format: generic    # or neurips, iclr, ieee_access, nature_mi
  output_dir: ./outputs
```

### Examples

| Example | Provider used | Topic |
|---|---|---|
| [`integrator_bakeoff`](examples/integrator_bakeoff/config.yaml) | (any) | Three numerical integrators on a damped harmonic oscillator (RK4 / Velocity-Verlet / forward Euler). Original validation topic from the DS-wrapper era. |
| [`euv_mor_shot_noise`](examples/euv_mor_shot_noise/config.yaml) | `ollama` → cloud-routed reasoning models | Theoretical LER floor imposed by Poisson photon shot noise in metal-oxide EUV resists at production doses (10–60 mJ/cm²). |
| [`bernstein_vazirani_noise`](examples/bernstein_vazirani_noise/config.yaml) | `claude_cli` (reuses `claude login` OAuth) | Bernstein-Vazirani algorithm under per-gate depolarizing noise — pure-numpy state-vector simulator for n ∈ {2..10}, MC validated against closed-form fidelity. |

Each example is a single `config.yaml` and produces a `paper.md`, figures, `slides.md`, `poster.tex`, and `talk.md` under `outputs/<quest_id>/`. PDFs require pandoc / Marp CLI / pdflatex on PATH.

---

## Running the full flow

The DAG runs unchanged whether your Axon corpus is empty or huge — knowledge retrieval falls through three layers automatically:

```
asearch(query)
  ├─ 1. Pinned local_papers (always first)
  ├─ 2. Axon (your long-term store; falls through on empty)
  └─ 3. External router: agent picks 1–5 sources from the catalog
        ├─ openalex / arxiv / crossref / semantic_scholar / pubmed / core / google_scholar
        └─ (optional) full-text fetch via host network
```

After a quest is reviewed and **accepted**, the write-back lays down a structured bundle so Axon stays title-searchable and topic-linkable (not just chunk-soup):

| kind | Purpose |
|---|---|
| `fi_paper_spine` | One tight chunk per paper — title / authors / DOI / abstract / key claims. Title queries hit this directly. |
| `fi_quest_paper` | Full paper body with a 1-line `[Title · Year · Venue · DOI]` citation header on every chunk. |
| `fi_quest_summary` | Structured-findings JSON (hypothesis, key_findings, result_json, verdict, score, model). |
| `fi_topic_event` | One pointer per accepted quest keyed by topic slug — enables "what do we know about X" rollups. |
| `fi_external_ref_spine` × N | Curated card-catalog entries for cited papers. Only refs from *accepted* quests persist. |

### Empty corpus (cold start)

Just run a quest. The router auto-fires when Axon returns nothing:

```bash
python launch.py --config examples/integrator_bakeoff/config.yaml
```

Watch the log: `[literature] retrieved N docs` followed by `source-router picked [openalex, crossref] (rationale: …)` means the LLM routed off the topic.

### Paywalled venue (e.g. SPIE for EUV / lithography)

Two options, often used together:

**A. Manual drop.** Download the PDF on a network with access; list it under `local_papers`. It is pinned to the head of every retrieval AND ingested permanently:

```yaml
knowledge:
  local_papers:
    - ~/papers/grenville_2015_inpria.pdf
```

**B. Opportunistic fetch.** If the host is already authenticated to the publisher (institutional VPN, Shibboleth, EZproxy at the OS level), set:

```yaml
knowledge:
  try_fetch_full_text: true
```

FI will GET the publisher PDF; a login wall returns HTML which the `Content-Type` + `%PDF-` magic-bytes check rejects cleanly. The quest is never blocked.

### Ingesting papers outside a quest

```bash
python launch.py --ingest paper1.pdf paper2.md ~/notes/lit/*.txt
```

Permanent ingest into Axon as `kind=fi_local_paper`. Useful for bootstrapping the corpus before running quests.

### Inspecting what landed

After an accepted quest, the run log shows e.g. `[write-back] axon ingest=True (verdict=accept, score=4)`. Two demo scripts illustrate the doc layout without needing Axon installed:

```bash
python scripts/demo_structured_ingest.py    # shows the 4-doc + N-ref-spine bundle
python scripts/demo_paywall_fetch.py        # arXiv success vs SPIE login-wall rejection
python scripts/demo_local_papers.py         # pinned-to-head behavior
```

---

## Honest about scope

**What's intentionally not in scope.** A web UI / Docker-deploy / one-click installer (Phase ≥7 if ever). A custom embedding service or vector store — Axon owns that. A Rust rewrite — orchestration is LLM-network-bound, so Rust gives ~5% wall-time win at best (within noise); the hybrid Python+Rust route via `maturin` is reserved for the day Phase H profiling surfaces a real CPU hot spot. See `docs/plan.md`.

**What's structural, not yet user-validated.** Live runs against `claude_code` and `github_copilot_*` proxies require their respective auth/install prerequisites; the spawn paths are wired and use `GET /v1/models` for readiness. The Phase-1 paper templates ship `generic` + `neurips` fully; `iclr`, `ieee_access`, `nature_mi` are stubs that fall back to pandoc's default. Slides require the Marp CLI; poster requires pdflatex.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
