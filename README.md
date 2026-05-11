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
| Paper PDF via pandoc + LaTeX (`generic` and `neurips` templates ship; others stub) | ✅ |
| Slides via Marp; poster via `beamerposter`; speech script via single LLM call | ✅ |
| SQLite-checkpointed state for resumability (`<quest_root>/.fi/state.sqlite`) | ✅ |
| `--fleet` runner with bounded concurrency, ref-counted proxies, `--memory-cap-mb`, optional `viztracer --profile` | ✅ |

11/11 pytest tests pass on Windows-native Python 3.11.9 (no WSL2). See [`TEST_RESULTS.md`](TEST_RESULTS.md).

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

output:
  kinds: [paper_md, paper_pdf, slides, poster, speech]
  paper_format: generic    # or neurips, iclr, ieee_access, nature_mi
  output_dir: ./outputs
```

See [`examples/integrator_bakeoff/config.yaml`](examples/integrator_bakeoff/config.yaml).

---

## Honest about scope

**What's intentionally not in scope.** A web UI / Docker-deploy / one-click installer (Phase ≥7 if ever). A custom embedding service or vector store — Axon owns that. A Rust rewrite — orchestration is LLM-network-bound, so Rust gives ~5% wall-time win at best (within noise); the hybrid Python+Rust route via `maturin` is reserved for the day Phase H profiling surfaces a real CPU hot spot. See `docs/plan.md`.

**What's structural, not yet user-validated.** Live runs against `claude_code` and `github_copilot_*` proxies require their respective auth/install prerequisites; the spawn paths are wired and use `GET /v1/models` for readiness. The Phase-1 paper templates ship `generic` + `neurips` fully; `iclr`, `ieee_access`, `nature_mi` are stubs that fall back to pandoc's default. Slides require the Marp CLI; poster requires pdflatex.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
