# Frontier Insight — Test Results

> Latest results live at the top; Phase-0 (DS-wrapper era) results are
> retained below for historical reference.

## CLI-exec providers — live probe (claude_cli, codex_cli, copilot_cli)

**3 / 3 providers respond** against their real local CLIs on Windows
(`scripts/probe_cli_providers.py`, one-shot `{"a": 1}` JSON prompt):

| Provider | Binary | Wall-time | Result |
|---|---|---|---|
| `claude_cli` | `claude` 2.1.138 (Claude Code) | 10.8 s | `{"a": 1}` |
| `codex_cli` | `codex-cli` 0.128.0 | 9.6 s | `{"a":1}` |
| `copilot_cli` | GitHub Copilot CLI 1.0.40 | 12.6 s | `{"a": 1}` |

The probe required one fix to land first: on Windows
`asyncio.create_subprocess_exec("codex", ...)` raises FileNotFoundError
even though `codex.CMD` is on PATH, because the subprocess family does
not honor `PATHEXT`. `_run_cli` now resolves the binary name via
`shutil.which()` before spawning, so the qualified path (e.g.,
`...\codex.CMD`) is passed in instead.

## Post-DS redesign — Phases A through H landed

**Status:** structural validation complete on Windows-native (no WSL2),
no real LLM API spend in the test suite.

- **11/11 pytest tests pass** (`python -m pytest -v`, ~3 min total):
  - `tests/test_config.py` (5/5) — schema, tilde expansion, inline AxonConfig dict, rejected providers/sandboxes.
  - `tests/test_execution.py` (3/3) — VenvExecutor creates a venv, runs a script, honors a timeout.
  - `tests/test_engine_smoke.py` (1/1) — full 8-node LangGraph DAG runs end-to-end with a fake LLM. Real per-quest venv, real `pip install matplotlib`, real subprocess execution of agent-generated code, real `figures/result.png` on disk, real `paper.md` written, `RESULT_JSON: {"score": 0.987}` parsed back from stdout. Review verdict round-trips.
  - `tests/test_fleet.py` (1/1) — two Engines run concurrently via `asyncio.gather`; distinct quest_ids and quest_roots, both produce independent papers and figures, no port/file collisions.
  - `tests/test_knowledge_writeback.py` (1/1) — confirms `Knowledge.add_quest_artifacts(quest_id, paper_md_path, summary)` is invoked after a successful quest with the analysis summary + key findings.
- The engine compiles with `AsyncSqliteSaver` at `<quest_root>/.fi/state.sqlite`; resumability is wired via LangGraph's `thread_id` mechanism.
- Per-quest venv resolves `Scripts/python.exe` on Windows and `bin/python` on POSIX (smoke test exercised the Windows path).
- Knowledge layer (Axon) imports lazily; falls back to no-op + warning when not installed.
- Provider proxy spawn paths in `core/provider.py::ProxySupervisor._spawn` use the verbatim install/run commands per upstream READMEs and probe `GET /v1/models` for readiness rather than raw TCP. Live auth + bake-off runs against `claude_code` and `github_copilot_*` are user-validated when the prereqs are installed.
- Generators ship: `paper.py` (pandoc + LaTeX, `generic`/`neurips` templates), `slides.py` (Marp), `poster.py` (beamerposter, fixed 36×48), `speech.py` (one LLM call). Each skips cleanly when its external tool is missing.

### What this suite does NOT exercise

- A real LLM-driven quest end-to-end (needs API key + ~30 min wall-time + budget).
- Live proxy auth (`claude_code` requires `claude auth login` + a wrapper checkout; `github_copilot_*` requires `npx copilot-api@latest auth`).
- Docker sandbox at runtime (the code is exercised by import; no Docker daemon was available during the suite run).
- Paper PDF, slides, poster compilation (require pandoc/MiKTeX/Marp/pdflatex on PATH).
- Linux/macOS — the code is portable but only Windows was used for the validation run.

To reproduce structural validation on a fresh box:
```bash
pip install -r requirements.txt
pip install pytest pytest-asyncio    # dev extras
python -m pytest -v
```

To run a real quest (needs `OPENAI_API_KEY` or another provider's key):
```bash
python launch.py --config examples/integrator_bakeoff/config.yaml
```


---

## Live research-quest run (post-redesign) — EUV / MOR / shot-noise

**Topic.** Theoretical lower bound on Line-Edge Roughness (LER) imposed by
Poisson photon shot noise in metal-oxide EUV resists at production
doses (10–60 mJ/cm²). Narrowed to a theory-forward question with a
single executable Monte Carlo.

**Setup.**
- Provider: `ollama` against `qwen3-coder:480b-cloud` (engine nodes) +
  `gpt-oss:120b-cloud` (speech generator, after upstream 502s on the
  former for the longest prompt).
- Sandbox: `venv`. Knowledge layer disabled (no local Axon brain).
- Outputs requested: paper_md, paper_pdf, slides, poster, speech.

**Run wall-time.** ~5 minutes for the 8-node DAG; speech generator
re-invoked separately after a transient cloud 502.

**Artifacts produced** under `outputs/1778452404-euv-mor-...-e6bfe5/`:
| File | Size | Notes |
|---|---|---|
| `paper/paper.md` | 7.7 KB | IMRAD; Poisson absorption + threshold model; predicts $\sqrt{\alpha_{\text{CAR}}/\alpha_{\text{MOR}}} \approx 0.577$ for the shot-noise prefactor ratio; identifies ~1.2 nm of typical 3 nm MOR LER as shot-noise-attributable. |
| `figures/*.png` | 4×60 KB | 3-panel composite (edge profiles + LER-vs-dose log-log + shot-noise scaling exponent) + separate LER-vs-dose plot. |
| `code/experiment.py` | 4.9 KB | Numpy + matplotlib Monte Carlo; runs in <2s. Verified manually to emit `RESULT_JSON: {...}`. |
| `slides.md` | 3.2 KB | Marp markdown, 8-slide deck. |
| `poster.tex` | 7.0 KB | beamerposter 36×48-in template populated by the LLM. |
| `talk.md` | 11.7 KB | ~10-minute spoken script with `[slide: N]` cues. |

**`paper_pdf`, `slides.html/pdf`, `poster.pdf` not produced** — pandoc,
Marp CLI, and pdflatex are not installed on this validation machine.
The generators correctly skipped with warnings per the contract.

**Caveats surfaced by the live run** (worth addressing in a follow-up):
1. Filename drift between design-node figure plan and implement-node
   actual `savefig` paths — the paper referenced names the script didn't
   write. Worked around manually by copying `combined_plots.png` to the
   missing referenced names.
2. The first engine attempt got rc=2 with 0 s duration on the freshly-
   created venv's first script invocation. Manual re-run of the same
   script succeeds — likely a Windows venv-startup race. Not yet
   reproducible deterministically.
3. Ollama cloud-routed models (`qwen3-coder:480b-cloud`,
   `gpt-oss:120b-cloud`) returned occasional 502 Bad Gateway on the
   longest prompts. Mitigated by adding tenacity retry to `LLMClient.chat`
   (exponential backoff, 4 attempts).

---

## Phase 0 (DS-wrapper era) — historical

The pre-redesign prototype wrapped DeepScientist over REST and ran two
end-to-end integrator-bake-off quests on WSL2 (Ubuntu 24.04.3, Python
3.14.4, DS 1.5.17, Codex CLI 0.130.0). Both reached closure with
finalized PDFs (~360–450 KB), 3 figures each, and correct numerical
conclusions (RK4 RMS-x error 1.87 × 10⁻⁶ at h=0.1; recovered
log-log convergence slopes 1.65 / 4.00 / 2.00 for Euler / RK4 / V-Verlet).
The `test_runs/quest-001` and `test_runs/quest-002` directories that
held those artifacts were removed during the redesign cleanup; the
research itself is no longer reproducible without restoring DS. Three
Phase-0 fixes that informed the new design are retained in code: manual
tilde expansion in `Config` path validators, glob-with-priority artifact
collection, and a clear-error pattern for unsupported providers.
