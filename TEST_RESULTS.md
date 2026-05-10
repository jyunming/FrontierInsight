# Frontier Insight — Test Results

> Latest results live at the top; Phase-0 (DS-wrapper era) results are
> retained below for historical reference.

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

# Phase 0 (DS-wrapper era) — historical

> Validation of the Phase-0 prototype on the bake-off topic
> *"three numerical integrators on a 1-D damped harmonic oscillator"*.
> Plan and architecture: [`docs/plan.md`](docs/plan.md) ·
> [`docs/architecture.md`](docs/architecture.md).
> Test artifacts: [`test_runs/quest-001/`](test_runs/quest-001/) ·
> [`test_runs/quest-002/`](test_runs/quest-002/).

## Headline

**End-to-end works.** Two real DeepScientist quests were run on this
prototype, both produced a finalized PDF paper with real numerical
results, real figures, and real primary references. Quest 002 was
created and ridden to closure through `launch.py`, exercising the
Phase-0 FI integration path.

| | Quest 001 (manual via curl) | Quest 002 (FI launch.py) |
|---|---|---|
| Created via | `curl POST /api/quests` | `launch.py` → `_create_quest` |
| Wall time | ~31 min | ~2 hr (with a 1.5 hr ChatGPT-Plus quota stall mid-run; ~25 min of actual codex work) |
| Anchor progression | stayed in `baseline` (DS state-machine bug; codex self-repaired) | clean `baseline → write` transition |
| Paper markdown | 7,468 B (5 inline refs with DOIs) | 7,109 B (full IMRAD; references in `references.bib`) |
| Paper PDF | 449 KB (TinyTeX) | 360 KB (TinyTeX) |
| Figures | 3 PNG + 3 SVG | 3 PNG (real `experiments/main/` run) |
| Bundle manifest | yes | yes |
| Closure decision | `decision-591a68ee` → `export_pdf` → `decision-347cc923` → `approve_completion` | `decision-7749159c` → `approve_completion` |

The paper itself reaches the right conclusion: at h = 0.1, RK4 RMS-x
error is **1.87 × 10⁻⁶**, Velocity-Verlet **9.65 × 10⁻⁴**, forward Euler
**0.652** (and **6.4 × 10¹⁴** at h = 0.5 — i.e. unstable). Empirical
log–log convergence-order slopes recovered: **1.65 / 4.00 / 2.00** for
Euler / RK4 / V-Verlet, matching the textbook 1st / 4th / 2nd-order
expectation.

## Environment under test

- WSL2 Ubuntu 24.04.3 LTS (kernel 6.6.87.2-microsoft-standard-WSL2),
  Python 3.14.4, Node 22.22.0, npm 11.9.0
- DeepScientist 1.5.17 (`@researai/deepscientist`)
- Codex CLI 0.130.0 (gpt-5.5; ChatGPT Plus auth, no incremental cost)
- TinyTeX (pdflatex / xelatex / lualatex / bibtex under
  `~/.TinyTeX/bin/x86_64-linux/`)
- DS daemon: `127.0.0.1:21500`, home `~/deepscientist-wsl`
  (Linux-native FS, not the Windows-mount path)

## Quest 001 — manual end-to-end via DS REST

Used to (a) validate the WSL2 fix kills the Windows-only `cp1252` /
broken-pipe failure mode that blocked Windows-native DS, and (b) confirm
DS produces real research from the bake-off prompt before wiring up FI.

- 35 min wall-time (pre-codex-quota era)
- Hit one DS state-transition bug at gate confirmation: codex called
  `artifact.confirm_baseline` and the runtime threw on the gate update.
  Codex recovered autonomously, repairing the quest's state files so
  `baseline_gate` flipped to `confirmed`. Notable resilience: the bug is
  DS-internal, not Windows-specific.
- Final outputs in
  [`test_runs/quest-001/`](test_runs/quest-001/) — `integrator_bakeoff.md`,
  `integrator_bakeoff.pdf`, `figures/{energy_vs_t,error_vs_h_loglog,trajectory_h0.1}.{png,svg}`.

## Quest 002 — FI launch.py path

```
python launch.py --config examples/integrator_bakeoff/config.yaml
```

What FI did, exercised:

1. ✅ `core.platform.detect_system()` → `wsl2`
2. ✅ `core.platform.ensure_daemon(...)` → daemon already healthy on 21500
3. ✅ `core.runner.DeepScientistBackend._create_quest()` → `quest_id="002"`
   with `auto_start=true`, `workspace_mode=autonomous`
4. ✅ `_poll_until_closure(...)` polled every 15 s; saw codex transition
   anchor cleanly `baseline → write` (cleaner than 001 — no
   state-machine repair needed)
5. ⏸ Mid-write, **codex hit the ChatGPT Plus daily usage cap** ("try
   again at 2:18 PM"). DS retried 5/5, exhausted, parked the quest. FI's
   `_poll_until_closure` correctly hit its 1 h timeout and raised
   `TimeoutError`. (This is a **rate-limit** issue, not a code bug.)
6. ✅ After the cap reset, sent a single resume chat to quest 002 via
   the same chat REST endpoint. Codex re-entered the `write` skill and
   finished the paper bundle in ~20 min.
7. ✅ `_collect_artifacts()` resolved `paper_md`, `paper_pdf`,
   `figures_dir`, `bundle_manifest` correctly (after one Phase-0
   patch — see *Findings* below).
8. ✅ `generation.paper.PaperGenerator.generate()` wrote everything to
   [`test_runs/quest-002/`](test_runs/quest-002/) — `paper.pdf`,
   `integrator_bakeoff_imrad.md`, `figures/`,
   `paper_bundle_manifest.json`.

## Findings & Phase-0 fixes

1. **`Path("~/...")` is not auto-expanded by pydantic.** YAML loaded
   `home: ~/deepscientist-wsl` as the literal string `~/deepscientist-wsl`,
   so `quest_root = home / "quests" / "002"` failed to resolve. Fixed in
   `core/config.py` with `field_validator(..., mode="before")` that
   `Path(v).expanduser()`s `home` and `output_dir` on load.

2. **DS PDF/figures paths drift across quests.**
   - Quest 001: PDF at `quest_root/.ds/worktrees/<paper-line>/paper/build/*.pdf`,
     figures at `baselines/local/<id>/figures/`.
   - Quest 002: PDF at `quest_root/paper/build/*.pdf` directly (no
     worktree indirection), figures at
     `experiments/main/<run>/outputs/figures/`.
   Fixed `_collect_artifacts` in `core/runner.py` to glob both
   conventions and prefer directories that actually contain raster /
   vector figure files (PNG / SVG / PDF / JPG) over JSON-only catalog
   directories like `paper/figures/figure_catalog.json`.

3. **DS `claude` runner is reserved-slot, not implemented.** Phase-0
   `core/providers.py` declares `anthropic` but raises
   `NotImplementedError`. Plumbing Claude into DS will need either a
   litellm proxy with an Anthropic API key, or a custom adapter that
   wraps Claude Code as a fake-codex runner. Not in Phase-0 scope.

4. **No state-machine bug in quest 002.** The closure path was clean:
   single `decision_request` (`decision-7749159c`) → `approve_completion`.
   Suggests the quest-001 anchor-confirmation bug was either a
   first-quest-of-home edge case or a transient DS state-machine glitch.

## What is *not* yet validated

- ARC backend (Phase 6).
- Pandoc + journal-template paper formats (Phase 1) — Phase-0 just
  copies the DS PDF as-is.
- Slides (Phase 2), poster (Phase 3), speech script (Phase 4).
- Cross-quest memory (Phase 5).
- Local provider (Ollama, vLLM) — only `provider: codex` was exercised.
- Windows-host-launching-WSL2-daemon path (FI itself was run inside WSL2).

## How to reproduce

```bash
git clone https://github.com/jyunming/FrontierInsight.git
cd FrontierInsight

# WSL2 prerequisites (one-time):
#   npm install -g @researai/deepscientist
#   ds latex install-runtime    # for the PDF export step

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python launch.py --config examples/integrator_bakeoff/config.yaml
# → ~30 min, writes ./outputs/<quest_id>/{paper.md,paper.pdf,figures/,bundle_manifest.json}
```

## Open questions for Phase 1

- Which paper formats to ship first? (NeurIPS + IEEE Access seem highest
  signal-to-effort for the user's stated journals/conferences.)
- Slides: Marp (markdown-native, no LaTeX) vs Beamer? Marp is faster to
  ship and more portable.
- For Claude-backed runs, do we accept the litellm-proxy path
  (needs Anthropic API key, separate from Claude Pro/Max)?
