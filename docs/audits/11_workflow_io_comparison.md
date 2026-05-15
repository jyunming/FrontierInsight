# Workflow / IO / CLI Comparison — FrontierInsight vs 2026 Peers

**Date**: 2026-05-15
**Scope**: FI's user-facing surfaces — config schema, output package, CLI shape, IDE integration, run visualization, sharing — against eight peer tools across four families (IDE assistants, notebooks, code-agent CLIs, minimal LLM CLIs).
**Verdict**: FI is competitive on depth (review panel, fleet runs, cross-check, no-simulation flow) but trailing on **discoverability** and **shareability**. The biggest wins available in <1 week of effort are (a) a subcommand-style `fi` entrypoint, (b) a `fi export` tarball for sharing a quest as a single file, and (c) a stable JSON spec for `frontier_insight_summary.json` so external tooling can render quests without scraping the dir.

---

## Findings

### 1. Cursor + Continue + Claude Desktop — IDE-first model

**Cursor.** The 2026 Cursor surface revolves around two artifacts: project-rooted `.cursor/rules/*.md` files (granular, file-scoped behavior rules — the successor to `.cursorrules`) and `environment.json` for cloud Background Agents. Settings themselves live in a SQLite blob, not a flat JSON file — a deliberate trade-off Cursor accepted to keep their settings GUI authoritative. Background Agents are framed as "asynchronous workers that edit and run code in a remote environment" — the same shape as an FI quest, but Cursor punts state visualization into their cloud dashboard.

**Continue.dev.** Plain `~/.continue/config.json` plus an escape hatch `config.ts` for programmatic extension. Slash commands are first-class: `/edit`, `/comment`, `/share`, plus user-defined entries in a `customCommands` array. Continue's `/share` command is particularly interesting — it emits a Markdown transcript of a chat session into a configured `outputDir`. That's a much lighter-weight sharing surface than FI's per-quest directory, and it points to a missing FI primitive: "give me a single shareable file."

**Claude Desktop / claude.ai.** Web/desktop UI; sharing is via a conversation-level "Share" link that produces a read-only HTML view of the thread. The closest analogue in FI is `--serve` (FastAPI on `127.0.0.1:8765`), but FI's `--serve` is single-user, local-only, and not designed to be linked.

**Takeaways for FI:** (a) `.cursor/rules/*.md`-style per-directory behavior rules don't map cleanly — FI's behavior is config-driven not chat-driven. (b) Continue's `/share` → Markdown output is a pattern FI should copy verbatim for the VSCode chat extension. (c) Cursor's `environment.json` Docker-image-pinning idea is a future direction for FI's `execution.sandbox: docker` mode.

### 2. Jupyter / Marimo / Streamlit — research-notebook surfaces

**Jupyter.** The dominant baseline, but the reproducibility numbers are damning: 75% of public Jupyter notebooks don't run, 96% don't reproduce. The `.ipynb` JSON format mixes code, output, and cell metadata — bad for diff, worse for re-execution.

**Marimo.** The 2026 alternative most relevant to FI. Marimo notebooks are stored as **pure Python files** that double as scripts, dataflow-graph apps, and reactive notebooks. Reactive execution removes hidden-state bugs; the file-as-Python design makes them trivial to version-control and AI-edit. Marimo positions itself explicitly as "AI-native, built to work with tools like Claude Code."

**Streamlit.** App-first, not notebook-first. Streamlit Community Cloud + Streamlit in Snowflake give a one-click "deploy from GitHub" path; sharing is a URL, access control is role-based. A multi-agent research/writing system in Streamlit is the closest functional analog to a single FI quest — but it produces an *app*, not a *paper*.

**Relationship to FI:** Notebooks are complements, not competitors. A useful future direction: FI could emit a Marimo `quest.py` notebook alongside `paper.md` so the user can re-execute the experiment interactively. That's the exact gap Marimo's "first-class SQL + reproducibility" pitch addresses.

### 3. Aider / Codex CLI / Claude Code / Gemini CLI — file-system-aware agents

| Tool | Config file | Format | Subcommand style | Session resume |
|------|-------------|--------|------------------|----------------|
| **Aider** | `.aider.conf.yml` (project) + `.aider.model.settings.yml` | YAML | Flag-based (`--model`, `--architect`, `--watch`) + in-REPL `/` commands | Implicit via git commit history |
| **Codex CLI** | `~/.codex/config.toml` + `.codex/config.toml` | TOML | Subcommand-based (`codex mcp`, `codex resume`, `codex fork`, `codex update`) | Explicit `codex resume <id>` |
| **Claude Code** | `~/.claude/settings.json` + `.claude/settings.json` + `.claude/settings.local.json` | JSON | Subcommand-based (`claude auth`, `claude rc`, `claude doctor`) + slash commands | Implicit via session id |
| **Gemini CLI** | `~/.gemini/settings.json` + per-command `.toml` files | JSON for settings, TOML for commands | Subcommand-based (`gemini settings`, `gemini mcp`, `gemini export`) | `gemini export` writes session to .md/.json |

Three patterns are remarkably consistent in the 2026 cohort:
1. **Hierarchical config layering.** Every tool layers user-global ⇒ project ⇒ local-override. FI today has one layer (single YAML passed via `--config`) — no project default, no user defaults.
2. **Subcommand-first CLI.** Aider remains the holdout with flag-driven invocation; Codex, Claude Code, and Gemini CLI are all `tool verb [args]`. The verbs that recur: `auth`, `mcp`, `export`, `resume`, `doctor`/`settings`.
3. **Custom commands as files.** Gemini CLI's `~/.gemini/commands/<name>.toml` and Claude Code's `.claude/commands/*.md` (or Skills) lower the bar for users to define their own workflows. FI's `--proposal`/`--critique` are baked into Python — there's no plug-in slot.

**Takeaway:** FI's current `launch.py --mode-flag` shape is closest to Aider, the least modern of the four. A subcommand `fi <verb>` design would slot FI into the 2026 idiom.

### 4. Minimal CLIs — aichat, shell-gpt

**aichat.** A single Rust binary; config in `~/.config/aichat/config.yaml`. The interesting bits for FI are the session model (`-s name` opens or resumes a named session; `save_session: true|false|null` controls implicit save behavior) and `repl_prelude` / `cmd_prelude` / `agent_prelude` (role+session presets per launch mode). aichat treats "REPL" and "one-shot CMD" as distinct entry points with their own defaults — FI conflates them.

**shell-gpt / `sgpt`.** Stdlib + `--repl` flag + `~/.config/shell_gpt/.sgptrc` (INI-style). Tiny config surface (~10 keys). Not a real comparison target for FI, but a useful reminder that not every LLM CLI needs 25 flags.

**Takeaway:** A `fi repl` subcommand modeled on aichat's REPL — load a quest's state, drop into an interactive prompt that can re-run nodes, replay clarify, dump intermediate state — would be a strong companion to the current "batch quest" flow.

### 5. LangSmith Studio / Weave / Anthropic Console — state visualization

LangSmith Studio renders a LangGraph's `StateGraph` as a visual flowchart and lets developers (a) inspect the State at any node, (b) edit past execution states, (c) re-run from any prior point. The pitch in 2026 is "your agents' development environment, not just observability." Crucially, this works for *any* LangGraph project — FI's `core/engine.py` is already LangGraph-based, so the Studio integration is mostly a matter of conforming to the Agent Server API protocol.

Weave (W&B) sits adjacent — trace-level observability, MCP auto-logging via `@weave.op()`, guardrails for runtime control. Heavier-weight than Studio; aimed at production deployments.

Anthropic's **Agent View** (announced 2026-05) is a CLI dashboard in Claude Code that shows multiple concurrent sessions on one screen — status, last response, last interaction timestamp. The closest FI analog is `--serve`'s quest list view at `/api/quests`. FI's GUI is more capable per-quest (clarify panel, log SSE, paper preview) but **lacks the at-a-glance multi-session table** that Agent View nails.

**Takeaway:** FI's `--serve` is a working prototype but is reinventing what LangSmith Studio already does well. Worth investigating whether FI can expose a Studio-compatible Agent Server protocol so users get the upstream visualization for free.

### 6. Comparison table

| Surface | FI | Cursor | Continue | Claude Code | Codex CLI | Gemini CLI | Aider | Marimo |
|---------|----|----|----|----|----|----|----|----|
| Config format | YAML | JSON (SQLite-stored) + `.cursor/rules/*.md` | JSON (+ TS) | JSON | TOML | JSON + TOML cmds | YAML | Python file |
| Config layering | single file | user + project + rules | user + project | user + project + local | user + project | user + project | user + project | n/a |
| Entry style | `launch.py --mode` (11 modes, ~25 flags) | IDE chat | IDE chat + `/` | `claude <verb>` | `codex <verb>` | `gemini <verb>` | `aider` + flags | `marimo edit` |
| Resume | `--resume <quest_id>` | Cloud Background Agents | session implicit | session id | `codex resume <id>` | session id | git history | n/a |
| Output | per-quest dir | inline IDE edits | inline + `/share` md | inline + transcript | inline + session | inline + `export` | inline + git commits | `.py` notebook |
| Single-file share | none | none | `/share` md | conversation share | session export | `gemini export` | git push | `.py` file |
| State viz | FastAPI `--serve` | cloud dashboard | none | Agent View | none | none | none | reactive UI |
| Custom commands | hardcoded modes | rules files | `customCommands` JSON | Skills / `.claude/commands/*.md` | none (yet) | `.toml` files | `/cmd` plug-ins | n/a |
| Multi-run | `--fleet` (concurrent) | Background Agents (cloud) | n/a | Agent Teams (`--agents`) | n/a | n/a | n/a | n/a |
| LLM bridge | 13 providers + VSCode LM API | own | many | own | many providers | Google | many | n/a |
| Reproducibility | YAML + SQLite checkpoint | environment.json + Docker | n/a | n/a | n/a | n/a | git history | reactive guarantee |

---

## Recommendations

1. **Add a `fi` subcommand wrapper around `launch.py`. `[impact: high] [effort: low]`** — Ship a thin `fi` Python entry point in `pyproject.toml`'s `[project.scripts]` that maps `fi new <topic>`, `fi run <config.yaml>`, `fi fleet a.yaml b.yaml`, `fi resume <id>`, `fi summarize <dir>`, `fi proposal "<topic>"`, `fi digest`, `fi portfolio`, `fi critique <id>`, `fi serve`, `fi ingest <files>`. Keep `python launch.py --config ...` working as a legacy alias. This is a pure surface change — argparse subparsers wrap the existing dispatch in `launch.py`. The discoverability win is large: `fi --help` prints a 10-line verb list instead of 25 flags.

2. **Publish `frontier_insight_summary.json` as a versioned public spec. `[impact: high] [effort: low]`** — The summary file already exists per-quest. Add `"schema_version": "1"` at the top, document the shape in `docs/spec/summary_v1.md`, and treat it as a stable contract. External tools (notebooks, dashboards, the VSCode extension) can then render quests without scraping the directory. Mirror Continue's pattern of treating its config schema as an external integration point.

3. **Add `fi export <quest_id>` producing a `.fi-quest` tarball. `[impact: high] [effort: low]`** — Bundle `config.yaml` + `paper.md` + figures + `frontier_insight_summary.json` + `.fi/state.sqlite` into one gzipped tar. Add `fi import <file.fi-quest>` to round-trip. This is the missing shareability primitive — closes the gap with Continue's `/share`, Codex's session export, Gemini's `export`, and Claude Desktop's conversation-share link. Email-friendly, git-tree-friendly, no need to push a whole `outputs/` directory.

4. **Adopt hierarchical config layering. `[impact: medium] [effort: medium]`** — Add `~/.config/frontier-insight/defaults.yaml` (user) and `.fi.yaml` (project) layers that merge under the explicit `--config` file. Adopt the precedence rule used by every peer: CLI flag > project > user > engine defaults. Also accept `[tool.frontier_insight]` table in `pyproject.toml` for projects that already standardize on TOML. Pydantic handles the merge cleanly; sample YAMLs in `examples/` stay valid.

5. **Add a Marimo `quest.py` output alongside `paper.md`. `[impact: medium] [effort: medium]`** — When the engine finishes, render the experiment's code + result_json into a single reactive Marimo notebook. The user gets one file they can re-execute, modify, and re-publish — closing the "did the experiment really run as the paper claims?" reproducibility hole that bites every Jupyter user. Gated behind `output.kinds: [..., marimo]` so existing quests don't pay the cost.

6. **Wire the engine into LangSmith Studio as a fallback visualization. `[impact: medium] [effort: medium]`** — FI's `core/engine.py` is already a LangGraph `StateGraph`. Conform to the Agent Server API protocol so `langgraph dev` opens FI quests in Studio. FI keeps `--serve` (lower-friction, no cloud, no LangChain account) but power users get edit-state-and-rerun for free. Document the choice in `docs/USAGE.md`.

7. **Add `fi repl` interactive mode. `[impact: low] [effort: medium]`** — Modeled on aichat's REPL: load a quest by id, drop into a prompt that can re-run any node (`run review_panel`), inspect state (`show review`), dump intermediate artifacts (`dump cross_check.md`), and submit clarify answers without going through the GUI. Useful for quick iteration on a single high-cost quest where the fleet runner is overkill. Lives at `cli/repl.py`; uses `prompt_toolkit` (already a transitive dep).

8. **Add a multi-session table view to `--serve`. `[impact: low] [effort: low]`** — Anthropic's Agent View shows N concurrent agents on one screen with status + last response + timestamp. FI's `/api/quests` endpoint returns the data; the static HTML just doesn't render it as a sortable table. One-evening change in `web/static/index.html`.

9. **Custom-command plug-in slot. `[impact: low] [effort: medium]`** — Mirror Gemini CLI's `~/.gemini/commands/*.toml` and Claude Code's `.claude/commands/*.md`. Let users drop a `~/.config/frontier-insight/commands/my_review.md` (system prompt + slot for `{topic}`) that becomes `fi my-review <topic>`. Cleanly extensible without forking FI for one-off PM workflows. Lower-priority than 1-3.

10. **Keep YAML — don't migrate to TOML. `[impact: n/a] [effort: avoid]`** — Tempting after seeing Codex (TOML) and the wider Python world's `pyproject.toml` standardization, but YAML's multi-line strings + comments + anchors fit FI's research-config use case better than TOML's table-heavy layout. The right move is to *accept* `[tool.frontier_insight]` in `pyproject.toml` as a secondary layer (rec 4), not to switch the primary format.

---

## References

**FI source paths (repo-relative):**
- `launch.py` — 11-mode entrypoint (lines 38-277 cover argparse).
- `core/config.py` — Pydantic schema; ~17 EngineConfig fields (lines 80-200).
- `web/server.py` — `--serve` FastAPI status GUI.
- `vscode-frontier-insight/` — chat extension, 4.6k LOC TS.
- `examples/` — sample YAML configs (`bernstein_vazirani_noise`, `euv_mor_shot_noise`, `integrator_bakeoff`).
- `docs/USAGE.md` — current CLI documentation.

**Peer documentation:**
- Cursor configuration reference — https://cursor.com/docs/cli/reference/configuration
- Cursor Environments / environment.json — https://stevekinney.com/courses/ai-development/cursor-environment-configuration
- Continue.dev config schema — https://docs.continue.dev/reference/json-reference
- Continue.dev slash commands — https://docs.continue.dev/customization/slash-commands
- Claude Code settings — https://code.claude.com/docs/en/settings
- Claude Code 2026 command reference — https://smartscope.blog/en/generative-ai/claude/claude-code-reference-guide/
- Codex CLI configuration — https://developers.openai.com/codex/config-reference
- Codex CLI command-line options — https://developers.openai.com/codex/cli/reference
- Codex config.toml writeup — https://codex.danielvaughan.com/2026/04/08/codex-cli-configuration-reference/
- Gemini CLI configuration — https://geminicli.com/docs/reference/configuration/
- Gemini CLI custom commands — https://geminicli.com/docs/cli/custom-commands/
- Aider configuration — https://aider.chat/docs/config.html
- Aider YAML config — https://aider.chat/docs/config/aider_conf.html
- aichat configuration guide — https://github.com/sigoden/aichat/wiki/Configuration-Guide
- Marimo vs Jupyter — https://marimo.io/features/vs-jupyter-alternative
- Marimo dataflow graphs — https://marimo.io/blog/dataflow
- LangSmith Studio docs — https://docs.langchain.com/langsmith/studio
- LangGraph Studio announcement — https://www.langchain.com/blog/langgraph-studio-the-first-agent-ide
- Weave + Anthropic integration — https://github.com/wandb/weave/blob/master/docs/docs/guides/integrations/anthropic.md
- Anthropic Agent View — https://en.ain.ua/2026/05/12/anthropic-launched-agent-view-mode-for-claude-code-this-will-allow-managing-multiple-ai-agents-at-once/
- pyproject.toml `[tool]` table spec — https://packaging.python.org/en/latest/specifications/pyproject-toml/
- Streamlit Community Cloud — https://streamlit.io/cloud
