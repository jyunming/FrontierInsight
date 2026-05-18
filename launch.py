"""Frontier Insight — orchestrator entry point.

Single quest:
    python launch.py --config examples/integrator_bakeoff/config.yaml

Fleet:
    python launch.py --fleet a.yaml b.yaml c.yaml --max-concurrent 4

Optional fleet hardening:
    --memory-cap-mb 4096   throttle new starts when RSS approaches the cap
    --profile              dump a per-quest viztracer trace if viztracer is installed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Make sibling packages importable when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config
from core.engine import Engine, QuestArtifacts
from core.provider import ProxySupervisor
from generation.paper import PaperGenerator
from generation.poster import PosterGenerator
from generation.slides import SlideGenerator
from generation.speech import SpeechGenerator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="frontier-insight",
        description="End-to-end automated research pipeline.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path, help="YAML config for a single quest.")
    mode.add_argument(
        "--fleet",
        type=Path,
        nargs="+",
        help="Run multiple quests concurrently (one config path per quest).",
    )
    mode.add_argument(
        "--ingest",
        type=Path,
        nargs="+",
        help="Ingest one or more PDF / Markdown / TXT files into Axon "
             "(kind=fi_local_paper) and exit. Requires `axon` to be "
             "installed; optionally pass --axon-config to point at a "
             "non-default Axon corpus. PDF support requires `pypdf` installed.",
    )
    mode.add_argument(
        "--serve",
        action="store_true",
        help="Start the status GUI (FastAPI server + HTMX frontend). "
             "Use --output-root to point at the quest output directory, "
             "and --host / --port to bind elsewhere than 127.0.0.1:8765. "
             "Requires FastAPI + uvicorn installed.",
    )
    mode.add_argument(
        "--summarize",
        type=Path,
        help="Summarize a folder of mixed content (papers, source code, "
             "study notes, experiment logs, prior quest outputs). Walks "
             "the folder, classifies each file, calls the LLM once, and "
             "writes outputs/<summary_id>/summary.md. Input files plus "
             "the produced summary are ingested into Axon as "
             "fi_summary_input / fi_summary so future quests can "
             "retrieve them. Pair with --summarize-kind to override "
             "content-type auto-detection.",
    )
    mode.add_argument(
        "--digest",
        action="store_true",
        help="Generate a weekly PM-style digest across the quests in "
             "--output-root. Writes outputs/_digests/<YYYY-Www>.md with "
             "a structured diff against the most-recent prior digest "
             "(promoted / new / still-in-progress / stalled / dropped). "
             "Pair with --days N to widen or narrow the window "
             "(default 7). Ingests into Axon as kind=fi_digest so "
             "future quests can retrieve prior-week context.",
    )
    mode.add_argument(
        "--portfolio",
        action="store_true",
        help="Generate a cross-quest portfolio synthesis across every "
             "quest under --output-root (no time window). Writes "
             "outputs/_portfolio/<YYYY-MM-DD>.md with topic clusters, "
             "near-duplicate detection, meta-paper candidates, coverage "
             "gaps, and prioritized next-quest suggestions. Ingests "
             "into Axon as kind=fi_portfolio.",
    )
    mode.add_argument(
        "--critique",
        type=str,
        help="Adversarial second-pass review of a completed quest. "
             "Pass the quest_id (e.g. "
             "1778452404-euv-mor-photon-shot-noise-ler-e6bfe5). "
             "Reads the quest's paper.md + experiment code + the "
             "original in-quest review, then asks the LLM — ideally "
             "via a different provider — to find what the in-quest "
             "reviewer missed. Writes outputs/<quest_id>/critique.md "
             "and ingests as kind=fi_critique. Pair with "
             "--critique-provider to route through a different model "
             "family from the one that wrote the paper.",
    )
    mode.add_argument(
        "--proposal",
        type=str,
        help="Pre-quest planning doc. Pass a topic string (1-3 "
             "paragraphs). The LLM produces a structured proposal "
             "(background, hypothesis, plan, success criteria, "
             "risks, scope limits, recommended next step) for the "
             "user to review BEFORE committing compute to the full "
             "quest. Writes outputs/_drafts/<id>-proposal.md plus a "
             "companion outputs/_drafts/<id>.yaml ready to feed into "
             "--config. Ingests as kind=fi_proposal.",
    )
    mode.add_argument(
        "--analyze",
        type=Path,
        help="Run a no-simulation quest with pre-staged data. Pass a "
             "directory path; every file under it (recursive, "
             "symlinks skipped) is copied into the new quest's "
             "data/ directory before the engine starts. The engine "
             "then routes auto_collect_data (passthrough) → "
             "wait_for_data (passthrough, data is already there) → "
             "data_load → analyze → cross_check → write → review. "
             "Requires --analyze-topic to describe what to analyze. "
             "Use when you already have the dataset and just want a "
             "paper analyzing it — the inverse of --proposal.",
    )
    mode.add_argument(
        "--install-tectonic",
        action="store_true",
        help="Download the tectonic LaTeX binary (~70 MB) into "
             "tools/ (tools/tectonic.exe on Windows, tools/tectonic "
             "on macOS/Linux) so paper_pdf works without an admin "
             "install of MiKTeX. Tectonic is self-bootstrapping — on "
             "first compile it downloads required CTAN packages into "
             "the user's home dir (~30 s, no GUI prompts). Picks the "
             "right asset for your OS+arch from GitHub Releases and "
             "verifies the SHA-256 from the release's SHA256SUMS file.",
    )
    mode.add_argument(
        "--list-drafts",
        action="store_true",
        help="List proposal drafts (YAML companions) that ``--proposal`` "
             "previously wrote to ``outputs/_drafts/``. Pairs with "
             "``--config <yaml>`` to launch the chosen quest. Web UI "
             "exposes the same surface as the 'Continue a draft' "
             "picker on /interview; VSCode chat as ``@fi /drafts``.",
    )
    mode.add_argument(
        "--new",
        action="store_true",
        help="Run the interactive interview to set up a new quest. "
             "Asks topic / title / output_kinds / paper_format / "
             "research approach / study_depth / clarify slots / "
             "review panel / knowledge / provider / model. Writes the "
             "resulting `outputs/_drafts/<id>.yaml` and (unless "
             "--draft-only is set) starts the quest immediately. "
             "Mirrors the VSCode `@fi /new` flow for headless use.",
    )
    mode.add_argument(
        "--update",
        type=str,
        default=None,
        metavar="QUEST_ID",
        help="Mid-quest re-entry. Pass the quest_id of an existing "
             "quest under --output-root. The interview opens "
             "pre-filled with the quest's current YAML; only "
             "research-shaping fields are editable (no topic / "
             "title / provider). Affected LangGraph stages are "
             "invalidated based on what changed, then the quest "
             "resumes via the existing --resume path. Mirrors "
             "`@fi /update` in the VSCode extension.",
    )
    p.add_argument(
        "--summarize-kind",
        type=str,
        choices=["auto", "literature", "code", "study", "execution", "mixed"],
        default="auto",
        help="Content-type hint for --summarize. Default 'auto' detects "
             "from the file mix. Override when you know better.",
    )
    p.add_argument(
        "--summarize-provider",
        type=str,
        default="vscode_extension",
        help="Provider for --summarize (when not launched from the VSCode "
             "extension). Defaults to vscode_extension; use openai / "
             "claude_cli / codex_cli / gemini_cli for headless runs.",
    )
    p.add_argument(
        "--days",
        type=int,
        default=7,
        help="Window size in days for --digest. Default 7 (rolling "
             "week). Use a non-7 value to get a custom-window digest "
             "filename (YYYY-MM-DD-to-YYYY-MM-DD.md) instead of the "
             "ISO-week canonical form (YYYY-Www.md).",
    )
    p.add_argument(
        "--digest-provider",
        type=str,
        default="vscode_extension",
        help="Provider for --digest. Same conventions as "
             "--summarize-provider — vscode_extension when launched "
             "from chat, headless CLI providers otherwise.",
    )
    p.add_argument(
        "--portfolio-provider",
        type=str,
        default="vscode_extension",
        help="Provider for --portfolio. Same conventions as "
             "--digest-provider.",
    )
    p.add_argument(
        "--critique-provider",
        type=str,
        default="vscode_extension",
        help="Provider for --critique. Defaults to vscode_extension; "
             "explicitly set to a DIFFERENT provider from the quest's "
             "original (e.g. claude_cli for an openai-written paper) "
             "to get the strongest adversarial second-pass effect.",
    )
    p.add_argument(
        "--proposal-provider",
        type=str,
        default="vscode_extension",
        help="Provider for --proposal. Same conventions as the other "
             "PM-command providers — vscode_extension when launched "
             "from chat, headless CLI providers otherwise.",
    )
    p.add_argument(
        "--proposal-ensemble",
        type=str,
        default="",
        metavar="M1,M2,M3",
        help="Multi-model ensemble for --proposal. Comma-separated "
             "list of model identifiers. Each model independently "
             "drafts a proposal; a moderator picks the best (default) "
             "or merges (with --proposal-ensemble-merge synthesize). "
             "Cost: N+1 LLM calls. Worth it because the proposal "
             "frames every downstream quest decision. Example: "
             "--proposal-ensemble gpt-5.5,claude-opus-4-7,gemini-2.5-pro",
    )
    p.add_argument(
        "--proposal-ensemble-merge",
        type=str,
        choices=("tournament", "synthesize"),
        default="tournament",
        help="How to combine the ensemble candidates. tournament "
             "(default): moderator picks the single best plan, "
             "preserving the winner's voice. synthesize: moderator "
             "merges all N into one coherent plan that flags any "
             "disagreement between candidates.",
    )
    p.add_argument(
        "--proposal-ensemble-moderator",
        type=str,
        default="",
        help="Model to perform the merge step. Defaults to the first "
             "model in --proposal-ensemble.",
    )
    p.add_argument(
        "--critique-ensemble",
        type=str,
        default="",
        metavar="M1,M2,M3",
        help="Multi-model ensemble for --critique. Same shape as "
             "--proposal-ensemble but default merge is synthesize — "
             "adversarial critique benefits from preserving every "
             "critic's perspective, not picking one winner.",
    )
    p.add_argument(
        "--critique-ensemble-merge",
        type=str,
        choices=("tournament", "synthesize"),
        default="synthesize",
        help="How to combine the critique candidates. synthesize "
             "(default): merge all N adversarial critiques into one "
             "report with disagreement flags. tournament: moderator "
             "picks the most rigorous single critic.",
    )
    p.add_argument(
        "--critique-ensemble-moderator",
        type=str,
        default="",
        help="Model to perform the merge step. Defaults to the first "
             "model in --critique-ensemble.",
    )
    p.add_argument(
        "--analyze-topic",
        type=str,
        default=None,
        help="Required companion to --analyze: a one-sentence "
             "description of what the analysis should produce "
             "(\"Compare ridership trends across regions\", "
             "\"Find common failure modes in these logs\"). The "
             "engine's analyze + write nodes read this verbatim as "
             "the quest topic.",
    )
    p.add_argument(
        "--analyze-provider",
        type=str,
        default="vscode_extension",
        help="Provider for --analyze. Same conventions as the other "
             "PM-command providers — vscode_extension when launched "
             "from chat, headless CLI providers otherwise.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs"),
        help="Quest output root for --serve mode. The server scans this "
             "directory for existing quests and writes new ones here.",
    )
    p.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Bind host for --serve. Default 127.0.0.1.",
    )
    p.add_argument(
        "--port", type=int, default=8765,
        help="Bind port for --serve. Default 8765.",
    )
    p.add_argument(
        "--axon-config",
        type=Path,
        default=None,
        help="YAML AxonConfig for --ingest mode (the same shape the engine "
             "passes via knowledge.axon_config). Defaults to AxonConfig() if omitted.",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=min(4, (os.cpu_count() or 4)),
        help="Cap on concurrent quests in --fleet mode.",
    )
    p.add_argument(
        "--memory-cap-mb",
        type=int,
        default=None,
        help="Throttle new fleet quest starts when RSS exceeds this many MB.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Dump a per-quest viztracer trace to .fi/profile.json if viztracer is installed.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override config.output.output_dir (single-quest mode only).",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="When `engine.clarify_mode: interactive` is set in the YAML, "
             "pause at the clarify node and read answers from stdin. "
             "Single-quest mode only — fleet runs are headless.",
    )
    p.add_argument(
        "--draft-only",
        action="store_true",
        help="With --new: write the generated `outputs/_drafts/<id>.yaml` "
             "and exit, instead of immediately launching the quest. "
             "Useful when you want to hand-edit the YAML before "
             "starting (e.g. set knowledge.local_papers, swap "
             "provider.model, tune cost knobs). Run "
             "`python launch.py --config <path>` afterward to start.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a previously-failed quest. Pass the quest_id of an "
             "existing quest directory under config.output.output_dir. "
             "The LangGraph SqliteSaver checkpoint at "
             "<output_dir>/<quest_id>/.fi/state.sqlite is reused so the "
             "run picks up at the last completed node instead of starting "
             "over. Single-quest mode only (use --config). Useful when a "
             "long Copilot outage exhausts the bridge retry budget mid-quest.",
    )
    p.add_argument(
        "--vscode-bridge-port",
        type=int,
        default=0,
        help="TCP port the FI VSCode extension is listening on for "
             "the LLM bridge. The extension passes this when it spawns FI; "
             "setting it forces provider.name=vscode_extension regardless of "
             "what the YAML says. Do not pass this from a regular terminal "
             "run — use copilot_cli or similar for headless Copilot usage.",
    )
    p.add_argument(
        "--no-axon-sidecar",
        action="store_true",
        help="Skip the Axon sidecar health check + auto-launch. By default, "
             "every FI launch ensures a long-lived ``python -m axon.api`` "
             "process is running on AXON_HOST:AXON_PORT (default "
             "127.0.0.1:8000) so the embedding model + vector indexes "
             "stay hot across quests. Pass this in CI / tests where the "
             "sidecar isn't wanted, or when running an Axon API server "
             "manually with non-default flags.",
    )
    args = p.parse_args(argv)
    if args.fleet and args.output is not None:
        p.error("--output cannot be combined with --fleet (per-quest output_dir comes from each YAML).")
    if args.ingest and args.output is not None:
        p.error("--output is irrelevant in --ingest mode.")
    if args.resume and not args.config:
        p.error("--resume requires --config (the YAML for the original quest).")
    if args.analyze and not args.analyze_topic:
        p.error(
            "--analyze requires --analyze-topic to describe what the "
            "analysis should produce (one sentence is fine).",
        )
    if args.analyze_topic and not args.analyze:
        p.error("--analyze-topic only makes sense with --analyze.")
    return args


# ---- single-quest path ----------------------------------------------------


# Same alphabet as `_new_quest_id` produces: timestamp + slug + 6-char nonce,
# joined by `-`. We don't enforce the full structural shape (digit+slug+hex)
# because users may have manually renamed a quest dir; we only enforce that
# the string is filesystem-safe (no separators, no traversal, no dot-files).
_RESUME_QUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")


def _validate_resume_quest_id(quest_id: str, output_dir: Path) -> str | None:
    """Validate `--resume <quest_id>` against path-traversal and existence.

    Returns `None` if the quest is safe to resume, or a human-readable error
    string if it isn't. Returning a string rather than raising keeps the
    caller in control of the exit code / error sink.

    Three checks, in order:
      1. The id matches the strict identifier alphabet (digits, letters,
         ``_``, ``-``, ``.``). This rejects ``..``, ``/``, ``\\``, and
         any other separator the OS might interpret.
      2. The resolved path ``output_dir / quest_id`` stays inside the
         resolved ``output_dir``. Defense-in-depth against unusual OS
         path-normalization (symlinks, UNC, 8.3 names, etc.) — the regex
         in (1) already catches the obvious cases.
      3. The checkpoint sqlite from the prior run actually exists. Without
         this, we'd silently create an empty quest dir under the supplied
         name and "resume" from a fresh state — confusing for the user.
    """
    if not _RESUME_QUEST_ID_RE.match(quest_id):
        return (
            f"--resume {quest_id!r}: invalid quest id. Only letters, digits, "
            f"`_`, `-`, and `.` are allowed (no path separators, no `..`)."
        )
    root = output_dir.resolve()
    candidate = (root / quest_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return (
            f"--resume {quest_id!r}: resolved path {candidate} is outside "
            f"the configured output_dir {root}."
        )
    checkpoint = candidate / ".fi" / "state.sqlite"
    if not checkpoint.is_file():
        return (
            f"--resume {quest_id!r}: no checkpoint at {checkpoint}. The quest "
            f"dir must exist and contain a .fi/state.sqlite from a prior run."
        )
    return None


def _apply_vscode_bridge_override(cfg: Config, port: int) -> None:
    """When launched with ``--vscode-bridge-port N``, force
    every quest's provider to route through the FI VSCode extension's
    ``vscode.lm.*`` bridge on that port. Overrides whatever ``provider``
    block the YAML carried (the YAML can still set ``provider.model``,
    ``provider.node_models``, etc. — those flow through unchanged so
    per-node model routing still works inside the chat session)."""
    if port <= 0:
        return
    cfg.provider.name = "vscode_extension"
    cfg.provider.extra = {**(cfg.provider.extra or {}), "bridge_port": port}


def _pick_clarify_callback(
    cfg: Config, engine: Engine, interactive: bool,
) -> object:
    """Select the right clarify-callback for this run.

    * Terminal `--interactive`: collect answers from stdin via
      ``_cli_clarify_callback``.
    * `provider.name == "vscode_extension"`: route the questions
      through the same bridge the LLM calls use, so the FI VSCode
      extension can present them as modals and post answers back.
    * Otherwise: no callback. If the YAML has
      ``engine.clarify_mode = "interactive"`` and we return None, the
      engine raises a clear RuntimeError at the clarify pause —
      that's the surface we want users to hit, not a silent hang.
    """
    if interactive:
        return _cli_clarify_callback
    if cfg.provider.name == "vscode_extension":
        # IMPORTANT: route the clarify call through the SAME bridge
        # the LLMClient already uses. The extension-side TCP server
        # only tracks one active socket (every new connection
        # overwrites it), so two clients on the same port would have
        # their responses cross-routed. Reading `engine._client._bridge`
        # works because by the time clarify_callback fires, the
        # clarify NODE has already made one LLM call (to generate the
        # questions), which lazy-initialized the bridge.
        port = int(cfg.provider.extra.get("bridge_port", 0))
        if port <= 0:
            return None

        async def callback(questions: dict[str, object]) -> dict[str, object]:
            assert engine._client is not None, (
                "_pick_clarify_callback was called before Engine.run() "
                "created the LLMClient"
            )
            bridge = engine._client._bridge
            if bridge is None:
                # Defensive: shouldn't happen (clarify-node LLM call
                # creates the bridge first), but if the engine ever
                # routes around the LLMClient we'd hit this. Build a
                # fresh client — it'll still go to the same port.
                from core.vscode_bridge import VSCodeBridgeClient
                bridge = VSCodeBridgeClient(host="127.0.0.1", port=port)
                await bridge.connect()
                engine._client._bridge = bridge
            return await bridge.clarify(dict(questions))

        return callback
    return None


async def _cli_clarify_callback(questions: dict[str, object]) -> dict[str, object]:
    """Terminal-based clarify-question collector. Prints each question
    with its agent-suggested default, reads one line of stdin per slot,
    accepts the default on blank input. Used by `launch.py --interactive`."""
    print()
    print("=" * 72)
    print("Pre-flight clarification — agent has 5 questions to sharpen the run.")
    print("Press Enter to accept the default in parentheses.")
    print("=" * 72)
    answers: dict[str, object] = {}
    for key, value in questions.items():
        if not isinstance(value, dict):
            continue
        question = value.get("question", key)
        default = value.get("default", "")
        # Pretty-print the default — lists get joined.
        default_str = (
            ", ".join(str(x) for x in default) if isinstance(default, list)
            else str(default)
        )
        print()
        print(f"  [{key}] {question}")
        try:
            user_input = input(f"    answer ({default_str}): ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = ""
        # Preserve list-typed defaults by parsing comma-separated input.
        if isinstance(default, list):
            answers[key] = (
                [s.strip() for s in user_input.split(",") if s.strip()]
                if user_input else default
            )
        else:
            answers[key] = user_input or default
    print()
    print("Thanks — proceeding with the autonomous loop.")
    print("=" * 72)
    print()
    return answers


async def run_one(
    cfg: Config,
    *,
    supervisor: ProxySupervisor,
    profile: bool = False,
    engine: Engine | None = None,
    interactive: bool = False,
    resume_quest_id: str | None = None,
    source_yaml_path: Path | None = None,
) -> dict[str, object]:
    # Engine may be constructed by the caller (e.g. `gated()` builds it
    # once so the status-line `quest_id` matches the quest that actually
    # runs — instead of creating a second Engine here with a fresh
    # quest_id and stranding the status-line one). When omitted, build.
    if engine is None:
        engine = Engine(cfg, supervisor=supervisor, resume_quest_id=resume_quest_id)
    if resume_quest_id is not None:
        print(f"[FI] resume quest_id={engine.quest_id} provider={cfg.provider.name}")
    else:
        print(f"[FI] start quest_id={engine.quest_id} provider={cfg.provider.name}")
    # Drop a copy of the source YAML into the quest dir so future
    # `--resume`s (and the VSCode `/resume` command) can find the
    # config trivially at `<quest_root>/config.yaml` instead of
    # slug-matching the drafts dir or asking the user to pick.
    if source_yaml_path is not None and source_yaml_path.is_file():
        engine.quest_root.mkdir(parents=True, exist_ok=True)
        dest = engine.quest_root / "config.yaml"
        if not dest.exists():
            try:
                shutil.copy2(source_yaml_path, dest)
            except OSError as e:
                # Non-fatal: the quest can still run. We just lose the
                # auto-resume convenience for THIS quest.
                print(f"[FI] couldn't copy config.yaml into quest dir: {e!r}", file=sys.stderr)
    # Pick the right clarify handler:
    #   --interactive          → terminal Q&A
    #   provider=vscode_extension → route through the bridge so the
    #                            extension shows VSCode modals
    #   otherwise              → None; clarify_mode=interactive crashes
    #                            (the engine catches this and produces
    #                            a clear RuntimeError).
    callback = _pick_clarify_callback(cfg, engine, interactive)
    art: QuestArtifacts = await _maybe_profiled(
        engine, profile=profile, clarify_callback=callback,
    )
    print(f"[FI] {art.quest_id} -> {art.quest_root}")
    written = await _run_generators(cfg, art, supervisor=supervisor)
    summary = {
        "quest_id": art.quest_id,
        "quest_root": str(art.quest_root),
        "provider": cfg.provider.name,
        "outputs": {k: str(v) for k, v in written.items()},
        "paper_md": str(art.paper_md) if art.paper_md else None,
        "paper_pdf": str(written.get("paper_pdf")) if written.get("paper_pdf") else None,
    }
    summary_path = art.quest_root / "frontier_insight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[FI] summary -> {summary_path}")
    return summary


# Derive the allowed values from `core.config.PaperFormat` so adding a
# new template stays a one-place change. `typing.get_args` returns the
# Literal members as a tuple of strings.
from typing import get_args as _get_args
from core.config import PaperFormat as _PaperFormat
_VALID_PAPER_FORMATS: frozenset[str] = frozenset(_get_args(_PaperFormat))


def _apply_paper_venue_override(cfg: Config, art: QuestArtifacts) -> None:
    """Honor the `paper_venue` clarify slot: if the user's YAML left
    `output.paper_format` at the default `"generic"` AND clarify picked
    a specific venue, apply the venue choice. We DON'T override an
    explicit YAML setting (the user's hand-written value wins) and
    we silently ignore unknown venue strings to avoid pydantic
    rejecting them on a Config validator."""
    venue = (
        (art.raw_state or {}).get("clarify_answers", {}).get("paper_venue", "")
    )
    if not isinstance(venue, str) or venue not in _VALID_PAPER_FORMATS:
        return
    if cfg.output.paper_format != "generic":
        # User set an explicit format in YAML; honor that.
        return
    if venue == "generic":
        return  # nothing to do
    print(
        f"[FI] paper_venue from clarify -> overriding paper_format: "
        f"generic -> {venue}",
    )
    cfg.output.paper_format = venue  # type: ignore[assignment]


async def _run_generators(
    cfg: Config,
    art: QuestArtifacts,
    *,
    supervisor: ProxySupervisor,
) -> dict[str, Path]:
    """Run each generator in turn; one failure does not abort the rest."""
    written: dict[str, Path] = {}
    _apply_paper_venue_override(cfg, art)
    # 1) Paper (sync — pandoc shell-out is fine without async).
    try:
        written.update(PaperGenerator(cfg).generate(art, art.quest_root))
    except Exception as e:  # pragma: no cover — defensive
        print(f"[FI] paper generator failed: {e!r}", file=sys.stderr)
        # Strict mode escape hatch: ``output.require_pdf=True`` is the
        # user's signal that a missing PDF is a hard failure for this
        # quest, not a soft-degrade. The default catch-all above
        # exists so a flaky pandoc / LaTeX engine doesn't lose the
        # slides+poster+speech outputs — that policy stays for the
        # default ``require_pdf=False`` case. When the user opted into
        # strict mode, re-raise so the quest exits non-zero and the
        # caller sees the failure instead of a "completed quest with
        # missing PDF".
        if cfg.output.require_pdf:
            raise
    # 2) Slides (LLM + Marp).
    try:
        written.update(
            await SlideGenerator(cfg).generate(art, art.quest_root, supervisor=supervisor)
        )
    except Exception as e:
        print(f"[FI] slide generator failed: {e!r}", file=sys.stderr)
    # 3) Poster (LLM + pdflatex).
    try:
        written.update(
            await PosterGenerator(cfg).generate(art, art.quest_root, supervisor=supervisor)
        )
    except Exception as e:
        print(f"[FI] poster generator failed: {e!r}", file=sys.stderr)
    # 4) Speech (one LLM call).
    try:
        written.update(
            await SpeechGenerator(cfg).generate(art, art.quest_root, supervisor=supervisor)
        )
    except Exception as e:
        print(f"[FI] speech generator failed: {e!r}", file=sys.stderr)
    for name, path in written.items():
        print(f"[FI] wrote {name} -> {path}")
    return written


async def _maybe_profiled(
    engine: Engine,
    *,
    profile: bool,
    clarify_callback: object = None,
) -> QuestArtifacts:
    if not profile:
        return await engine.run(clarify_callback=clarify_callback)
    try:
        from viztracer import VizTracer  # type: ignore[import-not-found]
    except ImportError:
        print("[FI] --profile requested but viztracer is not installed; skipping")
        return await engine.run(clarify_callback=clarify_callback)
    trace_path = engine.fi_dir / "profile.json"
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    with VizTracer(output_file=str(trace_path)):
        art = await engine.run(clarify_callback=clarify_callback)
    print(f"[FI] {art.quest_id} profile -> {trace_path}")
    return art


# ---- fleet path -----------------------------------------------------------


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


async def _await_under_cap(cap_mb: int, *, poll_s: float = 2.0) -> None:
    """Block until process RSS is under the cap. No-op if psutil missing."""
    while True:
        rss = _rss_mb()
        if rss is None or rss < cap_mb:
            return
        print(f"[FI fleet] memory {rss:.0f}MB >= cap {cap_mb}MB; waiting...")
        await asyncio.sleep(poll_s)


async def run_fleet(
    configs: list[Config] | list[tuple[Path, Config]],
    *,
    supervisor: ProxySupervisor,
    max_concurrent: int,
    memory_cap_mb: int | None,
    profile: bool,
) -> int:
    # Accept either bare configs (test-friendly) OR (yaml_path, config)
    # tuples (production: lets us drop config.yaml into each quest dir).
    # Normalize to the tuple form.
    norm: list[tuple[Path | None, Config]] = [
        (None, c) if isinstance(c, Config) else (c[0], c[1])  # type: ignore[arg-type]
        for c in configs
    ]
    sem = asyncio.Semaphore(max_concurrent)
    total = len(norm)
    state = {"done": 0, "failed": 0, "running": 0}
    started_at = time.monotonic()

    def _status_line(quest_id: str, event: str) -> None:
        elapsed = int(time.monotonic() - started_at)
        rss = _rss_mb()
        rss_str = f" rss={rss:.0f}MB" if rss is not None else ""
        print(
            f"[FI fleet] {event} {quest_id} "
            f"running={state['running']} done={state['done']}/{total} "
            f"failed={state['failed']} elapsed={elapsed}s{rss_str}"
        )

    async def gated(yaml_path: Path | None, cfg: Config) -> dict[str, object] | Exception:
        async with sem:
            # Memory cap is checked at actual start (after semaphore admit),
            # not at entry — otherwise queued tasks could pass the early
            # check and start later when RSS has grown past the cap.
            if memory_cap_mb is not None:
                await _await_under_cap(memory_cap_mb)
            state["running"] += 1
            # Construct the Engine ONCE and reuse it in run_one so the
            # status-line quest_id matches the quest that actually runs.
            # Previously a second Engine (with a fresh quest_id) was
            # built inside run_one, leaving a stranded sibling quest_dir
            # on disk and breaking per-quest accounting.
            engine = Engine(cfg, supervisor=supervisor)
            _status_line(engine.quest_id, "start")
            try:
                summary = await run_one(
                    cfg, supervisor=supervisor, profile=profile, engine=engine,
                    source_yaml_path=yaml_path,
                )
                state["running"] -= 1
                state["done"] += 1
                _status_line(str(summary.get("quest_id")), "done ")
                return summary
            except Exception as e:
                state["running"] -= 1
                state["failed"] += 1
                _status_line(engine.quest_id, "FAIL ")
                return e

    results = await asyncio.gather(*(gated(p, c) for p, c in norm))
    failed = [r for r in results if isinstance(r, Exception)]
    for r in failed:
        print(f"[FI fleet] FAILURE: {r!r}", file=sys.stderr)
    return 1 if failed else 0


# ---- entry ---------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    # Ensure Axon sidecar is up before anything that touches the
    # knowledge layer. Idempotent: returns fast if already running.
    # Skipped for the install-tectonic / list-drafts modes that
    # never touch Axon, plus when --no-axon-sidecar is passed.
    _axon_inert_modes = bool(
        getattr(args, "install_tectonic", False)
        or getattr(args, "list_drafts", False)
    )
    if not args.no_axon_sidecar and not _axon_inert_modes:
        from core.axon_sidecar import ensure_axon_up
        ensure_axon_up()
    supervisor = ProxySupervisor()
    try:
        if args.serve:
            # We're already inside an event loop (main_async), so use
            # the async helper rather than the blocking uvicorn.run.
            # Forward --max-concurrent + --vscode-bridge-port so the
            # web UI's subprocess launcher inherits the same caps +
            # transport the rest of FI was started with.
            from web.server import serve_async
            await serve_async(
                output_root=args.output_root,
                host=args.host,
                port=args.port,
                max_concurrent=args.max_concurrent,
                vscode_bridge_port=args.vscode_bridge_port,
            )
            return 0

        if args.ingest:
            return _ingest_papers(args.ingest, axon_config_path=args.axon_config)

        if args.install_tectonic:
            return _install_tectonic()

        if args.list_drafts:
            return _list_drafts(args.output_root)

        if args.new:
            return await _run_new(
                output_root=args.output_root,
                draft_only=args.draft_only,
                vscode_bridge_port=args.vscode_bridge_port,
                interactive=args.interactive,
                supervisor=supervisor,
            )

        if args.update:
            # Phase 3 — see core/interview_update.py for the
            # invalidation-matrix wiring. The launch-side dispatch
            # is implemented in _run_update below.
            return await _run_update(
                quest_id=args.update,
                output_root=args.output_root,
                vscode_bridge_port=args.vscode_bridge_port,
                interactive=args.interactive,
                supervisor=supervisor,
            )

        if args.digest:
            return await _run_digest(
                output_root=args.output_root,
                days=args.days,
                provider_name=args.digest_provider,
                axon_config_path=args.axon_config,
                vscode_bridge_port=args.vscode_bridge_port,
                supervisor=supervisor,
            )

        if args.portfolio:
            return await _run_portfolio(
                output_root=args.output_root,
                provider_name=args.portfolio_provider,
                axon_config_path=args.axon_config,
                vscode_bridge_port=args.vscode_bridge_port,
                supervisor=supervisor,
            )

        if args.critique:
            return await _run_critique(
                quest_id=args.critique,
                output_root=args.output_root,
                provider_name=args.critique_provider,
                axon_config_path=args.axon_config,
                vscode_bridge_port=args.vscode_bridge_port,
                supervisor=supervisor,
                ensemble_models=args.critique_ensemble,
                ensemble_merge=args.critique_ensemble_merge,
                ensemble_moderator=args.critique_ensemble_moderator,
            )

        if args.proposal:
            return await _run_proposal(
                topic=args.proposal,
                output_root=args.output_root,
                provider_name=args.proposal_provider,
                axon_config_path=args.axon_config,
                vscode_bridge_port=args.vscode_bridge_port,
                supervisor=supervisor,
                ensemble_models=args.proposal_ensemble,
                ensemble_merge=args.proposal_ensemble_merge,
                ensemble_moderator=args.proposal_ensemble_moderator,
            )

        if args.analyze:
            return await _run_analyze(
                data_path=args.analyze.resolve(),
                topic=args.analyze_topic,
                output_root=args.output_root,
                provider_name=args.analyze_provider,
                vscode_bridge_port=args.vscode_bridge_port,
                supervisor=supervisor,
                profile=args.profile,
            )

        if args.summarize:
            return await _run_summarize(
                folder=args.summarize.resolve(),
                kind=args.summarize_kind,
                provider_name=args.summarize_provider,
                output_root=args.output_root,
                axon_config_path=args.axon_config,
                vscode_bridge_port=args.vscode_bridge_port,
                supervisor=supervisor,
            )

        if args.config:
            cfg = Config.from_yaml(args.config)
            if args.output is not None:
                cfg.output.output_dir = args.output
            _apply_vscode_bridge_override(cfg, args.vscode_bridge_port)
            if args.resume:
                resume_err = _validate_resume_quest_id(
                    args.resume, cfg.output.output_dir,
                )
                if resume_err is not None:
                    print(f"[FI] {resume_err}", file=sys.stderr)
                    return 1
            await run_one(
                cfg, supervisor=supervisor, profile=args.profile,
                interactive=args.interactive,
                resume_quest_id=args.resume,
                source_yaml_path=args.config.resolve(),
            )
            return 0

        configs = [(p.resolve(), Config.from_yaml(p)) for p in args.fleet]
        for _, c in configs:
            _apply_vscode_bridge_override(c, args.vscode_bridge_port)
        return await run_fleet(
            configs,
            supervisor=supervisor,
            max_concurrent=args.max_concurrent,
            memory_cap_mb=args.memory_cap_mb,
            profile=args.profile,
        )
    finally:
        await supervisor.shutdown()


async def _run_summarize(
    *,
    folder: Path,
    kind: str,
    provider_name: str,
    output_root: Path,
    axon_config_path: Path | None,
    vscode_bridge_port: int,
    supervisor: ProxySupervisor,
) -> int:
    """Top-level wiring for ``python launch.py --summarize <folder>``.

    Two responsibilities: (1) build the provider config the same way
    a quest would (so the user gets the same VSCode-bridge / CLI /
    HTTP plumbing), and (2) open an Axon brain so the summarizer can
    ingest input docs + the produced summary. Returns 0 on success,
    1 on any failure that prevented writing the summary file.
    """
    from core.config import ProviderConfig as _ProviderConfig, KnowledgeConfig
    from core.summarizer import summarize_folder

    if not folder.is_dir():
        print(f"[FI] --summarize: not a directory: {folder}", file=sys.stderr)
        return 1

    provider = _ProviderConfig(name=provider_name)  # type: ignore[arg-type]
    # When launched from the VSCode extension the bridge port is set;
    # honor it the same way quests do.
    if vscode_bridge_port > 0:
        provider.name = "vscode_extension"
        provider.extra = {**(provider.extra or {}), "bridge_port": vscode_bridge_port}

    # Best-effort Knowledge handle. The summarizer tolerates
    # `knowledge=None` (or a disabled Knowledge) so an Axon-less
    # environment still produces summary.md; only the ingest step
    # is skipped.
    knowledge = None
    try:
        from core.knowledge import Knowledge
        knowledge = Knowledge(KnowledgeConfig(
            enabled=True,
            axon_config=axon_config_path if axon_config_path else None,
            seed_source_catalog=False,
        ))
    except Exception as e:  # pragma: no cover — Axon may not be installed
        print(f"[FI] --summarize: Axon unavailable ({e!r}); summary will still be written.", file=sys.stderr)

    print(f"[FI] summarize {folder} (kind={kind}, provider={provider.name})")
    try:
        art = await summarize_folder(
            folder, provider=provider, output_dir=output_root,
            supervisor=supervisor, kind=kind, knowledge=knowledge,
        )
    except Exception as e:
        print(f"[FI] --summarize failed: {e!r}", file=sys.stderr)
        return 1

    print(f"[FI] summary -> {art.summary_path}")
    print(
        f"[FI] {art.file_count} files; detected_kind={art.detected_kind}; "
        f"ingested_to_axon={art.ingested_to_axon}",
    )
    return 0


async def _run_digest(
    *,
    output_root: Path,
    days: int,
    provider_name: str,
    axon_config_path: Path | None,
    vscode_bridge_port: int,
    supervisor: ProxySupervisor,
) -> int:
    """Top-level wiring for ``python launch.py --digest``.

    Mirrors ``_run_summarize`` — builds the provider config the same
    way a quest would, opens a best-effort Axon handle for ingest,
    and dispatches to :func:`core.digest.generate_digest`. Returns 0
    on success, 1 if the digest call raises.
    """
    from core.config import ProviderConfig as _ProviderConfig, KnowledgeConfig
    from core.digest import generate_digest

    if days <= 0:
        print(f"[FI] --digest: --days must be positive (got {days})", file=sys.stderr)
        return 1

    provider = _ProviderConfig(name=provider_name)  # type: ignore[arg-type]
    if vscode_bridge_port > 0:
        provider.name = "vscode_extension"
        provider.extra = {**(provider.extra or {}), "bridge_port": vscode_bridge_port}

    knowledge = None
    try:
        from core.knowledge import Knowledge
        knowledge = Knowledge(KnowledgeConfig(
            enabled=True,
            axon_config=axon_config_path if axon_config_path else None,
            seed_source_catalog=False,
        ))
    except Exception as e:  # pragma: no cover
        print(
            f"[FI] --digest: Axon unavailable ({e!r}); digest will still be "
            f"written but not ingested.",
            file=sys.stderr,
        )

    print(f"[FI] digest window=last {days} days (provider={provider.name})")
    try:
        art = await generate_digest(
            output_root, days=days, provider=provider,
            supervisor=supervisor, knowledge=knowledge,
        )
    except Exception as e:
        print(f"[FI] --digest failed: {e!r}", file=sys.stderr)
        return 1

    print(f"[FI] digest -> {art.digest_path}")
    print(
        f"[FI] {art.quest_count} quests touched "
        f"({art.completed_count} completed, "
        f"{art.in_progress_count} in-progress); "
        f"ingested_to_axon={art.ingested_to_axon}",
    )
    if art.diff.prev_digest_id:
        print(
            f"[FI] vs {art.diff.prev_digest_id}: "
            f"{len(art.diff.promoted)} promoted, "
            f"{len(art.diff.new_in_progress)} new, "
            f"{len(art.diff.still_in_progress)} still-in-progress, "
            f"{len(art.diff.stalled)} stalled, "
            f"{len(art.diff.dropped)} dropped",
        )
    return 0


async def _run_portfolio(
    *,
    output_root: Path,
    provider_name: str,
    axon_config_path: Path | None,
    vscode_bridge_port: int,
    supervisor: ProxySupervisor,
) -> int:
    """Top-level wiring for ``python launch.py --portfolio``.

    Same provider + Axon plumbing as ``_run_digest``; dispatches to
    :func:`core.portfolio.generate_portfolio`. Returns 0 on success,
    1 on any failure that prevented writing the portfolio file.
    """
    from core.config import ProviderConfig as _ProviderConfig, KnowledgeConfig
    from core.portfolio import generate_portfolio

    provider = _ProviderConfig(name=provider_name)  # type: ignore[arg-type]
    if vscode_bridge_port > 0:
        provider.name = "vscode_extension"
        provider.extra = {**(provider.extra or {}), "bridge_port": vscode_bridge_port}

    knowledge = None
    try:
        from core.knowledge import Knowledge
        knowledge = Knowledge(KnowledgeConfig(
            enabled=True,
            axon_config=axon_config_path if axon_config_path else None,
            seed_source_catalog=False,
        ))
    except Exception as e:  # pragma: no cover
        print(
            f"[FI] --portfolio: Axon unavailable ({e!r}); portfolio will "
            f"still be written but not ingested.",
            file=sys.stderr,
        )

    print(f"[FI] portfolio synthesis (provider={provider.name})")
    try:
        art = await generate_portfolio(
            output_root, provider=provider,
            supervisor=supervisor, knowledge=knowledge,
        )
    except Exception as e:
        print(f"[FI] --portfolio failed: {e!r}", file=sys.stderr)
        return 1

    print(f"[FI] portfolio -> {art.portfolio_path}")
    print(
        f"[FI] {art.completed_count + art.in_progress_count} quests; "
        f"{art.completed_count} completed, "
        f"{art.in_progress_count} in-progress; "
        f"ingested_to_axon={art.ingested_to_axon}",
    )
    return 0


def _parse_ensemble_flag(
    models_csv: str, merge: str, moderator: str,
) -> "NodeEnsembleConfig | None":  # type: ignore[name-defined]
    """Convert ``--{proposal,critique}-ensemble M1,M2,M3`` flag triple
    into a NodeEnsembleConfig, or None when the flag was omitted.

    Empty / whitespace-only strings → None (no ensemble; default
    single-call path). Otherwise split on comma, strip, drop empties.
    """
    csv = (models_csv or "").strip()
    if not csv:
        return None
    models = [m.strip() for m in csv.split(",") if m.strip()]
    if not models:
        return None
    from core.config import NodeEnsembleConfig
    return NodeEnsembleConfig(
        models=models,
        merge=merge,  # type: ignore[arg-type]
        moderator=(moderator.strip() or None),
    )


async def _run_critique(
    *,
    quest_id: str,
    output_root: Path,
    provider_name: str,
    axon_config_path: Path | None,
    vscode_bridge_port: int,
    supervisor: ProxySupervisor,
    ensemble_models: str = "",
    ensemble_merge: str = "synthesize",
    ensemble_moderator: str = "",
) -> int:
    """Top-level wiring for ``python launch.py --critique <quest_id>``.

    Same provider + Axon plumbing as the other PM commands; dispatches
    to :func:`core.critique.generate_critique`. Returns 0 on success,
    1 on any failure that prevented writing the critique file.
    """
    from core.config import ProviderConfig as _ProviderConfig, KnowledgeConfig
    from core.critique import generate_critique

    provider = _ProviderConfig(name=provider_name)  # type: ignore[arg-type]
    if vscode_bridge_port > 0:
        provider.name = "vscode_extension"
        provider.extra = {**(provider.extra or {}), "bridge_port": vscode_bridge_port}

    knowledge = None
    try:
        from core.knowledge import Knowledge
        knowledge = Knowledge(KnowledgeConfig(
            enabled=True,
            axon_config=axon_config_path if axon_config_path else None,
            seed_source_catalog=False,
        ))
    except Exception as e:  # pragma: no cover
        print(
            f"[FI] --critique: Axon unavailable ({e!r}); critique will still "
            f"be written but not ingested.",
            file=sys.stderr,
        )

    ensemble = _parse_ensemble_flag(
        ensemble_models, ensemble_merge, ensemble_moderator,
    )
    if ensemble is not None:
        print(f"[FI] critique {quest_id} (provider={provider.name}; "
              f"ensemble: {len(ensemble.models)} critics, merge={ensemble.merge})")
    else:
        print(f"[FI] critique {quest_id} (provider={provider.name})")
    try:
        art = await generate_critique(
            quest_id, output_root, provider=provider,
            supervisor=supervisor, knowledge=knowledge,
            ensemble=ensemble,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[FI] --critique: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[FI] --critique failed: {e!r}", file=sys.stderr)
        return 1

    print(f"[FI] critique -> {art.critique_path}")
    print(
        f"[FI] critique_provider={art.critique_provider}; "
        f"ingested_to_axon={art.ingested_to_axon}",
    )
    return 0


async def _run_analyze(
    *,
    data_path: Path,
    topic: str,
    output_root: Path,
    provider_name: str,
    vscode_bridge_port: int,
    supervisor: ProxySupervisor,
    profile: bool = False,
) -> int:
    """Top-level wiring for ``python launch.py --analyze <data_path>
    --analyze-topic "..."``.

    Builds a no-simulation Config, stages the user's data into the
    quest's ``data/`` directory, then runs the engine through
    ``run_one``. The engine's no-simulation path (auto_collect_data
    passthrough → wait_for_data passthrough since data exists →
    data_load → analyze → write → review) handles the rest."""
    from core.analyze_cli import prepare_analyze_quest

    if vscode_bridge_port > 0:
        provider_name = "vscode_extension"

    try:
        cfg, quest_id, files_staged = prepare_analyze_quest(
            data_path=data_path, topic=topic, output_root=output_root,
            provider_name=provider_name,
        )
    except ValueError as e:
        print(f"[FI] --analyze: {e}", file=sys.stderr)
        return 1

    if files_staged == 0:
        print(
            f"[FI] --analyze: no files copied from {data_path} (empty "
            f"directory, or every file matched the skip list). Aborting.",
            file=sys.stderr,
        )
        return 1

    _apply_vscode_bridge_override(cfg, vscode_bridge_port)

    print(
        f"[FI] --analyze: quest_id={quest_id} "
        f"files_staged={files_staged} provider={cfg.provider.name}",
    )
    # Pass the minted quest_id through as ``resume_quest_id`` so the
    # Engine reuses it instead of generating a fresh one. Without
    # this, the data we just staged into
    # ``<output_root>/<quest_id>/data/`` would be orphaned —
    # ``Engine`` would mint its own quest_id, write to a different
    # quest_root, and pause for "missing" data files. The resume
    # gate inside Engine.run treats a quest_id with no prior
    # checkpoint as a brand-new quest with that pre-specified id,
    # which is exactly the contract this path needs.
    await run_one(
        cfg, supervisor=supervisor, profile=profile,
        resume_quest_id=quest_id,
    )
    return 0


async def _run_new(
    *,
    output_root: Path,
    draft_only: bool,
    vscode_bridge_port: int,
    interactive: bool,
    supervisor: ProxySupervisor,
) -> int:
    """``python launch.py --new`` — interactive interview frontend.

    Walks the user through the question set defined in
    :mod:`core.interview`, calls the preflight clarify LLM once (with
    a visible spinner) for topic-tuned defaults, writes the resulting
    YAML to ``outputs/_drafts/<stamp>-<title>.yaml``, then (unless
    ``--draft-only``) immediately starts the quest via the existing
    ``--config`` path.

    Returns 0 on a completed interview, 0 on quest success, 1 on any
    user-cancellation or quest failure. Stdin EOF + Ctrl-C count as
    cancellation, not failure — the quest just doesn't start.
    """
    # Only the symbols _run_new actually uses. _cli_prompt_for
    # imports its own (smart_defaults + model_choices_for).
    from core.interview import (
        QUESTIONS, InterviewAnswers, answers_to_yaml,
        derive_tier2, derive_tier3, preflight_clarify,
        questions_for_tier, slugify,
    )

    # Choice labels include em-dashes / arrows / Unicode math. Windows
    # consoles default to cp1252 which can't encode them; force stdout
    # + stderr to UTF-8 so the prompts render and don't crash with
    # ``UnicodeEncodeError: 'charmap' codec can't encode character``.
    # No-op on POSIX (already UTF-8).
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        # Older Pythons / unusual stream wrappers — fall through;
        # the print calls below will still error on a non-UTF8
        # console, but at least we tried.
        pass

    print("=" * 72)
    print("Frontier Insight — set up a new research quest")
    print("Press Ctrl-C at any prompt to cancel.")
    print("=" * 72)
    print()

    partial: dict[str, object] = {}
    frontend = "cli"

    tier1 = questions_for_tier(1, frontend)
    tier2_qs = questions_for_tier(2, frontend)
    tier3_qs = questions_for_tier(3, frontend)

    # ---- Stage 1: ask tier-1 sequentially ----
    try:
        for q in tier1:
            answer = _cli_prompt_for(q, partial, {})
            if answer is None:
                print()
                print("— interview cancelled.")
                return 1
            partial[q.id] = answer
    except (KeyboardInterrupt, EOFError):
        print()
        print("— interview cancelled (Ctrl-C / EOF).")
        return 1

    # ---- Stage 2: preflight LLM + derive tier-2 + tier-3 ----
    print()
    print("⏳ Calling clarify-preflight to suggest topic-tuned defaults...")
    try:
        preflight_cache = await preflight_clarify(
            topic=str(partial.get("topic", "")),
            paper_format=str(partial.get("paper_format", "generic")),
            provider_name=str(partial.get("provider") or "openai"),
            provider_model=partial.get("provider_model"),  # type: ignore[arg-type]
        )
        print("✓ done")
    except Exception as e:
        print(f"⚠ preflight failed ({e}); proceeding with empty advanced defaults.")
        preflight_cache = {}
    print()

    derived = derive_tier2(partial)
    advanced = derive_tier3(partial)
    # Overlay preflight-suggested values onto tier-3 placeholders so
    # the user sees the LLM's suggestions in the review screen.
    for k in ("comparative_baseline", "success_metric", "budget"):
        if preflight_cache.get(k):
            advanced[k] = preflight_cache[k]

    # ---- Stage 3: review-and-edit loop ----
    show_advanced = False
    try:
        while True:
            rows = _build_review_rows(derived, advanced, tier2_qs, tier3_qs, show_advanced)
            _print_review(rows, show_advanced=show_advanced)
            choice = input(
                "\nEdit which? [number to edit, "
                + ("'h' to hide" if show_advanced else "'a' to show advanced")
                + ", Enter to launch] > "
            ).strip().lower()
            if choice == "":
                break
            if choice == "a" and not show_advanced:
                show_advanced = True
                continue
            if choice == "h" and show_advanced:
                show_advanced = False
                continue
            if not choice.isdigit():
                print(f"  ⚠ unrecognized input {choice!r}; press Enter or pick a number.")
                continue
            idx = int(choice) - 1
            if idx < 0 or idx >= len(rows):
                print(f"  ⚠ number out of range; pick 1-{len(rows)}.")
                continue
            row = rows[idx]
            new_val = _cli_prompt_for(row["question"], {**partial, **derived, **advanced}, preflight_cache)
            if new_val is None:
                continue  # user aborted single-field edit; stay in review
            if row["tier"] == 2:
                derived[row["id"]] = new_val
            else:
                advanced[row["id"]] = new_val
    except (KeyboardInterrupt, EOFError):
        print()
        print("— interview cancelled at review screen (Ctrl-C / EOF).")
        return 1

    # ---- Build final answers ----
    answers = InterviewAnswers(
        topic=str(partial["topic"]),
        title=str(derived["title"]),
        output_kinds=list(partial["output_kinds"]),  # type: ignore[arg-type]
        paper_format=str(partial["paper_format"]),
        no_simulation=bool(derived["no_simulation"]),
        study_depth=str(partial["study_depth"]),
        comparative_baseline=str(advanced.get("comparative_baseline") or ""),
        success_metric=str(advanced.get("success_metric") or ""),
        budget=str(advanced.get("budget") or ""),
        clarify_mode=str(derived["clarify_mode"]),
        review_panel=list(derived["review_panel"]),  # type: ignore[arg-type]
        knowledge_enabled=bool(derived["knowledge_enabled"]),
        provider=str(partial.get("provider", "openai")),
        provider_model=str(partial.get("provider_model")) if partial.get("provider_model") else None,
        audience=str(derived.get("audience", "external")),
        knowledge_top_k=int(derived.get("knowledge_top_k", 8) or 8),
        knowledge_external_top_k=int(advanced.get("knowledge_external_top_k", 20) or 20),
        ensemble_profile=str(advanced.get("ensemble_profile") or "off"),
        max_iterations=int(advanced.get("max_iterations", 2) or 2),
    )

    yaml_text = answers_to_yaml(answers, frontend="cli")
    drafts_dir = output_root / "_drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d-%H%M")
    yaml_path = drafts_dir / f"{stamp}-{slugify(answers.title)}.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print()
    print("=" * 72)
    print(f"YAML written: {yaml_path}")
    print("=" * 72)

    if draft_only:
        print()
        print("--draft-only: not launching the quest. Run:")
        print(f"  python launch.py --config {yaml_path}")
        return 0

    # Launch the quest using the existing --config code path
    # (``run_one``). The interview already pinned clarify_overrides
    # in the YAML, so the engine skips its own clarify LLM call.
    print()
    print("Launching quest...")
    cfg = Config.from_yaml(yaml_path)
    _apply_vscode_bridge_override(cfg, vscode_bridge_port)
    try:
        await run_one(
            cfg,
            supervisor=supervisor,
            profile=False,
            interactive=interactive,
            source_yaml_path=yaml_path.resolve(),
        )
        return 0
    except Exception as e:
        print(f"[FI] quest failed: {e!r}", file=sys.stderr)
        return 1


def _build_review_rows(
    derived: dict[str, object],
    advanced: dict[str, object],
    tier2_qs: tuple,  # type: ignore[type-arg]
    tier3_qs: tuple,  # type: ignore[type-arg]
    show_advanced: bool,
) -> list[dict[str, object]]:
    """Pack tier-2 (always) + tier-3 (when ``show_advanced``) into a
    numbered list the review screen prints. Each row carries the
    Question so the edit handler can re-prompt that exact slot."""
    rows: list[dict[str, object]] = []
    for q in tier2_qs:
        rows.append({
            "id": q.id, "tier": 2, "label": q.label,
            "value": derived.get(q.id), "question": q,
        })
    if show_advanced:
        for q in tier3_qs:
            rows.append({
                "id": q.id, "tier": 3, "label": q.label,
                "value": advanced.get(q.id), "question": q,
            })
    return rows


def _print_review(rows: list[dict[str, object]], *, show_advanced: bool) -> None:
    """Render the numbered review table. Tier-2 rows always show;
    tier-3 rows appear under an "Advanced" subheading when the user
    toggled them on with `a`."""
    print()
    print("─── Review before launch ──────────────────────────────")
    in_advanced_section = False
    for i, r in enumerate(rows, 1):
        if r["tier"] == 3 and not in_advanced_section:
            print()
            print("  ── Advanced (preflight-LLM suggested) ──")
            in_advanced_section = True
        val = r["value"]
        if isinstance(val, list):
            val_str = f"[{', '.join(str(v) for v in val)}]" if val else "(none)"
        elif val == "":
            val_str = "(empty)"
        else:
            val_str = str(val)
        print(f"  {i:>2}. {r['label']:<28} {val_str}")
    if not show_advanced:
        print()
        print("      (type 'a' to show advanced fields)")


def _cli_prompt_for(
    q,  # type: ignore[no-untyped-def]
    partial: dict[str, object],
    preflight: dict[str, str],
) -> object | None:
    """Render one interview question on stdout and read the answer
    from stdin. Returns the parsed answer (already coerced to the
    question's value type), or None when the user cancels.

    For ``single``-kind questions we print a numbered choice list and
    accept either the number or a unique prefix of the label. The
    default (if any) is shown in parentheses and accepted on blank
    input. Smart defaults are recomputed each call so cascades work
    (paper_format flowing into no_simulation, etc).
    """
    from core.interview import (
        Choice, build_smart_defaults, model_choices_for,
    )

    # Recompute smart defaults given everything answered so far.
    smart = build_smart_defaults(partial)
    default = smart.get(q.id, q.default)

    # Preflight overrides for the three topic-tuned slots — they live
    # in preflight, not the smart_defaults table.
    if q.id in ("comparative_baseline", "success_metric", "budget"):
        if preflight.get(q.id):
            default = preflight[q.id]

    # Provider-model is a derived question — choices depend on the
    # already-picked provider. Build the choice list at render time.
    if q.id == "provider_model":
        provider = partial.get("provider")
        choices = model_choices_for(str(provider)) if provider else ()
    else:
        choices = q.choices

    print()
    print(f"  [{q.label}] {q.prompt}")
    if q.placeholder:
        print(f"      example: {q.placeholder}")

    if q.kind == "text":
        default_str = "" if default in (None, "") else str(default)
        prompt_str = f"    answer ({default_str}): " if default_str else "    answer: "
        try:
            raw = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return raw or default_str or ""

    # single-select (or single-with-allow_other for provider_model)
    if not choices:
        # Fall back to free text — happens when allow_other and no
        # curated list (e.g. unknown provider).
        try:
            raw = input("    answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return raw

    # Render the choice list.
    for i, c in enumerate(choices, start=1):
        marker = " (default)" if c.value == default else ""
        print(f"    {i}. {c.label}{marker}")
        if c.description:
            print(f"        {c.description}")
    if q.allow_other:
        print(f"    {len(choices) + 1}. Other (type your own)")

    while True:
        try:
            raw = input(f"    select 1-{len(choices) + (1 if q.allow_other else 0)} (or label prefix): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw and default is not None:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1].value
            if q.allow_other and idx == len(choices) + 1:
                try:
                    custom = input("    type the value: ").strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                return custom or None
            print(f"    (not a valid choice — pick 1 to {len(choices) + (1 if q.allow_other else 0)})")
            continue
        # Match by unique label prefix (case-insensitive).
        lower = raw.lower()
        matches = [c for c in choices if c.label.lower().startswith(lower)]
        if len(matches) == 1:
            return matches[0].value
        if len(matches) > 1:
            print(f"    (ambiguous — matches {[c.label for c in matches]})")
            continue
        print(f"    (no match for '{raw}' — pick a number or a unique prefix)")


async def _run_update(
    *,
    quest_id: str,
    output_root: Path,
    vscode_bridge_port: int,
    interactive: bool,
    supervisor: ProxySupervisor,
) -> int:
    """``python launch.py --update <quest_id>`` — mid-quest re-entry.

    Loads the existing quest's ``config.yaml``, runs the interview
    pre-filled with current values (and limited to the editable
    field subset), computes a diff, invalidates affected LangGraph
    stages, writes the updated YAML, then resumes the quest via the
    existing ``--resume`` path. Implementation in
    :mod:`core.interview_update`.
    """
    from core.interview_update import run_update_flow
    return await run_update_flow(
        quest_id=quest_id,
        output_root=output_root,
        vscode_bridge_port=vscode_bridge_port,
        interactive=interactive,
        supervisor=supervisor,
        run_one=run_one,
        apply_vscode_bridge_override=_apply_vscode_bridge_override,
    )


async def _run_proposal(
    *,
    topic: str,
    output_root: Path,
    provider_name: str,
    axon_config_path: Path | None,
    vscode_bridge_port: int,
    supervisor: ProxySupervisor,
    ensemble_models: str = "",
    ensemble_merge: str = "tournament",
    ensemble_moderator: str = "",
) -> int:
    """Top-level wiring for ``python launch.py --proposal "<topic>"``.

    Returns 0 on success, 1 on validation or LLM failure.
    """
    from core.config import ProviderConfig as _ProviderConfig, KnowledgeConfig
    from core.proposal import generate_proposal

    provider = _ProviderConfig(name=provider_name)  # type: ignore[arg-type]
    if vscode_bridge_port > 0:
        provider.name = "vscode_extension"
        provider.extra = {**(provider.extra or {}), "bridge_port": vscode_bridge_port}

    knowledge = None
    try:
        from core.knowledge import Knowledge
        knowledge = Knowledge(KnowledgeConfig(
            enabled=True,
            axon_config=axon_config_path if axon_config_path else None,
            seed_source_catalog=False,
        ))
    except Exception as e:  # pragma: no cover
        print(
            f"[FI] --proposal: Axon unavailable ({e!r}); proposal will still "
            f"be written but not ingested.",
            file=sys.stderr,
        )

    # Build optional ensemble config from CLI flags.
    ensemble = _parse_ensemble_flag(
        ensemble_models, ensemble_merge, ensemble_moderator,
    )
    if ensemble is not None:
        print(f"[FI] proposal (provider={provider.name}; "
              f"ensemble: {len(ensemble.models)} models, merge={ensemble.merge})")
    else:
        print(f"[FI] proposal (provider={provider.name})")
    try:
        art = await generate_proposal(
            topic, output_root, provider=provider,
            supervisor=supervisor, knowledge=knowledge,
            ensemble=ensemble,
        )
    except ValueError as e:
        print(f"[FI] --proposal: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[FI] --proposal failed: {e!r}", file=sys.stderr)
        return 1

    print(f"[FI] proposal -> {art.proposal_path}")
    print(f"[FI] companion YAML -> {art.yaml_path}")
    print(f"[FI] ingested_to_axon={art.ingested_to_axon}")
    print(
        f"[FI] To run the quest: python launch.py --config "
        f"{art.yaml_path}",
    )
    return 0


# Tectonic LaTeX engine — single self-bootstrapping binary used as a
# fallback to MiKTeX/TeX Live for users who can't install LaTeX
# without admin rights. The release tag is the only version pin we
# carry; the SHA-256 of each platform archive is fetched from the
# official `SHA256SUMS` file in the same release. That removes the
# staleness problem of baking per-asset hashes into source: upgrading
# tectonic is a one-line `_TECTONIC_VERSION` bump.
_TECTONIC_VERSION = "0.16.9"
# Map (sys.platform, normalized_machine_arch) → tectonic release-asset
# filename. Used by `_install_tectonic` to pick the right download.
_TECTONIC_ASSET_NAMES: dict[tuple[str, str], str] = {
    ("win32", "AMD64"): f"tectonic-{_TECTONIC_VERSION}-x86_64-pc-windows-msvc.zip",
    ("darwin", "arm64"): f"tectonic-{_TECTONIC_VERSION}-aarch64-apple-darwin.tar.gz",
    ("darwin", "x86_64"): f"tectonic-{_TECTONIC_VERSION}-x86_64-apple-darwin.tar.gz",
    ("linux", "x86_64"): f"tectonic-{_TECTONIC_VERSION}-x86_64-unknown-linux-musl.tar.gz",
}


def _install_tectonic() -> int:
    """Download the tectonic LaTeX binary for the current OS+arch into
    ``<repo_root>/tools/`` so paper_pdf works without an admin install
    of MiKTeX.

    Verification: fetches the release's official ``SHA256SUMS`` file
    over HTTPS and compares against the archive. Two pieces have to
    line up — the archive AND the checksum file — and both arrive
    over HTTPS, so a MITM would have to break TLS in both. Removes
    the need to bake per-asset hashes into source.

    Tectonic is a single Rust binary (~70 MB) that downloads required
    CTAN packages on first run into
    ``%LOCALAPPDATA%/TectonicProject/Tectonic/``. No GUI, no admin,
    no install step. The ``_find_pdf_engine`` helper in
    ``generation/paper.py`` picks it up automatically when pdflatex
    isn't on PATH.

    Returns 0 on success, 1 on any failure (network, checksum
    mismatch, write error, unsupported platform).
    """
    import hashlib
    import platform
    import tarfile
    import tempfile
    import urllib.error
    import urllib.request
    import zipfile

    repo_root = Path(__file__).resolve().parent
    tools_dir = repo_root / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "tectonic.exe" if sys.platform == "win32" else "tectonic"
    dest = tools_dir / exe_name
    if dest.is_file():
        print(f"[FI] tectonic already installed at {dest}")
        return 0

    arch = platform.machine()
    # Normalize the few common machine-arch spellings to the keys our
    # asset table uses.
    arch_norm = {
        "aarch64": "arm64",
        "AMD64": "AMD64",
        "x86_64": "x86_64",
        "arm64": "arm64",
    }.get(arch, arch)
    key = (sys.platform, arch_norm)
    if key not in _TECTONIC_ASSET_NAMES:
        print(
            f"[FI] --install-tectonic: unsupported platform "
            f"({sys.platform}/{arch}). Manual download from "
            f"github.com/tectonic-typesetting/tectonic/releases.",
            file=sys.stderr,
        )
        return 1
    asset_name = _TECTONIC_ASSET_NAMES[key]
    base_url = (
        f"https://github.com/tectonic-typesetting/tectonic/releases/"
        f"download/tectonic@{_TECTONIC_VERSION}"
    )
    archive_url = f"{base_url}/{asset_name}"
    sums_url = f"{base_url}/SHA256SUMS"

    print(f"[FI] downloading {archive_url}")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / asset_name
            with urllib.request.urlopen(archive_url, timeout=180) as r:
                archive.write_bytes(r.read())

            # Verification: tectonic releases publish per-asset
            # archives but NOT a SHA256SUMS aggregation file — only
            # the .zip / .tar.gz / .AppImage assets are present in
            # the release. Fetching a SHA256SUMS file would 404 and
            # bail the install before extracting the binary.
            #
            # We fall back to TLS-only verification: github.com's
            # certificate authenticates the asset. That's the same
            # trust model `pip` and most package managers use for
            # downloads without a separate checksum channel. If
            # tectonic starts publishing SHA256SUMS in a future
            # release, re-enable the strict path below.
            try:
                with urllib.request.urlopen(sums_url, timeout=10) as r:
                    sums_text = r.read().decode("utf-8", errors="replace")
                expected: str | None = None
                for line in sums_text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and parts[-1].endswith(asset_name):
                        expected = parts[0]
                        break
                if expected is not None:
                    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
                    if actual.lower() != expected.lower():
                        print(
                            f"[FI] --install-tectonic: SHA-256 mismatch.\n"
                            f"  expected: {expected}\n"
                            f"  got:      {actual}\n"
                            f"  Aborting — possible network corruption or MITM.",
                            file=sys.stderr,
                        )
                        return 1
                    print("[FI] SHA-256 verification passed")
                else:
                    print(
                        f"[FI] note: {asset_name} not listed in "
                        f"SHA256SUMS — proceeding with TLS-only trust.",
                    )
            except urllib.error.HTTPError as sums_err:
                # 404 is the expected case today; just log and proceed.
                print(
                    f"[FI] note: SHA256SUMS unavailable ({sums_err}); "
                    f"proceeding with TLS-only trust. github.com's "
                    f"certificate authenticates the asset.",
                )

            # Extract just the `tectonic` / `tectonic.exe` binary; the
            # archives also contain LICENSE and a README we don't need.
            # Write to a sibling temp file in tools/ first, then
            # ``os.replace`` it to the final dest. ``os.replace`` is
            # atomic on a single filesystem on both Windows and POSIX,
            # so a Ctrl-C / disk-full during extraction can't leave a
            # half-written ``tectonic`` that future runs would happily
            # exec. Sibling-not-tmpdir because cross-device replace
            # falls back to copy on some setups and breaks atomicity.
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=f".{exe_name}.partial-", dir=tools_dir,
            )
            os.close(tmp_fd)
            tmp_dest = Path(tmp_path)
            try:
                if asset_name.endswith(".zip"):
                    with zipfile.ZipFile(archive) as zf:
                        for member in zf.namelist():
                            if member.endswith(exe_name):
                                with zf.open(member) as src, open(tmp_dest, "wb") as dst:
                                    dst.write(src.read())
                                break
                        else:
                            print("[FI] --install-tectonic: archive missing tectonic binary", file=sys.stderr)
                            return 1
                elif asset_name.endswith((".tar.gz", ".tgz")):
                    with tarfile.open(archive, "r:gz") as tf:
                        for member in tf.getmembers():
                            if member.name.endswith(exe_name):
                                extracted = tf.extractfile(member)
                                if extracted is None:
                                    continue
                                tmp_dest.write_bytes(extracted.read())
                                break
                        else:
                            print("[FI] --install-tectonic: archive missing tectonic binary", file=sys.stderr)
                            return 1
                if sys.platform != "win32":
                    tmp_dest.chmod(0o755)
                os.replace(tmp_dest, dest)
            finally:
                if tmp_dest.exists():
                    try:
                        tmp_dest.unlink()
                    except OSError:
                        pass
    except (urllib.error.URLError, OSError) as e:
        print(f"[FI] --install-tectonic: download failed: {e!r}", file=sys.stderr)
        return 1

    print(f"[FI] tectonic installed at {dest}")
    print(
        "[FI] On next paper_pdf run, FI will auto-detect tectonic when "
        "pdflatex isn't on PATH. First compile downloads CTAN packages "
        "(~30 s, one-time)."
    )
    return 0


def _ingest_papers(paths: list[Path], *, axon_config_path: Path | None) -> int:
    """Permanently ingest paper files into Axon (kind=fi_local_paper)
    outside of any quest. Useful before launching a new quest that
    should benefit from manually-downloaded paywalled PDFs."""
    from core.config import KnowledgeConfig
    from core.knowledge import Knowledge

    cfg = KnowledgeConfig(
        enabled=True,
        axon_config=axon_config_path if axon_config_path else None,
        local_papers=list(paths),
        # Avoid re-seeding the source catalog every time we run --ingest.
        seed_source_catalog=False,
    )
    try:
        k = Knowledge(cfg)
    except Exception as e:
        print(f"[FI ingest] Axon init failed: {e}", file=sys.stderr)
        return 1
    if not k.enabled:
        print(
            "[FI ingest] Axon not available (is the `axon` package installed?). "
            "Files were parsed but not ingested into a persistent store.",
            file=sys.stderr,
        )
        return 1
    loaded = [d.metadata["filename"] for d in k._local_papers]
    if not loaded:
        print("[FI ingest] no files loaded — check paths / file types.", file=sys.stderr)
        return 1
    print(f"[FI ingest] ingested {len(loaded)} file(s) into Axon: {loaded}")
    return 0


def _list_drafts(output_root: Path) -> int:
    """Top-level handler for ``python launch.py --list-drafts``.

    Prints a table of proposal YAMLs in ``outputs/_drafts/`` so the
    user can pick one to feed into ``--config`` without remembering
    paths. Mirrors the Web `/api/drafts` + /interview drafts picker
    and the VSCode `@fi /drafts` chat command.
    """
    drafts_dir = output_root / "_drafts"
    if not drafts_dir.is_dir():
        print(f"[FI] no drafts directory at {drafts_dir}", file=sys.stderr)
        print("    Run `--proposal \"<topic>\"` to create one.")
        return 0
    files = sorted(
        drafts_dir.glob("*.yaml"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not files:
        print(f"[FI] no proposal drafts in {drafts_dir}")
        print("    Run `--proposal \"<topic>\"` to create one.")
        return 0
    print(f"[FI] {len(files)} proposal draft(s) in {drafts_dir}:\n")
    import yaml as _yaml
    for p in files[:40]:
        try:
            doc = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            doc = {}
        title = (doc.get("title") or "").strip() or "(no title)"
        topic = (doc.get("topic") or "").strip().replace("\n", " ")
        topic_short = topic[:80] + ("…" if len(topic) > 80 else "")
        mtime = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(p.stat().st_mtime),
        )
        print(f"  {mtime}  {title}")
        if topic_short:
            print(f"              {topic_short}")
        print(f"              python launch.py --config {p}")
        print()
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        # Ctrl-C on a long-running mode (--serve, --fleet, a single
        # quest mid-LLM-call) cleanly cancels uvicorn / the LangGraph
        # loop. The asyncio runner then re-raises KeyboardInterrupt
        # at the top level, which without this catch dumps a
        # CancelledError-then-KeyboardInterrupt traceback on the
        # user's screen — looks like a crash even though it's a
        # cooperative shutdown. Exit code 130 is the conventional
        # "terminated by SIGINT" status on POSIX; Windows callers
        # treat any non-zero as "interrupted".
        print()  # newline so the prompt doesn't fight with `^C`
        print("[FI] interrupted (Ctrl-C). Goodbye.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
