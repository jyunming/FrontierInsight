"""Frontier Insight — orchestrator entry point.

Single quest:
    python launch.py --config examples/integrator_bakeoff/config.yaml

Fleet (Phase H):
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
        help="Start the Phase J status GUI (FastAPI server + HTMX frontend). "
             "Use --output-root to point at the quest output directory, "
             "and --host / --port to bind elsewhere than 127.0.0.1:8765. "
             "Requires FastAPI + uvicorn installed.",
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
        help="Phase P — TCP port the FI VSCode extension is listening on for "
             "the LLM bridge. The extension passes this when it spawns FI; "
             "setting it forces provider.name=vscode_extension regardless of "
             "what the YAML says. Do not pass this from a regular terminal "
             "run — use copilot_cli or similar for headless Copilot usage.",
    )
    args = p.parse_args(argv)
    if args.fleet and args.output is not None:
        p.error("--output cannot be combined with --fleet (per-quest output_dir comes from each YAML).")
    if args.ingest and args.output is not None:
        p.error("--output is irrelevant in --ingest mode.")
    if args.resume and not args.config:
        p.error("--resume requires --config (the YAML for the original quest).")
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
    """Phase P — when launched with ``--vscode-bridge-port N``, force
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


async def _run_generators(
    cfg: Config,
    art: QuestArtifacts,
    *,
    supervisor: ProxySupervisor,
) -> dict[str, Path]:
    """Run each generator in turn; one failure does not abort the rest."""
    written: dict[str, Path] = {}
    # 1) Paper (sync — pandoc shell-out is fine without async).
    try:
        written.update(PaperGenerator(cfg).generate(art, art.quest_root))
    except Exception as e:  # pragma: no cover — defensive
        print(f"[FI] paper generator failed: {e!r}", file=sys.stderr)
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
    supervisor = ProxySupervisor()
    try:
        if args.serve:
            # We're already inside an event loop (main_async), so use
            # the async helper rather than the blocking uvicorn.run.
            from web.server import serve_async
            await serve_async(
                output_root=args.output_root, host=args.host, port=args.port,
            )
            return 0

        if args.ingest:
            return _ingest_papers(args.ingest, axon_config_path=args.axon_config)

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


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
