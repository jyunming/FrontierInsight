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
    args = p.parse_args(argv)
    if args.fleet and args.output is not None:
        p.error("--output cannot be combined with --fleet (per-quest output_dir comes from each YAML).")
    if args.ingest and args.output is not None:
        p.error("--output is irrelevant in --ingest mode.")
    return args


# ---- single-quest path ----------------------------------------------------


async def run_one(
    cfg: Config,
    *,
    supervisor: ProxySupervisor,
    profile: bool = False,
    engine: Engine | None = None,
) -> dict[str, object]:
    # Engine may be constructed by the caller (e.g. `gated()` builds it
    # once so the status-line `quest_id` matches the quest that actually
    # runs — instead of creating a second Engine here with a fresh
    # quest_id and stranding the status-line one). When omitted, build.
    if engine is None:
        engine = Engine(cfg, supervisor=supervisor)
    print(f"[FI] start quest_id={engine.quest_id} provider={cfg.provider.name}")
    art: QuestArtifacts = await _maybe_profiled(engine, profile=profile)
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


async def _maybe_profiled(engine: Engine, *, profile: bool) -> QuestArtifacts:
    if not profile:
        return await engine.run()
    try:
        from viztracer import VizTracer  # type: ignore[import-not-found]
    except ImportError:
        print("[FI] --profile requested but viztracer is not installed; skipping")
        return await engine.run()
    trace_path = engine.fi_dir / "profile.json"
    engine.fi_dir.mkdir(parents=True, exist_ok=True)
    with VizTracer(output_file=str(trace_path)):
        art = await engine.run()
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
    configs: list[Config],
    *,
    supervisor: ProxySupervisor,
    max_concurrent: int,
    memory_cap_mb: int | None,
    profile: bool,
) -> int:
    sem = asyncio.Semaphore(max_concurrent)
    total = len(configs)
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

    async def gated(cfg: Config) -> dict[str, object] | Exception:
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

    results = await asyncio.gather(*(gated(c) for c in configs))
    failed = [r for r in results if isinstance(r, Exception)]
    for r in failed:
        print(f"[FI fleet] FAILURE: {r!r}", file=sys.stderr)
    return 1 if failed else 0


# ---- entry ---------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    supervisor = ProxySupervisor()
    try:
        if args.ingest:
            return _ingest_papers(args.ingest, axon_config_path=args.axon_config)

        if args.config:
            cfg = Config.from_yaml(args.config)
            if args.output is not None:
                cfg.output.output_dir = args.output
            await run_one(cfg, supervisor=supervisor, profile=args.profile)
            return 0

        configs = [Config.from_yaml(p) for p in args.fleet]
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
