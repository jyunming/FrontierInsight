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
    return p.parse_args(argv)


# ---- single-quest path ----------------------------------------------------


async def run_one(
    cfg: Config,
    *,
    supervisor: ProxySupervisor,
    profile: bool = False,
) -> dict[str, object]:
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
        if memory_cap_mb is not None:
            await _await_under_cap(memory_cap_mb)
        async with sem:
            state["running"] += 1
            engine = Engine(cfg, supervisor=supervisor)
            _status_line(engine.quest_id, "start")
            try:
                summary = await run_one(cfg, supervisor=supervisor, profile=profile)
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


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
