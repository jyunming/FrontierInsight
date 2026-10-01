"""Running a benchmark quest: recorded, copied and run again from a step, or replayed.

A benchmark run lives in its own folder::

    <run>/quest/<quest_id>/   the quest itself (FI's folder, as any quest's)
    <run>/bench/              the benchmark's own files, kept out of the quest's sealed records:
        config.json             the config the quest ran with
        calls.jsonl             every model call of this run (the recording a full replay reads)
        replay_events.jsonl     planted and unrecorded calls (core/replay.py)
        crossref.json           the recorded retraction lookups
        plant.json              what was planted (plant.py)
        run.json                how the run went: mode, stops, time

The model client is swapped by :class:`BenchEngine` (an ``Engine`` whose ``_connect_llm`` wraps the real client in a
:class:`core.replay.ReplayClient`); nothing in the engine changes. One replay client serves a run's every resume, so a
step called on both sides of a pause numbers its calls on.

Benchmark settings (:data:`BENCH_SETTINGS`): explore-then-confirm is off, because a quest run again from a step after its
confirm run was seen is never more than "statistically adequate" (``confirm_reused``), which would make every forked
run look caught; a passing review is accepted automatically (no person), so "would be published" is the evidence level
one below publication-ready with no-one's-accept as its only gap; every model call is kept; nothing is written back to
the knowledge store. The plan pause is answered the way a person who approves the plan unchanged would.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from core.config import Config
from core.engine import Engine
from core.replay import CrossrefReplay, Recording, ReplayClient

#: What every benchmark quest runs with (see the module docstring).
BENCH_SETTINGS: dict[str, Any] = {
    "engine": {"phased": False, "auto_accept_on_pass": True},
    # Only the paper: slides and a poster call a model of their own, outside any replay.
    "output": {"save_model_calls": True, "kinds": ["paper_md"]},
    "knowledge": {"write_back_quests": False},
    "pauses": {"papers": False, "review": "off"},
}
PLAN_PAUSE = "read and edit the plan"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def bench_config(config: Config | dict[str, Any], *, output_dir: Path, ask: str = "",
                 extra: dict[str, Any] | None = None) -> Config:
    """``config`` with the benchmark's settings, the task's ``ask`` sentence added to the topic, and ``extra``."""
    data = config.model_dump(mode="json") if isinstance(config, Config) else dict(config)
    data = _merge(data, BENCH_SETTINGS)
    if ask and ask.strip() not in str(data.get("topic") or ""):
        data["topic"] = str(data.get("topic") or "").rstrip() + "\n\n" + ask.strip() + "\n"
    data = _merge(data, extra or {})
    # After the models' settings: a kind of output made outside the engine calls a model no replay answers.
    data = _merge(data, {"output": {"output_dir": str(output_dir), "kinds": BENCH_SETTINGS["output"]["kinds"]}})
    return Config.model_validate(data)


class BenchEngine(Engine):
    """An engine whose model calls go through a :class:`ReplayClient` (``replay``), one shared by every resume."""

    def __init__(self, config: Config, *, replay: ReplayClient, extra_hits: list[dict[str, Any]] | None = None,
                 **kw: Any) -> None:
        super().__init__(config, **kw)
        self._bench_replay = replay
        if extra_hits:
            self._add_search_hits(extra_hits)

    def _add_search_hits(self, hits: list[dict[str, Any]]) -> None:
        """Every literature search of this run also returns ``hits`` (a planted source, plant.py::plant_l1)."""
        from core.knowledge import RetrievedDoc

        search = self.knowledge.asearch

        async def asearch(*a: Any, **kw: Any) -> list[Any]:
            found = list(await search(*a, **kw) or [])
            return found + [RetrievedDoc(content=h["content"], metadata=dict(h["metadata"])) for h in hits]

        self.knowledge.asearch = asearch  # type: ignore[method-assign]

    async def _connect_llm(self) -> None:
        replay = self._bench_replay
        if replay.mode != "replay":
            await super()._connect_llm()
            replay.real = self._client
        else:
            # Nothing reaches a model. The connection named the model that answered when the recorded run's did.
            self._reports_model = any(bool(c.get("reported")) for c in replay.recording.calls.values())
            self._log.info("[bench] every model call is answered from the recording (%d calls)", len(replay.recording))
        self._client = replay


def paths(run_dir: Path) -> tuple[Path, Path]:
    """``(output_dir, bench_dir)`` of a benchmark run folder."""
    run_dir = Path(run_dir)
    return run_dir / "quest", run_dir / "bench"


def quest_root(run_dir: Path) -> Path:
    """The one quest folder inside a run folder."""
    out, _bench = paths(run_dir)
    found = [p for p in out.iterdir() if p.is_dir()] if out.is_dir() else []
    if len(found) != 1:
        raise FileNotFoundError(f"{out} should hold exactly one quest folder, it holds {len(found)}")
    return found[0]


def _stop_text(root: Path) -> str:
    for name in ("NEXT_STEP.md", "quest_failed.md"):
        path = root / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


async def run(config: Config, run_dir: Path, *, mode: str, recording: Recording | None = None,
              plants: dict[tuple[str, int], str] | None = None, quest_id: str | None = None,
              from_step: str | None = None, approved_by: str | None = None,
              crossref: CrossrefReplay | None = None, extra_hits: list[dict[str, Any]] | None = None,
              passes: int = 5) -> dict[str, Any]:
    """Run (or resume, or rerun from ``from_step``) the quest in ``run_dir`` with model calls served as ``mode`` says,
    past the plan pause, until it finishes or stops anywhere else. ``extra_hits``: sources every literature search
    also returns (a planted one). Returns the run's ``run.json``."""
    out_dir, bench_dir = paths(run_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "config.json").write_text(json.dumps(config.model_dump(mode="json"), indent=1), encoding="utf-8")
    replay = ReplayClient(mode=mode, recording=recording, plants=plants, out_dir=bench_dir,
                          real=None if mode == "replay" else _Deferred())
    started = time.time()
    stops: list[str] = []
    error = ""
    kw: dict[str, Any] = {"replay": replay, "extra_hits": extra_hits}
    engine = BenchEngine(config, resume_quest_id=quest_id, **kw) if quest_id else BenchEngine(config, **kw)
    with (crossref.installed() if crossref is not None else nullcontext()):
        try:
            await engine.run(from_step=from_step, approved_by=approved_by) if from_step else await engine.run()
            for _ in range(passes):
                text = _stop_text(engine.quest_root)
                if not text or not (engine.quest_root / "NEXT_STEP.md").is_file():
                    break
                headline = text.splitlines()[0] if text else ""
                if PLAN_PAUSE not in text:
                    stops.append(headline)
                    break
                (engine.quest_root / "NEXT_STEP.md").unlink()
                engine = BenchEngine(config, resume_quest_id=engine.quest_id, **kw)
                await engine.run()
        except Exception as e:  # noqa: BLE001 -- a quest that fails is an outcome the scorer reads, not a crash of the bench
            error = f"{type(e).__name__}: {e}"[:2000]
    # Every call this run asked for that its recording did not have: the model's, and Crossref's retraction lookups.
    from core.replay import read_events

    looked_up = sum(1 for e in read_events(bench_dir) if e.get("event") == "divergence" and e.get("node") == "retractions"
                    and float(e.get("ts") or 0) >= started)
    record = {
        "mode": mode, "quest_id": engine.quest_id, "from_step": from_step, "started_at": started,
        "seconds": round(time.time() - started, 1),
        "stops": stops, "error": error, "divergences": replay.divergences + looked_up,
        "planted_calls": [f"{n}#{i}" for (n, i) in (plants or {})],
    }
    (bench_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


class _Deferred:
    """Stands for the real client until the engine connects it (``BenchEngine._connect_llm``)."""

    async def chat(self, *a: Any, **kw: Any) -> str:  # pragma: no cover -- replaced before any call
        raise RuntimeError("the model client was not connected")


def run_sync(*args: Any, **kw: Any) -> dict[str, Any]:
    return asyncio.run(run(*args, **kw))


def fork(source_run: Path, dest_run: Path) -> Path:
    """A copy of a finished benchmark run's quest (its virtual environment left out: the copy makes its own) in a new
    run folder. Returns the copied quest folder."""
    src = quest_root(source_run)
    out_dir, bench_dir = paths(dest_run)
    if out_dir.exists():
        raise FileExistsError(f"{dest_run} already holds a quest")
    bench_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / src.name
    shutil.copytree(src, target, ignore=shutil.ignore_patterns(".venv"))
    _out, src_bench = paths(source_run)
    if (src_bench / "config.json").is_file():
        shutil.copy2(src_bench / "config.json", bench_dir / "source_config.json")
    return target


def config_of(run_dir: Path, *, source: bool = False) -> Config:
    """The config a run's quest ran with (``source``: the one it was copied from), pointed at this run's folder."""
    _out, bench_dir = paths(run_dir)
    data = json.loads((bench_dir / ("source_config.json" if source else "config.json")).read_text(encoding="utf-8"))
    data = _merge(data, {"output": {"output_dir": str(paths(run_dir)[0])}})
    return Config.model_validate(data)


async def fork_start_time(run_dir: Path, step: str) -> float | None:
    """When the checkpoint a rerun from ``step`` resumes from was taken (seconds since the epoch), read from the quest's
    checkpoint history: the calls of the source run after it are the ones a rerun from ``step`` makes again."""
    from datetime import datetime

    from core import rerun_from

    root = quest_root(run_dir)
    engine = Engine(config_of(run_dir), resume_quest_id=root.name)
    opened = await engine._open_readonly_graph()
    if opened is None:
        return None
    graph, conn = opened
    try:
        cfg = await rerun_from.checkpoint_before(graph, {"configurable": {"thread_id": root.name}}, step)
        if cfg is None:
            return None
        snap = await graph.aget_state(cfg)
        created = getattr(snap, "created_at", None)
        return datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp() if created else None
    finally:
        await conn.close()


def segment_after(recording_dir: Path, after: float) -> Recording:
    """The calls of a recorded run started after ``after``, numbered per step from 1 in the order they were asked: the
    recording a rerun from a step replays. Read from the run's own ``calls.jsonl`` (the whole answers, in the order they
    were asked); a quest recorded outside the benchmark falls back to the calls it kept in ``.fi/io/``."""
    calls = paths(recording_dir)[1] / "calls.jsonl"
    if calls.is_file():
        rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows and all(r.get("ts") for r in rows):  # an older recording has no start times: read the quest's own
            return Recording(rows).after(after)
    return Recording.from_quest(quest_root(recording_dir), after=after)
