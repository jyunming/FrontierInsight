"""Direct tests for launch.py — CLI parsing, generator failure isolation,
and fleet counter accounting. No real LLM calls; no real quest run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import launch
from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import QuestArtifacts


# ---- parse_args ----------------------------------------------------------


def test_parse_args_config_only() -> None:
    args = launch.parse_args(["--config", "x.yaml"])
    assert args.config == Path("x.yaml")
    assert args.fleet is None


def test_parse_args_fleet_multiple() -> None:
    args = launch.parse_args(["--fleet", "a.yaml", "b.yaml"])
    assert args.fleet == [Path("a.yaml"), Path("b.yaml")]
    assert args.config is None


def test_parse_args_no_mode_raises_systemexit() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args([])


def test_parse_args_config_and_fleet_mutually_exclusive() -> None:
    """Both --config and --fleet at once must be rejected."""
    with pytest.raises(SystemExit):
        launch.parse_args(["--config", "x.yaml", "--fleet", "a.yaml"])


def test_parse_args_profile_and_memory_cap() -> None:
    args = launch.parse_args(
        ["--config", "x.yaml", "--profile", "--memory-cap-mb", "2048"]
    )
    assert args.profile is True
    assert args.memory_cap_mb == 2048


def test_parse_args_resume_requires_config() -> None:
    """--resume without --config is rejected (fleet/serve/ingest can't resume)."""
    with pytest.raises(SystemExit):
        launch.parse_args(["--fleet", "a.yaml", "--resume", "some-quest-id"])


def test_parse_args_resume_with_config_ok() -> None:
    args = launch.parse_args(["--config", "x.yaml", "--resume", "q-123"])
    assert args.resume == "q-123"


# ---- _run_generators -----------------------------------------------------


def _make_cfg(tmp_path: Path) -> Config:
    return Config(
        topic="launch unit",
        title="launch-unit",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=30),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _make_artifacts(tmp_path: Path) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    quest_root.mkdir(parents=True, exist_ok=True)
    return QuestArtifacts(
        quest_id="qid-test",
        quest_root=quest_root,
        paper_md=quest_root / "paper" / "paper.md",
        figures_dir=quest_root / "figures",
    )


async def test_run_generators_continues_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One generator raising must not abort the other three."""
    cfg = _make_cfg(tmp_path)
    art = _make_artifacts(tmp_path)
    supervisor = MagicMock()

    def boom_paper(self, art, out_dir):  # noqa: ANN001
        raise RuntimeError("paper exploded")

    async def ok_slides(self, art, out_dir, *, supervisor):  # noqa: ANN001
        return {"slides_md": out_dir / "slides.md"}

    async def boom_poster(self, art, out_dir, *, supervisor):  # noqa: ANN001
        raise RuntimeError("poster exploded")

    async def ok_speech(self, art, out_dir, *, supervisor):  # noqa: ANN001
        return {"speech_md": out_dir / "speech.md"}

    monkeypatch.setattr("launch.PaperGenerator.generate", boom_paper)
    monkeypatch.setattr("launch.SlideGenerator.generate", ok_slides)
    monkeypatch.setattr("launch.PosterGenerator.generate", boom_poster)
    monkeypatch.setattr("launch.SpeechGenerator.generate", ok_speech)

    written = await launch._run_generators(cfg, art, supervisor=supervisor)

    # Both surviving generators contributed; failed ones did not abort the run.
    assert "slides_md" in written
    assert "speech_md" in written
    assert "paper_md" not in written  # paper raised
    assert "poster_md" not in written  # poster raised


# ---- _maybe_profiled -----------------------------------------------------


async def test_maybe_profiled_no_profile_calls_run_directly() -> None:
    engine = MagicMock()
    engine.run = AsyncMock(return_value="sentinel")
    result = await launch._maybe_profiled(engine, profile=False)
    assert result == "sentinel"
    engine.run.assert_awaited_once()


async def test_maybe_profiled_viztracer_missing_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If viztracer isn't installed, _maybe_profiled prints a warning and
    still runs the engine."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):  # noqa: ANN001
        if name == "viztracer":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    engine = MagicMock()
    engine.run = AsyncMock(return_value="sentinel-no-viz")
    result = await launch._maybe_profiled(engine, profile=True)
    assert result == "sentinel-no-viz"
    engine.run.assert_awaited_once()


# ---- _await_under_cap ----------------------------------------------------


async def test_await_under_cap_returns_immediately_when_psutil_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """psutil-missing path: _rss_mb returns None, _await_under_cap returns."""
    monkeypatch.setattr(launch, "_rss_mb", lambda: None)
    # Tight cap that would otherwise block; should still return immediately.
    await launch._await_under_cap(cap_mb=1, poll_s=0.01)


async def test_await_under_cap_returns_when_under_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launch, "_rss_mb", lambda: 100.0)
    await launch._await_under_cap(cap_mb=4096, poll_s=0.01)


# ---- run_fleet counters --------------------------------------------------


async def test_run_fleet_counters_at_end_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fake quests both succeed: done=2, failed=0, return code 0."""
    cfg_a = _make_cfg(tmp_path)
    cfg_b = _make_cfg(tmp_path)

    async def fake_run_one(cfg, *, supervisor, profile, engine=None):  # noqa: ANN001
        # Engine is now constructed in gated() and passed through — accept it.
        qid = engine.quest_id if engine is not None else "q-" + cfg.title
        return {"quest_id": qid, "ok": True}

    monkeypatch.setattr(launch, "run_one", fake_run_one)

    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *a, **kw: printed.append(" ".join(str(x) for x in a)),
    )

    rc = await launch.run_fleet(
        [cfg_a, cfg_b],
        supervisor=MagicMock(),
        max_concurrent=2,
        memory_cap_mb=None,
        profile=False,
    )
    assert rc == 0
    done_lines = [ln for ln in printed if "done " in ln and "/2" in ln]
    assert any("done=2/2" in ln for ln in done_lines)
    assert all("failed=0" in ln for ln in done_lines)


async def test_run_fleet_failure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One quest raises, the other succeeds: rc=1, failed counted, both ran."""
    cfg_a = _make_cfg(tmp_path)
    cfg_b = _make_cfg(tmp_path)

    # First call raises, subsequent calls succeed. max_concurrent=1
    # serializes the gather so order is deterministic.
    state = {"first": True}

    async def fake_run_one(cfg, *, supervisor, profile, engine=None):  # noqa: ANN001
        if state["first"]:
            state["first"] = False
            raise RuntimeError("first quest exploded")
        qid = engine.quest_id if engine is not None else "q-ok"
        return {"quest_id": qid, "ok": True}

    monkeypatch.setattr(launch, "run_one", fake_run_one)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    rc = await launch.run_fleet(
        [cfg_a, cfg_b],
        supervisor=MagicMock(),
        max_concurrent=1,
        memory_cap_mb=None,
        profile=False,
    )
    assert rc == 1
