"""A quest that stopped without findings (needs/STUCK.json) makes no paper, slides or poster through the launch path,
and a rerun or a refine that goes on from it leaves no stale stop behind."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core import todo
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _classify, _fake_response_for

UNREADABLE = "I looked at the results but cannot put them in the shape asked for"


def _cfg(tmp_path: Path, **engine: object) -> Config:
    return Config(
        topic="a topic for the stop without findings", title="no-findings", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, oracle_check="off", **engine),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _chat(readable: list[bool]):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Analysis" and not readable[0]:
            return UNREADABLE
        return _fake_response_for(prompt)

    return fake_chat


@pytest.mark.asyncio
async def test_the_launch_path_makes_no_paper_slides_or_poster_after_the_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import launch
    from core.provider import ProxySupervisor

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr("core.engine.LLMClient.chat", _chat([False]))
    cfg = _cfg(tmp_path)
    cfg.output.kinds = ["paper_md", "paper_pdf", "slides", "poster"]
    eng = Engine(cfg)
    art = await eng.run()
    assert art.paper_md is None and (eng.quest_root / "needs" / "STUCK.json").is_file()
    # An old paper left on disk by an earlier round is no more used than the one the run would write.
    (eng.quest_root / "paper").mkdir(exist_ok=True)
    (eng.quest_root / "paper" / "paper.md").write_text("# an old paper\n", encoding="utf-8")
    assert eng._collect_artifacts({}).paper_md is None

    summary = await launch._finish_outputs(cfg, art, supervisor=ProxySupervisor())
    assert summary["outputs"] == {} and summary["paper_md"] is None
    root = eng.quest_root
    for name in ("paper.pdf", "slides.md", "slides.html", "slides.pdf", "poster.pdf", "poster.html", "poster.md"):
        assert not (root / name).exists(), name
    out = capsys.readouterr().out
    assert "no paper, slides or poster were made" in out and "needs/STUCK.json" in out


@pytest.mark.asyncio
async def test_a_rerun_from_the_analysis_moves_the_stop_aside_and_its_card_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    readable = [False]
    monkeypatch.setattr("core.engine.LLMClient.chat", _chat(readable))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    assert any(i.kind == "stuck" for i in todo.waiting(first.quest_root))

    readable[0] = True
    second = Engine(cfg, resume_quest_id=first.quest_id)
    art = await second.run(from_step="analysis")
    root = second.quest_root
    assert not (root / "needs" / "STUCK.json").exists()
    assert not any(i.kind == "stuck" for i in todo.waiting(root))
    kept = list((root / ".fi" / "previous").glob("*/needs/STUCK.json"))
    assert len(kept) == 1  # kept for the history
    assert art.paper_md is not None and art.paper_md.exists()  # the stop does not hold the paper back


@pytest.mark.asyncio
async def test_a_refine_that_reopens_the_quest_moves_the_stop_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    readable = [False]
    monkeypatch.setattr("core.engine.LLMClient.chat", _chat(readable))
    cfg = _cfg(tmp_path)
    eng = Engine(cfg)
    await eng.run()
    assert (eng.quest_root / "needs" / "STUCK.json").is_file()

    readable[0] = True

    async def accept(snapshot: dict) -> dict:
        return {"action": "accept", "answer": "yes"}

    art = await asyncio.wait_for(eng.run(reopen=True, human_feedback_callback=accept), timeout=600)
    assert not (eng.quest_root / "needs" / "STUCK.json").exists()
    assert len(list((eng.fi_dir / "set_aside_by_stuck").glob("*/STUCK.json"))) == 1
    assert not any(i.kind == "stuck" for i in todo.waiting(eng.quest_root))
    assert art.paper_md is not None and art.paper_md.exists()
