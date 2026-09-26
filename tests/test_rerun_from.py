"""Rerunning a quest from a chosen step (core/rerun_from.py, ``--resume <id> --from <step>``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import rerun_from
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_EXPERIMENT_CODE, _classify, _fake_response_for


def test_step_names_are_plain_and_the_node_names_work_too() -> None:
    assert rerun_from.resolve("writing") == "writing" and rerun_from.resolve("write") == "writing"
    assert rerun_from.resolve("Code") == "code" and rerun_from.resolve("execute") == "run"
    assert rerun_from.resolve("design") is None and rerun_from.resolve("") is None
    assert rerun_from.choices() == "code, run, analysis, writing, review"


def test_the_backup_takes_what_the_step_and_later_ones_made_and_leaves_its_inputs(tmp_path: Path) -> None:
    for rel in ("paper/paper.md", "paper.pdf", "slides.pdf", "figures/a.png", "code/experiment.py", "plan.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x", encoding="utf-8")
    where, moved = rerun_from.back_up(tmp_path, "writing")
    assert set(moved) == {"paper", "paper.pdf", "slides.pdf"}
    assert where is not None and (where / "paper" / "paper.md").is_file() and (where / "paper.pdf").is_file()
    # The writing's inputs stay: the figures, the code and the plan.
    assert (tmp_path / "figures" / "a.png").is_file() and (tmp_path / "code" / "experiment.py").is_file()
    assert (tmp_path / "plan.md").is_file() and (tmp_path / "paper").is_dir()
    assert rerun_from.back_up(tmp_path, "review") == (None, [])


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="smoke-test topic for rerunning from a step",
        title="rerun-from",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_a_finished_quest_is_written_again_from_the_writing_without_running_the_code_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        calls.append(_classify(prompt))
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    artifacts = await first.run()
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    first_paper = artifacts.paper_md.read_text(encoding="utf-8")
    code_before = (first.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    implements = calls.count("Implementation")
    writes_before = len([c for c in calls if c == "Writing"])

    second = Engine(cfg, resume_quest_id=first.quest_id)
    again = await second.run(from_step="writing")

    assert calls.count("Implementation") == implements, "the code was written again"
    assert len([c for c in calls if c == "Writing"]) > writes_before, "the paper was not written again"
    assert again.paper_md is not None and again.paper_md.exists()
    assert (second.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == code_before
    previous = list((second.fi_dir / "previous").iterdir())
    assert len(previous) == 1 and (previous[0] / "paper" / "paper.md").read_text(encoding="utf-8") == first_paper
    assert "rerunning from the writing step" in (second.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_step_the_quest_never_reached_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str]) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    # Pretend the no-simulation path's first step is asked for on a quest that simulated: "run" looks for execute
    # first, so use a step whose nodes this quest never ran by removing them from the table for this test.
    monkeypatch.setitem(rerun_from.STEPS, "run", ("data_load",))
    second = Engine(cfg, resume_quest_id=first.quest_id)
    await second.run(from_step="run")
    assert "never reached the run step" in capsys.readouterr().out
    assert not (second.fi_dir / "previous").exists()


_ = (_FAKE_EXPERIMENT_CODE, Any)


class _Snap:
    def __init__(self, nxt: tuple[str, ...], n: int) -> None:
        self.next, self.config = nxt, {"configurable": {"checkpoint_id": str(n)}}


class _Graph:
    def __init__(self, nexts: list[tuple[str, ...]]) -> None:
        self.nexts = nexts  # oldest first, as the quest ran

    async def aget_state_history(self, _config):  # newest first, as LangGraph gives it
        for n, nxt in reversed(list(enumerate(self.nexts))):
            yield _Snap(nxt, n)


@pytest.mark.asyncio
async def test_a_step_of_several_nodes_is_done_again_from_its_first_node_of_the_latest_pass() -> None:
    # Two passes of a no-simulation quest; the second is paused at wait_for_data.
    graph = _Graph([("design",), ("auto_collect_data",), ("wait_for_data",), ("data_load",), ("analyze",),
                    ("review",), ("design",), ("auto_collect_data",), ("wait_for_data",)])
    found = await rerun_from.checkpoint_before(graph, {}, "run")
    assert found == {"configurable": {"checkpoint_id": "7"}}, "the second pass's collection, not the first pass's"
    graph = _Graph([("design",), ("implement_outline",), ("implement",), ("execute",), ("analyze",)])
    assert (await rerun_from.checkpoint_before(graph, {}, "code"))["configurable"]["checkpoint_id"] == "1"
    assert await rerun_from.checkpoint_before(graph, {}, "review") is None
    # A crashed script repaired and run again is still the one run step: it restarts at the first execute.
    graph = _Graph([("implement",), ("execute",), ("execute_reflect",), ("execute",), ("analyze",)])
    assert (await rerun_from.checkpoint_before(graph, {}, "run"))["configurable"]["checkpoint_id"] == "1"


@pytest.mark.asyncio
async def test_a_quest_waiting_at_the_review_is_written_again_from_the_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web offers Redo at an answer pause: the review's pending question is left behind, the paper is written
    again, and the quest comes back to the review."""
    from core.config import PausesConfig

    calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        calls.append(_classify(prompt))
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = _cfg(tmp_path)
    cfg.pauses = PausesConfig(review="ask")
    cfg.engine.auto_accept_on_pass = False
    first = Engine(cfg)
    await first.run()
    assert json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "review"
    writes = calls.count("Writing")
    second = Engine(cfg, resume_quest_id=first.quest_id)
    await second.run(from_step="writing")
    assert calls.count("Writing") > writes, "the paper was written again"
    assert json.loads((second.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "review", "back at the review"
    assert list((second.fi_dir / "previous").iterdir()), "the first paper is kept"
