"""Resume capability: Engine(resume_quest_id=...) reuses the prior
quest_id (and therefore quest_root and state.sqlite), so a quest that
died mid-pipeline on a transient upstream outage can be re-entered
without losing checkpointed work.

We test the small, deterministic bits here: id reuse and quest_root
reuse. The actual LangGraph state-resume mechanism is owned by
LangGraph's AsyncSqliteSaver — we just have to pass the same thread_id.
"""

from __future__ import annotations

from pathlib import Path

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="resume unit",
        title="resume-unit",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(),
        execution=ExecutionConfig(),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def test_engine_without_resume_generates_fresh_quest_id(tmp_path: Path) -> None:
    e1 = Engine(_cfg(tmp_path))
    e2 = Engine(_cfg(tmp_path))
    assert e1.quest_id != e2.quest_id


def test_engine_with_resume_quest_id_reuses_id_and_root(tmp_path: Path) -> None:
    prior_id = "1700000000-resume-unit-deadbe"
    e = Engine(_cfg(tmp_path), resume_quest_id=prior_id)
    assert e.quest_id == prior_id
    assert e.quest_root.name == prior_id
    # state.sqlite path is what LangGraph keys checkpoints by; reuse
    # is the whole point.
    assert (e.fi_dir / "state.sqlite").parent == e.quest_root / ".fi"


def test_engine_resume_two_engines_same_id_share_checkpoint_path(
    tmp_path: Path,
) -> None:
    """The contract: two Engine instances built with the same
    resume_quest_id resolve to the same on-disk checkpoint. That is
    sufficient for LangGraph's AsyncSqliteSaver to load the prior
    state when ainvoke is called with thread_id=quest_id."""
    qid = "1700000000-resume-unit-cafe11"
    a = Engine(_cfg(tmp_path), resume_quest_id=qid)
    b = Engine(_cfg(tmp_path), resume_quest_id=qid)
    assert a.fi_dir / "state.sqlite" == b.fi_dir / "state.sqlite"
