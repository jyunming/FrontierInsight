"""Recording that a quest used a skill — the return leg of the loop.

Selection ranks partly on a skill's track record and distillation needs to
know which quests a skill took part in, so neither works unless use is
written down.

Three properties are load-bearing, and the last one cannot be shown by a
single-threaded test:

**Writing must not lapse approval.** ``provenance.json`` is outside the
content hash precisely so bookkeeping does not look like the skill changing.

**Nothing here may fail a quest.** By the time this runs the quest has already
produced an accepted paper.

**``--fleet`` writes concurrently.** A dropped record is invisible — the count
is simply lower than the truth — so the lock is verified with real processes.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core.skills import Skill, Status, evaluate
from core.skills import approval
from core.skills.base import PROVENANCE_JSON, SKILL_MD
from core.skills import usage


@pytest.fixture
def skill(tmp_path: Path) -> Skill:
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / SKILL_MD).write_text("# demo\n\nUse it.\n", encoding="utf-8")
    (d / "selftest.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    (d / PROVENANCE_JSON).write_text(
        json.dumps({"origin": "authored", "taught_by_projects": []}), encoding="utf-8"
    )
    return Skill(name="demo", path=d)


def history(skill: Skill) -> list[dict]:
    return json.loads(
        (skill.path / PROVENANCE_JSON).read_text(encoding="utf-8")
    )["taught_by_projects"]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_a_use_is_appended_with_quest_and_outcome(skill: Skill) -> None:
    assert usage.record_use(skill.path, "quest-1", "accept") is True
    got = history(skill)
    assert len(got) == 1
    assert got[0]["quest"] == "quest-1" and got[0]["outcome"] == "accept"
    assert got[0]["at"]


def test_multiple_quests_accumulate(skill: Skill) -> None:
    for q in ("q1", "q2", "q3"):
        usage.record_use(skill.path, q, "accept")
    assert [h["quest"] for h in history(skill)] == ["q1", "q2", "q3"]


def test_the_same_quest_twice_updates_rather_than_duplicates(skill: Skill) -> None:
    """A resumed quest must not inflate a skill's track record."""
    usage.record_use(skill.path, "q1", "revise")
    usage.record_use(skill.path, "q1", "accept")
    got = history(skill)
    assert len(got) == 1 and got[0]["outcome"] == "accept"


def test_other_provenance_fields_survive(skill: Skill) -> None:
    usage.record_use(skill.path, "q1", "accept")
    data = json.loads((skill.path / PROVENANCE_JSON).read_text(encoding="utf-8"))
    assert data["origin"] == "authored"


def test_missing_provenance_is_created(tmp_path: Path) -> None:
    d = tmp_path / "fresh"
    d.mkdir()
    assert usage.record_use(d, "q1", "accept") is True
    assert json.loads((d / PROVENANCE_JSON).read_text("utf-8"))["taught_by_projects"]


def test_corrupt_provenance_does_not_raise(skill: Skill) -> None:
    (skill.path / PROVENANCE_JSON).write_text("{not json", encoding="utf-8")
    assert usage.record_use(skill.path, "q1", "accept") is True
    assert len(history(skill)) == 1


# ---------------------------------------------------------------------------
# The property the whole design rests on
# ---------------------------------------------------------------------------


def test_recording_use_does_not_lapse_approval(skill: Skill, tmp_path: Path) -> None:
    """`provenance.json` is outside the content hash for exactly this reason.

    If bookkeeping lapsed approval, every accepted quest would silently
    un-trust the skill it just used.
    """
    ledger = tmp_path / "approvals.json"
    approval.approve("demo", skill.content_hash(), approved_by="tester", path=ledger)
    assert evaluate(skill, ledger=ledger).status is Status.TRUSTED

    usage.record_use(skill.path, "q1", "accept")

    assert evaluate(skill, ledger=ledger).status is Status.TRUSTED
    assert len(history(skill)) == 1


def test_the_lock_file_is_not_written_into_the_skill(skill: Skill) -> None:
    """The lock must live outside the skill, on every platform.

    ``filelock`` removes its lock file on release on Windows but leaves it
    behind on POSIX, so a lock beside ``provenance.json`` only lapsed
    approval on Linux — green on the developer's machine, red in CI. Assert
    on the location rather than on the leftover, so the property holds
    wherever the test runs.
    """
    provenance = skill.path / PROVENANCE_JSON
    lock = usage._lock_path(provenance)
    assert skill.path not in lock.parents

    before = skill.content_hash()
    assert usage.record_use(skill.path, "q1", "accept") is True
    assert sorted(p.name for p in skill.path.rglob("*.lock")) == []
    assert skill.content_hash() == before


# ---------------------------------------------------------------------------
# Concurrency — real processes, because a lock cannot be tested in one
# ---------------------------------------------------------------------------


def test_parallel_writers_lose_no_records(skill: Skill, tmp_path: Path) -> None:
    """`--fleet` is the primary production invocation, so several quests
    finishing with one skill at once is the normal case, not an edge one.

    A last-writer-wins update would drop records silently: the count would
    simply be lower than the truth, with nothing to notice.
    """
    script = tmp_path / "writer.py"
    script.write_text(
        textwrap.dedent(f'''
            import sys
            sys.path.insert(0, {str(Path.cwd())!r})
            from pathlib import Path
            from core.skills import usage
            usage.record_use(Path({str(skill.path)!r}), sys.argv[1], "accept")
        '''),
        encoding="utf-8",
    )
    procs = [
        subprocess.Popen([sys.executable, str(script), f"q{i}"])
        for i in range(8)
    ]
    for p in procs:
        assert p.wait(timeout=120) == 0

    quests = {h["quest"] for h in history(skill)}
    assert quests == {f"q{i}" for i in range(8)}, (
        f"records were lost under concurrency: {sorted(quests)}"
    )


# ---------------------------------------------------------------------------
# record_quest
# ---------------------------------------------------------------------------


def test_record_quest_writes_every_named_skill(skill: Skill, monkeypatch) -> None:
    monkeypatch.setenv("FI_SKILLS_DIR", str(skill.path.parent))
    assert usage.record_quest(["demo"], "q9", "accept") == ["demo"]
    assert history(skill)[0]["quest"] == "q9"


def test_record_quest_skips_a_skill_that_vanished(skill: Skill, monkeypatch) -> None:
    monkeypatch.setenv("FI_SKILLS_DIR", str(skill.path.parent))
    assert usage.record_quest(["demo", "gone"], "q9", "accept") == ["demo"]


def test_record_quest_with_no_skills_is_a_no_op() -> None:
    assert usage.record_quest([], "q9", "accept") == []


def test_registry_failure_never_raises(monkeypatch) -> None:
    import core.skills.registry as reg

    monkeypatch.setattr(
        reg, "discover", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert usage.record_quest(["demo"], "q9", "accept") == []


# ---------------------------------------------------------------------------
# Engine gating
# ---------------------------------------------------------------------------


def _engine(quest_id: str = "q1"):
    from types import SimpleNamespace

    logs: list = []
    return SimpleNamespace(
        quest_id=quest_id,
        _log=SimpleNamespace(
            info=lambda *a, **k: logs.append(("info", a)),
            warning=lambda *a, **k: logs.append(("warn", a)),
        ),
        logs=logs,
    )


def test_engine_records_only_on_accept(skill: Skill, monkeypatch) -> None:
    from core.engine import Engine

    monkeypatch.setenv("FI_SKILLS_DIR", str(skill.path.parent))
    eng = _engine()
    Engine._record_skill_usage(  # type: ignore[arg-type]
        eng, {"selected_skills": ["demo"], "review": {"verdict": "accept"}},
    )
    assert len(history(skill)) == 1


@pytest.mark.parametrize("verdict", ["revise", "reject", ""])
def test_engine_does_not_record_a_rejected_quest(
    skill: Skill, monkeypatch, verdict: str
) -> None:
    """A rejected quest says nothing about the skill — only about that attempt."""
    from core.engine import Engine

    monkeypatch.setenv("FI_SKILLS_DIR", str(skill.path.parent))
    Engine._record_skill_usage(  # type: ignore[arg-type]
        _engine(), {"selected_skills": ["demo"], "review": {"verdict": verdict}},
    )
    assert history(skill) == []


def test_engine_no_skills_selected_is_a_no_op(skill: Skill, monkeypatch) -> None:
    from core.engine import Engine

    monkeypatch.setenv("FI_SKILLS_DIR", str(skill.path.parent))
    Engine._record_skill_usage(  # type: ignore[arg-type]
        _engine(), {"selected_skills": [], "review": {"verdict": "accept"}},
    )
    assert history(skill) == []


def test_engine_swallows_a_bookkeeping_failure(monkeypatch) -> None:
    """The quest has already produced an accepted paper by this point."""
    from core.engine import Engine
    import core.skills.usage as usage_mod

    monkeypatch.setattr(
        usage_mod, "record_quest",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk on fire")),
    )
    eng = _engine()
    Engine._record_skill_usage(  # type: ignore[arg-type]
        eng, {"selected_skills": ["demo"], "review": {"verdict": "accept"}},
    )
    assert any(kind == "warn" for kind, _ in eng.logs)
