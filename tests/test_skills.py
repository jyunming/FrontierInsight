"""Tests for the skills subsystem — the unit of accumulated capability.

FI is moving from regenerating every experiment to keeping what it learned.
A skill is what it knows about driving one piece of scientific software, and
two gates stand between learning something and being allowed to use it: a
self-test the skill carries, and a person's approval.

The property most of these defend is that the *human* gate stays meaningful.
An approval that survives the skill being rewritten is a one-time rubber
stamp on a moving target, which is exactly the failure a learning system
invites.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from core.skills import Maturity, Skill, Status, discover, evaluate, loadable_skills
from core.skills import approval
from core.skills.base import SKILL_MD


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    """An isolated approval ledger. No test may touch the real one."""
    return tmp_path / "approvals.json"


def make_skill(
    root: Path,
    name: str = "demo",
    *,
    selftest: str | None = "import sys; sys.exit(0)",
    api_surface: bool = False,
    skill_py: bool = False,
    provenance: dict | None = None,
    scripts: dict[str, str] | None = None,
    references: dict[str, str] | None = None,
) -> Skill:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / SKILL_MD).write_text(f"# {name}\n\nWhen to use it.\n", encoding="utf-8")
    if selftest is not None:
        (d / "selftest.py").write_text(textwrap.dedent(selftest), encoding="utf-8")
    if api_surface:
        (d / "api_surface.md").write_text("sig()\n", encoding="utf-8")
    if skill_py:
        (d / "skill.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    if provenance is not None:
        (d / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    for sub, files in (("scripts", scripts), ("references", references)):
        for rel, body in (files or {}).items():
            f = d / sub / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(body, encoding="utf-8")
    return Skill(name=name, path=d)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


def test_maturity_is_derived_from_files_not_declared(tmp_path: Path) -> None:
    """A skill hardens by gaining files, never by editing a version field —
    so notes can become an adapter with no migration."""
    assert make_skill(tmp_path, "a").maturity is Maturity.NOTES
    assert make_skill(tmp_path, "b", api_surface=True).maturity is Maturity.GUIDED
    assert make_skill(
        tmp_path, "c", api_surface=True, skill_py=True
    ).maturity is Maturity.ADAPTER


def test_a_directory_without_instructions_is_not_a_skill(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert discover(tmp_path) == []


def test_skill_carries_its_own_range_assertions(tmp_path: Path) -> None:
    """A skill knows its valid domain, so the design need not restate it."""
    s = make_skill(
        tmp_path, "amb",
        provenance={"result_assertions": [{"path": "k1", "min": 0, "max": 1}]},
    )
    assert s.assertions() == [{"path": "k1", "min": 0, "max": 1}]


def test_malformed_provenance_degrades_to_empty(tmp_path: Path) -> None:
    d = tmp_path / "bad"
    d.mkdir()
    (d / SKILL_MD).write_text("# bad\n", encoding="utf-8")
    (d / "provenance.json").write_text("{not json", encoding="utf-8")
    s = Skill(name="bad", path=d)
    assert s.provenance() == {} and s.assertions() == []


# ---------------------------------------------------------------------------
# The self-test gate
# ---------------------------------------------------------------------------


def test_passing_selftest_reaches_proposed(tmp_path: Path, ledger: Path) -> None:
    s = make_skill(tmp_path)
    state = evaluate(s, ledger=ledger)
    assert state.status is Status.PROPOSED
    assert not state.loadable


def test_failing_selftest_is_quarantined_not_merely_unapproved(
    tmp_path: Path, ledger: Path
) -> None:
    """Order matters: a broken skill should be reported as broken, not as
    awaiting a signature it could never usefully get."""
    s = make_skill(tmp_path, selftest="import sys; sys.exit(1)")
    state = evaluate(s, ledger=ledger)
    assert state.status is Status.QUARANTINED
    assert not state.loadable


def test_missing_selftest_can_never_be_promoted(tmp_path: Path, ledger: Path) -> None:
    """Writing the test is part of learning the software, not a later chore."""
    s = make_skill(tmp_path, selftest=None)
    state = evaluate(s, ledger=ledger)
    assert state.status is Status.UNTESTED
    assert "cannot demonstrate" in state.reason


def test_quarantine_survives_approval(tmp_path: Path, ledger: Path) -> None:
    """Approval must not override a failing self-test — a person signing off
    on a skill that provably does not work is the one case where the human
    gate should not be the last word."""
    s = make_skill(tmp_path, selftest="import sys; sys.exit(2)")
    approval.approve("demo", s.content_hash(), approved_by="someone", path=ledger)
    assert evaluate(s, ledger=ledger).status is Status.QUARANTINED


def test_crashing_selftest_does_not_take_the_caller_down(
    tmp_path: Path, ledger: Path
) -> None:
    s = make_skill(tmp_path, selftest="raise SystemExit(3)")
    assert evaluate(s, ledger=ledger).status is Status.QUARANTINED


# ---------------------------------------------------------------------------
# The human gate — and why it binds to content
# ---------------------------------------------------------------------------


def test_approval_makes_a_skill_loadable(tmp_path: Path, ledger: Path) -> None:
    s = make_skill(tmp_path)
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)
    state = evaluate(s, ledger=ledger)
    assert state.status is Status.TRUSTED and state.loadable


def test_editing_a_skill_lapses_its_approval(tmp_path: Path, ledger: Path) -> None:
    """The property the whole design rests on.

    A distilled skill rewritten from newer work is a different skill wearing
    the same name. If approval survived that, the human gate would be a
    one-time stamp on something that keeps changing underneath it.
    """
    s = make_skill(tmp_path)
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)
    assert evaluate(s, ledger=ledger).status is Status.TRUSTED

    (s.path / SKILL_MD).write_text("# demo\n\nRewritten by distillation.\n", "utf-8")

    state = evaluate(s, ledger=ledger)
    assert state.status is Status.PROPOSED
    assert "re-approval required" in state.reason


def test_provenance_updates_do_not_lapse_approval(tmp_path: Path, ledger: Path) -> None:
    """Recording that a skill was used in one more quest is not a change to
    the skill — if it lapsed approval, the ledger would churn constantly."""
    s = make_skill(tmp_path, provenance={"taught_by_projects": []})
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)
    (s.path / "provenance.json").write_text(
        json.dumps({"taught_by_projects": ["quest-1", "quest-2"]}), encoding="utf-8"
    )
    assert evaluate(s, ledger=ledger).status is Status.TRUSTED


def test_approval_requires_a_named_person(tmp_path: Path, ledger: Path) -> None:
    """There is no anonymous or system approver: the point of the gate is
    that a person decided."""
    s = make_skill(tmp_path)
    for who in ("", "   "):
        with pytest.raises(ValueError):
            approval.approve("demo", s.content_hash(), approved_by=who, path=ledger)


def test_revoke_returns_a_skill_to_proposed(tmp_path: Path, ledger: Path) -> None:
    s = make_skill(tmp_path)
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)
    assert approval.revoke("demo", ledger) is True
    assert evaluate(s, ledger=ledger).status is Status.PROPOSED
    assert approval.revoke("demo", ledger) is False


# ---------------------------------------------------------------------------
# Resolution for a quest
# ---------------------------------------------------------------------------


def test_unapproved_skill_is_rejected_not_silently_used(
    tmp_path: Path, ledger: Path, monkeypatch
) -> None:
    """An unattended run must fall back to guided generation rather than
    quietly adopt capability nobody approved."""
    make_skill(tmp_path, "demo")
    ok, rejected = loadable_skills(["demo"], skills_dir=tmp_path, ledger=ledger)
    assert ok == []
    assert len(rejected) == 1 and rejected[0].status is Status.PROPOSED


def test_missing_skill_is_reported_rather_than_raised(
    tmp_path: Path, ledger: Path
) -> None:
    ok, rejected = loadable_skills(["nope"], skills_dir=tmp_path, ledger=ledger)
    assert ok == []
    assert "no skill by that name" in rejected[0].reason


def test_only_trusted_status_is_loadable() -> None:
    """One place decides the policy, so no call site can forget it."""
    assert Status.TRUSTED.loadable
    for s in (Status.PROPOSED, Status.QUARANTINED, Status.UNTESTED):
        assert not s.loadable


# ---------------------------------------------------------------------------
# The seed skill
# ---------------------------------------------------------------------------


def test_fi_ships_no_skills(tmp_path: Path, monkeypatch) -> None:
    """A fresh install must start with nothing — not even a skill FI's own
    authors wrote.

    A skill that arrives with the install is a curated library again, and
    curation does not compound. Capability is meant to be acquired from work
    done with this user, on this machine.
    """
    repo = Path(__file__).resolve().parent.parent
    assert not (repo / "skills").exists(), (
        "the repository ships a skills directory — skills are machine state, "
        "not repository content"
    )
    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "empty"))
    assert discover() == []


def test_skills_root_is_outside_the_repository(monkeypatch) -> None:
    """Skills accumulate per machine. A root inside the checkout would make
    them things you commit and ship."""
    from core.skills.registry import skills_root

    monkeypatch.delenv("FI_SKILLS_DIR", raising=False)
    repo = Path(__file__).resolve().parent.parent
    root = skills_root()
    assert repo not in root.parents and root != repo


def test_skills_root_is_overridable(tmp_path: Path, monkeypatch) -> None:
    from core.skills.registry import skills_root

    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "elsewhere"))
    assert skills_root() == tmp_path / "elsewhere"


# ---------------------------------------------------------------------------
# Agent Skills standard layout
#
# The standard (https://agentskills.io/) puts executable code in ``scripts/``
# and documentation in ``references/``. FI reads both so a skill written for
# another agent works here unchanged. The tests that matter are not about
# reading the directories — they are about the approval gate still covering
# what is inside them.
# ---------------------------------------------------------------------------


def test_bundled_scripts_and_references_are_listed(tmp_path: Path) -> None:
    s = make_skill(
        tmp_path,
        scripts={"run.py": "print(1)\n", "helpers/io.py": "x = 1\n"},
        references={"api.md": "# API\n"},
    )
    assert s.bundled_scripts() == ["scripts/helpers/io.py", "scripts/run.py"]
    assert s.reference_files() == ["references/api.md"]


def test_a_scripts_only_skill_is_a_tool_not_a_library(tmp_path: Path) -> None:
    """A bundled script is something you run, not something you import.

    Inferring LIBRARY here would put "import it and call its functions" in
    front of a model holding a shell script, which teaches it to write code
    that cannot work.
    """
    from core.skills import Kind

    s = make_skill(tmp_path, scripts={"run.sh": "echo hi\n"})
    assert s.kind is Kind.TOOL


def test_editing_a_bundled_script_lapses_approval(
    tmp_path: Path, ledger: Path
) -> None:
    """The hole this closes: the hash covered four named files while the
    importer copied whole ``scripts/`` trees in beside them, so an approved
    skill's executable content could be swapped with the approval intact.

    That is the human gate guarding only the files someone remembered to
    list — which is why the hash is now an exclusion list, not an allow list.
    """
    s = make_skill(tmp_path, scripts={"run.py": "print('original')\n"})
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)
    assert evaluate(s, ledger=ledger).status is Status.TRUSTED

    (s.path / "scripts" / "run.py").write_text("print('swapped')\n", encoding="utf-8")

    state = evaluate(s, ledger=ledger)
    assert state.status is Status.PROPOSED
    assert "re-approval required" in state.reason


def test_adding_a_new_bundled_file_lapses_approval(
    tmp_path: Path, ledger: Path
) -> None:
    """Adding a file is as much a change as editing one — a skill that gains
    a script after approval is not the skill that was approved."""
    s = make_skill(tmp_path, scripts={"run.py": "print(1)\n"})
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)

    (s.path / "scripts" / "extra.py").write_text("print(2)\n", encoding="utf-8")

    assert evaluate(s, ledger=ledger).status is Status.PROPOSED


def test_renaming_a_script_lapses_approval_even_with_identical_bytes(
    tmp_path: Path, ledger: Path
) -> None:
    """The path is hashed alongside the bytes. Instructions reference scripts
    by name, so moving one breaks the skill without changing any content."""
    s = make_skill(tmp_path, scripts={"run.py": "print(1)\n"})
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)

    (s.path / "scripts" / "run.py").rename(s.path / "scripts" / "main.py")

    assert evaluate(s, ledger=ledger).status is Status.PROPOSED


SELFTEST_THAT_WRITES_BESIDE_ITSELF = chr(10).join([
    # Importing a sibling module makes CPython write __pycache__ next to it.
    "import helper, pathlib, sys",
    # A plotting self-test dropping a PNG in its cwd is the real-world case.
    "pathlib.Path('artifact.png').write_bytes(b'not-really-a-png')",
    "sys.exit(0)",
])


def test_running_the_selftest_does_not_lapse_approval(
    tmp_path: Path, ledger: Path
) -> None:
    """Checking a skill must not modify the skill.

    The hash covers the whole folder, so anything evaluation leaves behind
    revokes the approval it was checking -- an intermittent "content
    changed since approval" that reads like tampering. Two things used to
    land there: __pycache__ from importing a module beside selftest.py,
    and, in the wild, the PNG a plotting self-test writes to its cwd.

    So the gate runs against a copy, and the assertion is the strong form:
    evaluation leaves the directory byte-identical. Reverting to cwd=skill
    fails this on the listing, not just on the hash.
    """
    s = make_skill(
        tmp_path,
        selftest=SELFTEST_THAT_WRITES_BESIDE_ITSELF,
        scripts={"noop.py": "x = 1"},
    )
    # Importable beside selftest.py, so the child really writes bytecode.
    (s.path / "helper.py").write_text("VALUE = 1", encoding="utf-8")

    before_hash = s.content_hash()
    before_files = sorted(p.relative_to(s.path).as_posix()
                          for p in s.path.rglob("*"))

    approval.approve("demo", before_hash, approved_by="jyunming", path=ledger)

    first = evaluate(s, ledger=ledger)
    assert first.status is Status.TRUSTED, first.selftest_output

    after_files = sorted(p.relative_to(s.path).as_posix()
                         for p in s.path.rglob("*"))
    assert after_files == before_files, (
        "evaluation left files in the skill directory: "
        f"{sorted(set(after_files) - set(before_files))}"
    )
    assert s.content_hash() == before_hash, (
        "the skill's content hash moved because the gate wrote to it"
    )

    second = evaluate(s, ledger=ledger)
    assert second.status is Status.TRUSTED, second.reason


def test_provenance_still_does_not_lapse_approval_under_the_new_hash(
    tmp_path: Path, ledger: Path
) -> None:
    """Re-pinned against the exclusion list: hashing everything-but would
    catch provenance too, and the ledger would churn on every quest."""
    s = make_skill(
        tmp_path, provenance={"taught_by_projects": []}, scripts={"r.py": "x=1\n"}
    )
    approval.approve("demo", s.content_hash(), approved_by="jyunming", path=ledger)
    (s.path / "provenance.json").write_text(
        json.dumps({"taught_by_projects": [{"quest": "q1", "outcome": "accept"}]}),
        encoding="utf-8",
    )
    assert evaluate(s, ledger=ledger).status is Status.TRUSTED
