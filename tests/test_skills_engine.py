"""Skills must reach the prompts, and only when trusted.

``test_skills.py`` pins discovery and the promotion gate. These pin the half
that makes it worth having: a skill FI knows about but never tells ``design``
and ``implement`` about changes nothing — the model keeps re-deriving physics
it could have imported.

The other property here is the safety one. An unattended ``--fleet`` run must
never load capability nobody approved, so "not trusted" has to mean "absent
from the prompt", not "mentioned with a warning".
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine import Engine, _assertion_violations
from core.skills import approval
from core.skills.base import SKILL_MD


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills" / "demo"
    d.mkdir(parents=True)
    (d / SKILL_MD).write_text(
        "# demo\n\nUse for widget imaging.\n", encoding="utf-8"
    )
    (d / "api_surface.md").write_text("demo.run(x: float) -> float\n", encoding="utf-8")
    (d / "selftest.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    (d / "provenance.json").write_text(
        json.dumps({"result_assertions": [{"path": "widget_q", "min": 0, "max": 1}]}),
        encoding="utf-8",
    )
    return d


def _engine(names: list[str], skills_dir: Path, ledger: Path) -> SimpleNamespace:
    """An engine stub exposing only what the skills helpers touch."""
    logs: list[str] = []
    return SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(skills=names, skills_exclude=[]),
        ),
        _log=SimpleNamespace(
            info=lambda *a, **k: logs.append(("info", a)),
            warning=lambda *a, **k: logs.append(("warn", a)),
        ),
        logs=logs,
    )


def _state(*names: str, reasons: dict | None = None) -> dict:
    """The state `select_skills` would have produced.

    The block now renders what selection chose, not what the config lists —
    which is the whole point of the change: a growing library must not grow
    the prompt.
    """
    return {
        "selected_skills": list(names),
        "skill_selection": {"reasons": reasons or {}},
    }


@pytest.fixture
def patched(monkeypatch, skill_dir: Path, tmp_path: Path):
    """Redirect skill discovery and approvals at the fixtures."""
    import core.skills.registry as reg

    ledger = tmp_path / "approvals.json"
    real = reg.loadable_skills
    monkeypatch.setattr(
        reg, "loadable_skills",
        lambda names, **kw: real(
            names, skills_dir=skill_dir.parent, ledger=ledger,
        ),
    )
    monkeypatch.setattr("core.skills.loadable_skills", reg.loadable_skills)
    return ledger


def test_untrusted_skill_never_reaches_the_prompt(patched, skill_dir) -> None:
    """The safety property: an unapproved skill is absent, not merely noted."""
    eng = _engine(["demo"], skill_dir.parent, patched)
    block = Engine._skills_block(eng, _state("demo"))  # type: ignore[arg-type]
    assert block == ""
    assert any(kind == "warn" for kind, _ in eng.logs)


def test_approved_skill_reaches_the_prompt_with_its_api(patched, skill_dir) -> None:
    from core.skills import Skill

    s = Skill(name="demo", path=skill_dir)
    approval.approve("demo", s.content_hash(), approved_by="tester", path=patched)

    eng = _engine(["demo"], skill_dir.parent, patched)
    block = Engine._skills_block(eng, _state("demo", reasons={"demo": "needs widget imaging"}))  # type: ignore[arg-type]

    assert "## Skill: demo" in block
    assert "*Selected because:* needs widget imaging" in block
    assert "widget imaging" in block          # instructions
    assert "demo.run(x: float)" in block      # api surface
    # This fixture carries an api_surface.md, so it is inferred to be a
    # library, and the preamble must be the library one.
    assert "## Skill: demo (library)" in block
    assert "import it and call its functions" in block
    assert "drive the tool" not in block


def test_a_tool_skill_is_not_told_to_import_itself(
    tmp_path: Path, monkeypatch
) -> None:
    """The bug this guards: telling a model to import pandoc teaches it to
    write code that cannot work. A skill with no importable surface must be
    described as something FI drives, not something it calls."""
    import core.skills.registry as reg
    from core.skills import Skill

    root = tmp_path / "skills"
    d = root / "cli-thing"
    d.mkdir(parents=True)
    (d / SKILL_MD).write_text("# cli-thing\n\nRun it from the shell.\n", "utf-8")
    (d / "selftest.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")

    ledger = tmp_path / "l.json"
    approval.approve(
        "cli-thing", Skill(name="cli-thing", path=d).content_hash(),
        approved_by="tester", path=ledger,
    )
    real = reg.loadable_skills
    monkeypatch.setattr(
        reg, "loadable_skills",
        lambda names, **kw: real(names, skills_dir=root, ledger=ledger),
    )
    monkeypatch.setattr("core.skills.loadable_skills", reg.loadable_skills)

    block = Engine._skills_block(
        _engine(["cli-thing"], root, ledger), _state("cli-thing"),
    )  # type: ignore[arg-type]
    assert "## Skill: cli-thing (tool)" in block
    assert "drive the tool" in block
    assert "import it and call its functions" not in block


def test_nothing_selected_is_empty_not_an_error(patched, skill_dir) -> None:
    """An empty selection is a correct and common answer."""
    eng = _engine([], skill_dir.parent, patched)
    assert Engine._skills_block(eng, _state()) == ""  # type: ignore[arg-type]


def test_registry_failure_falls_back_to_generation(monkeypatch, skill_dir, tmp_path):
    """A broken registry must degrade to the pre-skills behaviour, never
    stall a quest."""
    import core.skills.registry as reg

    monkeypatch.setattr(
        reg, "loadable_skills",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("registry boom")),
    )
    monkeypatch.setattr("core.skills.loadable_skills", reg.loadable_skills)
    eng = _engine(["demo"], skill_dir.parent, tmp_path / "l.json")
    # A non-empty selection, so the broken registry is actually reached — an
    # empty one short-circuits before it and would test nothing.
    assert Engine._skills_block(eng, _state("demo")) == ""  # type: ignore[arg-type]
    assert any(kind == "warn" for kind, _ in eng.logs)


def test_trusted_skill_contributes_its_range_assertions(patched, skill_dir) -> None:
    from core.skills import Skill

    s = Skill(name="demo", path=skill_dir)
    approval.approve("demo", s.content_hash(), approved_by="tester", path=patched)

    eng = _engine(["demo"], skill_dir.parent, patched)
    got = Engine._skill_assertions(eng, _state("demo"))  # type: ignore[arg-type]
    assert got == [{"path": "widget_q", "min": 0, "max": 1}]


def test_untrusted_skill_contributes_no_assertions(patched, skill_dir) -> None:
    eng = _engine(["demo"], skill_dir.parent, patched)
    assert Engine._skill_assertions(eng, _state("demo")) == []  # type: ignore[arg-type]


def test_skill_assertions_are_enforced_by_the_plausibility_gate() -> None:
    """The join between skills and A3: a skill's declared domain sends a
    violating run back for repair, without the design restating anything."""
    state = {
        "result_json": {"widget_q": 1.8},
        "design": {"hypothesis": "h"},          # design declares nothing
        "_skill_assertions": [{"path": "widget_q", "min": 0, "max": 1}],
    }
    violations = _assertion_violations(state)
    assert len(violations) == 1
    assert "widget_q" in violations[0].describe()


def test_design_assertions_survive_the_merge() -> None:
    """Merging must not drop what the design itself declared."""
    state = {
        "result_json": {"cd_nm": -1.0, "widget_q": 0.5},
        "design": {"result_assertions": [{"path": "cd_nm", "min": 0}]},
        "_skill_assertions": [{"path": "widget_q", "min": 0, "max": 1}],
    }
    violations = _assertion_violations(state)
    assert len(violations) == 1 and violations[0].path == "cd_nm"


def test_prompts_expose_the_slot_and_ask_for_it_to_be_used() -> None:
    """A slot nothing instructs the model to use is decoration."""
    root = Path(__file__).resolve().parent.parent / "agents"
    for name in ("design.md", "implement.md", "implement_outline.md",
                 "implement_body.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert "$skills_block" in text, f"{name} has no skills slot"
    # The guidance must cover both kinds. "Import and call it" alone would be
    # wrong for an external tool, which has no importable API.
    flat = " ".join((root / "design.md").read_text(encoding="utf-8").split())
    assert "design the experiment around them" in flat
    assert "library" in flat and "tool" in flat
    impl = " ".join((root / "implement.md").read_text(encoding="utf-8").split())
    assert "imported and called" in impl
    assert "invoked as an external command" in impl


def test_bundled_files_are_named_but_never_inlined(
    tmp_path: Path, monkeypatch
) -> None:
    """A standard-layout skill ships ``scripts/`` and ``references/``, and a
    references directory can be larger than the whole quest.

    Inlining them would rebuild the exact cost selection exists to remove —
    the measured ~465 tokens per skill per node that made a growing library
    its own largest expense. So the block names the paths and stops; the
    generated code opens what it needs, which is what the standard calls
    loading at execution time.
    """
    import core.skills.registry as reg
    from core.skills import Skill

    root = tmp_path / "skills"
    d = root / "bundled"
    (d / "scripts").mkdir(parents=True)
    (d / "references").mkdir(parents=True)
    (d / SKILL_MD).write_text("# bundled\n\nDrive it.\n", encoding="utf-8")
    (d / "selftest.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    (d / "scripts" / "convert.py").write_text(
        "SENTINEL_SCRIPT_BODY = 'must not appear in the prompt'\n", encoding="utf-8"
    )
    (d / "references" / "manual.md").write_text(
        "SENTINEL_REFERENCE_BODY must not appear in the prompt\n" * 200,
        encoding="utf-8",
    )

    ledger = tmp_path / "l.json"
    approval.approve(
        "bundled", Skill(name="bundled", path=d).content_hash(),
        approved_by="tester", path=ledger,
    )
    real = reg.loadable_skills
    monkeypatch.setattr(
        reg, "loadable_skills",
        lambda names, **kw: real(names, skills_dir=root, ledger=ledger),
    )
    monkeypatch.setattr("core.skills.loadable_skills", reg.loadable_skills)

    eng = _engine(["bundled"], root, ledger)
    block = Engine._skills_block(eng, _state("bundled"))  # type: ignore[arg-type]

    assert "scripts/convert.py" in block
    assert "references/manual.md" in block
    assert "SENTINEL_SCRIPT_BODY" not in block
    assert "SENTINEL_REFERENCE_BODY" not in block
    # And the model is told where they are, or naming them is useless.
    assert str(d) in block
