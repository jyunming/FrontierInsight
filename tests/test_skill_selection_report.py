"""The selection report — what makes "why did it not use ambit?" answerable.

Without it the reasoning exists only in a log line that scrolls past, and an
unapproved skill can sit unnoticed forever while every quest quietly does
without it.

Also covers the `--why-skills` CLI wiring, which previews a selection for a
topic without running a quest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills import Skill, SkillState, Status
from core.skills import selection as sel


def _state(tmp: Path, name: str, *, desc: str = "x", loadable: bool = True,
           reason: str = "") -> SkillState:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n\n# {name}\n", encoding="utf-8"
    )
    return SkillState(
        skill=Skill(name=name, path=d),
        status=Status.TRUSTED if loadable else Status.PROPOSED,
        reason=reason,
    )


def test_report_names_what_was_chosen_and_why(tmp_path: Path) -> None:
    cat = sel.build_catalogue([_state(tmp_path, "ambit")])
    s = sel.Selection(chosen=["ambit"], reasons={"ambit": "computes the image"})
    out = sel.render_selection_report(cat, s, [])
    assert "Selected:" in out
    assert "ambit: computes the image" in out


def test_report_says_plainly_when_nothing_was_selected(tmp_path: Path) -> None:
    """An empty selection is a correct answer, and must read as a decision
    rather than as an absence."""
    cat = sel.build_catalogue([_state(tmp_path, "ambit")])
    out = sel.render_selection_report(cat, sel.Selection(), [])
    assert "Selected: none" in out
    assert "generated its own code" in out


def test_report_lists_candidates_that_lost(tmp_path: Path) -> None:
    """"Why did this quest not use ambit?" needs ambit to appear somewhere."""
    cat = sel.build_catalogue([
        _state(tmp_path, "ambit"), _state(tmp_path, "pandoc"),
    ])
    out = sel.render_selection_report(cat, sel.Selection(chosen=["ambit"]), [])
    assert "Considered but not selected:" in out
    assert "pandoc" in out


def test_report_surfaces_skills_that_could_not_be_candidates(tmp_path: Path) -> None:
    """The other half of the learning loop: it tells you there is something
    to approve."""
    cat = sel.build_catalogue([_state(tmp_path, "ambit")])
    near = [_state(tmp_path, "semulator", loadable=False, reason="awaiting approval")]
    out = sel.render_selection_report(cat, sel.Selection(chosen=["ambit"]), near)
    assert "semulator (proposed): awaiting approval" in out
    assert "--approve-skill" in out


def test_report_flags_names_the_model_invented(tmp_path: Path) -> None:
    cat = sel.build_catalogue([_state(tmp_path, "ambit")])
    s = sel.Selection(chosen=["ambit"], unknown=["imaginary"])
    out = sel.render_selection_report(cat, s, [])
    assert "not candidates (ignored): imaginary" in out


def test_report_handles_a_missing_reason(tmp_path: Path) -> None:
    cat = sel.build_catalogue([_state(tmp_path, "ambit")])
    out = sel.render_selection_report(cat, sel.Selection(chosen=["ambit"]), [])
    assert "(no reason given)" in out


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_why_skills_is_a_standalone_mode() -> None:
    import inspect
    import re

    import launch

    src = inspect.getsource(launch)
    m = re.search(r"(\w+)\.add_argument\(\s*\n\s*\"--why-skills\"", src)
    assert m and m.group(1) == "mode", "must be usable on its own"


def test_why_skills_has_its_own_provider_flags() -> None:
    """It cannot borrow --config: that is itself a mode, so the two can never
    appear together."""
    import inspect
    import re

    import launch

    src = inspect.getsource(launch)
    for flag in ("--skills-provider", "--skills-model"):
        m = re.search(r"(\w+)\.add_argument\(\s*\n\s*\"" + re.escape(flag) + r"\"", src)
        assert m and m.group(1) == "p", f"{flag} must be a plain option"


@pytest.mark.asyncio
async def test_why_skills_reports_when_there_are_no_candidates(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """With an empty library it must say so rather than making a call."""
    import launch

    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "empty"))
    rc = await launch._why_skills("anything", "claude_cli", "")
    assert rc == 0
    assert "No candidate skills" in capsys.readouterr().out


def test_report_shows_why_a_candidate_was_declined(tmp_path: Path) -> None:
    """A decline with a reason is the only evidence the model read the entry.

    Without it, "the model considered this skill and judged it wrong for the
    topic" and "the model never engaged with the catalogue at all" produce
    identical output — and telling those apart is exactly what is needed when
    a skill is silently never selected. Measured: the earlier prompt, which
    asked only for selections, returned an empty list from two different
    models on a topic the skill plainly fitted.
    """
    cat = sel.build_catalogue([_state(tmp_path, "ambit"), _state(tmp_path, "pandoc")])
    selection = sel.Selection(
        chosen=["ambit"],
        reasons={"ambit": "computes the aerial image this quest sweeps"},
        declined={"pandoc": "this quest writes no document of its own"},
    )
    out = sel.render_selection_report(cat, selection, [])
    assert "this quest writes no document of its own" in out


def test_report_marks_a_candidate_the_model_never_accounted_for(
    tmp_path: Path,
) -> None:
    """A missing reason is shown as missing rather than smoothed over — it is
    the signature of a selection call that skimmed the catalogue."""
    cat = sel.build_catalogue([_state(tmp_path, "ambit"), _state(tmp_path, "pandoc")])
    out = sel.render_selection_report(cat, sel.Selection(chosen=["ambit"]), [])
    assert "pandoc: (no reason given)" in out
