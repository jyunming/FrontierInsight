"""The first write of a figure function is given FI's drawing calls (version 5 frame), as the repair is; a quest of an older frame
is told what it was. Made-up topic: a colony's growth. No real model is called."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import analysis_frame as af
from core import run_checks, skeleton
from tests.test_analysis_frame import _design

AGENTS = Path(__file__).parent.parent / "agents"
BODIES = ["derive_doubling_time", "figure_growth_curve", "figure_split"]
RULE = skeleton._FIGURE_RULE
AXES_WORDS = ("draws what the plan says the figure shows on `fig.axes`", "Draw on `fig.axes`", "index `fig.axes`",
              "draws one on each of `fig.axes`")


def _template(name: str = "implement_body") -> str:
    return (AGENTS / f"{name}.md").read_text(encoding="utf-8")


def _mixin(root: Path) -> Any:
    mixin = run_checks.RunChecksMixin.__new__(run_checks.RunChecksMixin)
    mixin.quest_root = root  # type: ignore[attr-defined]
    return mixin


def test_the_rules_of_a_version_5_first_write_hold_the_repair_rule_and_no_axes_drawing() -> None:
    rules = run_checks.analysis_bodies_rules(True)
    assert RULE in "\n".join(line.strip() for line in rules.split("\n")) or all(line.strip() in rules for line in RULE.split("\n") if line.strip())
    for line in RULE.split("\n"):
        assert line.rstrip() == "" or ("  " + line).rstrip() in rules
    assert "fig.line(panel" in rules and "fig.band(" in rules and "fig.image(" in rules
    assert not any(w in rules for w in AXES_WORDS)
    assert "`fig.panel(0, m=200)`" in rules and "do not type a title for such a panel" in rules
    assert "`fig.axes` (one axes" not in rules


def test_a_version_4_first_write_is_byte_identical_to_the_base() -> None:
    assert run_checks.analysis_bodies_rules(False) is run_checks._ANALYSIS_BODIES_RULES
    assert run_checks.analysis_bodies_rules() == run_checks._ANALYSIS_BODIES_RULES
    for name in ("implement_body", "implement"):
        base = run_checks.frame_template_text(_template(name), False, BODIES, {"figure_growth_curve": 2})
        assert run_checks.frame_template_text(_template(name), False, BODIES, {"figure_growth_curve": 2}, False) == base
    assert run_checks.panel_statement(3, False) == run_checks.panel_statement(3)
    assert run_checks._frame_reply(False, BODIES, {"figure_split": 2}, False) == run_checks._frame_reply(False, BODIES, {"figure_split": 2})


def test_a_version_5_template_and_reply_give_the_calls_and_never_the_axes() -> None:
    for name in ("implement_body", "implement"):
        text = run_checks.frame_template_text(_template(name), False, BODIES, {"figure_growth_curve": 1, "figure_split": 3}, True)
        assert not any(w in text for w in AXES_WORDS), name
        assert "own panel (the `panel` of a drawing call)" in text
        assert ("A drawing call (`fig.line`" in text) == (name == "implement"), "the bullet is in implement.md only"
        assert "give it a short descriptive title" not in text
    reply = run_checks._frame_reply(False, BODIES, {"figure_growth_curve": 1, "figure_split": 3}, True)
    assert "ONE panel for it, panel 0" in reply and "numbered 0 to 2" in reply
    assert "fig.axes" not in reply and "`fig.panel(i, setting=value)`" in reply, "a panel says which setting it shows"


def test_the_panel_statement_without_calls_is_as_it_was() -> None:
    assert "fig.axes[0]" in run_checks.panel_statement(1) and "fig.panel(i, setting=value)" in run_checks.panel_statement(2)


def test_the_stub_docstring_names_the_calls_under_version_5_and_is_as_it_was_otherwise() -> None:
    plan = af.plan_of(_design())
    old = af._stub(plan, "figure_fig1")
    assert "Draw on `fig.axes`" in old and af._stub(plan, "figure_fig1", False) == old
    new = af._stub(plan, "figure_fig1", True)
    assert "Draw on `fig.axes`" not in new and "fig.axes" not in new
    assert ("fig.panel(0, setting=value)" in new) == (plan["figures"]["fig1"]["panels"] > 1)
    for call in ("line", "points", "bars", "band", "ref", "note", "curve", "image"):
        assert f"`fig.{call}`" in new and f"fig.{call}(" in RULE
    assert new.endswith('raise NotYet("figure_fig1 is not written yet")\n')
    compile(new, "stub", "exec")


def test_derive_source_and_compose_pass_the_choice_to_the_stubs() -> None:
    design = _design()
    assert "Draw on `fig.axes`" in af.derive_source(design)
    assert "FI's drawing calls" in af.derive_source(design, "", True) and "fig.axes" not in af.derive_source(design, "", True)
    text, _ = af.compose(design, "", "", (), calls=True)
    assert "FI's drawing calls" in text


def test_files_written_with_the_figure_calls_have_the_new_stub(tmp_path: Path) -> None:
    design = _design()
    af.ensure_files(tmp_path / "new", design, "")
    assert "FI's drawing calls" in (tmp_path / "new" / af.DERIVE_FILE).read_text(encoding="utf-8")
    assert "Draw on `fig.axes`" not in (tmp_path / "new" / af.DERIVE_FILE).read_text(encoding="utf-8")


def test_a_quest_is_given_the_calls_from_its_runtime_or_frame_version(tmp_path: Path) -> None:
    mixin = _mixin(tmp_path)
    assert mixin._figure_calls_on() is True, "no runtime yet: the frame will be this version's"
    code = tmp_path / "code"
    code.mkdir()
    for version, want in ((3, False), (4, False), (5, True)):
        (code / skeleton.RUNTIME_FILE).write_text(
            skeleton.RUNTIME.replace("FRAME_RUNTIME_VERSION = 5", f"FRAME_RUNTIME_VERSION = {version}"), encoding="utf-8")
        assert mixin._figure_calls_on() is want, version


def test_an_older_frame_without_a_runtime_file_is_not_given_the_calls(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("# >>> FI frame, version 4: written by FI from the plan; a repair does not change it\n",
                                      encoding="utf-8")
    assert _mixin(tmp_path)._figure_calls_on() is False


@pytest.mark.parametrize("calls", [False, True])
def test_the_template_text_is_stable_for_each_choice(calls: bool) -> None:
    a = run_checks.frame_template_text(_template("implement_body"), False, BODIES, None, calls)
    assert a == run_checks.frame_template_text(_template("implement_body"), False, BODIES, None, calls)
