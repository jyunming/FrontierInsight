"""The requests and repairs of a study whose analysis FI writes too (``execution.code_skeleton`` and a plan in the contract form):
the model is asked for ``trial`` and the functions of derive.py (the quantities and figures the plan declares), never for
experiment.py, and a failed function is repaired by itself. Every text the flag off, or a plan in an older form, gets is what it
was (sha256 at df4ff1ad). Made-up topic: a colony's growth. No real model is called."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from core import code_layout, run_checks, skeleton
from tests.test_skeleton_ask import DESIGN, _engine
from tests.test_skeleton_ask_repairs import INSIDE, SEARCH
from tests.test_skeleton_repairs import FRAMED, UNFRAMED

AGENTS = Path(__file__).parent.parent / "agents"
#: What the plan declares that FI cannot work out from its own record: a quantity, a reference and a figure.
BODIES = ["derive_doubling_time", "reference_growth", "figure_growth_curve"]
FRAME_LINE = "# >>> FI analysis frame, version 1: written by FI from the plan; a repair does not change it"
DERIVE = (f'"""Derived quantities."""\n{FRAME_LINE}\nRESULTS = ["growth"]\n# <<< FI analysis frame\n\n'
          "def derive_doubling_time(table):\n    raise NotYet('not yet')\n")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _agent(name: str) -> str:
    return (AGENTS / f"{name}.md").read_text(encoding="utf-8")


# ---- the request of a study whose analysis FI writes ------------------------------------------------------------------------------


def test_the_reply_format_asks_for_each_body_by_name_and_never_for_experiment_py() -> None:
    reply = run_checks._frame_reply(False, BODIES)
    assert "# file: experiment.py" not in reply and "experiment.py" in reply, "it says FI writes experiment.py"
    assert "# file: simulate.py" in reply and "# file: derive.py" in reply
    for sig in ("def derive_doubling_time(table)", "def reference_growth(setting)", "def figure_growth_curve(fig, table)"):
        assert sig in reply
    assert reply.index("# file: simulate.py") < reply.index("# file: derive.py") < reply.index("DEPS:")
    assert "# file: submit.py" in run_checks._frame_reply(True, BODIES)
    assert run_checks._frame_reply(True, BODIES).index("# file: derive.py") < run_checks._frame_reply(True, BODIES).index("# file: submit.py")


def test_a_plan_with_nothing_for_the_model_to_write_gets_no_derive_block() -> None:
    reply = run_checks._frame_reply(False, [])
    assert "# file: derive.py" not in reply and "# file: experiment.py" not in reply and "# file: simulate.py" in reply


@pytest.mark.parametrize("name", ["implement", "implement_body"])
def test_the_code_template_asks_for_the_bodies_and_drops_the_rules_of_the_analysis(name: str) -> None:
    on = run_checks.frame_template_text(_agent(name), False, BODIES)
    assert "# file: experiment.py" not in on and "# file: derive.py" in on and "def figure_growth_curve(fig, table)" in on
    for gone in ("Save figures to", "Stratify when", "Write a one-line JSON", "Report the trials behind a probability",
                 "A figure holds at most 3 panels", "Honour `FI_PILOT`"):
        assert gone not in on, gone
    off = run_checks.frame_template_text(_agent(name))
    assert "Stratify when" in off or name == "implement_body", "the rules are still sent when FI does not write the analysis"
    assert on != off


def test_the_outline_request_lists_the_bodies_and_not_the_analysis() -> None:
    text = _agent("implement_outline")
    on = run_checks.frame_outline_text(text, BODIES)
    assert "`derive_doubling_time`" in on and "FI writes experiment.py" in on
    assert "for experiment.py, the functions of the analysis" not in on and "and experiment.py, constrained to" not in on
    assert "(none: FI works out" in run_checks.frame_outline_text(text, [])
    assert on != run_checks.frame_outline_text(text)


@pytest.mark.parametrize("design,search,inside", [(DESIGN, False, {}), (SEARCH, True, {}),
                                                  (INSIDE, False, {"method": ["slow", "fast"]})], ids=["plain", "search", "inside"])
def test_the_protocol_text_of_the_bodies_has_the_rules_of_the_bodies_and_no_experiment_py_block(
        tmp_path: Path, design: dict[str, Any], search: bool, inside: dict[str, Any]) -> None:
    engine = _engine(tmp_path, skeleton_on=True)
    on = engine._frame_protocols({"design": design}, search, inside, BODIES)
    assert "# file: experiment.py" not in on and "`# file: derive.py`" in on
    assert "**The analysis is FI's.**" in on and "figure_<id>(fig, table)" in on and "table.missing(" in on
    for old in ("FI_TRIALS", "FI_BEST_DESIGN", "FI_OPTIMISATION", "FI_OPTIMUM_CHECK", "RESULT_JSON"):
        assert old not in on, old
    if search:
        assert "FI also draws the search's own figures" in on
    if inside:
        assert "table.levels(id, factor, key)" in on
    assert on != engine._frame_protocols({"design": design}, search, inside)


def test_with_nothing_to_write_the_protocol_text_says_so(tmp_path: Path) -> None:
    on = _engine(tmp_path, skeleton_on=True)._frame_protocols({"design": DESIGN}, False, {}, [])
    assert "nothing of it for you to write" in on and "`# file: derive.py`" not in on


def test_the_cluster_text_still_comes_with_the_bodies(tmp_path: Path) -> None:
    engine = _engine(tmp_path, skeleton_on=True)
    engine.config.execution.background_jobs = True
    assert "# file: submit.py" in engine._frame_protocols({"design": DESIGN}, False, {}, BODIES)


def test_the_package_layout_names_derive_py_and_not_experiment_py() -> None:
    for text in (code_layout.prompt_block("colony", ["E1"], frame="prompt", analysis=True),
                 code_layout.prefilled_block("colony", "def g(): pass", frame="prompt", analysis=True)):
        assert "functions of derive.py" in text and "experiment.py" not in text
    plain = code_layout.prefilled_block("colony", "def g(): pass", frame="prompt")
    assert "and experiment.py." in plain, "the frame request without the analysis still names experiment.py"


# ---- the condition 4e calls ---------------------------------------------------------------------------------------------------------


def _fake_analysis_frame(monkeypatch: pytest.MonkeyPatch, applies: str = "") -> None:
    mod = types.ModuleType("core.analysis_frame")
    mod.applies = lambda design: applies  # type: ignore[attr-defined]
    mod.plan_of = lambda design: {"names": BODIES}  # type: ignore[attr-defined]
    mod.body_names = lambda plan: list(plan["names"])  # type: ignore[attr-defined]
    mod.analysis_problem = lambda old, new, name: ("the function `x` is missing" if "missing" in new else "")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "core.analysis_frame", mod)
    import core

    monkeypatch.setattr(core, "analysis_frame", mod, raising=False)


def test_the_bodies_are_asked_only_for_a_plan_in_the_contract_form(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_analysis_frame(monkeypatch)
    on = _engine(tmp_path, skeleton_on=True)
    assert on._analysis_bodies({"design": DESIGN}) == BODIES
    off = _engine(tmp_path, skeleton_on=False)
    assert off._analysis_bodies({"design": DESIGN}) is None, "the flag is off"
    _fake_analysis_frame(monkeypatch, applies="the plan is not in the contract form")
    assert on._analysis_bodies({"design": DESIGN}) is None, "an older-form plan keeps the analysis the model writes"


def test_without_the_analysis_frame_there_are_no_bodies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "core.analysis_frame", None)  # the import fails
    assert _engine(tmp_path, skeleton_on=True)._analysis_bodies({"design": DESIGN}) is None


# ---- the repair of a body -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body,sig,says", [
    ("derive_doubling_time", "def derive_doubling_time(table)", "table.missing"),
    ("reference_growth", "def reference_growth(setting)", "one setting"),
    ("figure_growth_curve", "def figure_growth_curve(fig, table)", "never calls `savefig`")])
def test_the_repair_of_a_body_names_it_and_gives_its_signature(body: str, sig: str, says: str) -> None:
    text = skeleton.reflect_derive(body)
    assert f"`{body}`" in text and sig in text and says in text
    assert "give no experiment.py" in text and "RESULT_JSON" not in text
    assert skeleton.body_signature(body) == sig


def test_a_repair_that_cannot_name_the_body_still_says_what_to_give() -> None:
    text = skeleton.reflect_derive("")
    assert "A function of derive.py failed" in text and "whole" in text
    assert skeleton.body_signature("run_trial") == "" and skeleton.body_signature("derive") == ""


def test_the_repair_request_puts_what_may_change_first_and_replaces_the_intro() -> None:
    prompt = _agent("execute_reflect")
    assert skeleton.REFLECT_INTRO in prompt
    on = skeleton.derive_prompt(DERIVE, prompt)
    assert on.startswith(skeleton.DERIVE_REPAIR_NOTE) and skeleton.REFLECT_INTRO_DERIVE in on
    assert "emits `RESULT_JSON" not in on
    assert skeleton.derive_prompt("import os\n", prompt) is prompt and skeleton.derive_prompt("", prompt) is prompt


def test_the_notes_of_a_rerun_and_an_extension_name_derive_py() -> None:
    assert "derive.py" in skeleton.RERUN_NOTE_ANALYSIS and "derive.py" in skeleton.EXTEND_REPLY_ANALYSIS
    assert "experiment.py whole" not in skeleton.EXTEND_REPLY_ANALYSIS and "experiment.py" in skeleton.EXTEND_REPLY


def test_repair_problem_sends_the_analysis_files_to_the_analysis_check(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_analysis_frame(monkeypatch)
    assert skeleton.repair_problem(DERIVE, DERIVE + "\n# a helper\n", "code/derive.py") == ""
    assert skeleton.repair_problem(DERIVE, "the function missing", "derive.py") == "the function `x` is missing"
    assert skeleton.repair_problem("x = 1\n", "the function missing", "experiment.py") == "the function `x` is missing"
    # the simulation and the runtime keep their own checks
    assert skeleton.repair_problem(FRAMED, UNFRAMED, "simulate.py") != ""
    assert skeleton.repair_problem("anything", "else", "notes.py") == ""


def test_repair_problem_without_the_analysis_frame_checks_nothing_of_these_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "core.analysis_frame", None)
    assert skeleton.repair_problem("x = 1\n", "x = 2\n", "experiment.py") == ""
    assert skeleton.repair_problem("x = 1\n", "x = 2\n", "derive.py") == ""


# ---- flag off, and a plan in an older form: every text is what it was (sha256 at df4ff1ad) -------------------------------------


def test_with_no_analysis_by_fi_every_text_is_what_it_was(tmp_path: Path) -> None:
    engine = _engine(tmp_path, skeleton_on=True)
    engine.config.execution.background_jobs = False
    pins: dict[str, str] = {}
    for name in ("implement", "implement_body"):
        pins[name] = _sha(run_checks.frame_template_text(_agent(name)))
        pins[name + "_cluster"] = _sha(run_checks.frame_template_text(_agent(name), True))
    pins["outline"] = _sha(run_checks.frame_outline_text(_agent("implement_outline")))
    pins["reply"] = _sha(run_checks._frame_reply())
    pins["reply_cluster"] = _sha(run_checks._frame_reply(True))
    pins["protocols_plain"] = _sha(engine._frame_protocols({"design": DESIGN}, False, {}))
    pins["protocols_search"] = _sha(engine._frame_protocols({"design": SEARCH}, True, {}))
    pins["protocols_inside"] = _sha(engine._frame_protocols({"design": INSIDE}, False, {"method": ["slow", "fast"]}))
    pins["note"] = _sha(engine._frame_note({"design": DESIGN}))
    engine.config.execution.background_jobs = True
    pins["protocols_cluster"] = _sha(engine._frame_protocols({"design": DESIGN}, False, {}))
    pins["layout_prompt"] = _sha(code_layout.prompt_block("colony", ["E1"], frame="prompt") + code_layout.prompt_block("colony", ["E1"]))
    pins["layout_prefilled"] = _sha(code_layout.prefilled_block("colony", "def g(): pass", frame="prompt")
                                    + code_layout.prefilled_block("colony", "def g(): pass"))
    pins["repair_problem"] = _sha(json.dumps([skeleton.repair_problem(FRAMED, UNFRAMED, "simulate.py"),
                                              skeleton.repair_problem(UNFRAMED, FRAMED, "simulate.py"),
                                              skeleton.repair_problem("x = 1\n", "x = 2\n", "experiment.py"),
                                              skeleton.repair_problem("x = 1\n", "x = 2\n", "derive.py")]))
    assert pins == PINS_AT_DF4FF1AD, json.dumps(pins, indent=1)


#: sha256 (first 16 hex digits) of the same texts as the code before this change made them (commit df4ff1ad).
PINS_AT_DF4FF1AD = {
    "implement": "a70893e8a2235784", "implement_cluster": "49b3f279b0a52b4b", "implement_body": "d02bdddb32e99d50",
    "implement_body_cluster": "251751d486f7962c", "outline": "fa01183231dce8c8", "reply": "71cdb845e7baa014",
    "reply_cluster": "f1e703776014745e", "protocols_plain": "dad3557413701e1d", "protocols_search": "4713448bfebdf253",
    "protocols_inside": "2659fc6972c46c6c", "note": "a617e362566b77f2", "protocols_cluster": "4a61049737d99ef8",
    "layout_prompt": "a8e42eea942bd234", "layout_prefilled": "3222bc6d76f04d56", "repair_problem": "75789d950a008886"}
