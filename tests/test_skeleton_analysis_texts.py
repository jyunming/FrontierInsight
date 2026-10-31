"""The requests and repairs of a study whose analysis FI writes too (``execution.code_skeleton`` and a plan in the contract form):
the model is asked for ``trial`` and the functions of derive.py (the quantities and figures the plan declares), never for
experiment.py, and a failed function is repaired by itself. Every text the flag off, or a plan in an older form, gets is what it
was (sha256 at df4ff1ad). Made-up topic: a colony's growth. No real model is called."""

from __future__ import annotations

import hashlib
import re
import importlib
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
    assert "`derive_doubling_time`" in on and "all of experiment.py" in on
    assert "for experiment.py, the functions of the analysis" not in on and "and experiment.py, constrained to" not in on
    assert "needs no function of derive.py" in run_checks.frame_outline_text(text, [])
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


def test_the_repair_notes_carry_the_level_rule_and_the_reference_rule() -> None:
    assert "once for each setting" in skeleton.reflect_derive("reference_growth")
    assert skeleton.DERIVE_LEVEL_RULE in skeleton.reflect_derive("derive_doubling_time", compared=True)
    assert skeleton.DERIVE_LEVEL_RULE not in skeleton.reflect_derive("derive_doubling_time")


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
    mod.frame_problem = lambda src: ""  # type: ignore[attr-defined]
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


# ---- no text of the new path speaks of a script the model no longer writes ---------------------------------------------------------

#: Phrases FI itself wrote for the single-script path; none may reach the model when FI writes the analysis.
FORBIDDEN = ("# file: experiment.py", "FI_TRIALS", "FI_BEST_DESIGN", "FI_OPTIMISATION", "FI_OPTIMUM_CHECK", "RESULT_JSON", "figures/",
             "figures-dir", "plt.subplots(", "under `\"why_missing\"`", "emit `null`", "run_grid", "THIRD fenced", "after the other two",
             "give the whole corrected script", "Save figures to", "Stratify when", "matplotlib.use", "os.makedirs",
             "raw results files", "the analysis holds", "single-file", "Single Python file", "Honour `FI_PILOT`",
             "Honour `FI_REPLICATE_SEED`")


def _new_path_texts(tmp_path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    both = {"protocol": {**INSIDE["protocol"], "optimisation": SEARCH["protocol"]["optimisation"]}}
    inside = {"method": ["slow", "fast"]}
    for name in ("implement", "implement_body"):
        for cluster in (False, True):
            texts[f"{name}/{cluster}"] = run_checks.frame_template_text(_agent(name), cluster, BODIES)
        texts[f"{name}/empty"] = run_checks.frame_template_text(_agent(name), False, [])
    texts["outline"] = run_checks.frame_outline_text(_agent("implement_outline"), BODIES)
    texts["outline/empty"] = run_checks.frame_outline_text(_agent("implement_outline"), [])
    engine = _engine(tmp_path, skeleton_on=True)
    for label, design, search, comp in (("plain", DESIGN, False, {}), ("search", SEARCH, True, {}), ("inside", INSIDE, False, inside),
                                        ("search+inside", both, True, inside)):
        texts["protocols/" + label] = engine._frame_protocols({"design": design}, search, comp, BODIES)
    texts["protocols/empty"] = engine._frame_protocols({"design": DESIGN}, False, {}, [])
    engine.config.execution.background_jobs = True
    texts["protocols/cluster"] = engine._frame_protocols({"design": DESIGN}, False, {}, BODIES)
    texts["protocols/cluster+empty"] = engine._frame_protocols({"design": DESIGN}, False, {}, [])
    for body in BODIES:
        texts["repair/" + body] = skeleton.reflect_derive(body, compared=True)
    texts["repair/request"] = skeleton.derive_prompt(DERIVE, _agent("execute_reflect"), "derive_doubling_time")
    texts["rerun"] = skeleton.RERUN_NOTE_ANALYSIS
    texts["extend"] = skeleton.EXTEND_REPLY_ANALYSIS
    texts["layout"] = (code_layout.prompt_block("colony", ["E1"], frame="prompt", analysis=True)
                       + code_layout.prefilled_block("colony", "def g(): pass", frame="prompt", analysis=True))
    return texts


def test_no_text_of_the_new_path_speaks_of_a_script_the_model_no_longer_writes(tmp_path: Path) -> None:
    texts = _new_path_texts(tmp_path)
    # submit.py's own pending/done lines are `RESULT_JSON` lines of the job (the cluster section is checked by its other phrases)
    cluster = run_checks._cluster_bodies()
    bad = [(name, phrase) for name, text in texts.items() for phrase in FORBIDDEN
           if phrase in (text.replace(cluster, "") if phrase == "RESULT_JSON" else text)]
    assert not bad, bad
    # the replaced rules are there instead
    assert "table.missing(" in texts["implement_body/False"] and "plt.subplots" not in texts["implement_body/False"]
    assert "one more fenced block" in texts["protocols/cluster+empty"] and "experiment.py" not in cluster and "FI_TRIALS" not in texts["protocols/cluster"]


#: The only sentences of the new path that name experiment.py: FI says it writes it, or that a reply must not contain it.
EXPERIMENT_PY_SAYINGS = (
    "FI writes experiment.py, you do not", "FI wrote experiment.py, which", "all of experiment.py (the statistics",
    "FI writes experiment.py and runs it once", "FI writes experiment.py; if the problem", "give experiment.py or touch",
    "(FI writes experiment.py)", "FI writes experiment.py, so leave out", "give no experiment.py")


def test_the_new_path_names_experiment_py_only_to_say_that_FI_writes_it(tmp_path: Path) -> None:
    cluster = run_checks._cluster_bodies()
    assert "experiment.py" not in cluster
    for name, text in _new_path_texts(tmp_path).items():
        rest = text
        for ok in EXPERIMENT_PY_SAYINGS:
            rest = rest.replace(ok, "")
        assert "experiment.py" not in rest, (name, rest[max(0, rest.index("experiment.py") - 80):][:160])
        # a word of the schema, not a value: the only `null` is the one in the give-up line of the repair request
        assert "null" not in rest.replace("leave empty/null otherwise", ""), name
        # no sentence that starts in lower case after a full stop (a broken join), except a file name
        flat = re.sub(r"\. (simulate|derive|experiment)\.py", ". X", rest)
        assert not re.search(r"\. [a-z]", flat), (name, re.search(r".{40}\. [a-z].{20}", flat, re.S))


def test_the_level_rule_gives_the_key_form_the_analysis_accepts() -> None:
    rule = skeleton.DERIVE_LEVEL_RULE
    assert "written as text" in rule and "not `method=bh`" in rule and "EVERY value" in rule and "table.levels shows" not in rule
    assert ". its " not in run_checks._IN_EACH_RUN_BODIES and rule in run_checks._IN_EACH_RUN_BODIES


def test_the_repair_opening_says_which_function_failed() -> None:
    on = skeleton.derive_prompt(DERIVE, _agent("execute_reflect"), "derive_doubling_time")
    assert "The function `derive_doubling_time` of derive.py just failed" in on and "experiment script just failed" not in on
    assert "A function of derive.py just failed" in skeleton.derive_prompt(DERIVE, _agent("execute_reflect"))


def test_the_report_bullet_says_in_place_what_trial_does_with_a_missing_value() -> None:
    text = run_checks.frame_template_text(_agent("implement_body"), False, BODIES)
    assert "as said above" not in text and "in `trial`, return `None` and give the reason beside it" in text


def test_the_repair_request_of_a_body_has_no_rule_of_a_script_that_prints_its_results() -> None:
    on = skeleton.derive_prompt(DERIVE, _agent("execute_reflect"))
    assert "patched function of derive.py" in on and "table.missing(" in on and "<leave empty: give the functions only>" in on
    assert "$result_json_present" not in on and "$figures_count" not in on


def test_body_params_are_the_signatures_of_the_analysis_frame() -> None:
    try:
        analysis_frame = importlib.import_module("core.analysis_frame")
    except ModuleNotFoundError:
        pytest.skip("core.analysis_frame is not in this branch")
    assert {k: [x.strip() for x in v.split(",")] for k, v in skeleton.BODY_PARAMS.items()} == analysis_frame.SIGNATURES


def test_a_quest_with_an_older_frame_keeps_the_analysis_the_model_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_analysis_frame(monkeypatch)
    sys.modules["core.analysis_frame"].frame_problem = lambda src: "the frame is version 2" if "version 2" in src else ""  # type: ignore[attr-defined]
    engine = _engine(tmp_path, skeleton_on=True)
    assert engine._analysis_bodies({"design": DESIGN}) == BODIES
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    # A frame an older FI wrote for the same plan: the quest keeps it (skeleton.frame_for), and with it the model's analysis
    older = skeleton.frame_source(DESIGN, stochastic=True).replace(f"version {skeleton.VERSION}", "version 2")
    (code / "simulate.py").write_text(older, encoding="utf-8")
    assert engine._analysis_bodies({"design": DESIGN}) is None


def test_a_fault_inside_the_analysis_frame_is_not_taken_for_its_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Broken:
        def find_spec(self, name, path=None, target=None):  # noqa: ANN001
            if name == "core.analysis_frame":
                raise ModuleNotFoundError("No module named 'a_package_it_needs'", name="a_package_it_needs")
            return None

    monkeypatch.delitem(sys.modules, "core.analysis_frame", raising=False)
    import core

    monkeypatch.delattr(core, "analysis_frame", raising=False)
    finder = _Broken()
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(ModuleNotFoundError):
            _engine(tmp_path, skeleton_on=True)._analysis_bodies({"design": DESIGN})
        with pytest.raises(ModuleNotFoundError):
            skeleton.repair_problem("x = 1\n", "x = 2\n", "derive.py")
    finally:
        sys.meta_path.remove(finder)


# ---- flag off, and a plan in an older form: every text is what it was (sha256 at df4ff1ad) -------------------------------------


def test_with_no_analysis_by_fi_every_text_is_what_it_was(tmp_path: Path) -> None:
    # "note" shows the frame FI writes, so it carries the frame's version line: 4 since the runtime measures whether a trial drew random numbers (skeleton.VERSION; it was 3 before);
    # it differs from df4ff1ad only in that line (the frame is written only with the frame on), and in the sentences that say
    # which names a trial may return (one list for the frame and the rules; the frame is written only with the frame on).
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
    "protocols_inside": "2659fc6972c46c6c", "note": "c097e36a2cf46a3b", "protocols_cluster": "4a61049737d99ef8",
    "layout_prompt": "a8e42eea942bd234", "layout_prefilled": "3222bc6d76f04d56", "repair_problem": "75789d950a008886"}


def test_with_no_analysis_by_fi_the_methods_and_the_unpinned_texts_are_what_they_were(tmp_path: Path) -> None:
    import string

    engine = _engine(tmp_path, skeleton_on=True)
    engine.config.execution.background_jobs = False
    both = {"protocol": {**INSIDE["protocol"], "optimisation": SEARCH["protocol"]["optimisation"]}}
    state = {"design": DESIGN}
    pins = {
        "protocols_search_inside": _sha(engine._frame_protocols({"design": both}, True, {"method": ["slow", "fast"]})),
        "template_implement": _sha(engine._frame_template(state, string.Template(_agent("implement"))).template),
        "template_implement_body": _sha(engine._frame_template(state, string.Template(_agent("implement_body"))).template),
        "outline_template": _sha(engine._frame_outline_template(state, string.Template(_agent("implement_outline"))).template),
        "repair_fi_frame": _sha(json.dumps([skeleton.repair_problem("x", "y", "fi_frame.py"),
                                            skeleton.repair_problem(skeleton.RUNTIME_HEAD + "a", "y", "fi_frame.py")])),
        "reflect_prompt": _sha(skeleton.framed_prompt(FRAMED, _agent("execute_reflect"))),
    }
    assert pins == MORE_PINS_AT_DF4FF1AD, json.dumps(pins, indent=1)


MORE_PINS_AT_DF4FF1AD = {
    "protocols_search_inside": "224963aed9fe2a67", "template_implement": "a70893e8a2235784",
    "template_implement_body": "d02bdddb32e99d50", "outline_template": "fa01183231dce8c8",
    "repair_fi_frame": "2aca732a1c3b9c74", "reflect_prompt": "40e6696c66d0e48e"}
