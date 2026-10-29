"""Every request and repair of a quest where FI writes the frame of simulate.py (``execution.code_skeleton``) asks for the function
``trial`` and its helpers, never for an entry point, the seed, a loop or the whole file (core/skeleton.py): the first request
(plain, search, values compared inside each run, cluster), the repair of a crash, of an oracle check, of a reader's finding, the
update of a revised plan, the extension for a person's request and the rerun. With the frame off every one of them is what it
was. Made-up topic: a colony's growth. No real model is called."""

from __future__ import annotations

import hashlib
import json
import re
import string
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import in_run, oracle_check, skeleton, small_run, trial_checks
from core.engine import _SPLIT_RERUN_NOTE, _TRIAL_ORACLE_NOTE, _extend_directive
from tests.test_skeleton_ask import BOTH, DESIGN, OUTLINE, STATE, _engine, _implement, _Model
from tests.test_skeleton_repairs import BODY, FRAMED, UNFRAMED
from tests.test_skeleton_repairs import _crash_state
from tests.test_skeleton_repairs import _engine as _repair_engine

#: What a request must not ask of a simulate.py that has FI's frame (the frame's own listing and the note that forbids it are cut).
WHOLE = re.compile(r"run_trial|run_cell|return the whole|Return the whole|whole script|complete script|single Python file|"
                   r"Single Python file|FI_REPLICATE_SEED|FI_PILOT|FI_ORACLE")

INSIDE = {"protocol": {**DESIGN["protocol"], "in_each_run": {"method": ["slow", "fast"]}}}
SEARCH = {"protocol": {"optimisation": {
    "objective": {"quantity": "drag", "direction": "minimise"}, "constraints": [{"quantity": "mass", "limit": "<= 5"}],
    "design_variables": [{"name": "width", "low": 1, "high": 4}], "fixed": {"speed": 3.0},
    "numerical_settings": {"mesh": {"search": 8}}, "evaluation_budget": {"starts": 2, "per_start": 40},
    "baseline": {"values": {"width": 2}, "source": "the topic"}}}}


def _asked(prompt: str) -> str:
    """The request without FI's frame as shown to the model and without the repair note (which names what is forbidden)."""
    text = prompt.replace(skeleton.REPAIR_NOTE, "").replace(skeleton.REPAIR_NOTE_WHOLE, "")
    return re.sub(r"# >>> FI frame.*?# <<< FI frame", "<FRAME>", text, flags=re.S)


def _no_whole_ask(prompt: str, *, allow: str = "") -> None:
    text = _asked(prompt)
    for ok in filter(None, [allow]):
        text = text.replace(ok, "")
    hit = WHOLE.search(text)
    assert hit is None, f"{hit.group(0)!r}: {text[max(0, hit.start() - 150):hit.end() + 150]!r}"


# ---- the first request ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("design", [DESIGN, INSIDE, SEARCH], ids=["plain", "inside", "search"])
async def test_the_first_request_asks_for_nothing_of_the_frame(tmp_path: Path, design: dict[str, Any]) -> None:
    engine = _engine(tmp_path, skeleton_on=True)
    await engine.executor.setup(engine.quest_root)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    model = _Model([], BOTH)
    engine._client = model
    await engine._node_implement({"design": design, "implement_outline": OUTLINE})
    prompt = model.prompts[0]
    assert skeleton.has_frame(_frame_of(engine, design)), "the request is a frame-mode request"
    instructions = _asked(prompt)
    # the outline's own description of its program and the study's protocol text are the model's inputs, not asks
    instructions = instructions.split("## Outline")[0] + instructions.split("## Wall-time limit")[-1]
    assert not WHOLE.search(instructions), WHOLE.search(instructions).group(0)


def _frame_of(engine: Any, design: dict[str, Any]) -> str:
    return engine._skeleton_frame({"design": design})


@pytest.mark.asyncio
async def test_a_cluster_request_names_submit_py_in_the_reply_format(tmp_path: Path) -> None:
    engine = _engine(tmp_path, skeleton_on=True)
    engine.config.execution.background_jobs = True
    await engine.executor.setup(engine.quest_root)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    model = _Model([], BOTH + "\n```python\n# file: submit.py\nprint('RESULT_JSON: {}')\n```\n")
    engine._client = model
    await engine._node_implement(dict(STATE))
    prompt = model.prompts[0]
    fmt = prompt[prompt.index("# Output format"):prompt.index("# Inputs")]
    assert "`# file: submit.py`" in fmt and "EXACTLY these" in fmt, "the format says the third block is submit.py"
    assert fmt.index("simulate.py") < fmt.index("experiment.py") < fmt.index("submit.py") < fmt.index("DEPS:")
    plain = _engine(tmp_path / "plain", skeleton_on=True)
    assert "submit.py" not in skeleton_reply(plain)


def skeleton_reply(engine: Any) -> str:
    from core import run_checks

    return run_checks._frame_reply(bool(engine.config.execution.background_jobs))


# ---- the outline ---------------------------------------------------------------------------------------------------------------


def test_the_outline_request_lists_trial_and_its_helpers_and_no_loop_or_main_of_the_simulation(tmp_path: Path) -> None:
    on = _engine(tmp_path, skeleton_on=True)
    tmpl = on._prompts["implement_outline"]
    text = on._frame_outline_template(STATE, tmpl).template
    assert "FI writes the frame of simulate.py" in text and "no loop over settings or trials and no main" in text
    assert "single Python file" not in text and "belong to the scenario" not in text
    off = _engine(tmp_path / "off", skeleton_on=False)
    assert off._frame_outline_template(STATE, off._prompts["implement_outline"]) is off._prompts["implement_outline"]
    # an outline already written is shown with FI's own entry points left out, and one general sentence says the rest
    block = on._frame_protocols(STATE, False, {})
    assert "no loop over settings or trials and no main" in block


# ---- the repair of a crash -------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("design", [{"hypothesis": "h"}, SEARCH], ids=["plain", "search"])
async def test_a_crash_repair_of_a_framed_simulation_says_to_repair_trial(tmp_path: Path, design: dict[str, Any]) -> None:
    broken = FRAMED.replace('_scale(cell["rate"]', 'offset + _scale(cell["rate"]')
    eng, chat = _repair_engine(tmp_path, broken, {"functions": BODY, "patch_summary": "removed the undefined name"})
    state = _crash_state()
    state["design"] = design
    await eng._node_execute_reflect(state)
    prompt = chat.await_args.args[0][-1]["content"]
    assert prompt.startswith(skeleton.REPAIR_NOTE), "what may change comes first"
    assert skeleton.REFLECT_SIMULATE in prompt and "return the whole" not in prompt
    assert (skeleton.REFLECT_SEARCH in prompt) == ("optimisation" in design.get("protocol", {}))
    _no_whole_ask(prompt)


# ---- the repair of an oracle check ---------------------------------------------------------------------------------------------


def test_an_oracle_repair_of_a_framed_simulation_names_trial_and_asks_for_functions() -> None:
    oracles = [{"name": "zero_dose", "case": {"dose": 0}, "measure": "growth", "expected": 0.0, "tolerance": 0.01}]
    framed = oracle_check.directive(oracles, ["zero_dose: growth is 0.4"], trial=True, frame=True) + skeleton.ORACLE_NOTE
    _no_whole_ask(framed)
    assert "`trial`" in framed and "give each function you change whole" in framed
    old = oracle_check.directive(oracles, ["zero_dose: growth is 0.4"], trial=True) + _TRIAL_ORACLE_NOTE
    assert "return the whole of simulate.py" in old, "a simulation without a frame is asked as before"


@pytest.mark.asyncio
async def test_the_oracle_repair_request_is_chosen_by_the_file_it_repairs(tmp_path: Path) -> None:
    oracles = [{"name": "zero_dose", "case": {"dose": 0}, "measure": "growth", "expected": 0.0, "tolerance": 0.01}]
    for text, framed in ((FRAMED, True), (UNFRAMED, False)):
        eng, chat = _repair_engine(tmp_path / ("f" if framed else "u"), text, {"functions": BODY, "patch_summary": "x"})
        eng._trial_mode = True
        await eng._repair_script_for_oracle({"design": {"hypothesis": "h"}}, eng.quest_root / "code" / "simulate.py", oracles,
                                            ["zero_dose: growth is 0.4"])
        prompt = chat.await_args.args[0][-1]["content"]
        if framed:
            assert prompt.startswith(skeleton.REPAIR_NOTE)
            _no_whole_ask(prompt)
        else:
            assert "return the whole of simulate.py" in prompt and not prompt.startswith("FIRST, what you may change")


# ---- the repair of a reader's finding ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("function", ["trial", ""], ids=["one function", "no function named"])
def test_the_code_review_repair_of_a_framed_simulation_keeps_no_handling_of_the_pilot_or_the_seed_to_ask_for(
        tmp_path: Path, function: str) -> None:
    from tests.test_skeleton_repairs import FIXED_TRIAL, TEST, _block, _Run

    real = string.Template((Path(__file__).parent.parent / "agents" / "code_review_repair.md").read_text(encoding="utf-8"))
    framed = _Run(tmp_path / "f", FRAMED, {"code:test": [_block(TEST)], "code:repair": [_block(FIXED_TRIAL)]}, function)
    framed.env.prompts["code_review_repair"] = real
    framed.go()
    prompt = [p for p in framed.prompts if "# How to answer" in p][-1]
    assert "FI_PILOT" not in prompt and "FI_REPLICATE_SEED" not in prompt and "RESULT_JSON" not in prompt and "and the frame." in prompt
    assert prompt.startswith(skeleton.REPAIR_NOTE if function else skeleton.REPAIR_NOTE_WHOLE)
    plain_test = ("import simulate\n\ndef test_growth():\n    out = simulate.run_trial({'rate': 2.0, 'dose': 3.0}, 0, 1)\n"
                  "    assert out['growth'] > 6.0, 'growth lacks the offset'\n")
    fixed = "# file: simulate.py\n" + UNFRAMED.replace("0.0", "1.0")
    plain = _Run(tmp_path / "u", UNFRAMED, {"code:test": [_block(plain_test)], "code:repair": [_block(fixed)]}, "")
    plain.env.prompts["code_review_repair"] = real
    plain.go()
    assert "the handling of `FI_PILOT` and `FI_REPLICATE_SEED`" in [p for p in plain.prompts if "# How to answer" in p][-1], "no frame: asked as before"


# ---- the update of a revised plan, the extension for a person's request, the rerun ---------------------------------------------


def test_the_update_of_a_revised_plan_asks_for_functions_only(tmp_path: Path) -> None:
    eng = _engine(tmp_path, skeleton_on=True)
    tmpl = string.Template(eng._prompts["implement_update"].template)
    prompt = skeleton.repair_note(FRAMED) + tmpl.substitute(
        topic_block="t", changes_block="c", package_block="(none)", simulate_code=FRAMED, analysis_code="print(1)",
        design_block="d", clarify_block="c")
    _no_whole_ask(prompt)
    assert "ONLY those functions" in prompt


def test_the_extension_and_the_rerun_ask_for_trial_and_its_helpers_only() -> None:
    text = _extend_directive({"simulate.py": FRAMED, "experiment.py": "print(1)"}, ["a number for the dose 0"], frame=True)
    _no_whole_ask(text)
    assert skeleton.EXTEND_REPLY in text
    assert "Return the whole script(s)" in _extend_directive({"simulate.py": UNFRAMED}, ["a number"]), "unframed: as before"
    _no_whole_ask(skeleton.RERUN_NOTE)
    assert "`trial`" in skeleton.RERUN_NOTE and "Give simulate.py back exactly" in _SPLIT_RERUN_NOTE


# ---- ONE rule about a value that does not exist, from FI's own record -------------------------------------------------------------


def _missing_key_reason() -> str:
    """FI's own message for a run that gave a result for some compared values and not for all."""
    levels = in_run.level_cells({"method": ["slow", "fast"]})
    value, why = in_run.read_returned({"score": {"method=slow": 1.0}}, levels, "run_trial")
    assert value is None
    return "TypeError: " + why


def _failed(reason: str) -> small_run.SmallRun:
    s = small_run.Setting("rate=0.1", 3, 0, 3, False, [(reason, 3)])
    return small_run.SmallRun([s], "")


def test_a_failed_small_run_that_lacks_a_compared_value_gets_one_rule_that_names_the_keys() -> None:
    reason = _missing_key_reason()
    assert trial_checks.is_level_fault(reason) and not trial_checks.is_level_fault("NameError: name 'x' is not defined")
    rule = skeleton.level_rule(INSIDE["protocol"], FRAMED)
    assert "method=slow" in rule and "out.each(" in rule and "NaN (or None)" in rule
    text = _failed(reason).directive(rule)
    assert "method=fast" in text, "the missing key is named"
    assert text.count("Never return NaN") == 1 and text.count(rule) == 1 and "float('nan')" not in text
    # a failure of another kind, a load error, a study without compared values and a simulation without a frame: no rule
    assert rule not in _failed("NameError: name 'x' is not defined").directive(rule)
    assert rule not in small_run.SmallRun([], "SyntaxError: bad").directive(rule)
    assert skeleton.level_rule(DESIGN["protocol"], FRAMED) == "" and skeleton.level_rule(INSIDE["protocol"], UNFRAMED) == ""


def test_the_failures_of_a_study_run_say_the_same_rule() -> None:
    reason = _missing_key_reason()
    rule = skeleton.level_rule(INSIDE["protocol"], FRAMED)
    bad = [trial_checks.SettingFailures("rate=0.1", 3, 3, [(reason, 3)])]
    text = trial_checks.failures_directive(bad, rule)
    assert text.count(rule) == 1 and "Never return NaN" in text
    assert rule not in trial_checks.failures_directive([trial_checks.SettingFailures("rate=0.1", 3, 3, [("boom", 3)])], rule)


# ---- with the frame off, every changed request and message is what it was (sha256 at 6ad6c160) --------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_with_no_frame_every_changed_message_is_what_it_was() -> None:
    reason = _missing_key_reason()
    oracles = [{"name": "zero_dose", "case": {"dose": 0}, "measure": "growth", "expected": 0.0, "tolerance": 0.01}]
    pins = {
        "small_run": _sha(_failed(reason).directive()),
        "small_run_other": _sha(_failed("NameError: x").directive()),
        "failures": _sha(trial_checks.failures_directive([trial_checks.SettingFailures("rate=0.1", 3, 3, [(reason, 3)])])),
        "oracle": _sha(oracle_check.directive(oracles, ["zero_dose: growth is 0.4"], trial=True)),
        "oracle_script": _sha(oracle_check.directive(oracles, ["zero_dose: growth is 0.4"])),
        "extend": _sha(_extend_directive({"simulate.py": UNFRAMED, "experiment.py": "print(1)"}, ["a number"], "")),
        "repair_note": _sha(skeleton.repair_note(UNFRAMED) + skeleton.repair_note("")),
    }
    assert pins == PINS_AT_6AD6C160, json.dumps(pins)


#: sha256 (first 16 hex digits) of the same messages as the code before this change made them (commit 6ad6c160).
PINS_AT_6AD6C160 = {
    "small_run": "c46af295127b2217", "small_run_other": "7ad3e4770282781c", "failures": "a447049a7fe88a5e",
    "oracle": "c8b1dcea470a2f8e", "oracle_script": "89cff3048184af42", "extend": "8503839e96ba908c", "repair_note": "e3b0c44298fc1c14"}


# ---- round 4: the search path, the frame's own business, the outline's replace targets ---------------------------------------------


def test_a_search_that_compares_values_inside_each_run_gets_the_same_rule_for_a_missing_key(tmp_path: Path) -> None:
    import time

    from core import optimise

    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text(FRAMED, encoding="utf-8")
    protocol = {"optimisation": {"in_each_run": {"method": ["slow", "fast"]}}}
    reason = _missing_key_reason()
    rows = ([{"event": "evaluation", "stage": "start", "status": "ok", "design": {"x": n}} for n in range(3)]
            + [{"event": "evaluation", "stage": "start", "status": "failed", "design": {"x": 100 + n}, "reason": reason}
               for n in range(7)])
    runner = optimise.OptimisationRunner(None, quest_root=tmp_path, protocol=protocol, simulate=code / "simulate.py",
                                         analysis=code / "experiment.py", repairs_left=lambda: True)
    sent = runner._check_trials(optimise.SearchRun(record={}, rows=rows), time.monotonic())
    rule = skeleton.level_rule(protocol, FRAMED)
    assert sent is not None and sent.stderr.count(rule) == 1 and sent.stderr.count("Never return NaN") == 1
    # a failure of another kind, or a simulation without a frame: no rule
    other = [dict(r, reason="ZeroDivisionError: x") if r["status"] == "failed" else r for r in rows]
    assert rule not in runner._check_trials(optimise.SearchRun(record={}, rows=other), time.monotonic()).stderr
    (code / "simulate.py").write_text(UNFRAMED, encoding="utf-8")
    assert rule not in runner._check_trials(optimise.SearchRun(record={}, rows=rows), time.monotonic()).stderr


def test_with_the_frame_on_the_repairs_do_not_ask_to_keep_the_pilot_the_seed_or_the_final_result_line() -> None:
    from core.engine import _protocol_directive

    protocol = {"grid": {"a": [1]}}
    framed = _protocol_directive(protocol, [], frame=True)
    assert "FI_PILOT" not in framed and "FI_REPLICATE_SEED" not in framed and "RESULT_JSON" not in framed
    assert "and the frame FI wrote" in framed
    assert _sha(_protocol_directive(protocol, [])) == "c5a91f7539d93fb0", "no frame: what it was at 6ad6c160"
    intro = Path(__file__).parent.parent / "agents" / "execute_reflect.md"
    text = intro.read_text(encoding="utf-8")
    assert skeleton.REFLECT_INTRO in text, "the sentence the frame replaces is still in the prompt"
    on = skeleton.framed_prompt(FRAMED, text)
    assert "emits `RESULT_JSON" not in on and skeleton.REFLECT_INTRO_FRAME in on and on.startswith(skeleton.REPAIR_NOTE)
    assert skeleton.framed_prompt(UNFRAMED, text) is text and skeleton.framed_prompt("", text) is text


def test_the_replacements_of_the_frame_requests_find_their_text_in_the_prompt_files() -> None:
    from core import run_checks

    agents = Path(__file__).parent.parent / "agents"
    outline = (agents / "implement_outline.md").read_text(encoding="utf-8")
    changed = run_checks.frame_outline_text(outline)
    assert changed != outline
    assert "FI writes the frame of simulate.py" in changed and "belong to FI's frame" in changed
    assert "single Python file" not in changed and "belong to the scenario" not in changed
    review = (agents / "code_review_repair.md").read_text(encoding="utf-8")
    assert ", the handling of `FI_PILOT` and `FI_REPLICATE_SEED`, and the final `RESULT_JSON: {...}` line." in review
