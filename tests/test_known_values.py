"""A commissioned study's known values are computed from their sources before the plan (core/known_values.py).

The model is a fake: it returns small programs; FI runs them itself, apart from the quest, and judges what they printed.
A value both programs reproduce goes on to the plan, which is told so; a value two independent programs agree is other
than the commission says hands the study back (``needs/STUCK.json``, ``known_value_mismatch``) before the plan and the
design; anything FI cannot judge goes on, said to the plan. The value is never in a request for a program.

The topic of these tests is a made-up draining tank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from core import commission, known_values

CASE = "the reference tank, half full, outlet open"
TITLE = "A made-up tank note"
FIRST = "# Compute a Known Value From Its Source"


def _commission(folder: Path, value: float, tolerance: float) -> Path:
    """A valid commission whose one known value is ``value``, stated in its passage and its full text."""
    folder.mkdir(parents=True, exist_ok=True)
    passage = f"The drain constant of the reference tank is {value} per hour under the stated inlet conditions"
    (folder / "tank_note.md").write_text(("A made-up note on a draining tank, used only to test FI. " * 6) + passage
                                         + ". The level falls as the exponential of minus the drain constant times the time.",
                                         encoding="utf-8")
    data = {"question": "How fast does a tank drain through outlets of three sizes?", "scope": "three outlets",
            "limits": {"hours": 2, "model_calls": 60},
            "known_values": [{"name": "drain constant", "value": value, "tolerance": tolerance, "case": CASE,
                              "passage": passage, "source": {"title": TITLE, "doi": "10.0000/tank.1", "file": "tank_note.md"}}],
            "success": [{"condition": "the drain time falls with the outlet", "verify": "compare the runs"}],
            "failure": [{"condition": "the drain time does not depend on the outlet", "verify": "compare the runs"}],
            "doubts": []}
    path = folder / "commission.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert commission.problems(path) == [], "a commission FI's door lets in"
    return path


def _program(*routes: tuple[str, str, str], cannot: str = "") -> str:
    """A reply holding a program that prints one ``KNOWN_JSON`` line per (route, value expression, error expression)."""
    body = ["import json", "", "CASE = {\"outlet\": \"open\", \"fill\": \"half\"}", ""]
    for route, value, error in routes:
        body.append(f"print('KNOWN_JSON: ' + json.dumps({{'route': '{route}', 'value': {value}, 'error': {error}, "
                    "'how': 'made up for a test'}))")
    if cannot:
        body.append(f"print('KNOWN_JSON: ' + json.dumps({{'cannot': '{cannot}'}}))")
    return "```python\n" + "\n".join(body) + "\n```"


RIGHT = _program(("formula", "1 / 4", "0"), ("definition", "2501 / 10000", "2 / 10000"))  # 0.25 both ways


class Model:
    """The plan's model: each request for a program gets the next reply of ``programs``; the plan gets a draft."""

    def __init__(self, eng: Any, programs: list[str]) -> None:
        from tests.test_plan_optimisation import EXTRA, _NO_OBJECTIONS
        from tests.test_run_checks import _draft

        self.programs, self.asked, self.other = list(programs), [], []
        own = {"name": "drain constant", "kind": "special_case", "check": "the drain constant of the tank",
               "expected": 0.3, "tolerance": 0.5, "reference": "derivation: a guess", "case": {"n": 1}, "measure": "k"}
        draft = _draft({"n": [1, 2, 4]}, 4)
        draft["protocol"]["oracles"].append(own)
        self.draft = json.dumps({**draft, "plan": EXTRA})
        self.no_objections = _NO_OBJECTIONS
        eng._client = type("Stub", (), {"chat": AsyncMock(side_effect=self.chat), "aclose": AsyncMock()})()

    async def chat(self, messages: list[dict[str, str]], **_kw: Any) -> str:
        prompt = messages[-1]["content"]
        if prompt.startswith(FIRST):
            self.asked.append(prompt)
            return self.programs.pop(0)
        self.other.append(prompt)
        return self.draft if len(self.other) == 1 else self.no_objections


async def _plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: float, tolerance: float,
                programs: list[str]) -> tuple[Any, Model, dict[str, Any]]:
    from core.engine import Engine
    from tests.test_commission import _engine

    async def off(self: Any, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)
    path = _commission(tmp_path / "order", value, tolerance)
    eng = _engine(tmp_path, path)
    commission.stage(path, eng.quest_root)
    lit = await eng._node_literature({"topic": eng.config.topic})
    model = Model(eng, programs)
    out = await eng._node_plan({"topic": eng.config.topic, "iteration": 0, **lit})
    return eng, model, out


def _record(eng: Any) -> dict[str, Any]:
    return json.loads((eng.quest_root / ".fi" / known_values.RECORD).read_text(encoding="utf-8"))


def _hidden(model: Model, value: float) -> None:
    for prompt in model.asked:
        assert not commission.number_in_text(value, prompt), "the value is never in a request for a program"


@pytest.mark.asyncio
async def test_a_value_both_routes_reproduce_goes_on_and_the_plan_is_told_and_fi_writes_its_check(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [RIGHT])
    assert "stuck" not in out and len(model.asked) == 1, "one program, then the plan"
    assert (eng.quest_root / "plan.md").is_file() and not (eng.quest_root / "needs" / "STUCK.json").exists()
    value = _record(eng)["values"][0]
    assert value["verdict"] == "match" and [r["value"] for r in value["programs"][0]["routes"]] == [0.25, 0.2501]
    assert "WHAT FI COMPUTED OF THE KNOWN VALUES" in model.other[0] and "'drain constant' (0.25): confirmed" in model.other[0]
    _hidden(model, 0.25)
    assert TITLE in model.asked[0] and CASE in model.asked[0] and known_values.WITHHELD in model.asked[0]
    from core import plan

    check = next(o for o in plan.load_design(eng.quest_root)[0]["protocol"]["oracles"] if o["name"] == "drain constant")
    assert check["kind"] == "published_value" and check["expected"] == 0.25 and check["tolerance"] == 0.01, (
        "FI writes the commission's number over the one the plan's model typed")
    assert check["reference"].startswith("[1]") and check["case"] == {"n": 1} and check["measure"] == "k"
    assert "has no check in this plan" not in (eng.quest_root / "plan.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_two_routes_further_apart_than_their_errors_twice_hand_the_study_back_before_the_plan(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each route is within the commission's tolerance (0.05) of its value; only the two routes, held to their own errors
    # (0 and 0.003), tell that one of them is wrong: 0.6125 and 0.653 are 0.04 apart.
    apart = _program(("formula", "49 / 80", "0"), ("definition", "653 / 1000", "3 / 1000"))
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.6125, 0.05, [apart, apart])
    assert len(model.asked) == 2 and model.other == [], "a second program, and no plan is asked for"
    assert "Another program" not in model.asked[0] and "Another program computed this value" in model.asked[1]
    assert "653" not in model.asked[1], "the second program is not shown the first one's numbers"
    _hidden(model, 0.6125)
    record = json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert list(record)[0] == "return_reason" and record["return_reason"] == "known_value_mismatch"
    assert out == {"stuck": record} and not (eng.quest_root / "plan.md").exists()
    assert eng._route_after_plan(out) == "returned" and eng._route_after_plan({}) == "design"
    ev = record["evidence"][0]
    assert ev["value"] == 0.6125 and ev["tolerance"] == 0.05 and ev["passage"].startswith("The drain constant")
    assert [[r["value"] for r in p["routes"]] for p in ev["programs"]] == [[0.6125, 0.653], [0.6125, 0.653]]
    assert all(p["sha256"] and p["program"] for p in ev["programs"])
    assert "further apart than their errors" in record["problem"]


@pytest.mark.asyncio
async def test_a_changed_known_value_is_caught_and_handed_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The planted fault: the commission says 0.75 where its source's law gives 0.25, and both programs compute 0.25.
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.75, 0.01, [RIGHT, RIGHT])
    record = json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert record["return_reason"] == "known_value_mismatch" and out == {"stuck": record}
    assert "not 0.75 within 0.01" in record["problem"] and model.other == []


@pytest.mark.asyncio
async def test_a_first_program_off_and_a_second_that_reproduces_the_value_go_on_unconfirmed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = _program(("formula", "2 / 5", "0"), ("definition", "4001 / 10000", "1 / 10000"))
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [wrong, RIGHT])
    assert "stuck" not in out and not (eng.quest_root / "needs" / "STUCK.json").exists()
    value = _record(eng)["values"][0]
    assert value["verdict"] == "cannot_judge" and "a second, independent one reproduced it" in value["why"]
    assert "'drain constant' (0.25): not confirmed by FI's own computation" in model.other[0]


@pytest.mark.asyncio
async def test_a_program_that_types_the_value_in_is_asked_again_and_then_the_value_is_unconfirmed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    typed = _program(("formula", "0.25", "0"))
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [typed, typed])
    assert "stuck" not in out and len(model.asked) == 2
    assert "types in a number close to the value withheld" in model.asked[1]
    _hidden(model, 0.25)
    value = _record(eng)["values"][0]
    assert value["verdict"] == "cannot_judge" and "no usable program after two requests" in value["why"]


@pytest.mark.asyncio
async def test_a_program_that_says_the_text_cannot_give_the_value_goes_on_unconfirmed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [_program(cannot="the text gives no outlet area")])
    assert "stuck" not in out and len(model.asked) == 1
    value = _record(eng)["values"][0]
    assert value["verdict"] == "cannot_judge" and "the text gives no outlet area" in value["why"]
    assert "not confirmed by FI's own computation" in model.other[0]


@pytest.mark.asyncio
async def test_a_resume_reads_the_record_back_and_asks_for_no_program(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, model, _out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [RIGHT])
    (eng.quest_root / "plan.md").unlink()
    again = Model(eng, [])
    await eng._node_plan({"topic": eng.config.topic, "iteration": 0,
                          **await eng._node_literature({"topic": eng.config.topic})})
    assert again.asked == [] and "confirmed" in again.other[0]


@pytest.mark.asyncio
async def test_without_a_commission_nothing_runs(tmp_path: Path) -> None:
    from core.config import Config, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine

    eng = Engine(Config(topic="How fast does a made-up tank drain?", provider=ProviderConfig(name="openai"),
                        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out")))
    assert await eng._known_values_first({"topic": "t"}) is None
    assert known_values.plan_block(None) == ""
    assert not (eng.quest_root / ".fi" / known_values.RECORD).exists()


@pytest.mark.asyncio
async def test_a_study_handed_back_at_the_plan_never_reaches_the_design(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from tests.test_commission import _engine

    async def nothing(self: Any, state: Any) -> dict[str, Any]:
        return {}

    async def design(self: Any, state: Any) -> dict[str, Any]:
        raise AssertionError("the design step ran")

    for name in ("_node_clarify_then_phased", "_node_pause_after_literature", "_node_select_skills"):
        monkeypatch.setattr(Engine, name, nothing)
    monkeypatch.setattr(Engine, "_node_design", design)
    path = _commission(tmp_path / "order", 0.75, 0.01)
    eng = _engine(tmp_path, path)
    commission.stage(path, eng.quest_root)
    model = Model(eng, [RIGHT, RIGHT])
    final = await eng._build_graph().compile().ainvoke({"topic": eng.config.topic, "iteration": 0})
    assert final["stuck"]["return_reason"] == "known_value_mismatch" and model.other == []


F, D = "formula", "definition"


@pytest.mark.parametrize("routes,value,tolerance,verdict", [
    ([(F, 2.38462, 0.0)], 2.3846, 1e-4, "match"),  # one route is enough for a value no program types by chance
    ([(D, 0.5, 0.0)], 0.5, 0.05, "cannot_judge"),  # ... and is not for a round one
    ([(F, 0.5, 0.0), (D, 0.40, 0.04)], 0.5, 0.05, "match"),  # three of a route's own errors only widen the tolerance
    ([(F, 0.5, 0.0), (D, 0.40, 0.01)], 0.5, 0.05, "no_match"),
    ([(F, 0.5125, 0.0), (D, 0.5425, 0.005)], 0.5125, 0.05, "no_match"),  # each within the tolerance, apart beyond their errors
])
def test_a_route_is_held_to_the_larger_of_the_tolerance_and_three_of_its_errors(
        routes: list[tuple[str, float, float]], value: float, tolerance: float, verdict: str) -> None:
    given = [{"route": r, "value": v, "error": e} for r, v, e in routes]
    assert known_values.judge(given, value, tolerance)[0] == verdict


def test_the_value_is_masked_however_it_is_written_and_a_round_value_is_left() -> None:
    text = "it is 2.3846 here, 2.385 there, 2.38 in the abstract, -2.3846 below, 23846 never, and 0.5 as a level"
    withheld_text = known_values.withhold(text, 2.3846)
    assert withheld_text.count(known_values.WITHHELD) == 4 and "23846" in withheld_text and "0.5" in withheld_text
    assert known_values.withhold("a level of 0.5", 0.5) == "a level of 0.5"


def test_a_page_header_is_not_read_as_a_huge_number_when_the_value_is_withheld() -> None:
    assert known_values.withhold("2023\n\ne20230014", 1.01741) == "2023\n\ne20230014"
    assert known_values.withhold("a drain of 2e400 or 1.0174 per hour", 1.01741) == f"a drain of 2e400 or {known_values.WITHHELD} per hour"


def test_a_value_written_with_trailing_zeros_is_withheld_in_every_form() -> None:
    kv = {"value": 0.4, "passage": "Table 2 gives a drain constant of 0.400 for the open tank."}
    value = known_values.written(kv)
    assert known_values.decisive(value) and known_values._half_unit(value) == 0.0005
    shown = known_values.withhold("drains of 0.4, 0.40, 0.400, .4 and -0.4; not 0.41, 0.39 or 4.0", value)
    assert shown.count(known_values.WITHHELD) == 5 and "0.41" in shown and "0.39" in shown and "4.0" in shown
    assert known_values.written({"value": 0.5, "passage": "a level of 0.5 here"}) == known_values._decimal(0.5)
    assert known_values.withhold("a level of 0.5", known_values.written({"value": 0.5, "passage": "0.5"})) == "a level of 0.5"
    request = known_values.prompt(__import__("string").Template("$case | $passage | $excerpt"),
                                  {**kv, "name": "drain", "case": "the open tank", "source": {}}, "the text: 0.400 and 0.4")
    assert "0.4" not in request.replace(known_values.WITHHELD, ""), "the request never shows the value"


@pytest.mark.asyncio
async def test_a_program_that_could_not_be_asked_for_is_not_kept_so_the_next_pass_asks_again(tmp_path: Path) -> None:
    path = _commission(tmp_path / "order", 0.25, 0.01)
    quest = tmp_path / "quest"
    commission.stage(path, quest)

    async def refused(prompt: str, second: bool) -> str:
        raise RuntimeError("the provider refused the call")

    async def never(program: Path) -> tuple[int, str, str]:
        raise AssertionError("nothing is run")

    from core.engine_base import _load_prompts

    record = await known_values.check(commission.read(path), quest, template=_load_prompts()["known_value_program"],
                                      ask=refused, execute=never)
    assert record["values"][0]["verdict"] == "cannot_judge" and "could not be asked for" in record["values"][0]["why"]
    assert not (quest / ".fi" / known_values.RECORD).exists(), "a resume asks again instead of reading this back"


@pytest.mark.asyncio
async def test_a_quest_handed_back_at_the_plan_ends_as_a_return_not_a_failure(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from tests.test_commission import _engine

    async def nothing(self: Any, state: Any) -> dict[str, Any]:
        return {}

    async def design(self: Any, state: Any) -> dict[str, Any]:
        raise AssertionError("the design step ran")

    for name in ("_node_clarify_then_phased", "_node_pause_after_literature", "_node_select_skills"):
        monkeypatch.setattr(Engine, name, nothing)
    monkeypatch.setattr(Engine, "_node_design", design)
    path = _commission(tmp_path / "order", 0.75, 0.01)
    eng = _engine(tmp_path, path)
    model = Model(eng, [RIGHT, RIGHT])

    async def connect(self: Any) -> None:
        self._client = type("Stub", (), {"chat": AsyncMock(side_effect=model.chat), "aclose": AsyncMock()})()

    monkeypatch.setattr(Engine, "_connect_llm", connect)
    await eng.run()
    record = json.loads((eng.quest_root / "needs" / "STUCK.json").read_text(encoding="utf-8"))
    assert record["return_reason"] == "known_value_mismatch" and model.other == []
    assert not (eng.quest_root / "quest_failed.md").exists() and not (eng.quest_root / "plan.md").exists()
    quality = json.loads((eng.quest_root / ".fi" / "quality.json").read_text(encoding="utf-8"))
    assert quality["outcome"]["return_reason"] == "known_value_mismatch"


def test_a_passage_early_in_a_long_text_is_shown_with_what_follows_it() -> None:
    # A table near the start whose definitions come after the first 12,000 characters (a real text did this).
    text = ("Opening words of a made-up note. " * 150) + "Table 1 states the drain constant." + (
        " Filler of the text. " * 360) + "DEFINITION: the drain constant is the outflow over the level." + " More." * 2000
    shown = known_values.excerpt(text, "Table 1 states the drain constant.")
    assert "Table 1 states the drain constant." in shown and "DEFINITION: the drain constant" in shown
    assert len(shown) <= known_values.EXCERPT_CHARS + 10
    assert text.index("DEFINITION") > known_values.EXCERPT_CHARS, "the definition is past the first 12,000 characters"


@pytest.mark.asyncio
async def test_a_program_that_crashes_is_followed_by_a_second_one_that_decides(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    crash = "```python\nimport json\n\nCASE = {\"outlet\": \"open\"}\n\nraise RuntimeError(\"a made-up crash 7741\")\n```"
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [crash, crash, RIGHT])
    assert "stuck" not in out and len(model.asked) == 3, "written once more with its own error, then a second one"
    assert "a made-up crash <number>" in model.asked[1], "a program that stopped is told its own error, no number"
    assert "Another program computed this value" in model.asked[2] and "7741" not in model.asked[2]
    value = _record(eng)["values"][0]
    assert value["verdict"] == "match" and "a first program gave no value" in value["why"]


@pytest.mark.asyncio
async def test_a_value_too_common_to_hide_is_not_computed_and_no_model_is_asked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.5, 0.01, [])
    assert model.asked == [], "no program is asked for: its writer could not be kept from the value"
    value = _record(eng)["values"][0]
    assert value["verdict"] == "cannot_judge" and "too common a number to hide" in value["why"] and "stuck" not in out


def test_a_value_typed_in_as_text_counts_as_typed_in() -> None:
    for expression in ('float("0.25")', 'float("2.5e-1")'):
        script, why = known_values.program(_program(("formula", expression, "0")), 0.25, 0.01, given="")
        assert script == "" and "types in a number close to the value withheld" in why
    assert known_values.program(_program(("formula", 'float("1") / 4', "0")), 0.25, 0.01, given="")[0]


def test_a_value_assembled_from_pieces_of_text_counts_as_typed_in() -> None:
    for expression in ('float("0." + "25")', 'float("0" + "." + "25")', 'float(f"0.25")', 'float("0.2" "5")',
                       'float(("0." + "25") * 1)'):
        script, why = known_values.program(_program(("formula", expression, "0")), 0.25, 0.01, given="")
        assert script == "" and "types in a number close to the value withheld" in why, expression
    assert known_values.program(_program(("formula", 'float("1" + "0") / 40', "0")), 0.25, 0.01, given="")[0]


def test_a_negative_value_keeps_its_sign_when_its_passage_writes_more_digits() -> None:
    from decimal import Decimal

    value = known_values.written({"value": -0.4, "passage": "a drain of -0.400 per hour"})
    assert value == Decimal("-0.400") and str(value) == "-0.400"
    assert known_values.judge([{"route": "formula", "value": -0.4, "error": 0.0}], value, 0.0005)[0] == "match"


@pytest.mark.asyncio
async def test_a_program_that_stops_with_an_error_after_a_value_line_is_a_crash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stops = _program(("formula", "1 / 4", "0")).replace("\n```", "\nraise SystemExit(3)\n```")
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [stops, stops, RIGHT])
    assert len(model.asked) == 3, "written once more after it stopped, then a second program is asked for"
    value = _record(eng)["values"][0]
    first = value["programs"][0]
    assert first["routes"] == [] and first["printed_before_it_stopped"][0]["value"] == 0.25
    assert "stopped with an error" in first["why"] and value["verdict"] == "match" and "stuck" not in out


def _stops_with(error: str) -> str:
    return _program(("formula", "1 / 4", "0")).replace("```python\n", f"```python\nraise {error}\n", 1)


@pytest.mark.asyncio
async def test_a_program_that_stops_with_an_error_is_written_once_more_with_that_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A program calling a function the installed library no longer has stopped, and the value went unjudged though
    its method was sound (a live commission run; old code asked a second program and never repaired the first)."""
    gone = _stops_with('AttributeError("module \'numpy\' has no attribute \'trapz\'")')
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [gone, RIGHT])
    assert len(model.asked) == 2 and "Another program" not in model.asked[1]
    assert "stopped with an error before it finished" in model.asked[1] and "trapz" in model.asked[1]
    value = _record(eng)["values"][0]
    assert value["verdict"] == "match" and len(value["programs"]) == 1, "the repaired program decides; no second one"
    first = value["programs"][0]
    assert first["program"].endswith("_again.py") and first["stopped_first"]["returncode"] != 0
    assert "trapz" in first["stopped_first"]["why"] and (eng.quest_root / ".fi" / "known_values").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize("said", ['"not " + str(1 / 4)', '"share " + str(100 // 4) + "%"', '"ratio 1/4"',
                                  '"x" + str(1 / 4)', '"share " + chr(188)'])
async def test_the_error_a_program_stopped_with_is_told_with_no_number_in_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, said: str) -> None:
    """A number in the error is one the program computed; written another way (25%, 1/4) the value would get past
    the withholding (external review of the repair), so no number of the error is told."""
    eng, model, _ = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [_stops_with(f"ValueError({said})"), RIGHT])
    note = model.asked[1].split("stopped with an error before it finished:", 1)[1].split("Write the program", 1)[0]
    assert "ValueError" in note and not any(ch.isdigit() or ch == chr(188) for ch in note), note
    _hidden(model, 0.25)


@pytest.mark.asyncio
async def test_a_program_written_again_after_it_stopped_is_asked_for_once(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unusable reply to the repair request is not asked for again: one more call at most per program that stops."""
    typed = _program(("formula", "0.25", "0"))
    eng, model, _ = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [_stops_with('OverflowError("range")'), typed, RIGHT])
    assert len(model.asked) == 3 and "Another program computed this value" in model.asked[2]
    first = _record(eng)["values"][0]["programs"][0]
    assert first["returncode"] != 0 and "the reply could not be used" in first["why"]


@pytest.mark.asyncio
async def test_the_record_of_a_program_that_stopped_is_kept_whole(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stops = _program(("formula", "1 / 4", "0")).replace("\n```", "\nraise SystemExit(3)\n```")
    eng, model, _ = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [stops, RIGHT])
    first = _record(eng)["values"][0]["programs"][0]
    kept = first["stopped_first"]
    assert kept["printed_before_it_stopped"][0]["value"] == 0.25 and "KNOWN_JSON" in kept["stdout_tail"]
    assert first["verdict"] == "match" and first["program"].endswith("_again.py")


@pytest.mark.asyncio
async def test_a_second_program_that_stops_is_repaired_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wrong = _program(("formula", "3 / 10", "0"))
    gone = _stops_with('OverflowError("math range error")')
    eng, model, out = await _plan(tmp_path, monkeypatch, 0.25, 0.01, [wrong, gone, RIGHT])
    assert len(model.asked) == 3 and "Another program computed this value" in model.asked[2]
    assert "stopped with an error before it finished" in model.asked[2]
    value = _record(eng)["values"][0]
    assert [p["verdict"] for p in value["programs"]] == ["no_match", "match"]
    assert value["programs"][1]["stopped_first"]["returncode"] != 0


def test_a_route_counts_once_and_only_under_its_own_name() -> None:
    line = 'KNOWN_JSON: {"route": "%s", "value": 1, "error": 0}'
    routes, _ = known_values.printed("\n".join([line % "formula", line % "formula", line % "check", line % "Definition"]))
    assert [r["route"] for r in routes] == ["formula", "definition"]
    one, _ = known_values.printed("\n".join([line % "formula", line % "formula"]))
    assert len(one) == 1 and known_values.judge(one, 1.0, 0.1)[0] == "cannot_judge"


@pytest.mark.asyncio
async def test_each_known_value_is_kept_as_it_is_done_so_a_stop_asks_again_only_for_the_rest(tmp_path: Path) -> None:
    import asyncio
    import sys

    path = _commission(tmp_path / "order", 0.25, 0.01)
    quest = tmp_path / "quest"
    commission.stage(path, quest)
    data = commission.read(path)
    data["known_values"].append({**data["known_values"][0], "name": "outlet constant"})
    asked: list[str] = []

    async def ask(prompt: str, second: bool) -> str:
        asked.append(prompt)
        if len(asked) > 1:
            raise asyncio.CancelledError  # the quest stops while the second value's program is asked for
        return RIGHT

    async def execute(program: Path) -> tuple[int, str, str]:
        run = await asyncio.create_subprocess_exec(sys.executable, str(program), stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
        out, err = await run.communicate()
        return run.returncode or 0, out.decode(), err.decode()

    from core.engine_base import _load_prompts

    with pytest.raises(asyncio.CancelledError):
        await known_values.check(data, quest, template=_load_prompts()["known_value_program"], ask=ask, execute=execute)
    kept = json.loads((quest / ".fi" / known_values.RECORD).read_text(encoding="utf-8"))
    assert [v["name"] for v in kept["values"]] == ["drain constant"] and kept["values"][0]["verdict"] == "match"
    asked.clear()

    async def again(prompt: str, second: bool) -> str:
        asked.append(prompt)
        return RIGHT

    record = await known_values.check(data, quest, template=_load_prompts()["known_value_program"], ask=again,
                                      execute=execute)
    assert len(asked) == 1 and [v["verdict"] for v in record["values"]] == ["match", "match"]


def test_a_known_value_the_plan_has_no_check_for_gets_one_from_fi() -> None:
    data = {"known_values": [{"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": CASE,
                              "passage": "The drain constant is 0.25 per hour", "source": {"file": "tank_note.md"}}]}
    for oracles in ([], ["Drain constant"]):
        design = known_values.fill_checks({"protocol": {"oracles": oracles}}, data, {"tank_note.md": "2"})
        (check,) = design["protocol"]["oracles"]
        assert check["kind"] == "published_value" and check["expected"] == 0.25 and check["tolerance"] == 0.01
        assert check["reference"].startswith("[2]") and check["name"].lower() == "drain constant"
    added = known_values.fill_checks({"protocol": {}}, data, {})["protocol"]["oracles"][0]
    assert added["name"] == "drain constant" and CASE in added["check"]
    note = commission.oracle_gaps(data, {"oracles": []})
    assert len(note) == 1 and "FI added one from the commission" in note[0]


def test_a_value_the_source_gives_as_a_floor_is_matched_by_any_number_not_below_it() -> None:
    """A source that states only a floor ("the constant exceeds 0.25") is met by a correct 0.31: it was 'not reproduced'
    when every known value was a number to equal (old code), and a number below the floor still is not met."""
    above = [{"route": "definition", "value": 0.31, "error": 1e-6, "how": ""}]
    assert known_values.judge(above, 0.25, 0.01, "at_least")[0] == "match"
    assert known_values.judge(above, 0.25, 0.01)[0] == "no_match", "a value to equal is still held to its tolerance"
    below = [{"route": "definition", "value": 0.23, "error": 1e-6, "how": ""}]
    verdict, why = known_values.judge(below, 0.25, 0.01, "at_least")
    assert verdict == "no_match" and "at least 0.25" in why
    assert known_values.judge(above, 0.25, 0.01, "at_most")[0] == "no_match"
    assert known_values.judge(below, 0.25, 0.01, "at_most")[0] == "match"


def test_the_studys_check_of_a_floor_is_its_shortfall_and_is_filled_once_however_often_the_plan_is_written() -> None:
    """The check FI writes for a floor measures how far the quantity falls short of it, expecting 0, so every check of
    a number judges it on one side; a plan written again with FI's form is not wrapped twice, and a check with nothing
    to measure by has no expected value to be failed against."""
    from core import oracle_check, trial_runner

    data = {"known_values": [{"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": CASE, "bound": "at_least",
                              "passage": "The drain constant exceeds 0.25 per hour", "source": {"file": "tank_note.md"}}]}
    design = {"protocol": {"oracles": [{"name": "drain constant", "expected": 0.3, "tolerance": 0.2, "bound": "at_most",
                                        "case": {"outlet": "open"}, "measure": "k"}]}}
    once = known_values.fill_checks(design, data, {"tank_note.md": "2"})
    (check,) = once["protocol"]["oracles"]
    assert (check["measure"], check["expected"], check["tolerance"], check["bound"]) == ("max(0, 0.25 - (k))", 0.0, 0.01,
                                                                                         "at_least")
    assert "at least 0.25" in check["reference"]
    again = known_values.fill_checks(once, data, {"tank_note.md": "2"})
    assert again["protocol"]["oracles"][0]["measure"] == check["measure"]
    verdicts = []
    for k in (0.31, 0.245, 0.2):
        measured, _why = trial_runner._check_from(check["name"], check["measure"], {"k": k}, {"outlet": "open"})
        verdicts.append(oracle_check.judged([check], {"checks": [measured]})[0]["passed_by_engine"])
    assert verdicts == [True, True, False]
    added = known_values.fill_checks({"protocol": {}}, data, {})["protocol"]["oracles"][0]
    assert "expected" not in added and oracle_check.limit_of(added)[1] is None
    assert "at least 0.25" in commission.oracle_gaps(data, {"oracles": []})[0]


def test_a_commission_names_a_floor_or_a_ceiling_only_in_the_words_fi_reads(tmp_path: Path) -> None:
    path = _commission(tmp_path / "bound", 0.25, 0.01)
    data = json.loads(path.read_text(encoding="utf-8"))
    for word, ok in (("at_least", True), ("at most", True), ("above", False)):
        data["known_values"][0]["bound"] = word
        path.write_text(json.dumps(data), encoding="utf-8")
        found = [p for p in commission.problems(path) if "`bound`" in p]
        assert (found == []) is ok, (word, found)
    block = commission.prompt_block({**data, "known_values": [{**data["known_values"][0], "bound": "at_least"}]}, {})
    assert "at least 0.25" in block and "the quantity itself" in block


def test_a_limits_check_keeps_one_form_for_a_negative_limit_and_a_quantity_the_plan_already_clipped() -> None:
    """Review of the limit checks: a negative limit was wrapped twice when the plan was written again (a ceiling of
    -0.25 then failed a correct -1), and a quantity the plan clipped at the limit (``min(k, 0.25)``) hid a crossing."""
    for value, bound in ((-0.25, "at_least"), (-0.25, "at_most"), (0.25, "at_least")):
        once = known_values.wrap("k", value, bound)
        assert known_values.unwrap(once, value, bound) == "k", (value, bound)
    assert known_values.unwrap("min(k, 0.25)", 0.25, "at_most") == "k"
    assert known_values.unwrap("max(0.25, k)", 0.25, "at_least") == "k"
    assert known_values.unwrap("min(k, 0.3)", 0.25, "at_most") == "min(k, 0.3)", "another number is the plan's own"


def test_a_limits_check_with_nothing_to_measure_keeps_no_number_of_the_models_and_a_criterion_on_it_reads_the_shortfall(
) -> None:
    from core import oracle_check

    data = {"known_values": [{"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": CASE, "bound": "at_least",
                              "passage": "The drain constant exceeds 0.25 per hour", "source": {"file": "tank_note.md"}}]}
    design = {"protocol": {"oracles": [{"name": "drain constant", "expected": 42}],
                           "criteria": [{"name": "fast", "oracle": "drain constant", "use": "value", "direction": "higher",
                                         "target": 0.25, "tolerance": 0.01},
                                        {"name": "other", "oracle": "outlet", "use": "value", "direction": "higher",
                                         "target": 1, "tolerance": 0.1}]}}
    out = known_values.fill_checks(design, data, {})["protocol"]
    assert "expected" not in out["oracles"][0] and oracle_check.limit_of(out["oracles"][0])[1] is None
    assert {k: out["criteria"][0][k] for k in ("use", "direction", "target")} == {"use": "error", "direction": "lower",
                                                                                   "target": 0}
    assert out["criteria"][1] == design["protocol"]["criteria"][1]


def test_two_ways_of_computing_a_floor_that_differ_but_both_clear_it_hand_nothing_back() -> None:
    routes = [{"route": "formula", "value": 0.31, "error": 0.0, "how": ""},
              {"route": "definition", "value": 0.40, "error": 0.0, "how": ""}]
    assert known_values.judge(routes, 0.25, 0.01, "at_least")[0] == "cannot_judge"
    assert known_values.judge(routes, 0.25, 0.01)[0] == "no_match", "a value to equal is still not reproduced"


def test_a_proportions_shortfall_of_a_floor_is_judged_on_its_binomial_interval() -> None:
    """Ten successes in ten trials fall 0 short of a floor of 0.9 every time, a spread of 0 that passed the check; the
    proportion's interval says ten trials cannot show the rate is 0.9 or more."""
    from core import mc_error

    measure = known_values.wrap("hit", 0.9, "at_least")
    verdicts = []
    for n, hits in ((10, 10), (400, 400), (400, 300)):
        rows = [{"hit": 1.0}] * hits + [{"hit": 0.0}] * (n - hits)
        stat = mc_error.statistic(rows, measure, (), 0.0, 0.01, ["hit"], repeated=True)
        verdicts.append(mc_error.mean_verdict(stat, 0.0, 0.01))
    assert verdicts == [None, "pass", "fail"]


def test_a_floor_on_a_random_quantity_is_judged_on_the_shortfall_of_its_mean() -> None:
    """Trials of 0 and 0.06 have a mean of 0.03, above a floor of 0.02; the mean of each trial's shortfall (0.01) is
    not how far the mean falls short, and failed the check (second review of the limit checks)."""
    from core import mc_error

    rows = [{"share": 0.0}, {"share": 0.06}] * 200
    measure = known_values.wrap("share", 0.02, "at_least")
    stat = mc_error.statistic(rows, measure, (), 0.0, 0.0005, [], repeated=True)
    assert stat["on"] == "means" and abs(stat["inner_mean"] - (0.02 - 0.03)) < 1e-12
    assert mc_error.mean_verdict(stat, 0.0, 0.0005) == "pass"
    low = mc_error.statistic([{"share": 0.0}, {"share": 0.024}] * 200, measure, (), 0.0, 0.0005, [], repeated=True)
    assert mc_error.mean_verdict(low, 0.0, 0.0005) == "fail"
    plan = mc_error.statistic([{"hit": 1.0}, {"hit": 0.0}] * 200, "max(0, 0.9 - hit)", (), 0.45, 0.01, ["hit"], repeated=True)
    assert plan["on"] == "trials", "a formula the plan wrote expecting another number is still read trial by trial"


def test_a_criterion_with_another_bar_on_a_limits_value_is_left_out_and_said() -> None:
    data = {"known_values": [{"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": CASE, "bound": "at_least",
                              "passage": "The drain constant exceeds 0.25 per hour", "source": {"file": "tank_note.md"}}]}
    criteria = [{"name": "not too fast", "oracle": "drain constant", "use": "value", "direction": "lower", "target": 0.3,
                 "tolerance": 0.01},
                {"name": "well above", "oracle": "drain constant", "use": "value", "direction": "higher", "target": 0.4,
                 "tolerance": 0.01}]
    design = {"protocol": {"oracles": [{"name": "drain constant", "case": {"outlet": "open"}, "measure": "k"}],
                           "criteria": criteria}}
    assert known_values.fill_checks(design, data, {})["protocol"]["criteria"] == []
    notes = commission.oracle_gaps(data, design["protocol"])
    assert len(notes) == 2 and all("FI left it out" in n for n in notes)


@pytest.mark.asyncio
async def test_a_value_done_on_a_resume_is_kept_beside_the_values_already_on_disk(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A's program cannot be asked for and B is done; on the resume A is done: every write keeps B.
    import asyncio
    import sys

    path = _commission(tmp_path / "order", 0.25, 0.01)
    quest = tmp_path / "quest"
    commission.stage(path, quest)
    data = commission.read(path)
    data["known_values"] = [{**data["known_values"][0], "name": "outlet constant"}, data["known_values"][0]]

    async def first(prompt: str, second: bool) -> str:
        if "outlet constant" in prompt:
            raise RuntimeError("the provider refused the call")
        return RIGHT

    async def execute(program: Path) -> tuple[int, str, str]:
        run = await asyncio.create_subprocess_exec(sys.executable, str(program), stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
        out, err = await run.communicate()
        return run.returncode or 0, out.decode(), err.decode()

    from core.engine_base import _load_prompts

    template = _load_prompts()["known_value_program"]
    await known_values.check(data, quest, template=template, ask=first, execute=execute)
    record = quest / ".fi" / known_values.RECORD
    assert [v["name"] for v in json.loads(record.read_text(encoding="utf-8"))["values"]] == ["drain constant"]
    written: list[list[str]] = []
    real = Path.write_text

    def spy(self: Path, text: str, *a: Any, **kw: Any) -> int:
        if self == record:
            written.append([v["name"] for v in json.loads(text)["values"]])
        return real(self, text, *a, **kw)

    monkeypatch.setattr(Path, "write_text", spy)

    async def again(prompt: str, second: bool) -> str:
        assert "outlet constant" in prompt, "the value already done is not asked again"
        return RIGHT

    await known_values.check(data, quest, template=template, ask=again, execute=execute)
    assert written and all("drain constant" in names for names in written), written
    assert written[-1] == ["outlet constant", "drain constant"]


@pytest.mark.asyncio
async def test_the_repair_of_a_checks_numbers_still_names_a_published_value_and_a_new_check_does_not(
        tmp_path: Path) -> None:
    from tests.test_plan_parts_only import ORACLE, Scripted, _design, _planned

    eng, _path, _before = _planned(tmp_path)
    model = Scripted(eng, [{"protocol": {"oracles": [{"name": "no_gain", "expected": 0, "tolerance": 0.03}]}}])
    assert await eng._declare_oracles(["The check 'no_gain' has no numeric expected value."]) is True
    assert "convergence_rate, published_value)" in model.prompts[0], "a check a quest has keeps its kind"
    design = _design()
    design["protocol"].pop("oracles")
    eng, _path, _before = _planned(tmp_path / "new", design)
    model = Scripted(eng, [{"protocol": {"oracles": [ORACLE]}}])
    assert await eng._declare_oracles() is True
    assert "published_value" not in model.prompts[0], "a new check is never a published value the model types"


def test_the_note_on_a_plan_with_no_check_still_names_a_published_benchmark() -> None:
    from core import oracle_check

    assert "a published benchmark value" in oracle_check.problems([], None, 0)[0]


def test_only_the_whole_text_expression_is_read_and_constant_f_strings_and_deep_chains_are_handled() -> None:
    def program(expression: str) -> tuple[str, str]:
        return known_values.program(_program(("formula", expression, "0")), 0.25, 0.00001, given="")

    assert program('float("0." + "25" + "7")')[0], "0.257 is not the value withheld: its inner piece is not read alone"
    for typed in ('float(f"{0.2}{5}")', "float(f\"{'0.'!s}{'25'!s}\")"):
        assert program(typed)[0] == "", typed
    long = " + ".join(['"a"'] * 1500)  # past the depth cap: not folded, and nothing is raised
    assert program(f"len({long})")[0]
    assert program('len((("ab" * 20) * 20) * 20 * 20)')[0]  # text past the length cap is not folded either
