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

    for name in ("_node_clarify_then_phased", "_node_ideate", "_node_pause_after_literature", "_node_scope",
                 "_node_select_skills"):
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

    for name in ("_node_clarify_then_phased", "_node_ideate", "_node_pause_after_literature", "_node_scope",
                 "_node_select_skills"):
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
