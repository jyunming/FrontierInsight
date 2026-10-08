"""A model's numbers come from the plan or from its equations (core/typed_constants.py, read before the run through
core/static_checks.py). Neutral toy code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import typed_constants as tc
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

PROTOCOL = {
    "grid": {"x": [1, 2, 3]},
    "model": {"equations": [{"id": "E1", "formula": "y = a * f(x)", "role": "generates"}]},
    "fixed": {"a": 2.5},
}
PLAN = "The model: y = a * f(x) with a = 2.5 and a rolloff length of 4.2 from [1]. The topic is a toy."

SHAPED = '''\
import numpy as np


def response(x):
    """Typed surrogate."""
    centre = 3.7
    return np.exp(-((x - centre) / 1.3) ** 2) + 0.3 * x
'''
MANY = '''\
def response(x):
    return 0.37 * x + 0.81 * x ** 2 - 0.29
'''
TRACED = '''\
import math
import numpy as np


def response(x, a=2.5):
    return a * np.exp(-x / 4.2) + 1e-9 * x + 0.5 * math.pi + 60.0 / 3600.0 + 2 * x ** 3 + 128 + 1e3 / 4
'''
ONE_STRAY = '''\
def response(x):
    return x * 0.37
'''
INDEXED = '''\
import numpy as np


def response(x):
    grid = np.zeros(4096)
    y = np.linspace(0.0, 1.0, 777)
    return float(x[137]) + grid[0] + y[0]
'''
SIM = '''\
from pkg import model


def run_cell(cell):
    r = model.response(cell["x"])
    return {"y": r, "x_echo": cell["x"]}
'''
PKG_INIT = ""
LABELLED_SIM = '''\
import numpy as np

# E1
def rolloff(x):
    return 0.37 * x + 0.81 * x ** 2 - 0.29


def other(x):
    return 0.41 * x + 0.77 * x ** 2 - 0.63


def run_cell(cell):
    return {"y": rolloff(cell["x"]) + other(cell["x"])}
'''


def _problems(files: dict[str, str], plan: str = PLAN) -> list[tc.Typed]:
    return tc.problems(files, PROTOCOL, {"_text": plan})


def test_a_number_typed_into_a_smooth_functions_argument_is_caught_with_its_line() -> None:
    found = _problems({"pkg/__init__.py": "", "pkg/model.py": SHAPED})
    assert len(found) == 1 and found[0].function == "response" and found[0].script == "pkg/model.py"
    assert set(found[0].values) == {3.7, 1.3} and set(found[0].shape) == {1.3}
    assert "pkg/model.py line 4: `response` has" in found[0].says() and "3.7" in found[0].says()


def test_three_numbers_in_a_model_function_are_a_clear_case_and_one_is_not() -> None:
    assert [t.function for t in _problems({"pkg/__init__.py": "", "pkg/model.py": MANY})] == ["response"]
    assert _problems({"pkg/__init__.py": "", "pkg/model.py": ONE_STRAY}) == []


def test_numbers_the_plan_gives_and_plain_numerics_pass() -> None:
    assert _problems({"pkg/__init__.py": "", "pkg/model.py": TRACED}) == []
    # ... the same shape with its numbers stated in the plan passes; a percent of a stated number is the number
    assert _problems({"pkg/__init__.py": "", "pkg/model.py": SHAPED}, PLAN + " centre 3.7 width 1.3 gain 0.3") == []
    assert tc.traceable(0.037, {3.7}) and tc.traceable(370.0, {3.7}) and not tc.traceable(3.9, {3.7})


@pytest.mark.parametrize("value", [0, 1, 7, 12, 64, 4096, 1e-12, 1e6, 0.5, 0.25, 60, 3600, 3.141592653589793,
                                   6.283185307179586, 1.5707963267948966, 2.718281828459045, 1.4142135623730951])
def test_the_allow_list_is_plain_numerics(value: float) -> None:
    assert tc.simple(value)


@pytest.mark.parametrize("value", [0.3, 0.37, 2.5, 13, 95, 9.81, 1.436, 0.72])
def test_a_chosen_number_is_not_plain(value: float) -> None:
    assert not tc.simple(value)


def test_indices_and_sizes_are_not_model_numbers() -> None:
    assert _problems({"pkg/__init__.py": "", "pkg/model.py": INDEXED}) == []


def test_only_the_model_is_read() -> None:
    # an analysis script's function is not the model; the simulation's own function is read for a shape only
    assert _problems({"experiment.py": MANY}) == []
    assert _problems({"simulate.py": MANY}) == []
    assert [t.function for t in _problems({"simulate.py": SHAPED})] == ["response"]
    # a function labelled with the plan's equation is the model wherever it is
    found = _problems({"simulate.py": LABELLED_SIM})
    assert [t.function for t in found] == ["rolloff"]


def test_the_request_names_the_codes_numbers_and_says_what_to_do_in_plain_words() -> None:
    found = _problems({"pkg/__init__.py": "", "pkg/model.py": SHAPED})
    text = tc.directive(found, PROTOCOL)
    assert "3.7" in text and "1.3" in text and "2.5" not in text and "4.2" not in text, "the plan's numbers are not given"
    assert "Compute each from the model's equations; if it is a genuine constant of the study, define it once as a named constant at the top of the module" in text
    assert "state it in the plan" not in text
    assert text.rstrip().endswith("change nothing else.")
    assert "not in the plan" in tc.writer_note(found, PROTOCOL) and "does not describe" in tc.gap(found, PROTOCOL)


def test_a_number_the_plan_gives_through_a_small_integer_a_power_of_ten_or_a_ratio_is_traceable() -> None:
    assert tc.traceable(42 / 7, {42.0}) and tc.traceable(91 / 7, {91.0}) and tc.traceable(137 / 7, {137.0})
    assert tc.traceable(4200.0, {42.0}) and tc.traceable(0.42, {42.0})
    # two numbers combine only inside one structured place of the plan, never from free text
    assert not tc.traceable(6.5 * 3.25, {6.5, 3.25}) and not tc.traceable(6.5 / 3.25, {6.5, 3.25})
    assert tc.traceable(6.5 * 3.25, {6.5, 3.25}, [{6.5, 3.25}]) and tc.traceable(6.5 / 3.25, {6.5, 3.25}, [{6.5, 3.25}])
    assert not tc.traceable(6.5 * 3.25, {6.5, 3.25}, [{6.5}, {3.25}])
    assert not tc.traceable(19.5, {42.0}) and not tc.traceable(1234.567, {42.0, 7.7})
    code = "def response(x):\n    return x * 19.571428571428573 + 0.0\n"
    assert _problems({"pkg/__init__.py": "", "pkg/model.py": code}, "the gain is 137 per seven") == []


def test_the_structured_places_of_the_plan_are_each_their_own_group() -> None:
    protocol = {"model": {"equations": [{"id": "E1", "formula": "y = 3.5 * x / 8.5"}]},
                "oracles": [{"name": "c", "case": {"x": 1.5, "n": 7.25}}], "fixed": {"a": 2.0, "b": 9.0}, "grid": {"x": [1, 2, 3]}}
    groups = tc.structured_groups(protocol, {"variables": {"controls": [{"name": "g", "value": 4.25}]}})
    assert {3.5, 8.5, 1.0} <= set().union(*groups)
    assert any({3.5, 8.5} <= g for g in groups) and not any({3.5, 7.25} <= g for g in groups)
    # a ratio of two numbers of one equation is the plan's; of an equation's and a check's is not
    plan = "free text mentions 3.5 and 7.25"
    rate = "def response(x):\n    return x * (3.5 / 8.5) + x * x * 0.1234\n"
    assert tc.problems({"pkg/__init__.py": "", "pkg/model.py": rate}, protocol, {"_text": plan}) == []
    mixed = "def response(x):\n    return x * (3.5 * 7.25) + x * x * 0.1234 + x * 0.4321 + 0.5678 * x ** 3\n"
    assert [t.function for t in tc.problems({"pkg/__init__.py": "", "pkg/model.py": mixed}, protocol, {"_text": plan})] == ["response"]


def test_the_number_of_plan_values_does_not_change_the_answer() -> None:
    many = {float(n) for n in range(1000, 1600)} | {42.0}
    assert tc.traceable(6.0, many) and tc.traceable(1550.0 / 5.0, many)


def test_a_nested_helpers_return_does_not_decide_whether_the_entry_can_be_read() -> None:
    sim = ('from pkg import model\n\n\ndef run_cell(cell):\n    def helper(v):\n        return v + 1\n    r = helper(model.response(cell["x"]))\n'
           '    return {"y": r, "x_echo": cell["x"]}\n')
    files = {"pkg/__init__.py": "", "pkg/model.py": SHAPED, "simulate.py": sim}
    found = _problems(files)
    assert tc.tainted_outputs(files, found) == {"y"}
    assert tc.headline_hit(found, files, PROTOCOL, {"variables": {"dependent": ["y"]}}) is True
    # ... but an entry that itself returns something else is still not read
    bad = sim.replace('return {"y": r, "x_echo": cell["x"]}', "return r")
    assert tc.tainted_outputs({**files, "simulate.py": bad}, found) is None


def test_a_headline_name_must_match_exactly_and_an_unreadable_entry_is_no_hit() -> None:
    files = {"pkg/__init__.py": "", "pkg/model.py": SHAPED, "simulate.py": SIM.replace('"y": r', '"corporate_index": r')}
    found = _problems(files)
    assert tc.tainted_outputs(files, found) == {"corporate_index"}
    assert tc.headline_hit(found, files, PROTOCOL, {"variables": {"dependent": ["rate"]}}) is False
    assert tc.headline_hit(found, files, PROTOCOL, {"variables": {"dependent": ["Corporate-Index"]}}) is True
    # a file that does not parse, or an entry that returns a name, cannot be read with confidence
    assert tc.tainted_outputs({**files, "other.py": "def broken(:\n"}, found) is None
    opaque = {**files, "simulate.py": 'from pkg import model'+chr(10)*2+'def run_cell(cell):'+chr(10)+'    out = {"y": model.response(1)}'+chr(10)+'    return out'+chr(10)}
    assert tc.tainted_outputs(opaque, found) is None
    assert tc.headline_hit(found, opaque, PROTOCOL, {"variables": {"dependent": ["y"]}}) is False


def test_a_headline_quantity_that_rests_on_a_flagged_function_is_found() -> None:
    files = {"pkg/__init__.py": "", "pkg/model.py": SHAPED, "simulate.py": SIM}
    found = _problems(files)
    assert tc.tainted_outputs(files, found) == {"y"}
    design = {"variables": {"dependent": ["y"]}}
    assert tc.headline_hit(found, files, PROTOCOL, design) is True
    assert tc.headline_hit(found, files, PROTOCOL, {"variables": {"dependent": ["x_echo"]}}) is False
    assert tc.headline_hit(found, files, PROTOCOL, {}) is False


# ---- through the engine -------------------------------------------------------------------------------------------------

REPAIRED = '''\
import numpy as np


def response(x, a=2.5):
    """Computed from the plan's setting."""
    return a * np.exp(-x / 4.2) + 0.5 * x
'''


class _Client:
    def __init__(self, replies: list[str], asked: list[str]) -> None:
        self.replies, self.asked = replies, asked
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.replies[min(len(self.asked), len(self.replies)) - 1]


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[Engine, list[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    asked: list[str] = []
    eng = Engine(Config(
        topic="a neutral topic", title="shape", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = _Client(replies, asked)  # type: ignore[assignment]
    eng._protocol_block = lambda state: PROTOCOL  # type: ignore[method-assign]
    code = eng.quest_root / "code"
    (code / "pkg").mkdir(parents=True, exist_ok=True)
    (code / "pkg" / "__init__.py").write_text(PKG_INIT, encoding="utf-8")
    (code / "pkg" / "model.py").write_text(SHAPED, encoding="utf-8")
    (code / "simulate.py").write_text(SIM, encoding="utf-8")
    (eng.quest_root / "plan.md").write_text(PLAN, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_a_typed_in_shape_is_sent_back_and_a_computed_rewrite_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": REPAIRED, "patch_summary": "from the plan"})])
    assert await eng._static_gate({}) is None
    assert len(asked) == 1 and "numbers typed into them that the plan never gives" in asked[0]
    assert "4.2" not in asked[0], "no number of the plan is given to the repair"
    assert (eng.quest_root / "code" / "pkg" / "model.py").read_text(encoding="utf-8") == REPAIRED
    assert eng._static_check_gaps() == [] and eng._static_checks_note() == ""


@pytest.mark.asyncio
async def test_one_that_stays_is_a_note_for_the_paper_and_the_evidence_but_not_a_stop_unless_a_headline_rests_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": SHAPED, "patch_summary": "no change"})])
    assert await eng._static_gate({}) is None, "no headline quantity is named: a note"
    assert len(asked) == 2
    gap = eng._static_check_gaps()
    assert len(gap) == 1 and "numbers typed into it that the plan does not give" in gap[0]
    assert "numbers that are not in the plan" in eng._run_checks_note()
    record = json.loads((eng.quest_root / "needs" / "typed_constants_check.json").read_text(encoding="utf-8"))
    assert record["functions"][0]["function"] == "response" and 3.7 in record["functions"][0]["numbers"]

    stopped, _asked = _engine(tmp_path / "head", monkeypatch, [json.dumps({"code": SHAPED})])
    stop = await stopped._static_gate({"design": {"variables": {"dependent": ["y"]}}})
    assert stop is not None and "written into it" in stop["exec_result"]["cannot_repair"]
    assert stop["exec_give_up_reason"]


@pytest.mark.asyncio
async def test_a_model_whose_numbers_the_plan_gives_is_not_asked_about(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, ["{}"])
    (eng.quest_root / "code" / "pkg" / "model.py").write_text(REPAIRED, encoding="utf-8")
    assert await eng._static_gate({}) is None and asked == []


def test_an_unused_nested_function_that_calls_a_flagged_function_taints_nothing() -> None:
    sim = ('from pkg import model\n\n\ndef run_cell(cell):\n    def never_called(v):\n        return model.response(v)\n'
           '    return {"y": cell["x"] * 2.0, "x_echo": cell["x"]}\n')
    files = {"pkg/__init__.py": "", "pkg/model.py": SHAPED, "simulate.py": sim}
    found = _problems(files)
    assert tc.tainted_outputs(files, found) == set()
    assert tc.headline_hit(found, files, PROTOCOL, {"variables": {"dependent": ["y"]}}) is False
    called = sim.replace('"y": cell["x"] * 2.0', '"y": never_called(cell["x"])')
    assert tc.tainted_outputs({**files, "simulate.py": called}, found) == {"y"}
    # a flagged function reached only through an unused nested function of another function is not reached either
    other = ('from pkg import model\n\n\ndef wrapper(v):\n    def never_called():\n        return model.response(v)\n    return v\n\n\n'
             'def run_cell(cell):\n    return {"y": wrapper(cell["x"])}\n')
    assert tc.tainted_outputs({**files, "simulate.py": other}, found) == set()
