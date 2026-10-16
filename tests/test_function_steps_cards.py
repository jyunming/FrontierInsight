"""Step cards in the function steps (core/function_steps.py), the passages a function is written from
(core/step_passages.py), and the two static looks (core/step_static.py, core/step_coverage.py). Fake models, fake
knowledge, neutral topics (a damped spring); the one replay is the r6t kinetics quest's two helper steps."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import string
import sys
from pathlib import Path
from typing import Any

import pytest

from core import function_steps as fs
from core import plan_steps, step_coverage, step_passages, step_static

AGENTS = Path(__file__).resolve().parents[1] / "agents"
PROMPTS = {"fill": string.Template((AGENTS / "implement_function.md").read_text(encoding="utf-8")),
           "repair": string.Template((AGENTS / "implement_function_repair.md").read_text(encoding="utf-8"))}

EQUATIONS = [
    {"id": "E1", "formula": "w = sqrt(k / m)", "role": "generates", "source": "derivation", "derivation": "Newton's law",
     "example": {"inputs": {"k": 8.0, "m": 2.0}, "expected_formula": "sqrt(k / m)"}},   # 2.0
    {"id": "E2", "formula": "Q = m * w / c", "role": "generates", "source": "derivation",
     "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c"}},   # 12.0
]
STEPS = [
    {"id": "S1", "name": "mean quality", "does": "average two quality factors", "inputs": {"q1": "dimensionless",
     "q2": "dimensionless"}, "output": "dimensionless", "uses": ["E2"], "source": "Smith 2020",
     "example": {"inputs": {"q1": 2.0, "q2": 4.0}, "expected_formula": "(q1 + q2) / 2"}},   # 3.0
    {"id": "S2", "name": "scaled frequency", "does": "the natural frequency times a scale", "inputs": {
        "k": "N/m", "m": "kg", "scale": "1"}, "output": "rad/s", "uses": ["E1"], "relations": ["it doubles when scale doubles"]},
    {"id": "S3", "name": "bookkeeping", "does": "count the cycles", "inputs": {"n": "1"}, "output": "count"},
]
PROTOCOL = {"model": {"summary": "a mass on a damped spring", "equations": EQUATIONS, "steps": STEPS}}
OUTLINE = {"scaffold": "x = 1", "functions": [
    {"name": "mean_quality", "signature": "def mean_quality(q1, q2) -> float:", "one_line_purpose": "average",
     "implements": "S1", "depends_on": []},
    {"name": "quality", "signature": "def quality(m, w, c) -> float:", "one_line_purpose": "the quality factor",
     "implements": "E2", "depends_on": ["frequency"]},
    {"name": "frequency", "signature": "def frequency(k, m) -> float:", "one_line_purpose": "the natural frequency",
     "implements": "E1"},
    {"name": "scaled_frequency", "signature": "def scaled_frequency(k, m, scale) -> float:",
     "one_line_purpose": "scaled", "implements": "S2"},
    {"name": "count_cycles", "signature": "def count_cycles(n) -> int:", "one_line_purpose": "count", "implements": "S3"},
    {"name": "run_cell", "signature": "def run_cell(cell) -> dict:", "one_line_purpose": "one setting"},
]}
GOOD = {
    "frequency": "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k / m)\n",
    "quality": "# E2\ndef quality(m, w, c):\n    return m * w / c\n",
    "mean_quality": "# S1\ndef mean_quality(q1, q2):\n    return (q1 + q2) / 2\n",
    "scaled_frequency": "# S2\ndef scaled_frequency(k, m, scale):\n    return scale * frequency(k, m)\n",
    "count_cycles": "# S3\ndef count_cycles(n):\n    return int(n)\n",
}


def _which(prompt: str) -> str:
    found = re.search(r"# The function[^\n]*\n\n```python\n(?:#[^\n]*\n)*def (\w+)\(", prompt)
    assert found, prompt[-600:]
    return found.group(1)


class _Model:
    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies, self.prompts, self.count = {k: list(v) for k, v in replies.items()}, [], {}

    async def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        name = _which(prompt)
        n = self.count[name] = self.count.get(name, 0) + 1
        replies = self.replies[name]
        return "```python\n" + replies[min(n, len(replies)) - 1] + "```"


def _runner(root: Path):
    async def run(argv: list[str]) -> tuple[int, str, str]:
        for cache in (root / "code").glob("**/__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        proc = await asyncio.create_subprocess_exec(sys.executable, "-B", *argv, cwd=str(root),
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return proc.returncode or 0, out.decode(), err.decode()

    return run


def _filler(root: Path, model: _Model, **kw: Any) -> fs.FunctionFiller:
    return fs.FunctionFiller(quest_root=root, package="p", protocol=PROTOCOL, summary="a damped spring", prompts=PROMPTS,
                             chat=model.chat, run=_runner(root), log=logging.getLogger("t"), **kw)


# --- the specs -----------------------------------------------------------------------------------------------------------


def test_specs_merge_equations_and_steps_in_dependency_order() -> None:
    specs, notes = fs.model_functions(OUTLINE, PROTOCOL)
    names = [s.name for s in specs]
    assert set(names) == {"frequency", "quality", "mean_quality", "scaled_frequency", "count_cycles"}
    assert names.index("frequency") < names.index("quality") and names.index("frequency") < names.index("scaled_frequency")
    assert names.index("quality") < names.index("mean_quality"), "S1 uses E2, so it is written after it"
    step = next(s for s in specs if s.name == "scaled_frequency")
    assert step.kind == "step" and step.equation == "S2" and step.card["relations"] == ["it doubles when scale doubles"]
    assert "frequency" in step.depends_on, "the step's `uses` becomes a dependency"
    assert next(s for s in specs if s.name == "frequency").kind == "equation"
    assert not notes


def test_a_plan_with_no_step_cards_gives_what_it_always_gave() -> None:
    plain = {"model": {k: v for k, v in PROTOCOL["model"].items() if k != "steps"}}
    outline = {"functions": [f for f in OUTLINE["functions"] if f.get("implements", "").startswith("E")]}
    specs, notes = fs.model_functions(outline, plain)
    assert [s.kind for s in specs] == ["equation", "equation"] and notes == []
    assert fs.spec_hash(specs, plain) == fs.spec_hash(specs, PROTOCOL), "equation-only specs hash the same"
    # a step function the plan does not list is a note, not a spec
    ghost = {"functions": [*outline["functions"], {"name": "g", "signature": "def g(a):", "implements": "S9"}]}
    specs2, notes2 = fs.model_functions(ghost, plain)
    assert [s.name for s in specs2] == [s.name for s in specs] and any("S9" in n for n in notes2)


def test_a_step_without_a_function_or_with_the_wrong_parameters_is_a_note_not_a_stop() -> None:
    outline = {"functions": [
        {"name": "frequency", "signature": "def frequency(k, m) -> float:", "implements": "E1"},
        {"name": "quality", "signature": "def quality(m, w, c) -> float:", "implements": "E2"},
        {"name": "mean_quality", "signature": "def mean_quality(a, b) -> float:", "implements": "S1"}]}
    specs, notes = fs.model_functions(outline, PROTOCOL)
    assert {s.name for s in specs} == {"frequency", "quality"}
    assert any("mean_quality" in n and "q1" in n for n in notes) and any("S2" in n for n in notes)


def test_limits_and_call_scale() -> None:
    assert fs.MAX_FUNCTIONS == 16 and fs.max_calls_for(5) == 20 and fs.max_calls_for(16) == 64


# --- writing the functions -----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_steps_are_written_into_the_model_file_with_s_labels_and_recorded(tmp_path: Path) -> None:
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    model = _Model({k: [v] for k, v in GOOD.items()})
    outcome = await _filler(tmp_path, model).run_all(specs)
    assert outcome.status == "done", outcome
    text = (tmp_path / "code" / "p" / "model.py").read_text(encoding="utf-8")
    for label in ("# S1", "# S2", "# S3", "# E1", "# E2"):
        assert label in text
    record = fs.load(tmp_path)
    assert record["untested_by_example"] == ["S3"], "S3 has neither an example nor a relation"
    assert record["functions"]["scaled_frequency"]["kind"] == "step"
    # the layout's equation map reads the labels of the whole package
    from core import code_layout

    rows = code_layout.equation_map(PROTOCOL["model"] and {"model": {"equations": EQUATIONS}},
                                    code_layout.package_sources(tmp_path / "code"))
    assert {r["id"]: r["function"] for r in rows} == {"E1": "frequency", "E2": "quality"}


@pytest.mark.asyncio
async def test_the_literature_and_example_blocks_are_in_the_prompts_and_empty_when_there_is_none(tmp_path: Path) -> None:
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    model = _Model({k: [v] for k, v in GOOD.items()})
    passages = {"S1": "[Smith · 2020]\nThe mean quality is the average of the two."}
    assert (await _filler(tmp_path, model, passages=passages).run_all(specs)).status == "done"
    by = {_which(p): p for p in model.prompts}
    assert "The mean quality is the average of the two." in by["mean_quality"] and "# What the literature says" in by["mean_quality"]
    assert "q1 = 2" in by["mean_quality"] and "q2 = 4" in by["mean_quality"], "the example's inputs are shown"
    assert "# What the literature says" not in by["frequency"] and "# A worked case" not in by["frequency"]
    assert "# A worked case" not in by["scaled_frequency"], "no example, no block"
    assert "$literature_block" not in by["frequency"] and "$example_block" not in by["frequency"]


@pytest.mark.asyncio
async def test_a_steps_expected_output_is_never_shown_not_even_after_it_failed_its_example(tmp_path: Path) -> None:
    """review pass 2 N6: the repair after a failed example showed "FI computed the result ...: 3 (relative tolerance
    ...)", so the function could special-case the very example FI then judged. It is told the case did not match, and
    its inputs, only."""
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    wrong = "# S1\ndef mean_quality(q1, q2):\n    return q1 + q2\n"  # 6.0, not 3.0
    model = _Model({**{k: [v] for k, v in GOOD.items()}, "mean_quality": [wrong, GOOD["mean_quality"]]})
    assert (await _filler(tmp_path, model).run_all(specs)).status == "done"
    mine = [p for p in model.prompts if _which(p) == "mean_quality"]
    assert len(mine) == 2
    first, repair = mine
    shown = first.split("# A worked case")[1].split("# The function")[0]
    assert "q1 = 2" in shown and "q2 = 4" in shown and "FI computed the result" not in shown and "3" not in shown
    again = repair.split("# A worked case")[1].split("# The function")[0]
    assert "q1 = 2" in again and "q2 = 4" in again and "does not match" in again, again
    assert "FI computed the result" not in again and "3" not in again and "tolerance" not in again, again
    assert "implements step S1 of the method" in repair, "the failure is worded for a step"
    problem = repair.split("# What is wrong with it")[1].split("# Functions of the package")[0]
    assert "(3" not in problem, "the problem text itself never carries the expected value, as for an equation"


@pytest.mark.asyncio
async def test_a_step_that_returns_a_literal_or_ignores_its_inputs_fails_and_is_repaired(tmp_path: Path) -> None:
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    cheat = "# S1\ndef mean_quality(q1, q2):\n    return 3.0\n"  # passes the example by luck
    model = _Model({**{k: [v] for k, v in GOOD.items()}, "mean_quality": [cheat, GOOD["mean_quality"]]})
    assert (await _filler(tmp_path, model).run_all(specs)).status == "done"
    mine = [p for p in model.prompts if _which(p) == "mean_quality"]
    assert len(mine) == 2 and "returns a number written into the code" in mine[1]


@pytest.mark.asyncio
async def test_resume_does_not_write_a_done_step_again(tmp_path: Path) -> None:
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    first = _Model({k: [v] for k, v in GOOD.items()})
    assert (await _filler(tmp_path, first).run_all(specs)).status == "done"
    again = _Model({})
    assert (await _filler(tmp_path, again).run_all(specs)).status == "done" and again.prompts == []


# --- step_static ---------------------------------------------------------------------------------------------------------


def test_step_static() -> None:
    assert step_static.check("def f(a, b):\n    return 3.0\n", "f").startswith("`f` returns a number written")
    assert step_static.check("def f(a, b):\n    return -1\n", "f")
    assert step_static.check("def f(a, b):\n    return (1.0, 2.0)\n", "f")
    assert "never reads any of its parameters" in step_static.check("def f(a, b):\n    x = 2.0\n    return x * 3\n", "f")
    assert step_static.check("def f(a, b):\n    return a * 2\n", "f") == ""
    assert step_static.check("def f(a, b):\n    if a < 0:\n        return 0.0\n    return a ** b\n", "f") == ""
    assert step_static.check("def f(a):\n    def g():\n        return 1\n    return g() + a\n", "f") == ""
    assert step_static.check("def f():\n    return 4.0\n", "f"), "a literal return is wrong even with no parameter"
    assert step_static.check("def f():\n    import time\n    return time.time()\n", "f") == ""
    assert step_static.check("def f(a):\n    raise NotImplementedError('not written yet')\n", "f") == ""
    assert step_static.check("def f(a:\n", "f") == "" and step_static.check("def g(a):\n    return 1\n", "f") == ""
    assert step_static.check("def f(a):\n    return locals()\n", "f") == ""
    assert "number written" not in step_static.check("def f(a):\n    return True\n", "f"), "a boolean is not a number"


# --- step_coverage -------------------------------------------------------------------------------------------------------

SIM = '''
import numpy as np
from p import model
from p.model import frequency

def _helper(x):
    return model.quality(1.0, x, 2.0)

def run_cell(cell):
    vals = []
    for i in range(3):
        vals.append(frequency(cell["k"], 1.0))
    total = 0.0
    total += _helper(cell["w"])
    result = {"freq_mean": np.mean(vals), "q": total, "const": 0.5, "plain": cell["k"] * 2}
    result["late"] = model.mean_quality(1.0, 2.0)
    return result
'''


def test_step_coverage_flags_what_the_package_does_not_stand_behind() -> None:
    funcs = ["frequency", "quality", "mean_quality"]
    warnings = step_coverage.check(SIM, funcs, ["freq_mean", "q", "const", "plain", "late", "missing"])
    text = "\n".join(warnings)
    assert "`const`" in text and "written into the code" in text
    assert "`plain`" in text and "without calling any function" in text
    assert "`missing`" in text and "not among the numbers" in text
    assert "freq_mean" not in text and "`q`" not in text and "`late`" not in text, warnings
    assert len(warnings) == 3


def test_step_coverage_is_quiet_when_it_cannot_tell() -> None:
    dynamic = "def run_trial(cell, i, seed):\n    out = {}\n    for k in cell:\n        out[k] = 1\n    return {**out}\n"
    assert step_coverage.check(dynamic, ["f"], [{"id": "m"}]) == []
    assert step_coverage.check("def other():\n    pass\n", ["f"], ["m"]) == []
    assert step_coverage.check("def run_cell(c:\n", ["f"], ["m"]) == [] and step_coverage.check(SIM, [], ["const"]) == []


# --- passages ------------------------------------------------------------------------------------------------------------


class _Doc:
    def __init__(self, content: str, **meta: Any) -> None:
        self.content, self.metadata = content, meta


class _Knowledge:
    def __init__(self, hits: list[_Doc], fail: bool = False) -> None:
        self.hits, self.calls, self.fail = hits, [], fail

    async def asearch(self, query: str, **kw: Any) -> list[_Doc]:
        self.calls.append((query, kw))
        if self.fail:
            raise RuntimeError("down")
        return self.hits


CARD = {"id": "S1", "name": "mean quality", "does": "average two quality factors", "source": "Smith 2020"}
HITS = [_Doc("The mean quality factor is the average of two measurements. " * 5, title="On quality", authors=["Smith, J"],
             year=2020, doi="10.1/x"),
        _Doc("Something else entirely about birds. " * 5, title="Birds", authors=["Jones"], year=1999)]


@pytest.mark.asyncio
async def test_passages_are_limited_to_the_source_cut_and_cached(tmp_path: Path) -> None:
    cache = tmp_path / ".fi" / "step_passages.json"
    k = _Knowledge(HITS)
    text = await step_passages.for_card(CARD, k, cache)
    assert "[On quality" in text and "average of two measurements" in text and "birds" not in text.lower()
    assert len(text) <= step_passages.BUDGET_CHARS + 300
    assert k.calls[0][1] == {"top_k": 3, "sources": [], "web": False}, "no web search and no source routing (no model call)"
    again = _Knowledge([])
    assert await step_passages.for_card(CARD, again, cache) == text and again.calls == [], "a resume does not retrieve again"
    assert json.loads(cache.read_text(encoding="utf-8"))["S1"]["text"] == text


@pytest.mark.asyncio
async def test_no_source_no_hits_and_failures(tmp_path: Path) -> None:
    cache = tmp_path / "c.json"
    k = _Knowledge(HITS)
    assert await step_passages.for_card({**CARD, "source": ""}, k, cache) == "" and k.calls == []
    assert await step_passages.for_card({**CARD, "source": "Nobody 1850"}, _Knowledge(HITS), cache) == ""
    broken = _Knowledge(HITS, fail=True)
    assert await step_passages.for_card({**CARD, "id": "S2"}, broken, cache) == ""
    assert "S2" not in json.loads(cache.read_text(encoding="utf-8")), "a failed retrieval is asked again later"
    assert await step_passages.for_card(CARD, None, None) == ""


@pytest.mark.asyncio
async def test_a_source_line_with_nothing_to_filter_by_keeps_every_hit_and_definitions_are_added(tmp_path: Path) -> None:
    defs = [{"metric": "mean quality", "source": "[1]", "quote": "the average of the two factors"},
            {"metric": "unrelated", "source": "[2]", "quote": "something else"}]
    text = await step_passages.for_card({**CARD, "source": "[3]"}, _Knowledge(HITS), None, definitions=defs)
    assert "birds" in text.lower() and 'Definition of mean quality ([1]): "the average of the two factors"' in text
    assert "unrelated" not in text
    only = await step_passages.for_card(CARD, None, None, definitions=defs, metrics=["unrelated"])
    assert only.startswith("Definition of unrelated")


# --- replay: the r6t kinetics quest's two helper steps ---------------------------------------------------------------------

KINETICS_MODEL = {"summary": "first-order decay with Arrhenius rates", "equations": [
    {"id": "E1", "formula": "c(t) = c0 * exp(-k * t)", "role": "generates", "source": "derivation"},
    {"id": "E3", "formula": "k = A * exp(-Ea / (R * T))", "role": "generates", "source": "derivation"},
    {"id": "E5", "formula": "Ea = -R * beta", "role": "analyses", "source": "derivation"}],
    "steps": [
        {"id": "S1", "name": "fit rate", "does": "fit the first-order rate constant to one noisy concentration curve by "
         "weighted nonlinear least squares", "inputs": {"t_s": "s", "c_obs": "mol/L", "sigma_c": "mol/L", "k_true": "1/s"},
         "output": "fitted rate and its variance (1/s)", "uses": ["E1"], "source": "Nonlinear Regression Analysis",
         "relations": ["the fitted rate approaches k_true as sigma_c goes to 0"]},
        {"id": "S2", "name": "fit Arrhenius", "does": "centered weighted linear regression of log rate on 1/T, giving "
         "Ea and ln A", "inputs": {"temperatures": "K", "rates": "1/s", "variances": "1/s^2", "weighted": "bool"},
         "output": "Ea (J/mol) and ln A", "uses": ["E5"], "source": "Arrhenius 1889"}]}
KINETICS_OUTLINE = {"functions": [
    {"name": "first_order_concentration", "signature": "def first_order_concentration(c0_mol_L, k_s_inv, t_s) -> float:",
     "implements": "E1"},
    {"name": "arrhenius_rate", "signature": "def arrhenius_rate(A_s_inv, Ea_J_mol, R_J_mol_K, T_K) -> float:", "implements": "E3"},
    {"name": "activation_energy_from_slope", "signature": "def activation_energy_from_slope(R_J_mol_K, beta) -> float:",
     "implements": "E5"},
    {"name": "fit_arrhenius", "signature": "def fit_arrhenius(temperatures, rates, variances, weighted) -> tuple:",
     "implements": "S2", "depends_on": ["activation_energy_from_slope"]},
    {"name": "fit_rate", "signature": "def fit_rate(t_s, c_obs, sigma_c, k_true) -> tuple:", "implements": "S1"}]}


def test_replay_kinetics_helper_steps_parse_and_produce_specs() -> None:
    protocol = {"model": KINETICS_MODEL}
    cards, problems = plan_steps.parse(protocol)
    assert [c["id"] for c in cards] == ["S1", "S2"] and problems == []
    assert [c["untested_by_example"] for c in cards] == [False, True], "S1 has a relation, S2 has neither"
    specs, notes = fs.model_functions(KINETICS_OUTLINE, protocol)
    names = [s.name for s in specs]
    assert {"fit_rate", "fit_arrhenius"} <= set(names) and notes == []
    assert names.index("first_order_concentration") < names.index("fit_rate")
    assert names.index("activation_energy_from_slope") < names.index("fit_arrhenius")
    assert next(s for s in specs if s.name == "fit_arrhenius").card["untested_by_example"] is True


def test_replay_coverage_of_the_stored_kinetics_simulation_if_present() -> None:
    root = Path("C:/fi_runs/suite/rounds/r6t/first_order_kinetics/out")
    sims = sorted(root.glob("*/code/simulate.py")) if root.is_dir() else []
    if not sims:
        pytest.skip("the stored r6t run is not on this machine")
    funcs = ["first_order_concentration", "noisy_concentration", "arrhenius_rate", "centered_arrhenius_log_rate",
             "activation_energy_from_slope"]
    warnings = step_coverage.check(sims[0].read_text(encoding="utf-8"), funcs, ["definitely_not_a_metric"])
    assert isinstance(warnings, list)
