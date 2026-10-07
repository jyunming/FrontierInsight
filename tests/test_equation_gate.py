"""The plan's worked examples and the equation tests before the study, in the engine (core/equation_tests.py).
Fake models, neutral models (a mass on a damped spring)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from core import code_layout as cl, equation_tests as eqt, plan
from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_oracle_review import _Model as _PlanModel, _asked_about_checks, _config, _section

SPRING = {"summary": "a mass on a damped spring", "equations": [
    {"id": "E1", "formula": "w = sqrt(k / m)", "role": "generates", "source": "derivation", "derivation": "Newton's law",
     "example": {"inputs": {"k": 8.0, "m": 2.0}, "expected_formula": "sqrt(k / m)"}},
    {"id": "E2", "formula": "Q = m * w / c", "role": "generates", "source": "derivation", "derivation": "energy ratio",
     "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c"}},
]}
PROTOCOL = {"grid": {"c": [0.5, 1.0]}, "model": SPRING}
OUTLINE = {"scaffold": "x = 1", "functions": [
    {"name": "frequency", "signature": "def frequency(k, m) -> float:", "one_line_purpose": "natural frequency",
     "implements": "E1"},
    {"name": "quality", "signature": "def quality(m, w, c) -> float:", "one_line_purpose": "quality factor",
     "implements": "E2", "depends_on": ["frequency"]},
    {"name": "run_cell", "signature": "def run_cell(cell) -> dict:", "one_line_purpose": "one setting"},
], "constants": [{"name": "G", "value": "9.81", "source": "measured"}]}
GOOD = {"frequency": "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k / m)\n",
        "quality": "# E2\ndef quality(m, w, c):\n    return m * w / c\n"}
WRONG_Q = "# E2\ndef quality(m, w, c):\n    return w / c\n"  # a different normalisation: m dropped
SIMULATE = "from spring import model\n\n\ndef run_cell(cell):\n    return {'q': model.quality(2.0, 3.0, cell['c'])}\n"


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(topic="a mass on a damped spring", title="spring", provider=ProviderConfig(name="openai"),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)

    async def no_install(packages):  # noqa: ANN001
        return []

    engine._install_packages = no_install  # type: ignore[method-assign]
    return engine


class _Writer:
    """A fake code-writing model: answers a function request by the function it names, and counts the requests."""

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self.replies, self.prompts, self.seen = replies, [], {}

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        found = re.search(r"# The function[^\n]*\n\n```python\n(?:#[^\n]*\n)*def (\w+)\(", prompt)
        assert found, prompt[:200]
        name = found.group(1)
        n = self.seen[name] = self.seen.get(name, 0) + 1
        replies = self.replies[name]
        return "```python\n" + replies[min(n, len(replies)) - 1] + "```"

    async def aclose(self) -> None:
        return None


STATE: dict[str, Any] = {"design": {"protocol": PROTOCOL}}


def _plan_md(engine: Engine, protocol: dict[str, Any] = PROTOCOL) -> None:
    design = {"hypothesis": "h", "protocol": protocol}
    text = plan.render(engine.config.topic, {}, design)
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")


# --- Feature C in the engine -------------------------------------------------------------------------------------------


# --- the equation tests in the engine -------------------------------------------------------------------------------------------


def _write_code(engine: Engine, quality: str) -> None:
    code = engine.quest_root / "code"
    (code / "spring").mkdir(parents=True, exist_ok=True)
    (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (code / "spring" / "model.py").write_text(GOOD["frequency"] + "\n\n" + quality, encoding="utf-8")
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")


@pytest.mark.asyncio
async def test_an_equation_implemented_wrongly_is_caught_before_the_run_and_sent_to_the_existing_repair(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, WRONG_Q)
    engine._client = _Writer({"quality": [WRONG_Q]})  # whatever is asked of a model, the function stays wrong
    patch = await engine._equation_gate(STATE, None, None)
    assert patch is not None and patch["exec_result"]["returncode"] == 1 and patch["result_json"] == {}
    stderr = patch["exec_result"]["stderr_tail"]
    assert "`quality`" in stderr and "equation E2" in stderr and "Q = m * w / c" in stderr
    for secret in ("12.0", "3.0", "0.5", "'m'"):
        assert secret not in stderr, "the worked example's numbers never reach a request to the model"
    record = json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))
    assert record["failed"][0]["id"] == "E2" and record["tested"] == ["E1", "E2"]
    assert (engine.quest_root / "code" / eqt.TEST_PATH).is_file()


@pytest.mark.asyncio
async def test_a_correct_package_passes_and_asks_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, GOOD["quality"])
    writer = _Writer({})
    engine._client = writer
    assert await engine._equation_gate(STATE, None, None) is None and writer.prompts == []


@pytest.mark.asyncio
async def test_an_equation_that_cannot_be_tested_is_said_plainly_and_the_quest_goes_on(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    no_example = {**PROTOCOL, "model": {**SPRING, "equations": [
        SPRING["equations"][0],
        {**{k: v for k, v in SPRING["equations"][1].items() if k != "example"}},
        {"id": "E3", "formula": "t_half = ln(2) / g", "role": "analyses", "source": "derivation",
         "example": {"untestable": "read from a simulated trace"}},
    ]}}
    _plan_md(engine, no_example)
    _write_code(engine, WRONG_Q)  # wrong, but its equation has no example: nothing can show it here
    engine._client = _Writer({})
    assert await engine._equation_gate({"design": {"protocol": no_example}}, None, None) is None
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "equation E2 is not tested before the run: the plan gave no worked example" in log
    assert "equation E3 is not tested before the run: the plan says it has no closed form" in log
    record = json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))
    assert record["tested"] == ["E1"] and len(record["not_tested"]) == 2


@pytest.mark.asyncio
async def test_one_script_quests_and_quests_with_the_checks_off_are_not_tested(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    engine._client = _Writer({})
    assert await engine._equation_gate(STATE, None, None) is None, "no simulate.py: a one-script quest is unchanged"
    _write_code(engine, WRONG_Q)
    off = engine.config.model_copy(deep=True)
    off.engine.oracle_check = "off"
    other = Engine(off)
    assert await other._equation_gate(STATE, None, None) is None


# --- the worked examples are asked of the plan, once, in the request that already goes ----------------------------------


def _with_example(block: dict[str, Any]) -> dict[str, Any]:
    model = block["protocol"]["model"]
    eqs = [{**e, "example": {"inputs": {"a": 3.0}, "expected_formula": "a * a"}} if e["id"] == "E2" else
           {**e, "formula": "SOMETHING ELSE"} for e in model["equations"]]  # E1's formula must not be moved by this request
    return {**block, "protocol": {**block["protocol"], "model": {**model, "equations": eqs}}}


@pytest.mark.asyncio
async def test_a_missing_worked_example_goes_into_the_one_request_and_only_the_example_may_change(tmp_path: Path) -> None:
    protocol = {**PROTOCOL, "model": {**SPRING, "equations": [
        SPRING["equations"][0], {k: v for k, v in SPRING["equations"][1].items() if k != "example"}]},
        "oracles": [{"name": "w_check", "kind": "special_case", "check": "w at k=8, m=2", "expected": 2.0,
                     "expected_formula": "sqrt(8 / 2)", "tolerance": 1e-9, "case": {"c": 0.5}, "measure": "w",
                     "reference": "derivation: sqrt(4) = 2"}]}
    engine = Engine(_config(tmp_path))
    review = {"checks": [{"name": "w_check", "appropriate": "yes", "discriminating": "yes", "well_defined": "yes"}],
              "equations_tested": ["E1"], "equations_not_tested": [], "add": [], "summary": "fine"}
    model = _PlanModel(protocol, review, revise=_with_example)
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    asked = [r for r in model.revisions if "worked example" in r]
    assert len(asked) == 1, "one request, however many things it asks"
    assert "E2 (`Q = m * w / c`): give its `example`" in asked[0] and "E1 (" not in asked[0].split("give its `example`")[0][-200:]
    design = plan.load_design(engine.quest_root)[0]
    eqs = {e["id"]: e for e in design["protocol"]["model"]["equations"]}
    assert eqs["E2"]["example"]["expected_formula"] == "a * a"
    assert eqs["E1"]["formula"] == "w = sqrt(k / m)", "a request for examples never moves an equation"
    assert _asked_about_checks(model) == []
    # Asked once per quest: another pass asks nothing.
    n = len(model.revisions)
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len(model.revisions) == n


@pytest.mark.asyncio
async def test_an_example_the_plan_still_does_not_give_is_recorded_plainly(tmp_path: Path) -> None:
    protocol = {**PROTOCOL, "model": {**SPRING, "equations": [
        {k: v for k, v in e.items() if k != "example"} for e in SPRING["equations"]]},
        "oracles": [{"name": "w_check", "kind": "special_case", "check": "w", "expected": 2.0,
                     "expected_formula": "sqrt(8 / 2)", "tolerance": 1e-9, "case": {"c": 0.5}, "measure": "w",
                     "reference": "derivation: sqrt(4) = 2"}]}
    engine = Engine(_config(tmp_path))
    review = {"checks": [{"name": "w_check", "appropriate": "yes", "discriminating": "yes", "well_defined": "yes"}],
              "equations_tested": ["E1"], "equations_not_tested": [], "add": [], "summary": "fine"}
    model = _PlanModel(protocol, review)  # the plan changes nothing
    engine._client = model
    await engine._node_plan({"topic": engine.config.topic, "literature": []})
    assert len([r for r in model.revisions if "worked example" in r]) == 1
    section = _section(engine)
    assert "equation E1 is not tested before the run: the plan gave no worked example" in section
    assert "equation E2 is not tested before the run" in section


def test_the_code_layout_writes_the_equation_tests_with_the_other_files(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "spring").mkdir(parents=True)
    (code / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (code / "spring" / "model.py").write_text(GOOD["frequency"] + "\n\n" + GOOD["quality"], encoding="utf-8")
    files = cl.project_files(code, PROTOCOL, "spring")
    assert eqt.TEST_PATH in files and "def test_e1" in files[eqt.TEST_PATH] and "def test_e2" in files[eqt.TEST_PATH]
