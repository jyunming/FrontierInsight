"""The quest's code/ as a small research tool, by default (core/code_layout.py).

A quest that runs a simulation gets, besides simulate.py and experiment.py, the model's equations in a package of their
own (``code/<package>/``, each function labelled with its equation), the plan's checks as unit tests
(``tests/test_oracles.py``, written by FI) and ``METHODS.md`` (which function computes each equation, written by FI).
simulate.py stays the file FI imports and calls, so the trial runner and the oracle gate are unchanged.

What that costs is estimated at plan time and shown in plan.md; over ``execution.code_package_max_extra_lines`` /
``code_package_max_extra_calls`` the quest keeps two scripts and says so in plan.md and run.log. After the code is
written the layout is checked: a missing part is a warning, a stop only under ``rigor_profile: research``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import code_layout as cl
from core import plan as _plan
from core import trial_runner as tr
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_callable_simulation import ANALYSIS, MODEL, ORACLE, PROTOCOL, SIMULATE, _reply
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

PKG = "engine_callable"  # the package name the title "engine-callable" gives

MODEL_PY = """\
\"\"\"Classical RK4 on y' = -y.\"\"\"


def rhs(y):  # E1: y' = -y
    return -y


def rk4_step(y, h):
    k1 = rhs(y); k2 = rhs(y + h * k1 / 2); k3 = rhs(y + h * k2 / 2); k4 = rhs(y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
"""
SIM_PKG = f"""\
import math

from {PKG} import model


def run_cell(cell):
    dt = cell["dt"]
    y = 1.0
    for _ in range(int(round(1.0 / dt))):
        y = model.rk4_step(y, dt)
    return {{"error": abs(y - math.exp(-1.0))}}
"""


def _package_reply(model_py: str = MODEL_PY, simulate: str = SIM_PKG) -> str:
    return (f"```python\n# file: simulate.py\n{simulate}\n```\n"
            f"```python\n# file: {PKG}/__init__.py\n\"\"\"RK4 on y' = -y.\"\"\"\n```\n"
            f"```python\n# file: {PKG}/model.py\n{model_py}\n```\n"
            f"```python\n# file: experiment.py\n{ANALYSIS}\n```\nDEPS: matplotlib\n")


def _write_tool(code: Path, *, model_py: str = MODEL_PY, simulate: str = SIM_PKG) -> None:
    (code / PKG).mkdir(parents=True, exist_ok=True)
    (code / PKG / "__init__.py").write_text('"""RK4."""\n', encoding="utf-8")
    (code / PKG / "model.py").write_text(model_py, encoding="utf-8")
    (code / "simulate.py").write_text(simulate, encoding="utf-8")
    for name, text in cl.project_files(code, PROTOCOL, PKG).items():
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        (code / name).write_text(text, encoding="utf-8")


# --- the estimate and the cap ----------------------------------------------------------------------------------------


def test_the_package_name_comes_from_the_title_and_never_shadows_a_name_in_use() -> None:
    assert cl.package_name("engine-callable") == PKG
    assert cl.package_name("SIR outbreaks: how R0 sets the final size") == "sir_outbreaks_how_r0_sets_the"
    assert cl.package_name("") == cl.package_name("json") == cl.package_name("simulate") == "study_model"
    assert cl.package_name("3 body problem") == "study_3_body_problem"
    # A library the code imports is never hidden by a package of the same name beside simulate.py.
    assert cl.package_name("numpy") == cl.package_name("scipy") == cl.package_name("torch") == "study_model"
    assert cl.package_name("pytest") == "study_model", "an installed library"


def test_the_estimate_grows_with_the_equations_and_checks_and_the_cap_decides_the_shape() -> None:
    small = cl.estimate(PROTOCOL)
    assert small["extra_files"] == 4 and small["equations"] == 1 and small["checks"] == 1
    many = {**PROTOCOL, "model": {"equations": [{"id": f"E{i}", "role": "generates"} for i in range(1, 21)]}}
    assert cl.estimate(many)["extra_lines"] > small["extra_lines"]

    kept = cl.decide(PROTOCOL, enabled=True, max_extra_lines=400, max_extra_calls=3, package=PKG)
    assert kept["shape"] == cl.PACKAGE and kept["reason"] == "" and kept["limits"]["extra_calls"] == 3
    over = cl.decide(PROTOCOL, enabled=True, max_extra_lines=10, max_extra_calls=3, package=PKG)
    assert over["shape"] == cl.SINGLE and "execution.code_package_max_extra_lines" in over["reason"]
    # The request limit is a budget spent while the code is written, not a reason to leave the layout out.
    assert cl.decide(PROTOCOL, enabled=True, max_extra_lines=400, max_extra_calls=0, package=PKG)["shape"] == cl.PACKAGE
    off = cl.decide(PROTOCOL, enabled=False, max_extra_lines=400, max_extra_calls=3, package=PKG)
    assert off["shape"] == cl.SINGLE and "code_package is off" in off["reason"]


def test_the_request_budget_is_counted_across_the_quest(tmp_path: Path) -> None:
    assert cl.calls_left(tmp_path, 2) == 2
    cl.spend_call(tmp_path)
    cl.save(tmp_path, {"shape": cl.PACKAGE, "package": PKG})  # a later decision keeps the count
    assert cl.calls_left(tmp_path, 2) == 1 and cl.load(tmp_path)["package"] == PKG
    cl.spend_call(tmp_path)
    assert cl.calls_left(tmp_path, 2) == 0 and cl.calls_left(tmp_path, 5) == 3


def test_the_plan_says_in_plain_words_what_the_layout_costs_or_why_it_was_left_out() -> None:
    kept = cl.decide(PROTOCOL, enabled=True, max_extra_lines=400, max_extra_calls=3, package=PKG)
    text = "\n".join(cl.plan_lines(kept))
    assert f"## {cl.HEADING}" in text and f"code/{PKG}/" in text and "tests/test_oracles.py" in text
    assert "more lines of code" in text and "at most 3 more requests to the model in the whole quest" in text
    assert "METHODS.md" in text
    over = cl.decide(PROTOCOL, enabled=True, max_extra_lines=10, max_extra_calls=3, package=PKG)
    text = "\n".join(cl.plan_lines(over))
    assert "keep two scripts" in text and "Raise the limit" in text
    assert cl.plan_lines(None) == []
    rendered = _plan.render("t", {}, {"hypothesis": "h", "protocol": PROTOCOL}, code_layout=cl.plan_lines(kept))
    assert f"## {cl.HEADING}" in rendered
    assert _plan.parse(rendered).design is not None, "the section is prose: the design block still reads"


def test_the_defaults_turn_the_layout_on_with_a_cap() -> None:
    ex = ExecutionConfig()
    assert ex.code_package is True and ex.code_package_max_extra_lines > 0 and ex.code_package_max_extra_calls >= 3


# --- the reply and the files FI writes -------------------------------------------------------------------------------


def test_the_package_files_are_read_from_the_reply() -> None:
    from core.engine import _PY_FENCE_RE

    files = cl.reply_files(_package_reply(), _PY_FENCE_RE, PKG)
    assert set(files) == {f"{PKG}/__init__.py", f"{PKG}/model.py"} and cl.complete(files, PKG)
    assert "def rhs" in files[f"{PKG}/model.py"] and "# file:" not in files[f"{PKG}/model.py"]
    assert cl.reply_files(_reply(SIMULATE), _PY_FENCE_RE, PKG) == {}
    assert not cl.complete({}, PKG)


def test_methods_maps_each_equation_to_the_function_that_carries_its_label(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_tool(code)
    rows = cl.equation_map(PROTOCOL, cl.package_sources(code))
    assert rows == [{"id": "E1", "formula": "y' = -y", "file": f"{PKG}/model.py", "function": "rhs", "line": 4}]
    text = (code / cl.METHODS_NAME).read_text(encoding="utf-8")
    assert f"`{PKG}/model.py`, `rhs()`" in text and "E1" in text
    # A label above the def names the function below it; a docstring names its own.
    above = cl.equation_map(PROTOCOL, {"p/m.py": "# E1\ndef f(y):\n    return -y\n"})
    doc = cl.equation_map(PROTOCOL, {"p/m.py": 'def g(y):\n    """Implements E1."""\n    return -y\n'})
    assert above[0]["function"] == "f" and doc[0]["function"] == "g"
    assert cl.equation_map(PROTOCOL, {"p/m.py": "# E1\nX = 1\n"})[0]["function"] is None


def _pytest_file(code: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(code / cl.TEST_PATH)], cwd=code, capture_output=True, text=True,
                          timeout=60)


def test_the_plans_checks_become_unit_tests_that_run_the_simulation_like_fi(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_tool(code)
    text = (code / cl.TEST_PATH).read_text(encoding="utf-8")
    assert "def test_rk4_error_at_dt_0_1" in text
    ok = _pytest_file(code)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    # A wrong equation fails the test, with the plan's value in the message.
    _write_tool(tmp_path / "wrong", model_py=MODEL_PY.replace("return -y", "return -1.1 * y  # E1"))
    bad = _pytest_file(tmp_path / "wrong")
    assert bad.returncode == 1 and "the plan expects 0.0" in bad.stdout
    assert cl.oracle_tests({"oracles": [{"name": "no number", "check": "x"}]}) is None


def test_a_random_simulations_test_uses_the_seed_fi_gives_trial_0() -> None:
    oracle = {"name": "zero", "expected": 0, "tolerance": 0, "case": {"R0": 0.0, "n": 10}, "measure": "infected"}
    import ast
    import hashlib
    import os

    text = cl.oracle_tests({"oracles": [oracle]})
    seed_fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_seed")
    namespace: dict[str, Any] = {"hashlib": hashlib, "os": os}
    exec(compile(ast.Module(body=[seed_fn], type_ignores=[]), "tests", "exec"), namespace)
    assert namespace["_seed"]({"R0": 0.0, "n": 10}) == tr.trial_seed(0, tr.cell_key({"R0": 0.0, "n": 10}), 0)


# --- the check after the code is written -----------------------------------------------------------------------------


def test_the_layout_check_names_each_missing_part(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_tool(code)
    assert cl.check(code, PROTOCOL, PKG) == []
    (code / cl.METHODS_NAME).unlink()
    (code / cl.TEST_PATH).unlink()
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (code / PKG / "model.py").write_text(MODEL_PY.replace("  # E1: y' = -y", ""), encoding="utf-8")
    problems = " | ".join(cl.check(code, PROTOCOL, PKG))
    assert "does not use the package" in problems and "no unit tests" in problems
    assert "METHODS.md" in problems and "equation E1 of the model is not labelled" in problems
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "package code/engine_callable/ is missing" in cl.check(empty, PROTOCOL, PKG)[0]


def test_a_change_to_the_package_runs_the_trials_again(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_tool(code)
    before = tr._run_key(code / "simulate.py", PROTOCOL, 1, 0, True)
    assert tr._run_key(code / "simulate.py", PROTOCOL, 1, 0, True) == before
    (code / PKG / "model.py").write_text(MODEL_PY.replace("-y", "-(y)"), encoding="utf-8")
    assert tr._run_key(code / "simulate.py", PROTOCOL, 1, 0, True) != before, "the trials were run with other equations"
    (code / cl.TEST_PATH).write_text("# edited\n", encoding="utf-8")
    after = tr._run_key(code / "simulate.py", PROTOCOL, 1, 0, True)
    (code / cl.TEST_PATH).write_text("# edited again\n", encoding="utf-8")
    assert tr._run_key(code / "simulate.py", PROTOCOL, 1, 0, True) == after, "the tests are not the simulation"


# --- in the engine ---------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path, **execution: Any) -> Engine:
    return Engine(Config(
        topic="t", title="engine-callable", provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", **execution),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def test_the_layout_is_for_a_simulation_only(tmp_path: Path) -> None:
    state = {"design": {"protocol": PROTOCOL}}
    assert _engine(tmp_path / "a")._code_layout(state)["shape"] == cl.PACKAGE
    assert _engine(tmp_path / "b")._code_layout({**state, "no_simulation_resolved": True}) is None
    assert _engine(tmp_path / "c", split_analysis=False)._code_layout(state) is None
    assert _engine(tmp_path / "d")._split_block(state).count(f"# file: {PKG}/model.py") == 1
    assert f"{PKG}/" not in _engine(tmp_path / "e", split_analysis=False)._split_block(state)


def test_a_missing_part_is_a_warning_and_a_stop_only_under_research(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")  # two scripts, no package
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    stops: list[dict[str, Any]] = []
    monkeypatch.setattr(engine, "_pause_for_human", lambda **kw: stops.append(kw))
    # Code written before the quest decided a layout (a quest begun before it existed, resumed): left as it was.
    assert engine._code_layout({"design": {"protocol": PROTOCOL}}) is None
    assert engine._check_code_layout({"design": {"protocol": PROTOCOL}}) == [] and stops == []
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": PKG})  # this quest decided the layout at plan time
    problems = engine._check_code_layout({"design": {"protocol": PROTOCOL}})
    assert problems and stops == []
    assert "package code/engine_callable/ is missing" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    monkeypatch.setattr(engine.config, "rigor_profile", "research")
    engine._check_code_layout({"design": {"protocol": PROTOCOL}})
    assert len(stops) == 1 and stops[0]["kind"] == "code_layout" and stops[0]["payload"]["contract_stage"]
    assert any("execution.code_package: false" in s for s in stops[0]["steps"])
    from core import todo
    assert "package of their own" in todo.advice("code_layout")[0], "the to-do card speaks of this stop"
    # The person adds the missing parts and resumes: the folder is read again and nothing stops.
    _write_tool(code)
    assert engine._check_code_layout({"design": {"protocol": PROTOCOL}}) == [] and len(stops) == 1


def _cfg(tmp_path: Path, **execution: Any) -> Config:
    return Config(
        topic="the error of RK4 on y' = -y at two step sizes", title="engine-callable",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, shared_interpreter=False, system_site_packages=False,
                                  **execution),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    )


def _fake(prompts: list[str]):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["method"] = "integrate y' = -y to t = 1 with classical RK4 at each step size"
            body["protocol"] = PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            if "Implementation Outline" not in prompt[:200]:
                prompts.append(prompt)
            return _package_reply() if f"# file: {PKG}/model.py" in prompt else _reply(SIMULATE)
        return _fake_response_for(prompt)
    return fake_chat


@pytest.mark.asyncio
async def test_a_simulation_quest_gets_the_research_tool_layout_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    code = engine.quest_root / "code"
    assert (code / PKG / "model.py").is_file() and (code / PKG / "__init__.py").is_file()
    assert "from engine_callable import model" in (code / "simulate.py").read_text(encoding="utf-8")
    assert (code / cl.TEST_PATH).is_file() and "`engine_callable/model.py`, `rhs()`" in (
        code / cl.METHODS_NAME).read_text(encoding="utf-8")
    assert len(prompts) == 1, "the reply held the package: no second request"
    plan = (engine.quest_root / "plan.md").read_text(encoding="utf-8")
    assert f"## {cl.HEADING}" in plan and "more requests to the model" in plan
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "laid out as a small research tool" in log and "is missing" not in log
    oracle = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert oracle["status"] == "ok" and oracle.get("contract") == "trial", "FI still calls simulate.py's run_cell"
    readme = (code / "README.md").read_text(encoding="utf-8")
    assert f"`{PKG}/`" in readme and cl.TEST_PATH in readme
    # The unit tests FI wrote pass on the code the quest ran.
    ran = _pytest_file(code)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    record = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert record["levels"]["independently_validated"] is True, record["all_gaps"]


@pytest.mark.asyncio
async def test_a_quest_over_the_cap_keeps_two_scripts_and_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts))
    engine = Engine(_cfg(tmp_path, code_package_max_extra_lines=10))
    await engine.run()
    code = engine.quest_root / "code"
    assert not (code / PKG).exists() and not (code / cl.TEST_PATH).exists() and not (code / cl.METHODS_NAME).exists()
    assert (code / "simulate.py").is_file(), "the two scripts, as before"
    assert prompts and all(f"{PKG}/model.py" not in p for p in prompts), "the package is not asked for"
    plan = (engine.quest_root / "plan.md").read_text(encoding="utf-8")
    assert "The code will keep two scripts" in plan and "execution.code_package_max_extra_lines" in plan
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert re.search(r"the code keeps two scripts .*over the limit of 10", log), log[-3000:]
    oracle = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert oracle["status"] == "ok"


@pytest.mark.asyncio
async def test_a_reply_without_the_package_is_asked_once_more_then_keeps_two_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    async def never_a_package(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Implementation":
            if "Implementation Outline" not in prompt[:200]:
                prompts.append(prompt)
            return _reply(SIMULATE)
        return await _fake([])(self, messages, **kw)

    monkeypatch.setattr("core.engine.LLMClient.chat", never_a_package)
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    assert len(prompts) == 2 and "the reply did not hold the package" in prompts[1]
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "left out the model's package" in log and "no package of equations of its own" in log
    assert json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))["status"] == "ok"


# --- the package where a script is rewritten: an extension, a repair, the labels -------------------------------------


def _engine_with_tool(tmp_path: Path) -> tuple[Engine, Path, dict[str, Any]]:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    _write_tool(code)
    (code / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": PKG})  # decided at plan time
    return engine, code, {"design": {"protocol": PROTOCOL}}


@pytest.mark.asyncio
async def test_an_extension_sees_the_package_and_a_part_of_it_never_replaces_the_whole(tmp_path: Path) -> None:
    from core.engine import _PY_FENCE_RE, _extend_directive

    engine, code, state = _engine_with_tool(tmp_path)
    shown = engine._package_shown(state)
    assert shown is not None and shown[0] == PKG and f"{PKG}/model.py" in shown[1]
    directive = _extend_directive(engine._scripts_on_disk(), ["the error at dt = 0.05"], package=shown)
    assert f"### code/{PKG}/model.py" in directive and "def rk4_step" in directive and "back exactly as it is" in directive
    assert f"code/{PKG}/" not in _extend_directive(engine._scripts_on_disk(), ["x"]), "no package, nothing shown"

    # The extension gives model.py back with only its new function: the package is kept as it is.
    part = "def euler_step(y, h):\n    return y + h * rhs(y)\n"
    scripts = {"simulate": SIM_PKG, "analysis": ANALYSIS}
    _s, _t, files = await engine._code_package_reply(state, "prompt", _package_reply(model_py=part), scripts, extend=True)
    assert files == {}
    assert "without rhs, rk4_step" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    # The whole file with the new function added is taken.
    whole = MODEL_PY + "\n\n" + part
    _s, _t, files = await engine._code_package_reply(state, "prompt", _package_reply(model_py=whole), scripts, extend=True)
    assert "def euler_step" in files[f"{PKG}/model.py"] and "def rk4_step" in files[f"{PKG}/model.py"]
    assert cl.reply_files(_package_reply(model_py=whole), _PY_FENCE_RE, PKG) == files


def test_a_repair_of_the_simulation_sees_the_package_and_can_fix_it(tmp_path: Path) -> None:
    engine, code, state = _engine_with_tool(tmp_path)
    note = engine._package_repair_note(state)
    assert "package_files" in note and f"### code/{PKG}/model.py" in note and "def rhs" in note
    before = engine._package_snapshot()
    fixed = MODEL_PY.replace("return -y", "return -1.0 * y")
    wrote = engine._apply_package_repair(state, {"code": SIM_PKG, "package_files": {
        f"{PKG}/model.py": fixed,
        "../outside.py": "x = 1\n",          # not in the package: ignored
        f"{PKG}/broken.py": "def f(:\n",     # does not parse: ignored
    }}, node="oracle")
    assert wrote == [f"{PKG}/model.py"] and (code / PKG / "model.py").read_text(encoding="utf-8") == fixed
    assert not (code.parent / "outside.py").exists() and not (code / PKG / "broken.py").exists()
    assert engine._apply_package_repair(state, {"code": SIM_PKG}, node="oracle") == []
    # A repair that is undone takes its package change with it.
    (code / PKG / "extra.py").write_text("y = 2\n", encoding="utf-8")
    engine._restore_package(before)
    assert engine._package_snapshot() == before
    # Two scripts alone: nothing about a package is said.
    plain = _engine(tmp_path / "plain")
    (plain.quest_root / "code").mkdir(parents=True)
    (plain.quest_root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    assert plain._package_repair_note(state) == "" and plain._apply_package_repair(
        state, {"package_files": {f"{PKG}/model.py": fixed}}, node="oracle") == []


def test_the_labels_are_asked_for_in_the_package(tmp_path: Path) -> None:
    engine, code, state = _engine_with_tool(tmp_path)
    assert engine._label_target(state, code / "simulate.py") == code / PKG / "model.py"
    plain = _engine(tmp_path / "plain")
    assert plain._label_target(state, plain.quest_root / "code" / "simulate.py") == plain.quest_root / "code" / "simulate.py"


def test_a_plan_without_a_number_to_check_is_not_missing_unit_tests(tmp_path: Path) -> None:
    code = tmp_path / "code"
    no_number = {**PROTOCOL, "oracles": [{"name": "shape", "check": "the error falls as dt falls"}]}
    assert cl.oracle_tests(no_number) is None
    _write_tool(code)
    (code / cl.TEST_PATH).unlink()
    assert not any("unit tests" in p for p in cl.check(code, no_number, PKG))
    assert any("unit tests" in p for p in cl.check(code, PROTOCOL, PKG))


def test_what_the_package_imports_is_in_requirements(tmp_path: Path) -> None:
    from core import code_project

    code = tmp_path / "code"
    _write_tool(code, model_py="import scipy.integrate\n\nfrom . import helpers\n\n\n"
                               "def rhs(y):  # E1\n    return -y\n")
    (code / PKG / "helpers.py").write_text("X = 1\n", encoding="utf-8")
    lines = code_project.requirements_for(code, ["numpy"])
    assert "scipy" in lines and "numpy" in lines
    assert not {PKG, "model", "helpers", "simulate"} & set(lines), lines


# --- a later pass, an older quest, a name of the reply's own, the request budget -------------------------------------


def _decided(engine: Engine, **extra: Any) -> None:
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": PKG, **extra})


@pytest.mark.asyncio
async def test_a_redesign_without_the_package_is_asked_again_and_says_the_old_one_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, code, state = _engine_with_tool(tmp_path)
    _decided(engine)
    asked: list[str] = []

    async def no_package(prompt, **kw):  # noqa: ANN001
        asked.append(prompt)
        return _reply(SIM_PKG.replace("1.0 / dt", "2.0 / dt"))

    monkeypatch.setattr(engine, "_chat", no_package)
    new_sim = SIM_PKG.replace("1.0 / dt", "2.0 / dt")  # the simulation is written again, the package left out
    scripts = {"simulate": new_sim, "analysis": ANALYSIS}
    _s, _t, files = await engine._code_package_reply(state, "prompt", _reply(new_sim), scripts, extend=False)
    assert files == {} and len(asked) == 1 and "did not hold the package" in asked[0]
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "the package from before is kept" in log
    # Given back unchanged (a rerun that keeps the simulation): nothing to ask.
    _s, _t, files = await engine._code_package_reply(state, "prompt", _reply(SIM_PKG), {"simulate": SIM_PKG,
                                                                                         "analysis": ANALYSIS},
                                                     extend=False)
    assert len(asked) == 1
    # The budget: once the limit of extra requests is spent, the model is not asked again.
    engine.config.execution.code_package_max_extra_calls = 1
    _s, _t, files = await engine._code_package_reply(state, "prompt", _reply(new_sim), scripts, extend=False)
    assert len(asked) == 1 and "not asking again" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_an_older_quest_or_an_extension_of_two_scripts_is_not_restructured(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True)
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    state = {"design": {"protocol": PROTOCOL}, "refine_extend": ["the error at dt = 0.05"]}
    assert f"{PKG}/model.py" not in engine._split_block(state), "an older quest: no layout"
    _decided(engine)  # a quest that decided the layout but kept two scripts (the package never came)
    assert f"{PKG}/model.py" not in engine._split_block(state), "an extension does not ask for a package"
    assert f"{PKG}/model.py" in engine._split_block({"design": {"protocol": PROTOCOL}})
    _s, _t, files = await engine._code_package_reply(state, "p", _package_reply(), {"simulate": SIM_PKG,
                                                                                     "analysis": ANALYSIS}, extend=True)
    assert files == {} and not (code / PKG).exists()


@pytest.mark.asyncio
async def test_a_package_under_a_name_of_the_replys_own_is_kept(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code").mkdir(parents=True)
    state = {"design": {"protocol": PROTOCOL}}
    reply = _package_reply().replace(f"{PKG}/", "rk4tool/").replace(f"from {PKG} import", "from rk4tool import")
    scripts = {"simulate": SIM_PKG.replace(f"from {PKG} import", "from rk4tool import"), "analysis": ANALYSIS}
    _s, _t, files = await engine._code_package_reply(state, "p", reply, scripts, extend=False)
    assert set(files) == {"rk4tool/__init__.py", "rk4tool/model.py"}
    assert cl.load(engine.quest_root)["package"] == "rk4tool", "used from then on"


@pytest.mark.asyncio
async def test_a_repair_that_fixes_only_the_package_is_applied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, code, state = _engine_with_tool(tmp_path)
    _decided(engine)
    fixed = MODEL_PY.replace("return -y", "return -1.0 * y")
    sim_before = (code / "simulate.py").read_text(encoding="utf-8")

    async def package_only(prompt, **kw):  # noqa: ANN001
        assert f"### code/{PKG}/model.py" in prompt, "the repair is shown the package"
        return json.dumps({"code": "", "package_files": {f"{PKG}/model.py": fixed}, "patch_summary": "E1 sign"})

    monkeypatch.setattr(engine, "_chat", package_only)
    new, failed, outcome = await engine._repair_script_for_oracle(state, code / "simulate.py", [ORACLE],
                                                                  ["rk4 error at dt 0.1: off"], passing=[])
    assert outcome == "applied" and not failed and new == sim_before
    assert (code / PKG / "model.py").read_text(encoding="utf-8") == fixed
    assert (code / "simulate.py").read_text(encoding="utf-8") == sim_before


def test_the_package_is_checked_for_randomness_and_the_analysis_for_importing_it(tmp_path: Path) -> None:
    code = tmp_path / "code"
    _write_tool(code, model_py=MODEL_PY + "\nimport numpy as np\n_RNG = np.random.default_rng(0)\n")
    (code / "experiment.py").write_text(f"from {PKG} import model\n" + ANALYSIS, encoding="utf-8")
    problems = " | ".join(cl.check(code, PROTOCOL, PKG))
    assert "makes random numbers when it is loaded" in problems
    assert "experiment.py imports the model's package" in problems
    assert cl.module_level_randomness("import random\n\ndef f(seed):\n    return random.Random(seed).random()\n") == []


def test_the_generated_tests_take_any_check_name_and_the_scripts_names_are_not_package_files(tmp_path: Path) -> None:
    odd = {**ORACLE, "name": 'ends with "quote" and \\ backslash'}
    text = cl.oracle_tests({**PROTOCOL, "oracles": [odd]})
    compile(text, "test_oracles.py", "exec")
    from core.engine import _PY_FENCE_RE

    assert cl.reply_files(f"```python\n# file: {PKG}/simulate.py\nx = 1\n```\n", _PY_FENCE_RE, PKG) == {}


def test_a_change_to_the_package_makes_the_older_contracts_raw_files_stale(tmp_path: Path) -> None:
    from core import split_run

    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    assert split_run.simulation_sha(code / "simulate.py") == split_run.sha256_of(code / "simulate.py"), "no package"
    _write_tool(code)
    before = split_run.simulation_sha(code / "simulate.py")
    (code / PKG / "model.py").write_text(MODEL_PY.replace("-y", "-(y)"), encoding="utf-8")
    assert split_run.simulation_sha(code / "simulate.py") != before


def test_a_generator_the_caller_can_replace_is_not_a_problem_but_one_made_at_load_is() -> None:
    fallback = "import numpy as np\n\n\ndef step(y, rng=None):\n    rng = rng or np.random.default_rng()\n    return y\n"
    assert set(__import__("core.engine").engine.unseeded_rng_calls(fallback)) and cl.fallback_rng_lines(fallback) >= {5}
    ignores = "import numpy as np\n\n\ndef step(y, seed):\n    gen = np.random.default_rng()\n    return y\n"
    assert 5 not in cl.fallback_rng_lines(ignores), "a function that takes a seed and ignores it is still reported"
    idioms = [
        "    self_rng = rng if rng is not None else np.random.default_rng()\n",
        "    g = np.random.default_rng(seed) if seed is not None else np.random.default_rng()\n",
        "    return rng if rng is not None else np.random.default_rng()\n",
        "    gen = rng or np.random.default_rng()\n",
        "    return g(y, rng or np.random.default_rng())\n",
        "    if rng is None:\n        rng = np.random.default_rng()\n",
    ]
    for body in idioms:
        src = f"import numpy as np\n\n\ndef f(y, rng=None, seed=None):\n{body}"
        found = __import__("core.engine").engine.unseeded_rng_calls(src)
        assert all(line in cl.fallback_rng_lines(src) for line, _ in found), body
    nested = ("import numpy as np\n\n\ndef f(y, rng=None):\n    def inner():\n        rng = np.random.default_rng()\n"
              "        return rng\n    return inner()\n")
    assert 6 not in cl.fallback_rng_lines(nested), "a nested function is judged on its own"
    other_if = "import numpy as np\nif __name__ != '__main__':\n    R = np.random.default_rng()\n"
    assert cl.module_level_randomness(other_if) == [3]
    main = "import numpy as np\n\n\ndef f(y):\n    return y\n\n\nif __name__ == '__main__':\n    np.random.default_rng()\n"
    assert cl.module_level_randomness(main) == [], "runs only when the file is run by itself"
    default = "import numpy as np\n\n\ndef f(y, rng=np.random.default_rng(0)):\n    return y\n"
    assert cl.module_level_randomness(default) == [4], "a default argument is made at load"


def test_the_budget_starts_again_for_code_written_from_nothing_and_the_shape_holds(tmp_path: Path) -> None:
    engine, code, state = _engine_with_tool(tmp_path)
    cl.save(engine.quest_root, {"shape": cl.PACKAGE, "package": PKG, "extra_calls_spent": 3})
    # A redesign that adds many equations does not switch a quest whose package is on disk to two scripts.
    many = {**PROTOCOL, "model": {"equations": [{"id": f"E{i}", "role": "generates"} for i in range(1, 200)]}}
    assert engine._code_layout({"design": {"protocol": many}}, many)["shape"] == cl.PACKAGE
    engine.config.execution.code_package = False
    assert engine._code_layout(state)["shape"] == cl.SINGLE, "switching it off still does"
    engine.config.execution.code_package = True
    # Two scripts are decided again each time: raising the limit, as plan.md says, takes effect on a resume.
    cl.save(engine.quest_root, {"shape": cl.SINGLE, "package": PKG, "reason": "over the limit of 10"})
    assert engine._code_layout(state)["shape"] == cl.PACKAGE
    # A restart from the code step: code/ is empty, the budget is new.
    import shutil

    shutil.rmtree(code)
    code.mkdir()
    import asyncio

    asyncio.run(engine._code_package_reply(state, "p", _package_reply(), {"simulate": SIM_PKG, "analysis": ANALYSIS},
                                           extend=False))
    assert cl.calls_left(engine.quest_root, 3) == 3


def test_a_package_module_named_like_a_script_is_not_taken_for_it() -> None:
    from core import split_run
    from core.engine import _PY_FENCE_RE

    reply = (f"```python\n# file: simulate.py\nREAL = 1\n```\n```python\n# file: experiment.py\nA = 1\n```\n"
             f"```python\n# file: {PKG}/__init__.py\n```\n```python\n# file: {PKG}/simulate.py\nFAKE = 1\n```\n")
    scripts = split_run.parse_split_response(reply, _PY_FENCE_RE)
    assert scripts is not None and "REAL" in scripts["simulate"]
    plain = "```python\n# file: src/simulate.py\nS = 1\n```\n```python\n# file: src/experiment.py\nE = 1\n```\n"
    assert split_run.parse_split_response(plain, _PY_FENCE_RE) is not None, "a folder that is not a package still reads"
    one_folder = ("```python\n# file: project/simulate.py\nS = 1\n```\n```python\n# file: project/experiment.py\nE = 1\n```\n"
                  "```python\n# file: project/model.py\nM = 1\n```\n")
    assert split_run.parse_split_response(one_folder, _PY_FENCE_RE) is not None, "everything in one folder still reads"
    mixed = reply.replace(f"# file: {PKG}/__init__.py", f"# file: code/{PKG}/__init__.py")
    assert "REAL" in split_run.parse_split_response(mixed, _PY_FENCE_RE)["simulate"]


@pytest.mark.asyncio
async def test_simulate_importing_a_package_that_is_not_there_is_said_plainly(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code").mkdir(parents=True)
    state = {"design": {"protocol": PROTOCOL}}
    engine.config.execution.code_package_max_extra_calls = 0
    ghost_sim = SIM_PKG.replace(f"from {PKG} import", "from ghost import")
    # The reply wrote a package of its own name with no model.py, and simulate.py imports it.
    reply = _reply(ghost_sim) + "```python\n# file: ghost/__init__.py\n```\n```python\n# file: ghost/dynamics.py\nX = 1\n```\n"
    await engine._code_package_reply(state, "p", reply, {"simulate": ghost_sim, "analysis": ANALYSIS}, extend=False)
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "simulate.py imports the package `ghost`, but code/ghost/ was not written" in log


def test_the_improve_loop_may_change_the_models_package(tmp_path: Path) -> None:
    from core import improve

    quest = tmp_path / "q"
    code = quest / "code"
    _write_tool(code)
    (code / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    allowed = improve.editable(quest)
    assert allowed[0] == "simulate.py" and f"{PKG}/model.py" in allowed and "experiment.py" not in allowed
    assert not any(a.startswith("tests/") for a in allowed), "FI's unit tests are not the simulation"
    files = improve.snapshot(quest)
    on_disk = (code / PKG / "model.py").read_bytes().decode("utf-8")
    assert files[f"{PKG}/model.py"] == on_disk
    edit = improve.Edit(file=f"code/{PKG}/model.py", find="return -y", replace="return -1.0 * y", why="E1")
    new, why = improve.check_edit(files, edit, allowed, tried={improve.fingerprint(files)}, expected=[])
    assert new is not None and why == "" and improve.edited_path(edit) == f"{PKG}/model.py"
    improve.save_snapshot(quest, "original", files)
    improve.restore(quest, {**files, f"{PKG}/model.py": new})
    assert "-1.0 * y" in (code / PKG / "model.py").read_text(encoding="utf-8")
    improve.restore(quest, improve.load_snapshot(quest, "original"))
    assert (code / PKG / "model.py").read_bytes().decode("utf-8") == on_disk
    outside = improve.Edit(file=f"code/{PKG}/../experiment.py", find="x", replace="y", why="")
    assert improve.check_edit(files, outside, allowed, tried=set(), expected=[])[0] is None


def test_the_files_fi_writes_are_not_the_code_an_attempt_ran(tmp_path: Path) -> None:
    import hashlib

    from core import attempt_records

    quest = tmp_path / "q"
    code = quest / "code"
    _write_tool(code)
    record = {name: hashlib.sha256((code / name).read_bytes()).hexdigest() for name in (cl.METHODS_NAME, cl.TEST_PATH)}
    (quest / ".fi").mkdir(parents=True)
    (quest / ".fi" / "code_project.json").write_text(json.dumps(record), encoding="utf-8")
    hashes = attempt_records.script_hashes(quest)
    assert cl.METHODS_NAME not in hashes and cl.TEST_PATH not in hashes and f"{PKG}/model.py" in hashes
