"""A repair touches only the faulty part (core/agent_scope.py, FunctionFiller._repair_module). Fake agent, fake model; neutral code."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import pytest

from core import agent_scope
from core.engine import Engine
from tests.test_agent_build import _Agent, _engine, _first_write, _no_code_checks
from tests.test_function_steps import BAD, GOOD
from tests.test_function_steps_partial import _ModuleModel, _filler, _module
from tests.test_function_steps import _specs

CODE = (
    "def heat(x):\n    return x / 0\n\n\n"
    "def cool(x):\n    return x * 2\n\n\n"
    "def main():\n    return heat(1)\n\n\n"
    "def other_caller():\n    return cool(3)\n"
)
UTILS = "def tidy(x):\n    return x\n"


def _first(code: Path) -> None:
    code.mkdir(parents=True, exist_ok=True)
    (code / "experiment.py").write_text(CODE, encoding="utf-8")
    (code / "utils.py").write_text(UTILS, encoding="utf-8")


def _failed(eng: Engine) -> dict[str, Any]:
    path = eng.quest_root / "code" / "experiment.py"
    tb = (f'Traceback (most recent call last):\n  File "{path}", line 10, in main\n    return heat(1)\n'
          f'  File "{path}", line 2, in heat\n    return x / 0\nZeroDivisionError: division by zero')
    return {"returncode": 1, "stderr_tail": tb, "stdout_tail": ""}


def _round(edit: Callable[[Path], None], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Engine, dict[str, Any]]:
    agent = _Agent([_first, edit])
    eng = _engine(tmp_path, agent=agent)
    _first_write(eng, agent)
    monkeypatch.setattr("core.agent_build._frozen.load", lambda _root: None)
    patch = asyncio.run(eng._reflect_by_agent({"exec_reflect_history": []}, 0, 1, "1", "", _failed(eng), {}))
    assert patch["exec_patch_pending"] is True
    return eng, patch


def _text(eng: Engine, name: str = "experiment.py") -> str:
    return (eng.quest_root / "code" / name).read_text(encoding="utf-8")


def test_an_agent_repair_that_also_edits_an_unrelated_function_has_that_edit_put_back(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("x / 0", "x / 1").replace("x * 2", "x * 5"), encoding="utf-8")

    eng, patch = _round(edit, tmp_path, monkeypatch)
    text = _text(eng)
    assert "return x / 1" in text, "the fix of the failing function is kept"
    assert "return x * 2" in text and "x * 5" not in text, "the unrelated function is as it was"
    assert "x * 5" not in patch["code"]


def test_an_edit_to_a_direct_caller_of_the_failing_function_is_kept(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("x / 0", "x / y").replace("def heat(x)", "def heat(x, y=1)")
                                            .replace("heat(1)", "heat(1, 2)"), encoding="utf-8")

    eng, _ = _round(edit, tmp_path, monkeypatch)
    assert "heat(1, 2)" in _text(eng) and "def heat(x, y=1)" in _text(eng)


def test_a_changed_file_the_failure_is_not_in_is_put_back_and_a_new_file_is_kept(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("x / 0", "x / 1"), encoding="utf-8")
        (code / "utils.py").write_text("def tidy(x):\n    return None\n", encoding="utf-8")
        (code / "newhelper.py").write_text("VALUE = 1\n", encoding="utf-8")

    eng, _ = _round(edit, tmp_path, monkeypatch)
    assert _text(eng, "utils.py") == UTILS and "x / 1" in _text(eng)
    assert (eng.quest_root / "code" / "newhelper.py").is_file()


def test_a_failure_that_names_no_place_in_the_code_does_not_limit_the_round(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    assert agent_scope.scope_from_traceback("MemoryError", tmp_path) is None
    assert agent_scope.scope_from_traceback('File "/usr/lib/python3/x.py", line 3, in f', tmp_path) is None


def test_the_scope_is_the_failing_function_and_its_direct_callers(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "experiment.py").write_text(CODE, encoding="utf-8")
    scope = agent_scope.scope_from_traceback(f'File "{tmp_path / "code" / "experiment.py"}", line 2, in heat', tmp_path)
    assert scope is not None and scope.files == {"experiment.py": {"heat", "main"}}


def _agent_with(steps: list[Callable[[Path], None]], says: list[str], tmp_path: Path, monkeypatch) -> tuple[Engine, _Agent]:  # noqa: ANN001
    agent = _Agent(steps, says=says)
    eng = _engine(tmp_path, agent=agent)
    _first_write(eng, agent)
    monkeypatch.setattr("core.agent_build._frozen.load", lambda _root: None)
    return eng, agent


def _reflect(eng: Engine) -> dict[str, Any]:
    return asyncio.run(eng._reflect_by_agent({"exec_reflect_history": []}, 0, 1, "1", "", _failed(eng), {}))


def _write_code(text: str) -> Callable[[Path], None]:
    def step(code: Path) -> None:
        code.mkdir(parents=True, exist_ok=True)
        (code / "experiment.py").write_text(text, encoding="utf-8")
    return step


def test_a_frame_in_a_library_folder_that_has_a_code_part_is_not_the_quests_code(tmp_path: Path) -> None:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "experiment.py").write_text(CODE, encoding="utf-8")
    vendor = tmp_path / ".venv" / "Lib" / "site-packages" / "vendor" / "code" / "experiment.py"
    assert agent_scope.scope_from_traceback(f'File "{vendor}", line 2, in heat', tmp_path) is None
    assert agent_scope.scope_from_traceback('File "/x/y/code/experiment.py", line 2, in heat', tmp_path) is None


def test_a_deleted_function_is_put_back_where_it_was_so_the_module_still_runs() -> None:
    old = "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\nVALUE = b() + a()\n"
    new = "def a():\n    return 1\n\n\nVALUE = a()\n"
    fixed = agent_scope._put_back_units(old, new, ["b"])
    ns: dict[str, Any] = {}
    exec(compile(fixed, "m", "exec"), ns)  # noqa: S102 -- a NameError here is the fault
    assert fixed.index("def a") < fixed.index("def b") < fixed.index("VALUE")


def test_a_helper_the_failing_function_calls_may_change_with_it(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("def cool(x):", "def cool(x, y=1):"), encoding="utf-8")
    eng, agent = _agent_with([_write_code(CODE.replace("return x / 0", "return cool(x) / 0")), edit], [], tmp_path,
                             monkeypatch)
    _reflect(eng)
    assert "def cool(x, y=1)" in _text(eng) and len(agent.prompts) == 2, "a callee is in the scope: kept, no second round"


def test_an_edit_outside_the_scope_is_put_back_and_the_agent_is_sent_back_once_with_the_names(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("x / 0", "x / 1").replace("x * 2", "x * 5"), encoding="utf-8")
    eng, agent = _agent_with([_first, edit], [], tmp_path, monkeypatch)
    _reflect(eng)
    assert len(agent.prompts) == 3 and "cool" in agent.prompts[2] and "NEEDS:" in agent.prompts[2]
    assert "x * 2" in _text(eng) and "x / 1" in _text(eng)


def test_an_edit_outside_the_scope_the_agent_says_the_fix_needs_is_kept(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("x / 0", "x / 1").replace("x * 2", "x * 5"), encoding="utf-8")
    eng, agent = _agent_with([_first, edit], ["wrote it", "fixed\nNEEDS: cool - the fix scales by it"], tmp_path,
                             monkeypatch)
    _reflect(eng)
    assert "x * 5" in _text(eng) and len(agent.prompts) == 2, "claimed in the same message: kept, no second round"


def test_a_put_back_that_would_not_compile_is_undone(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "m.py").write_text("def f(:\n", encoding="utf-8")  # the snapshot's own text does not compile
    (code / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    said, kept = agent_scope.put_back(agent_scope.Violations(files=["m.py"]), {"m.py": b"def f(:\n"}, code)
    assert kept is False and said == [] and (code / "m.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


METHODS = (
    "class Model:\n    def run(self):\n        return 1 / 0\n\n    def other(self):\n        return 2\n\n\n"
    "def main():\n    return Model().run()\n"
)


def test_a_failure_in_a_method_scopes_that_method_not_the_class(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "experiment.py").write_text(METHODS, encoding="utf-8")
    scope = agent_scope.scope_from_traceback(f'File "{tmp_path / "code" / "experiment.py"}", line 3, in run', tmp_path)
    assert scope is not None and scope.files == {"experiment.py": {"Model.run", "main"}}

    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("1 / 0", "1 / 1").replace("return 2", "return 9"), encoding="utf-8")
    agent = _Agent([_write_code(METHODS), edit])
    eng = _engine(tmp_path / "q", agent=agent)
    _first_write(eng, agent)
    monkeypatch.setattr("core.agent_build._frozen.load", lambda _root: None)
    path = eng.quest_root / "code" / "experiment.py"
    failed = {"returncode": 1, "stdout_tail": "", "stderr_tail": f'File "{path}", line 3, in run\nZeroDivisionError'}
    asyncio.run(eng._reflect_by_agent({"exec_reflect_history": []}, 0, 1, "1", "", failed, {}))
    text = path.read_text(encoding="utf-8")
    assert "1 / 1" in text and "return 2" in text and "return 9" not in text


def test_a_round_that_leaves_invalid_python_is_a_failed_round_and_the_code_is_as_it_was(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        (code / "experiment.py").write_text("def heat(x:\n    return 1\n", encoding="utf-8")
    eng, agent = _agent_with([_first, edit], [], tmp_path, monkeypatch)
    patch = _reflect(eng)
    assert _text(eng) == CODE and len(agent.prompts) == 2, "the whole round is undone, no widened scope"
    assert "exec_give_up_reason" in patch and not patch.get("exec_patch_pending"), "the round is not recorded as a fix"


# --- the module repair asks for the failing functions only ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_module_repair_asks_for_the_failing_function_only_and_passing_ones_stay_as_they_were(tmp_path: Path) -> None:
    only_quality = "# E2\ndef quality(m, w, c):\n    return m * w / c\n"
    model = _ModuleModel({"frequency": [GOOD["frequency"]], "quality": [BAD["quality"]], "decay": [GOOD["decay"]]},
                         module_replies=[only_quality])
    outcome = await _filler(tmp_path, model).run_all(_specs())
    assert outcome.status == "done"
    assert "whole corrected module" not in model.module_prompts[0]
    text = (tmp_path / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert "return m * w / c" in text
    for name in ("frequency", "decay"):
        assert GOOD[name].split("\n\n\n", 1)[1].strip() in text, "a passing function is byte for byte as it was"


@pytest.mark.asyncio
async def test_a_passing_function_in_the_module_reply_is_ignored(tmp_path: Path) -> None:
    model = _ModuleModel({"frequency": [GOOD["frequency"]], "quality": [BAD["quality"]], "decay": [GOOD["decay"]]},
                         module_replies=[_module(frequency=BAD["frequency"])])
    outcome = await _filler(tmp_path, model).run_all(_specs())
    text = (tmp_path / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert outcome.status == "done" and "return math.sqrt(k / m)" in text and "math.sqrt(m / k)" not in text


# --- the second review ---------------------------------------------------------------------------------------------------------


def test_a_file_over_the_snapshot_size_limit_that_the_agent_breaks_is_still_put_back(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("core.agent_coder._KEEP_BYTES", 10)  # every file is "too large" for the snapshot to hold

    def edit(code: Path) -> None:
        (code / "experiment.py").write_text("def heat(x:\n    return 1\n", encoding="utf-8")
    eng, agent = _agent_with([_first, edit], [], tmp_path, monkeypatch)
    patch = _reflect(eng)
    assert _text(eng) == CODE and "exec_give_up_reason" in patch


def test_an_out_of_scope_edit_to_a_file_over_the_snapshot_limit_is_put_back(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("core.agent_coder._KEEP_BYTES", 10)

    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("x / 0", "x / 1").replace("x * 2", "x * 5"), encoding="utf-8")
        (code / "utils.py").write_text("def tidy(x):\n    return None\n", encoding="utf-8")
    eng, _agent = _agent_with([_first, edit], [], tmp_path, monkeypatch)
    _reflect(eng)
    assert "x * 2" in _text(eng) and _text(eng, "utils.py") == UTILS and "x / 1" in _text(eng)


ATTRS = (
    "class Model:\n    SCALE = 2\n    NAME = 'm'\n\n    def run(self):\n        return 1 / 0\n\n\n"
    "def main():\n    return Model().run()\n"
)


def test_a_class_attribute_changed_outside_the_failing_method_is_put_back(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    def edit(code: Path) -> None:
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(text.replace("1 / 0", "1 / 1").replace("SCALE = 2", "SCALE = 9"), encoding="utf-8")
    agent = _Agent([_write_code(ATTRS), edit])
    eng = _engine(tmp_path, agent=agent)
    _first_write(eng, agent)
    monkeypatch.setattr("core.agent_build._frozen.load", lambda _root: None)
    path = eng.quest_root / "code" / "experiment.py"
    failed = {"returncode": 1, "stdout_tail": "", "stderr_tail": f'File "{path}", line 6, in run\nZeroDivisionError'}
    asyncio.run(eng._reflect_by_agent({"exec_reflect_history": []}, 0, 1, "1", "", failed, {}))
    text = path.read_text(encoding="utf-8")
    assert "1 / 1" in text and "SCALE = 2" in text and "SCALE = 9" not in text


def test_the_scope_follows_bare_calls_and_self_calls_but_not_other_attribute_calls(tmp_path: Path) -> None:
    src = (
        "def helper(x):\n    return x\n\n\n"
        "class A:\n    def step(self):\n        return 1\n\n    def run(self):\n        helper(1)\n"
        "        self.step()\n        other.step()\n        return 1 / 0\n\n\n"
        "class B:\n    def step(self):\n        return 2\n\n    def helper(self):\n        return 3\n"
    )
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "experiment.py").write_text(src, encoding="utf-8")
    path = tmp_path / "code" / "experiment.py"
    scope = agent_scope.scope_from_traceback(f'File "{path}", line 13, in run', tmp_path)
    assert scope is not None and scope.files == {"experiment.py": {"A.run", "A.step", "helper"}}, (
        "B.step is another class's method and B.helper is a method, not the module function the bare call means")
