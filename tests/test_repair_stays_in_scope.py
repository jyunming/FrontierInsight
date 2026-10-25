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
