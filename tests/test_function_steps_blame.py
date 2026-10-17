"""A failed import of the model's module is read to its end and blamed where the error is (core/function_steps.py): the
function being written is not charged for a fault in another function, in a statement outside any function, or in the
environment. Neutral models only (a damped spring)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import pytest

from core import function_steps as fs
from tests.test_function_steps import GOOD, PROMPTS, PROTOCOL, _Model, _filler, _run, _specs, _which


def _long_traceback(depth: int = 60) -> str:
    frames = "".join(f'  File "C:\\py\\lib\\mod{i}.py", line {i}, in step{i}\r\n    call_{i}()\r\n' for i in range(depth))
    return "Traceback (most recent call last):\r\n" + frames + "ValueError: the final message\r\n"


def test_a_shortened_traceback_keeps_its_end_not_its_start() -> None:
    text = _long_traceback()
    short = fs.traceback_end(text, 400)
    assert len(short) <= 410 and short.endswith("ValueError: the final message")
    assert "mod59" in short and "mod0.py" not in short and "Traceback (most recent call last)" not in short
    assert short.startswith("...\n") and "\r" not in short
    assert fs.traceback_end("short\r\ntext", 400) == "short\ntext"
    assert fs.exception_line(text) == "ValueError: the final message"
    assert fs.exception_line("") == "no message"


def test_a_stub_imports_even_when_its_annotations_and_defaults_name_something_the_module_lacks() -> None:
    spec = fs.Spec(name="g", signature="def g(z: np.ndarray, n=np.float64(1), *, k: int = 3, **kw) -> np.ndarray:",
                   purpose="a purpose", equation="E1")
    module = fs.initial_module([spec])
    compiled = compile(module, "model.py", "exec")
    scope: dict[str, Any] = {}
    exec(compiled, scope)  # noqa: S102 -- importing the module is the point
    assert scope["g"].__name__ == "g" and fs.is_stub(module, "g")
    assert [a.arg for a in ast.parse(module).body[2].args.args] == ["z", "n"]
    with pytest.raises(NotImplementedError):
        scope["g"](1)


def _model_file(root: Path) -> Path:
    return root / "code" / "spring" / "model.py"


def _write_module(root: Path, text: str) -> None:
    (root / "code" / "spring").mkdir(parents=True, exist_ok=True)
    (root / "code" / "spring" / "__init__.py").write_text("from . import model  # noqa: F401\n", encoding="utf-8")
    _model_file(root).write_text(text, encoding="utf-8")


def _record(specs: list[fs.Spec]) -> dict[str, Any]:
    return {"calls_total": 0, "package": "spring", "calls": 0, "status": "running",
            "functions": {s.name: {"equation": s.equation, "status": fs.PENDING, "attempts": 0} for s in specs}}


@pytest.mark.asyncio
async def test_an_error_in_another_function_is_blamed_on_that_function_and_the_directive_names_the_place(tmp_path: Path) -> None:
    specs = {s.name: s for s in _specs()}
    other_broken = (GOOD["frequency"] + "\n\n# E2\ndef quality(m, w, c, extra=undefined_name):\n    return m * w / c\n")
    _write_module(tmp_path, other_broken)
    filler = _filler(tmp_path, _Model({}))
    known = fs._dumps(other_broken)
    problem, who, trace = await filler._check(specs["frequency"], _record(list(specs.values())), known)
    assert who == "quality", "the NameError is in quality's header, not in frequency"
    assert "frequency" in problem and "`quality`" in problem and "NameError" in problem
    assert trace.endswith("NameError: name 'undefined_name' is not defined")
    # What a function adds in this attempt is its own: the same error, but in a statement this attempt added.
    problem, who, _trace = await filler._check(specs["frequency"], _record(list(specs.values())), set())
    assert who == "frequency" and problem.startswith("importing it failed")


@pytest.mark.asyncio
async def test_a_syntax_error_in_another_function_costs_the_current_function_none_of_its_repairs(tmp_path: Path) -> None:
    specs = _specs()
    broken = GOOD["frequency"] + "\n\n# E2\ndef quality(m, w, c):\n    return m * (w / \n"
    _write_module(tmp_path, broken)
    filler = _filler(tmp_path, _Model({}))
    record = _record(specs)
    frequency = next(s for s in specs if s.name == "frequency")
    problem, who, _trace = await filler._check(frequency, record, set())
    assert who == "quality" and "not valid Python" in problem and "the function `quality`" in problem
    assert await filler._fix_elsewhere("quality", "SyntaxError", specs, record, set()) is True
    assert fs.is_stub(_model_file(tmp_path).read_text(encoding="utf-8"), "quality"), "back to a placeholder"
    assert record["functions"]["quality"]["status"] == fs.PENDING
    assert (await filler._check(frequency, record, set()))[0] == "", "the function being written now passes"
    assert not record.get("calls"), "no request to the model was spent on it"


@pytest.mark.asyncio
async def test_a_fault_in_a_placeholder_is_mended_and_the_function_being_written_keeps_its_repairs(tmp_path: Path) -> None:
    model = _Model({k: [v] for k, v in GOOD.items()})
    corrupted = {"done": False}

    async def run(argv: list[str], root: Path = tmp_path) -> tuple[int, str, str]:
        text = _model_file(root).read_text(encoding="utf-8") if _model_file(root).is_file() else ""
        if not corrupted["done"] and argv[0] == "-c" and "def quality" in text and "def frequency(k, m):\n    return" in text:
            corrupted["done"] = True  # the placeholder of the NEXT function is left with a fault, as if half written
            _model_file(root).write_text(text.replace("def quality(m, w, c)", "def quality(m, w, c, extra=undefined_name)"),
                                         encoding="utf-8")
        return await _run(argv, root)

    filler = fs.FunctionFiller(quest_root=tmp_path, package="spring", protocol=PROTOCOL, summary="a mass on a damped spring",
                               prompts=PROMPTS, chat=model.chat, run=run, log=logging.getLogger("t"))
    outcome = await filler.run_all(_specs())
    assert corrupted["done"]
    assert outcome.status == "done" and fs.load(tmp_path)["functions"]["frequency"]["attempts"] == 1
    assert [_which(p) for p in model.prompts] == ["frequency", "quality", "decay"], "no request was spent on the fault"


@pytest.mark.asyncio
async def test_a_function_written_earlier_that_breaks_the_import_is_repaired_alone_with_a_plain_directive(tmp_path: Path) -> None:
    model = _Model({"frequency": [GOOD["frequency"], GOOD["frequency"]], "quality": [GOOD["quality"]], "decay": [GOOD["decay"]]})
    corrupted = {"done": False}

    async def run(argv: list[str], root: Path = tmp_path) -> tuple[int, str, str]:
        text = _model_file(root).read_text(encoding="utf-8")
        if not corrupted["done"] and argv[0] == "-c" and not fs.is_stub(text, "frequency") and "quality" in argv[1]:
            corrupted["done"] = True  # frequency is written and checked; now it is left with a fault
            _model_file(root).write_text(text.replace("def frequency(k, m):", "def frequency(k, m, extra=undefined_name):"),
                                         encoding="utf-8")
        return await _run(argv, root)

    filler = fs.FunctionFiller(quest_root=tmp_path, package="spring", protocol=PROTOCOL, summary="a mass on a damped spring",
                               prompts=PROMPTS, chat=model.chat, run=run, log=logging.getLogger("t"))
    outcome = await filler.run_all(_specs())
    assert corrupted["done"] and outcome.status == "done"
    directive = next(p for p in model.prompts if "stops inside this function" in p)
    assert _which(directive) == "frequency" and "NameError" in directive and "undefined_name" in directive
    record = fs.load(tmp_path)
    assert record["functions"]["quality"]["attempts"] == 1, "the function being written kept its repairs"
    assert record["elsewhere"] == {"frequency": 1}


@pytest.mark.asyncio
async def test_a_cause_outside_any_function_ends_the_step_at_once_and_says_where(tmp_path: Path) -> None:
    model = _Model({k: [v] for k, v in GOOD.items()})

    async def run(argv: list[str], root: Path = tmp_path) -> tuple[int, str, str]:
        text = _model_file(root).read_text(encoding="utf-8")
        if argv[0] == "-c" and "RAISED = 1 / 0" not in text:
            # A statement outside any function that was there before this attempt, and fails when imported.
            _model_file(root).write_text(text.replace('from __future__ import annotations\n',
                                                       'from __future__ import annotations\n\nRAISED = 1 / 0\n'), encoding="utf-8")
        return await _run(argv, root)

    filler = fs.FunctionFiller(quest_root=tmp_path, package="spring", protocol=PROTOCOL, summary="a mass on a damped spring",
                               prompts=PROMPTS, chat=model.chat, run=run, log=logging.getLogger("t"))
    outcome = await filler.run_all(_specs())
    assert outcome.status == "fell_back" and len(model.prompts) == 1, "one request, and no repairs were spent on it"
    record = fs.load(tmp_path)
    assert "ZeroDivisionError" in record["functions"]["frequency"]["error"], "the end of the traceback is kept"
    assert "not in frequency" in record["reason"] and "not in a function" in record["reason"]


@pytest.mark.asyncio
async def test_a_function_that_is_wrong_itself_still_spends_its_own_repairs_and_keeps_the_end_of_the_error(tmp_path: Path) -> None:
    bad_import = "from math import not_there_at_all\n\n\n# E1\ndef frequency(k, m):\n    return 1.0\n"
    model = _Model({"frequency": [bad_import], "quality": [GOOD["quality"]], "decay": [GOOD["decay"]]})
    outcome = await _filler(tmp_path, model, repairs=2).run_all(_specs())
    assert outcome.status == "fell_back"
    assert len(model.prompts) == 5, "its own fault: one fill and two repairs, and the other two functions are still written"
    error = fs.load(tmp_path)["functions"]["frequency"]["error"]
    assert "ImportError" in error and "not_there_at_all" in error
