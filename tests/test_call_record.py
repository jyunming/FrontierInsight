"""What FI says about a function with no kept inputs: it separates "called, inputs not kept", "not called" (only where
the calls were observed) and "not observed" (core/second_impl.py::capture_inputs, _inputs_problem, dependence)."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from core import function_steps as fs
from core import second_impl as si

MODEL = '''def plain(level):
    return 2.0 * level


def takes_object(thing):
    return 1.0


def never_used(level):
    return 0.0
'''

SIMULATE = '''from pkg import model


def run_cell(cell):
    if cell.get("crash"):
        raise RuntimeError("this setting does not run")
    model.takes_object(object())
    return {"width": model.plain(cell["level"])}
'''

NAMES = ["plain", "takes_object", "never_used"]


def _quest(tmp_path: Path) -> Path:
    (tmp_path / "code" / "pkg").mkdir(parents=True)
    (tmp_path / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "code" / "pkg" / "model.py").write_text(MODEL, encoding="utf-8")
    (tmp_path / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    return tmp_path


def _run(root: Path):
    async def run(argv: list[str]) -> Any:
        done = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=120)
        return done.returncode, done.stdout, done.stderr
    return run


def _second(root: Path, captured: dict[str, list[dict[str, Any]]]) -> si.SecondImpl:
    async def nothing(*a: Any, **k: Any) -> Any:
        raise AssertionError("not used")

    return si.SecondImpl(quest_root=root, package="pkg", specs=[], protocol={"model": {"equations": []}}, prompts={},
                         chat=nothing, run=nothing, repair_main=nothing, still_passes=nothing, captured=captured)


def _problem(root: Path, captured: dict[str, list[dict[str, Any]]], name: str) -> str:
    spec = fs.Spec(name=name, signature=f"def {name}(thing)", purpose="p", equation="E9")
    return _second(root, captured)._inputs_problem(spec)


def test_a_function_that_ran_but_whose_inputs_were_not_kept_is_not_called_not_called(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    got = asyncio.run(si.capture_inputs(_run(root), root, "pkg", NAMES, [{"level": 1.0}, {"level": 2.0}], deterministic=True))
    assert "takes_object" not in got, "an object that is not plain data is not kept"
    record = si.read_captured(root)
    assert record["calls"]["takes_object"] == {"0": 1, "1": 1} and record["observed"] == [0, 1]
    assert _problem(root, got, "takes_object") == "called at 2 settings; FI could not keep its inputs"


def test_a_function_never_called_at_observed_settings_is_still_not_called(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    got = asyncio.run(si.capture_inputs(_run(root), root, "pkg", NAMES, [{"level": 1.0}], deterministic=True))
    assert "did not call it" in _problem(root, got, "never_used")
    out = asyncio.run(si.dependence(_run(root), root, "pkg", NAMES, {"width"}, deterministic=True))
    assert out["never_used"]["status"] == si.NOT_CALLED and out["never_used"]["settings_run"] == 1


def test_a_setting_that_did_not_run_is_not_observed_and_never_counts_as_not_called(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    got = asyncio.run(si.capture_inputs(_run(root), root, "pkg", NAMES, [{"level": 1.0, "crash": True}], deterministic=True))
    record = si.read_captured(root)
    assert record["observed"] == []
    why = _problem(root, got, "never_used")
    assert "could not observe" in why and "did not call it" not in why
    out = asyncio.run(si.dependence(_run(root), root, "pkg", NAMES, {"width"}, deterministic=True))
    assert all(e["status"] != si.NOT_CALLED for e in out.values())
    assert si.dependence_sentences(out) == []


def test_only_the_settings_that_ran_count_in_the_not_called_sentence(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    asyncio.run(si.capture_inputs(_run(root), root, "pkg", NAMES, [{"level": 1.0, "crash": True}, {"level": 2.0}],
                                  deterministic=True))
    assert si.read_captured(root)["observed"] == [1]
    out = asyncio.run(si.dependence(_run(root), root, "pkg", NAMES, {"width"}, deterministic=True))
    assert out["never_used"] == {"status": si.NOT_CALLED, "settings_run": 1}


def test_a_record_made_before_observed_settings_were_kept_still_reads(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    asyncio.run(si.capture_inputs(_run(root), root, "pkg", NAMES, [{"level": 1.0}], deterministic=True))
    path = root / si.CAPTURED
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("observed"), data.pop("calls")
    path.write_text(json.dumps(data), encoding="utf-8")
    out = asyncio.run(si.dependence(_run(root), root, "pkg", NAMES, {"width"}, deterministic=True))
    assert out["never_used"]["status"] == si.NOT_CALLED
    assert _problem(root, {}, "takes_object") == "called at 1 settings; FI could not keep its inputs"
