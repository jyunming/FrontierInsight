"""The repair of a static fault reads its reply the way the repair of a crashed script does: ``functions`` and ``code``."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from core import static_checks as sc
from core import undefined_names as un
from core.engine import Engine

OLD = "def simulate(n):\n    return n * SCALE\n\n\nif __name__ == '__main__':\n    print(simulate(3))\n"


class _Prompt:
    def substitute(self, **kw: object) -> str:
        return "repair"


class _Host(sc.StaticChecksMixin):
    _script_from_reply = Engine._script_from_reply

    def __init__(self, root: Path, reply: str) -> None:
        self.quest_root = root
        self.fi_dir = root / ".fi"
        self._prompts = {"execute_reflect": _Prompt()}
        self._log = logging.getLogger("test_static_repair_reply")
        self._reply = reply

    def _topic_block(self) -> str:
        return ""

    def _code_design_block(self, state: object) -> str:
        return ""

    async def _chat(self, prompt: str, node: str = "") -> str:
        return self._reply

    def _package_snapshot(self) -> dict:
        return {}

    async def _simulation_kept_or_back(self, *a: object, **k: object) -> bool:
        return False

    async def _keep_unrun_try(self, *a: object, **k: object) -> None:
        return None


def _run(tmp_path: Path, reply: dict | str) -> tuple[bool, str]:
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "sim.py").write_text(OLD, encoding="utf-8")
    host = _Host(tmp_path, reply if isinstance(reply, str) else json.dumps(reply))
    found = un.problems(host._static_files())
    assert len(found) == 1
    ok = asyncio.run(host._repair_static({}, un, found, None, {}))
    return ok, (tmp_path / "code" / "sim.py").read_text(encoding="utf-8")


def test_a_reply_with_the_fix_in_functions_is_put_into_the_script(tmp_path: Path) -> None:
    fixed = "def simulate(n):\n    SCALE = 2.0\n    return n * SCALE\n"
    ok, text = _run(tmp_path, {"functions": fixed, "code": ""})
    assert ok and "SCALE = 2.0" in text and "__main__" in text
    assert un.problems({"sim.py": text}) == []


def test_a_reply_with_the_fix_in_code_is_still_taken(tmp_path: Path) -> None:
    ok, text = _run(tmp_path, {"functions": "", "code": "SCALE = 2.0\n" + OLD})
    assert ok and text.startswith("SCALE = 2.0")


def test_a_reply_that_fixes_nothing_is_refused(tmp_path: Path) -> None:
    same = "def simulate(n):\n    return n * SCALE\n"
    ok, text = _run(tmp_path, {"functions": same, "code": ""})
    assert not ok and text == OLD


def test_an_empty_reply_is_refused(tmp_path: Path) -> None:
    ok, text = _run(tmp_path, {"functions": "", "code": ""})
    assert not ok and text == OLD


def test_the_typed_in_results_repair_reads_a_reply_that_gives_its_fix_as_functions(tmp_path: Path, monkeypatch) -> None:
    from core import run_checks
    from core import typed_results as _typed

    old = "def simulate(n):\n    return 42.0\n\n\nif __name__ == '__main__':\n    print(simulate(3))\n"
    (tmp_path / "code").mkdir()
    path = tmp_path / "code" / "sim.py"
    path.write_text(old, encoding="utf-8")

    class _Run(run_checks.RunChecksMixin, _Host):
        def _typed_result_exempt(self, state: object) -> set:
            return set()

    host = _Run(tmp_path, json.dumps({"functions": "def simulate(n):\n    return n * 2.0\n", "code": ""}))
    found = [_typed.TypedResult("sim.py", "simulate", "yield", 2, "42.0"), _typed.TypedResult("sim.py", "simulate", "rate", 2, "42.0")]
    monkeypatch.setattr(_typed, "typed_results", lambda code, name, exempt=None: [] if "n * 2.0" in code else list(found))
    ok = asyncio.run(host._repair_typed_results({}, path, found))  # type: ignore[arg-type]
    text = path.read_text(encoding="utf-8")
    assert ok and "n * 2.0" in text and "__main__" in text


PROSE = "I haven't run this patch; the `SCALE` line is a guess.\n\n"


def test_a_reply_with_a_sentence_before_its_object_is_read(tmp_path: Path) -> None:
    fixed = "def simulate(n):\n    SCALE = 2.0\n    return n * SCALE\n"
    ok, text = _run(tmp_path, PROSE + json.dumps({"functions": fixed, "code": "", "patch_summary": "defined SCALE"}))
    assert ok and "SCALE = 2.0" in text and "__main__" in text


def test_the_repair_reader_takes_the_object_after_the_sentence_and_in_a_fence() -> None:
    from core.engine import _repair_reply_json

    obj = {"functions": "def f():\n    return 1\n", "code": ""}
    assert _repair_reply_json(PROSE + json.dumps(obj), node="t") == obj
    assert _repair_reply_json(PROSE + "```json\n" + json.dumps(obj) + "\n```\n", node="t") == obj
    assert _repair_reply_json(json.dumps(obj), node="t") == obj


def test_the_repair_reader_gives_nothing_for_prose_or_a_whole_script() -> None:
    from core.engine import _repair_reply_json

    assert _repair_reply_json("I cannot run code here, so I have no fix. {braces} in prose.", node="t") == {}
    assert _repair_reply_json("Here:\n```python\nCFG = {'a': 1}\nprint(CFG)\n```\nDEPS: numpy\n", node="t") == {}


def test_a_reply_of_prose_only_is_still_refused(tmp_path: Path) -> None:
    ok, text = _run(tmp_path, "I could not run anything in this environment, so I have no fix to give.")
    assert not ok and text == OLD
