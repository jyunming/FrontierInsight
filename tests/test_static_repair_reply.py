"""The repair of a static fault reads its reply the way the repair of a crashed script does: ``functions`` and ``code``."""

from __future__ import annotations

from tests._engine_text import engine_text
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


def test_the_repair_reader_takes_the_last_object_that_carries_a_fix() -> None:
    from core.engine import _repair_reply_json

    fix = {"functions": "def f():\n    return 1\n", "code": "", "patch_summary": "x"}
    template = {"functions": "", "code": "", "patch_summary": ""}
    gave_up = {"give_up_reason": "none"}
    assert _repair_reply_json(PROSE + json.dumps(fix) + "\nFormat: " + json.dumps(template), node="t") == fix
    assert _repair_reply_json(PROSE + json.dumps(fix) + "\n" + json.dumps(gave_up), node="t") == fix
    assert _repair_reply_json(PROSE + json.dumps(gave_up), node="t") == gave_up


def test_a_dict_inside_a_script_block_is_not_the_reply() -> None:
    from core.engine import _repair_reply_json

    reply = 'Here:\n```python\nCFG = {"code": "A1", "n": 3}\nprint(CFG)\n```\nDEPS: numpy\n'
    assert _repair_reply_json(reply, node="t") == {}
    fix = {"functions": "def f():\n    return 1\n", "code": ""}
    assert _repair_reply_json(reply + "```json\n" + json.dumps(fix) + "\n```\n", node="t") == fix


def test_the_crashed_script_repair_reads_a_sentence_before_its_object() -> None:
    import inspect

    from core import engine

    assert '_repair_reply_json(text, node="execute_reflect")' in engine_text()


def test_an_object_whose_code_is_not_a_string_is_not_a_reply() -> None:
    from core.engine import _repair_reply_json

    assert _repair_reply_json('The config is {"code": 1}. Nothing else.', node="t") == {}
    assert _repair_reply_json('{"code": 1, "functions": ""}', node="t") == {}
    assert _repair_reply_json('See {"functions": ["a"]}.', node="t") == {}


def test_a_give_up_quoted_in_prose_never_beats_the_fence_that_fixes() -> None:
    from core.engine import _repair_reply_json

    fix = {"functions": "def f():\n    return 1\n", "code": ""}
    reply = 'I will not write {"give_up_reason": "cannot"} here.\n```json\n' + json.dumps(fix) + "\n```\n"
    assert _repair_reply_json(reply, node="t") == fix


def test_the_prompts_own_format_echoed_back_is_not_a_fix() -> None:
    import re

    from core.engine import PROMPTS_DIR, _repair_reply_json

    text = (PROMPTS_DIR / "execute_reflect.md").read_text(encoding="utf-8")
    fn = re.search(r'^\s*"functions":\s*"(<.*>)",?\s*$', text, re.MULTILINE).group(1)
    code = re.search(r'^\s*"code":\s*"(<.*>)",?\s*$', text, re.MULTILINE).group(1)
    echo = json.dumps({"functions": fn, "code": code, "deps": [], "patch_summary": "<one sentence>", "give_up_reason": None})
    assert _repair_reply_json("The format is:\n" + echo, node="t") == {}
    real = {"functions": "def f():\n    return 1\n", "code": code}
    assert _repair_reply_json("Done.\n" + json.dumps(real), node="t")["functions"].startswith("def f")


FN = "def f():\n    return 1\n"


def test_a_give_up_that_is_not_text_is_absent() -> None:
    from core.engine import _repair_reply_json

    for reply in ('{"give_up_reason": ["a", "b"], "functions": ' + json.dumps(FN) + "}",
                  'Sorry. {"give_up_reason": ["a"], "functions": ' + json.dumps(FN) + "}"):
        got = _repair_reply_json(reply, node="t")
        assert got["give_up_reason"] in ("a; b", "a") and got["functions"].startswith("def f")
    assert _repair_reply_json('{"give_up_reason": ["cannot"], "patch_summary": {"a": 1}}', node="t") == {"give_up_reason": "cannot"}
    assert _repair_reply_json('I give up {"give_up_reason": ["cannot", " because"]}', node="t") == {
        "give_up_reason": "cannot; because"}
    assert _repair_reply_json('I give up {"give_up_reason": ["cannot", 3]}', node="t") == {}
    assert _repair_reply_json('{"give_up_reason": [{"a": 1}]}', node="t").get("give_up_reason") is None
    assert _repair_reply_json('{"give_up_reason": 5}', node="t").get("give_up_reason") is None


def test_the_format_object_itself_is_blanked_too() -> None:
    import re

    from core.engine import PROMPTS_DIR, _repair_reply_json

    text = (PROMPTS_DIR / "execute_reflect.md").read_text(encoding="utf-8")
    fn = re.search(r'^\s*"functions":\s*"(<.*>)",?\s*$', text, re.MULTILINE).group(1)
    code = re.search(r'^\s*"code":\s*"(<.*>)",?\s*$', text, re.MULTILINE).group(1)
    fence = "`" * 3
    for echo in (json.dumps({"functions": fn, "code": code, "patch_summary": "x"}),
                 fence + "json\n" + json.dumps({"functions": fn, "code": code}) + "\n" + fence):
        got = _repair_reply_json(echo, node="t")
        assert got.get("functions") == "" and got.get("code") == ""
