"""A repair of a script is asked for the functions to change; FI splices them in and keeps every other byte."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import script_reply as sr
from core import trial_runner as tr

OLD = '''import math

LIMIT = 3


def helper(x):
    return x + 1


def run_cell(cell, seed):
    return {"value": helper(cell["x"]) * LIMIT}


def run_trial(cell, trial_id, seed):
    return {"value": helper(cell["x"])}
'''


def test_a_reply_of_one_function_is_spliced_and_the_rest_is_byte_identical() -> None:
    new_fn = "def run_trial(cell, trial_id, seed):\n    return {'value': helper(cell['x']) * 2}\n"
    made = sr.resolve(OLD, "", new_fn)
    assert made.kind == "functions" and made.replaced == ["run_trial"]
    head = OLD[:OLD.index("def run_trial")]
    assert made.code.startswith(head), "everything before the function is untouched"
    assert made.code[len(head):].strip() == new_fn.strip()
    assert tr.repair_breaks_contract(OLD, made.code, "simulate.py") == ""


def test_a_new_helper_in_the_reply_is_added_and_a_stand_in_is_refused() -> None:
    made = sr.resolve(OLD, "", "def extra(a):\n    return a\n\ndef helper(x):\n    return x + 2\n")
    assert made.kind == "functions" and "extra" in made.replaced and "def extra" in made.code
    assert "LIMIT = 3" in made.code and "def run_cell" in made.code
    assert sr.resolve(OLD, "", "PLACEHOLDER").why, "no function: nothing is made"
    whole = sr.resolve(OLD, "PLACEHOLDER", "")
    assert whole.kind == "whole" and tr.repair_breaks_contract(OLD, whole.code, "simulate.py")


def test_a_whole_script_is_still_taken_whole_and_the_contract_judges_it() -> None:
    made = sr.resolve(OLD, OLD.replace("LIMIT = 3", "LIMIT = 4"), "")
    assert made.kind == "whole" and "LIMIT = 4" in made.code


def test_a_whole_reply_that_drops_a_helper_it_still_calls_is_refused() -> None:
    # The old failure: a reply cut short kept the entry points and lost the helper they call.
    cut = "def run_cell(cell, seed):\n    return {'value': helper(cell['x'])}\n\ndef run_trial(cell, trial_id, seed):\n    return {}\n"
    why = tr.repair_breaks_contract(OLD, cut, "simulate.py")
    assert "helper" in why
    assert tr.repair_breaks_contract(OLD, cut.replace("helper(cell['x'])", "cell['x']"), "simulate.py") == ""


@pytest.mark.asyncio
async def test_an_oracle_repair_with_one_function_changes_only_that_function(tmp_path: Path, monkeypatch) -> None:
    from tests.test_code_layout import ORACLE, _engine_with_tool

    engine, code, state = _engine_with_tool(tmp_path)
    engine._trial_mode = True
    before = (code / "simulate.py").read_text(encoding="utf-8")
    import re

    name = re.search(r"^def (run_cell|run_trial)\(", before, re.M).group(1)
    node = __import__("ast").parse(before)
    fn = next(n for n in node.body if getattr(n, "name", "") == name)
    src = "\n".join(before.splitlines()[fn.lineno - 1:fn.end_lineno]) + "\n"
    changed = src.rstrip("\n") + "\n    _repaired = 1\n"

    async def answer(prompt, **kw):  # noqa: ANN001
        return json.dumps({"functions": changed, "patch_summary": "one function"})

    monkeypatch.setattr(engine, "_chat", answer)
    new, failed, outcome = await engine._repair_script_for_oracle(state, code / "simulate.py", [ORACLE], ["off"], passing=[])
    assert outcome == "applied" and not failed
    assert new == before.replace(src, changed)
    assert (code / "simulate.py").read_text(encoding="utf-8") == before.replace(src, changed)


def test_a_reply_with_functions_and_a_script_keeps_the_imports_and_constants_of_both() -> None:
    old = "import os\n\nX = 1\n\n\ndef a(p):\n    return 1\n\n\ndef b():\n    return 2\n"
    made = sr.resolve(old, "import math\nY = 5\n\ndef b():\n    return math.pi\n", "def b():\n    return math.pi\n")
    assert made.kind == "functions" and "import math" in made.code and "Y = 5" in made.code
    assert "def a(p):\n    return 1" in made.code and made.code.count("def b") == 1
    clash = sr.resolve(old, "def b():\n    return 1\n", "def b():\n    return 3\n")
    assert clash.code == "" and "`b`" in clash.why and "differ" in clash.why


def test_a_dropped_helper_is_found_though_another_function_defines_a_name_like_it() -> None:
    old = "def helper(x):\n    return x\n\n\ndef run_cell(cell, seed):\n    return helper(cell)\n"
    # `helper` is a nested def and a parameter in OTHER functions; the one the call reaches is gone
    new = ("def other(helper):\n    return helper\n\n\ndef third():\n    def helper(y):\n        return y\n    return helper\n\n\n"
           "def run_cell(cell, seed):\n    return helper(cell)\n")
    assert "helper" in tr.repair_breaks_contract(old, new, "simulate.py")
    own = "def run_cell(cell, seed):\n    def helper(y):\n        return y\n    return helper(cell)\n"
    assert tr.repair_breaks_contract(old, own, "simulate.py") == "", "a helper bound in its own scope is fine"


def test_a_reply_with_functions_and_a_script_keeps_its_classes_constants_and_main_block() -> None:
    old = "import os\n\nX = 1\n\n\ndef a(p):\n    return 1\n\n\ndef b():\n    return 2\n\n\nif __name__ == '__main__':\n    a(1)\n"
    code = ("import math\nX = 2\n\n\n@register\nclass Box:\n    size = 3\n\n\ndef b():\n    return math.pi\n\n\n"
            "if __name__ == '__main__':\n    a(2)\n")
    made = sr.resolve(old, code, "def b():\n    return math.pi\n")
    assert made.kind == "functions"
    assert "@register\nclass Box:\n    size = 3" in made.code, "a class with its decorator"
    assert "X = 2" in made.code and "X = 1" not in made.code, "an updated constant replaces the old one"
    assert "a(2)" in made.code and "a(1)" not in made.code and made.code.count("__main__") == 1
    compile(made.code, "x", "exec")


def test_a_dropped_helper_is_judged_by_pythons_own_scopes() -> None:
    old = "def helper(x):\n    return x\n\n\ndef run_cell(cell, seed):\n    return helper(cell)\n"
    imported = "from helper_mod import helper\n\n\ndef run_cell(cell, seed):\n    return helper(cell)\n"
    assert tr.repair_breaks_contract(old, imported, "simulate.py") == "", "a name imported at the top is bound"
    shadowed = "def run_cell(cell, seed):\n    xs = [helper for helper in cell]\n    return helper(0)\n"
    assert "helper" in tr.repair_breaks_contract(old, shadowed, "simulate.py"), "a comprehension's variable binds only inside it"
    in_class = "class K:\n    value = helper(1)\n\n\ndef run_cell(cell, seed):\n    return 0\n"
    assert "helper" in tr.repair_breaks_contract(old, in_class, "simulate.py")
