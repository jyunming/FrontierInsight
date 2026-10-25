"""A repair of one function may also change the functions it reaches by calls (core/finding_repair.py): the code that
computes a value a function only passes on is often in a helper it calls. Neutral code (a mass on a damped spring)."""
from __future__ import annotations

import ast
from pathlib import Path

from core import finding_repair as fr

SOURCE = '''import math


def report(m, k, c):
    return {"ratio": ratio(m, k, c), "label": "spring"}


def ratio(m, k, c):
    return crit(m, k) * 0 + c


def crit(m, k):
    return 2.0 * math.sqrt(m * k)


def unrelated(x):
    return x
'''

FIXED_RATIO = "def ratio(m, k, c):\n    return c / crit(m, k)\n"
PLAIN_REPORT = 'def report(m, k, c):\n    return {"ratio": ratio(m, k, c), "label": "spring"}\n'


def _block(text: str) -> str:
    return f"```python\n{text}```"


def test_the_functions_a_function_reaches_by_calls_are_its_scope() -> None:
    assert fr.reached_functions(SOURCE, "report") == ["ratio", "crit"]
    assert fr.reached_functions(SOURCE, "crit") == []
    assert fr.reached_functions(SOURCE, "missing") == []


def test_a_reply_that_rewrites_the_function_and_a_callee_it_reaches_keeps_both() -> None:
    reply = _block(PLAIN_REPORT + "\n\n" + FIXED_RATIO)
    scope = {"report", *fr.reached_functions(SOURCE, "report")}
    skipped: list[str] = []
    new, names, why = fr.splice(SOURCE, reply, scope, skipped)
    assert why == "" and sorted(names) == ["ratio", "report"] and skipped == []
    assert "return c / crit(m, k)" in new
    # the old scope (the function alone) dropped the callee: the fault that made the repair fail again
    old_skipped: list[str] = []
    old_new, _names, _why = fr.splice(SOURCE, reply, {"report"}, old_skipped)
    assert old_skipped == ["ratio"] and "c / crit" not in old_new


def test_a_reply_that_defines_only_a_reached_callee_is_applied_and_the_function_is_kept() -> None:
    scope = {"report", *fr.reached_functions(SOURCE, "report")}
    new, names, why = fr.splice(SOURCE, _block(FIXED_RATIO), scope, [])
    assert why == "" and names == ["ratio"]
    assert "def report(m, k, c):" in new and "return c / crit(m, k)" in new


def test_a_definition_nothing_in_the_repair_reaches_is_still_dropped() -> None:
    reply = _block(PLAIN_REPORT + "\n\ndef unrelated(x):\n    return x + 1\n\n" + FIXED_RATIO)
    scope = {"report", *fr.reached_functions(SOURCE, "report")}
    skipped: list[str] = []
    new, names, _why = fr.splice(SOURCE, reply, scope, skipped)
    assert skipped == ["unrelated"] and "unrelated" not in names
    assert "return x + 1" not in new and "c / crit" in new


def test_a_callee_that_gains_a_parameter_its_other_callers_do_not_pass_is_refused() -> None:
    callers = {"mod.py": {"report", "other"}}
    text = SOURCE + "\n\ndef other(m, k, c):\n    return ratio(m, k, c)\n"
    new_ratio = "def ratio(m, k, c, tol=0.1):\n    return c / crit(m, k) + tol\n"
    new_text = text.replace("def ratio(m, k, c):\n    return crit(m, k) * 0 + c\n", new_ratio)
    why = fr.unpassed_params(fr.function_source(text, "ratio"), fr.function_source(new_text, "ratio"), "ratio",
                             {"mod.py": new_text}, callers)
    assert "tol" in why
    assert ast.parse(new_text)


def test_the_request_shows_the_reached_functions_and_says_which_may_change() -> None:
    import inspect

    src = inspect.getsource(fr._attempt)
    assert "reached_functions(old, scope)" in src and "You may change" in src
