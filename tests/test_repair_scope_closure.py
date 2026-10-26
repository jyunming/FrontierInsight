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


# --- what a reached function must do: its return value flows into what the function produces -----------------------------

CHECKED = '''def compute(x):
    return x * 2


def validate(result):
    return None


def draw(result):
    return None


def helper_for_label(x):
    return str(x)


def run(x):
    value = compute(x)
    result = {"value": value}
    validate(result)
    draw(result)
    unused = helper_for_label(x)
    return result
'''


def test_a_function_called_only_for_its_effect_or_whose_result_goes_nowhere_is_not_in_scope() -> None:
    assert fr.reached_functions(CHECKED, "run") == ["compute"]
    only_check = fr.splice(CHECKED, _block("def validate(result):\n    return True\n"), {"run", "compute"}, [])
    assert only_check[0] is None, "a reply of a check alone defines neither the function nor a function it takes values from"


def test_a_value_stored_into_a_result_or_printed_counts_as_reaching_it() -> None:
    src = ("def a(x):\n    return x\n\n\ndef b(x):\n    return x\n\n\n"
           "def run(x, out):\n    out['k'] = a(x)\n    print(b(x))\n")
    assert fr.reached_functions(src, "run") == ["a", "b"]


def test_a_name_the_function_binds_itself_is_not_the_module_function() -> None:
    src = ("def check(v):\n    return v\n\n\ndef report(check):\n    return check(1)\n\n\n"
           "def other(x):\n    def check(v):\n        return v\n    return check(x)\n\n\n"
           "def real(x):\n    return check(x)\n")
    assert fr.reached_functions(src, "report") == []
    assert fr.reached_functions(src, "other") == []
    assert fr.reached_functions(src, "real") == ["check"]


def test_an_inner_function_that_is_never_called_does_not_carry_a_callee_in() -> None:
    src = "def g(x):\n    return x\n\n\ndef run(x):\n    def unused():\n        return g(x)\n    return x\n"
    assert fr.reached_functions(src, "run") == []


def test_async_functions_are_functions_of_the_file() -> None:
    src = "async def fetch(x):\n    return x\n\n\nasync def run(x):\n    return await fetch(x)\n\n\ndef sync(x):\n    return x\n"
    assert set(fr._top_functions(ast.parse(src))) == {"fetch", "run", "sync"}
    assert "async def" not in fr.signatures(src, "run") and "def fetch(x)" in fr.signatures(src, "run")
    new, names, why = fr.splice(src, _block("async def run(x):\n    return await fetch(x) + 1\n"), {"run"}, [])
    assert why == "" and names == ["run"] and "await fetch(x) + 1" in new


def test_the_parameter_check_reads_the_code_the_repair_leaves_including_its_new_callers(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def ratio(m):\n    return m\n\n\ndef run(m):\n    return ratio(m)\n", encoding="utf-8")
    staged = {"mod.py": "def ratio(m, tol=0.1):\n    return m + tol\n\n\ndef run(m):\n    return ratio(m)\n\n\n"
                        "def extra(m):\n    return ratio(m)\n"}
    assert fr.direct_callers(tmp_path, "mod.py", "ratio") == {"mod.py": {"run"}}
    assert fr.direct_callers(tmp_path, "mod.py", "ratio", staged) == {"mod.py": {"run", "extra"}}


def test_a_store_into_a_local_container_that_reaches_no_output_is_not_an_output() -> None:
    src = ("def validate(r):\n    return True\n\n\n"
           "def run(r):\n    flags = {}\n    flags['ok'] = validate(r)\n    if not flags['ok']:\n        raise ValueError\n    return r\n")
    assert fr.reached_functions(src, "run") == []
    kept = src.replace("    return r\n", "    return flags\n")
    assert fr.reached_functions(kept, "run") == ["validate"], "the container is returned, so its stores are outputs"
    param = src.replace("def run(r):\n    flags = {}\n", "def run(r, flags):\n")
    assert fr.reached_functions(param, "run") == ["validate"], "a store into a parameter reaches the caller"


def test_a_name_bound_by_a_match_pattern_is_the_functions_own() -> None:
    src = "def check(v):\n    return v\n\n\ndef run(obj):\n    match obj:\n        case check:\n            return check(1)\n"
    assert fr.reached_functions(src, "run") == []
    star = "def check(v):\n    return v\n\n\ndef run(obj):\n    match obj:\n        case [*check]:\n            return check(1)\n"
    assert fr.reached_functions(star, "run") == []
    rest = "def check(v):\n    return v\n\n\ndef run(obj):\n    match obj:\n        case {'a': 1, **check}:\n            return check(1)\n"
    assert fr.reached_functions(rest, "run") == []


def test_a_comprehension_target_shadows_only_inside_the_comprehension() -> None:
    src = ("def check(v):\n    return v\n\n\n"
           "def run(xs):\n    values = [check(x) for check in xs]\n    return check(1)\n")
    assert fr.reached_functions(src, "run") == ["check"]
    inside = "def check(v):\n    return v\n\n\ndef run(xs):\n    return [check(x) for check in xs]\n"
    assert fr.reached_functions(inside, "run") == []
