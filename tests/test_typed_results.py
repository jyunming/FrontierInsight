"""A reported result is computed, never a number typed into the code (core/typed_results.py)."""

from __future__ import annotations

from core import typed_results as tr


def _keys(source: str, **kw) -> list[str]:
    return [f.key for f in tr.typed_results(source, "simulate.py", **kw)]


CUP = '''
import random

def run_trial(cell, trial, seed):
    rng = random.Random(seed)
    temp = 90.0
    for _ in range(int(cell["steps"])):
        temp += (20.0 - temp) * 0.01 + rng.gauss(0, 0.1)
    return {"final_temp": temp, "lid_on": 1.0}
'''


def test_a_literal_returned_value_is_caught() -> None:
    found = tr.typed_results(CUP, "simulate.py")
    assert [f.key for f in found] == ["lid_on"]
    assert found[0].line == 9 and found[0].function == "run_trial" and "1.0" in found[0].value
    assert "`lid_on`" in found[0].says() and "typed into the code" in found[0].says()


def test_a_name_assigned_only_a_literal_is_caught() -> None:
    src = '''
def run_trial(cell, trial, seed):
    spill = 0.0
    width = 10.0
    t = 5.0
    for i in range(cell["n"]):
        t = t * 0.9 + seed % 7
    return {"spilled": spill, "width": width, "t_end": t}
'''
    assert _keys(src) == ["spilled", "width"]


def test_arithmetic_on_typed_numbers_is_caught_too() -> None:
    src = '''
def run_cell(cell):
    k = 2.0
    return {"stiffness_ratio": k * 3 / 4, "x": cell["x"]}
'''
    assert _keys(src) == ["stiffness_ratio"]


def test_a_computed_value_passes() -> None:
    src = '''
import math

def run_trial(cell, trial, seed):
    x = 1.0
    v = 0.0
    for _ in range(1000):
        a = -4.0 * x - 0.1 * v
        v += a * 0.01
        x += v * 0.01
    return {"x_end": x, "energy": 0.5 * v * v + 2.0 * x * x, "peak": max(abs(x), abs(v))}
'''
    assert _keys(src) == []


def test_a_value_derived_from_the_inputs_passes() -> None:
    src = '''
def run_trial(cell, trial, seed):
    gain = 3.0
    return {"echo": cell["gain"], "scaled": gain * cell["gain"], "trial_no": trial, "seeded": float(seed % 5),
            "same_as_input": float(cell["gain"])}
'''
    assert _keys(src) == []


def test_a_name_set_in_a_loop_or_a_branch_from_a_computation_passes() -> None:
    src = '''
def run_trial(cell, trial, seed):
    hit = 0.0
    count = 0.0
    flag = 0.0
    for i in range(cell["n"]):
        if (seed + i) % 3 == 0:
            hit = 1.0
        count += 1
    if cell["n"] > 5:
        flag = 1.0
    return {"hit": hit, "count": count, "flag": flag}
'''
    assert _keys(src) == [], "a literal set in a branch depends on the branch; a counter is changed by +="


def test_a_name_assigned_twice_passes() -> None:
    src = '''
def run_trial(cell, trial, seed):
    p = 0.0
    p = compute(cell)
    return {"p": p}
'''
    assert _keys(src) == []


def test_a_key_computed_in_another_return_passes() -> None:
    src = '''
def run_trial(cell, trial, seed):
    if cell["n"] < 1:
        return {"loss": 0.0, "ok": 0.0}
    loss = sum(range(cell["n"]))
    return {"loss": loss, "ok": 1.0}
'''
    assert _keys(src) == [], "loss is computed in one return; ok takes two values depending on the path"


def test_a_dictionary_built_up_by_key_is_followed() -> None:
    src = '''
def run_trial(cell, trial, seed):
    out = {}
    out["typed"] = 4.0
    out["computed"] = cell["n"] * 2.0
    total = 0.0
    for i in range(cell["n"]):
        total += i
    out["total"] = total
    return out
'''
    assert _keys(src) == ["typed"]
    branchy = '''
def run_trial(cell, trial, seed):
    out = {"a": 0.0}
    if cell["n"] > 2:
        out["a"] = 0.0
    return out
'''
    assert _keys(branchy) == [], "a write in a branch is not a straight line: not sure, not flagged"


def test_what_the_check_cannot_follow_is_left_alone() -> None:
    for src in (
        'def run_trial(cell, trial, seed):\n    return summarise(cell, seed)\n',
        'def run_trial(cell, trial, seed):\n    out = {"k": 1.0}\n    out.update(extra(cell))\n    return out\n',
        'def run_trial(cell, trial, seed):\n    base = {"z": cell["z"]}\n    return {**base, "k": 1.0}\n',
        'def run_trial(cell, trial, seed):\n    out = {"k": 1.0}\n    fill(out, cell)\n    return out\n',
        'def run_trial(cell, trial, seed):\n    if cell["a"]:\n        return {"k": 1.0}\n    return helper(cell)\n',
    ):
        assert _keys(src) == [], src


def test_the_plans_fixed_settings_are_exempt() -> None:
    src = '''
def run_cell(cell):
    mass = 2.0
    return {"mass": mass, "period": cell["k"] * 2.0, "cap": 5.0}
'''
    assert sorted(_keys(src)) == ["cap", "mass"]
    assert _keys(src, exempt={"mass"}) == ["cap"]


def test_a_module_constant_under_its_own_name_is_an_echoed_setting_under_another_name_it_is_a_result() -> None:
    src = '''
LIMIT = 5.0
SPRING_K = 2.0

def helper():
    return {"k": 1.0}

def run_trial(cell, trial, seed):
    return {"limit": LIMIT, "spring_k": SPRING_K, "ok_rate": LIMIT, "n": cell["n"]}
'''
    assert _keys(src) == ["ok_rate"], "only the entry functions are read; LIMIT as `limit` is a setting echoed back"
    assert _keys(src, exempt={"ok_rate"}) == []
    shadowed = ("LIMIT = 5.0\n\ndef run_trial(cell, trial, seed):\n    LIMIT = cell['x'] * 2\n"
                "    return {'ok_rate': LIMIT}\n")
    assert _keys(shadowed) == [], "a name the function binds itself is its own"


def test_a_counter_the_code_computes_passes_and_a_literal_zero_is_caught() -> None:
    computed = '''
def run_trial(cell, trial, seed):
    n = 0
    for i in range(cell["steps"]):
        if (seed + i) % 5 == 0:
            n += 1
    return {"count": n}
'''
    assert _keys(computed) == []
    assigned_in_loop = '''
def run_trial(cell, trial, seed):
    n = 0
    for i in range(cell["steps"]):
        n = n + 1
    return {"count": n}
'''
    assert _keys(assigned_in_loop) == []
    assert _keys("def run_trial(cell, trial, seed):\n    return {'count': 0}\n") == ["count"]
    never_counted = ("def run_trial(cell, trial, seed):\n    n = 0\n    for i in range(3):\n        pass\n"
                     "    return {'count': n}\n")
    assert _keys(never_counted) == ["count"], "a counter nothing ever increments is a typed-in zero"


def test_a_number_made_by_a_scalar_constructor_is_typed_in_too() -> None:
    src = '''
import numpy as np
from numpy import float32

def run_trial(cell, trial, seed):
    a = np.float64(0.0)
    return {"a": a, "b": float(0), "c": int(3), "d": numpy_like(1), "e": float32(2.5), "f": np.int32(4) * 2,
            "g": np.float64(cell["x"]), "h": float(seed)}
'''
    assert _keys(src) == ["a", "b", "c", "e", "f"]


def test_a_helper_whose_only_statement_returns_a_number_is_typed_in() -> None:
    src = '''
def _zero():
    return 0.0

def _two_steps():
    return _zero()

def _measured(x):
    return x * 2.0

def run_trial(cell, trial, seed):
    return {"a": _zero(), "b": _two_steps(), "c": _measured(cell["x"]), "d": _measured(1.0)}
'''
    assert _keys(src) == ["a", "b"], "a helper that takes an argument, or computes from one, is not followed"


def test_oracle_is_read_like_the_others() -> None:
    src = 'def oracle():\n    return {"decay": 1.0}\n'
    assert _keys(src) == ["decay"]


def test_a_script_that_does_not_parse_has_nothing_to_report() -> None:
    assert tr.typed_results("def run_trial(:\n") == []


def test_the_directive_names_each_key_and_line_and_holds_no_expected_value() -> None:
    found = tr.typed_results(CUP, "simulate.py")
    text = tr.directive(found)
    assert "simulate.py line 9" in text and "`lid_on`" in text and "compute it from the simulation" in text
    assert "if it is a setting, return it under the setting's own name" in text
    assert "tolerance" not in text.lower() and "expected" not in text.lower()


def test_strip_keys_removes_the_quantity_at_every_depth() -> None:
    value = {"lid_on": 1.0, "a": {"lid_on": [1, 2], "keep": 3}, "rows": [{"lid_on": 1, "x": 2}]}
    assert tr.strip_keys(value, {"lid_on"}) == {"a": {"keep": 3}, "rows": [{"x": 2}]}
