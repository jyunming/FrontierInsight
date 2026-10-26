"""A name the code uses that nothing in the code defines is found before the code runs (core/undefined_names.py)."""

from __future__ import annotations

from core import static_checks as sc
from core import undefined_names as un


def _names(files: dict[str, str]) -> list[str]:
    return sorted(u.name for u in un.problems(files))


def test_a_name_used_in_a_function_that_nothing_defines_is_found_with_its_script_and_scope() -> None:
    found = un.undefined_in("sim.py", "def simulate(n):\n    return Heater(n)\n")
    assert [(u.name, u.script, u.function) for u in found] == [("Heater", "sim.py", "simulate")]
    assert "first used on line 2" in found[0].says()


def test_names_python_binds_in_any_of_its_ways_are_defined() -> None:
    code = (
        "import math\n"
        "from os import path as p\n"
        "CONST = 3\n"
        "class A:\n"
        "    k = CONST\n"
        "    def m(self):\n"
        "        return self.k + A.k\n"
        "def f(xs, *a, **kw):\n"
        "    global G\n"
        "    G = 1\n"
        "    ys = [x * CONST for x in xs if x]\n"
        "    lam = lambda q: q + len(a)\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError as err:\n"
        "        print(err)\n"
        "    with open('x') as fh:\n"
        "        pass\n"
        "    return ys, lam, math.pi, p, G, super, __name__\n"
        "for i in range(3):\n"
        "    print(i, G)\n"
    )
    assert un.undefined_in("ok.py", code) == []


def test_a_name_bound_only_inside_a_function_is_not_defined_at_module_level() -> None:
    # The walrus binds w in f, so the module's use of w is a NameError in Python too.
    code = "def f(xs):\n    z = (w := 2)\n    return z\nprint(w)\n"
    assert [u.name for u in un.undefined_in("x.py", code)] == ["w"]


def test_a_module_name_set_by_a_script_of_the_folder_or_at_run_time_is_defined() -> None:
    # Scripts import one another, and a package's constants are set by the script that calls it.
    files = {"model.py": "def f():\n    return SCALE * 2\n", "sim.py": "import model\nmodel.SCALE = 3\n"}
    assert _names(files) == []
    assert _names({"model.py": "def f():\n    return SCALE * 2\n"}) == ["SCALE"]


def test_a_folder_that_sets_names_at_run_time_in_a_way_the_text_does_not_show_is_not_judged() -> None:
    files = {"model.py": "def f():\n    return SCALE\n", "sim.py": "for k in KEYS:\n    setattr(model, k, 1)\n"}
    assert _names(files) == []


def test_a_script_with_a_star_import_and_one_that_does_not_parse_are_not_judged() -> None:
    assert un.undefined_in("s.py", "from helpers import *\nprint(Heater)\n") == []
    assert un.undefined_in("s.py", "def (:\n") == []


def test_a_name_set_as_a_global_inside_a_function_is_defined_for_the_module() -> None:
    code = "def setup():\n    global LIMIT\n    LIMIT = 5\nprint(LIMIT)\n"
    assert un.undefined_in("g.py", code) == []


def test_the_check_is_a_stop_in_the_static_gate_and_is_read_before_the_others() -> None:
    checks = sc.static_checks()
    assert checks[0][0] == "undefined_names" and checks[0][2] == "stop"
    assert checks[0][1].problems({"s.py": "def f():\n    return Heater\n"}) != []


def test_the_fault_is_written_for_a_person_and_the_repair_request_names_it() -> None:
    found = un.problems({"sim.py": "def simulate(n):\n    return Heater(n)\n"})
    assert "Heater" in un.plain_note(found, None) and "`Heater`" in un.directive(found, None)
    assert un.record(found, None)["names"][0]["name"] == "Heater"


def test_every_static_check_finding_is_read_the_way_the_gate_reads_it() -> None:
    # The gate calls ``says([])`` on whatever finding a check returns; the undefined-name finding took no argument and
    # crashed the gate on a standing fault.
    found = un.problems({"sim.py": "def simulate(n):\n    return Heater(n)\n"})
    assert "Heater" in found[0].says([])
