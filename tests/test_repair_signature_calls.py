"""A repair that changes a function's parameters is accepted when every call of it binds to the new ones (no call that
worked is broken), and refused, with the same reason as before, when a call that bound to the old ones no longer does."""

from __future__ import annotations

from core import finding_repair as fr
from core import script_reply

# A helper with four parameters that every call gives five arguments: no call of it works.
BROKEN_CALLS = '''import numpy as np


def figure_a(fig, table):
    _plot_pair(fig, table, "speed", "load", 0)


def figure_b(fig, table):
    _plot_pair(fig, table, "speed", "drag", 0)
    _plot_pair(fig, table, "speed", "lift", 1)


def _plot_pair(fig, panel, xid, yid):
    return [], []


def _draw(fig, table, panel, xid, yid):
    fig.points(panel, yid, (xid, [1.0]), [2.0], yid)
'''

# The repair: the helper now takes what every call passes.
REPLY_FITS = '''```python
def _plot_pair(fig, table, xid, yid, panel):
    _draw(fig, table, panel, xid, yid)
```'''

# A file where the old helper is called right: (fig, panel, xid, yid).
WORKING_CALLS = '''def figure_a(fig, table):
    _plot_pair(fig, 0, "speed", "load")
    _plot_pair(fig, 1, "speed", "drag")


def _plot_pair(fig, panel, xid, yid):
    fig.points(panel, yid, (xid, [1.0]), [2.0], yid)
'''

REPLY_BREAKS = '''```python
def _plot_pair(fig, table, xid, yid, panel):
    fig.points(panel, yid, (xid, [1.0]), [2.0], yid)
```'''

REFUSED = "must keep its parameters (fig, panel, xid, yid)"


def test_new_parameters_that_every_call_fits_are_accepted():
    # elsewhere=[]: the caller holds the whole project and no other file uses the helper
    out, replaced, why = fr.splice(BROKEN_CALLS, REPLY_FITS, None, None, limit=None, elsewhere=[])
    assert why == ""
    assert replaced == ["_plot_pair"]
    assert "def _plot_pair(fig, table, xid, yid, panel):" in out


def test_the_script_repair_path_does_not_know_the_other_files_so_it_keeps_the_strict_rule():
    # was: accepted the new signature seeing only the script, though another file may call it the old way
    made = script_reply.resolve(BROKEN_CALLS, "", REPLY_FITS)
    assert made.code == ""
    assert REFUSED in made.why


def test_a_call_that_worked_and_no_longer_binds_is_still_refused():
    out, replaced, why = fr.splice(WORKING_CALLS, REPLY_BREAKS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_one_working_call_among_broken_ones_is_still_refused():
    mixed = BROKEN_CALLS.replace('_plot_pair(fig, table, "speed", "lift", 1)', '_plot_pair(fig, 1, "speed", "lift")')
    out, _replaced, why = fr.splice(mixed, REPLY_FITS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_a_call_in_another_file_that_does_not_bind_is_refused():
    other = "from analysis import _plot_pair\n\n\ndef side(fig):\n    _plot_pair(fig, 0, 'a', 'b')\n"
    out, _replaced, why = fr.splice(BROKEN_CALLS, REPLY_FITS, None, None, limit=None, elsewhere=[other])
    assert out is None
    assert REFUSED in why


def test_a_function_that_is_never_called_keeps_the_old_rule():
    alone = "def _plot_pair(fig, panel, xid, yid):\n    return None\n"
    out, _replaced, why = fr.splice(alone, REPLY_FITS.replace("_draw(fig, table, panel, xid, yid)", "return None"),
                                    None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_a_function_handed_on_keeps_the_old_rule():
    handed = BROKEN_CALLS + "\n\nHANDLERS = [_plot_pair]\n"
    out, _replaced, why = fr.splice(handed, REPLY_FITS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_a_function_fi_calls_itself_keeps_the_old_rule():
    body = ("def figure_a(fig, table):\n    return None\n\n\ndef helper(fig):\n    figure_a(fig, None, 0)\n")
    reply = "```python\ndef figure_a(fig, table, panel):\n    return None\n```"
    out, _replaced, why = fr.splice(body, reply, None, None, limit=None)
    assert out is None
    assert "must keep its parameters (fig, table)" in why


def test_a_call_with_starred_arguments_keeps_the_old_rule():
    starred = BROKEN_CALLS.replace('_plot_pair(fig, table, "speed", "load", 0)', '_plot_pair(*args)')
    out, _replaced, why = fr.splice(starred, REPLY_FITS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


# --- uses of the function that are not a plain call by its name -------------------------------------------------------

SIM = '''def f(x, y):
    return x + y
'''
REPLY_ONE = '''```python
def f(x):
    return x
```'''


def _change(other_text, other_name="use.py"):
    return fr.splice(SIM, REPLY_ONE, None, None, limit=None, elsewhere=[(other_name, other_text)], module="sim")


def test_an_import_under_another_name_is_followed_to_its_calls():
    out, _r, why = _change("from sim import f as g\n\n\ndef run():\n    return g(1, 2)\n")
    assert out is None
    assert "use.py line 5" in why


def test_an_aliased_call_that_binds_is_accepted():
    out, replaced, why = _change("from sim import f as g\n\n\ndef run():\n    return g(1)\n")
    assert why == ""
    assert replaced == ["f"]


def test_a_module_level_call_that_does_not_bind_is_refused():
    out, _r, why = _change("import sim\n\nVALUE = sim.f(1, 2)\n")
    assert out is None
    assert "use.py line 3" in why


def test_the_function_passed_as_a_value_is_refused():
    for text in ("import sim\n\nrows = list(map(sim.f, [1], [2]))\n",
                 "from sim import f\nimport functools\n\np = functools.partial(f, 1)\nf(1)\n",
                 "from sim import f\n\nTABLE = {'a': f}\nf(1)\n"):
        out, _r, why = _change(text)
        assert out is None, text
        assert "use.py" in why


def test_a_lookup_by_name_or_a_star_import_is_refused():
    for text in ("import sim\n\nsim.f(1)\ng = getattr(sim, 'f')\n", "from sim import *\n\nf(1)\n",
                 "from sim import\t*\n\nf(1)\n"):
        out, _r, why = _change(text)
        assert out is None, text
        assert "use.py" in why


def test_a_file_that_only_imports_the_function_under_another_name_is_looked_at(tmp_path):
    (tmp_path / "sim.py").write_text(SIM, encoding="utf-8")
    (tmp_path / "use.py").write_text("from sim import f as g\n\n\ndef run():\n    return g(1, 2)\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("x = 1\n", encoding="utf-8")
    assert fr.project_files_using(tmp_path, "sim.py", {"f"}) == ["use.py"]


# --- the loosened rule needs FI to hold every use: an unknown project keeps the strict rule --------------------------

LOCAL_SIM = "def f(x, y):\n    return x + y\n\n\ndef local():\n    return f(1, 2)\n"
LOCAL_REPLY = "```python\ndef f(x):\n    return x\n\n\ndef local():\n    return f(1)\n```"
OTHER_CALLS_OLD = "from sim import f\n\nf(1, 2)\n"
STRICT = "must keep its parameters (x, y)"


def test_without_the_other_files_a_signature_change_keeps_the_strict_rule():
    out, _r, why = fr.splice(LOCAL_SIM, LOCAL_REPLY, None, None, limit=None)
    assert out is None
    assert STRICT in why


def test_with_the_other_files_given_an_old_style_call_refuses_and_none_accepts():
    out, _r, why = fr.splice(LOCAL_SIM, LOCAL_REPLY, None, None, limit=None, elsewhere=[("other.py", OTHER_CALLS_OLD)], module="sim")
    assert out is None and "other.py" in why
    out, _r, why = fr.splice(LOCAL_SIM, LOCAL_REPLY, None, None, limit=None, elsewhere=[], module="sim")
    assert why == "" and "def f(x):" in out


def test_code_delta_entry_keeps_the_strict_rule():
    from core import code_delta

    out, _names, why = code_delta.splice_functions(LOCAL_SIM, LOCAL_REPLY)
    assert out is None
    assert STRICT in why


def test_script_reply_entry_keeps_the_strict_rule():
    made = script_reply.resolve(LOCAL_SIM, "", LOCAL_REPLY)
    assert made.code == ""
    assert STRICT in made.why


def _project(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")


def test_a_star_import_with_a_tab_is_found_by_parsing(tmp_path):
    _project(tmp_path, {"sim.py": SIM, "bridge.py": "from sim import\t*\n", "z.py": "x = 1\n"})
    assert fr.project_files_using(tmp_path, "sim.py", {"area"}) == ["bridge.py"]


def test_a_star_import_of_a_library_is_not_a_use(tmp_path):
    _project(tmp_path, {"sim.py": SIM, "z.py": "from math import *\nx = sqrt(4)\n"})
    assert fr.project_files_using(tmp_path, "sim.py", {"area"}) == []


def test_a_reexport_through_a_star_import_is_refused(tmp_path):
    _project(tmp_path, {"sim.py": SIM, "bridge.py": "from sim import f as g\n",
                        "down.py": "from bridge import *\n\nprint(g(1, 2))\n"})
    others = fr.project_files_using(tmp_path, "sim.py", {"f"})
    assert "down.py" in others
    out, _r, why = fr.splice(SIM, REPLY_ONE, None, None, limit=None,
                             elsewhere=[(o, (tmp_path / o).read_text(encoding="utf-8")) for o in others], module="sim")
    assert out is None
    assert "down.py" in why or "bridge.py" in why  # refused at the first file that may pass it on


def test_a_star_import_of_a_library_in_a_file_that_calls_it_is_refused_not_assumed():
    # FI cannot show from the code that the library does not bring `f` in; the loosened rule is not used (the strict one stays)
    lib = ("lib.py", "from math import *\n\nf(1, 2)\n")
    out, _r, why = fr.splice(SIM, REPLY_ONE, None, None, limit=None, elsewhere=[lib], module="sim")
    assert out is None
    assert "lib.py line 1" in why


def test_a_star_import_of_a_sibling_module_in_a_subfolder_is_a_use(tmp_path):
    (tmp_path / "pkg").mkdir()
    _project(tmp_path, {"sim.py": SIM})
    (tmp_path / "pkg" / "helper.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "use.py").write_text("from helper import *\n", encoding="utf-8")
    assert fr.project_files_using(tmp_path, "sim.py", {"area"}) == ["pkg/use.py"]


def _study(tmp_path, other_text=None):
    code = tmp_path / "code"
    code.mkdir()
    (code / "derive.py").write_text(BROKEN_CALLS, encoding="utf-8")
    if other_text is not None:
        (code / "other.py").write_text(other_text, encoding="utf-8")
    return code


def test_script_reply_with_a_code_folder_accepts_a_signature_every_use_binds_to(tmp_path):
    code = _study(tmp_path, "def unrelated():\n    return 1\n")
    made = script_reply.resolve(BROKEN_CALLS, "", REPLY_FITS, code_dir=code, rel="derive.py")
    assert made.why == ""
    assert "def _plot_pair(fig, table, xid, yid, panel):" in made.code


def test_script_reply_with_a_code_folder_refuses_when_another_file_uses_the_old_parameters(tmp_path):
    code = _study(tmp_path, "from derive import _plot_pair\n\n\ndef g(fig):\n    _plot_pair(fig, 0, 'a', 'b')\n")
    made = script_reply.resolve(BROKEN_CALLS, "", REPLY_FITS, code_dir=code, rel="derive.py")
    assert made.why and made.code == ""


def test_script_reply_without_a_code_folder_stays_strict(tmp_path):
    made = script_reply.resolve(BROKEN_CALLS, "", REPLY_FITS)
    assert "must keep its parameters" in made.why


def test_splice_functions_takes_the_code_folder_the_same_way(tmp_path):
    from core import code_delta
    code = _study(tmp_path)
    assert code_delta.splice_functions(BROKEN_CALLS, REPLY_FITS)[0] is None
    new, replaced, why = code_delta.splice_functions(BROKEN_CALLS, REPLY_FITS, code=code, rel="derive.py")
    assert why == "" and replaced == ["_plot_pair"]


# --- a use FI cannot prove is refused, never assumed away -------------------------------------------------------------

CALC = "def calculate(x, y):\n    return x + y\n"
CALC_FN = "def calculate(x):\n    return x\n"
CALC_CODE = "```python\n" + CALC_FN + "```"
FITTING_USE = "from sim import calculate\n\n\ndef run():\n    return calculate(1)\n"


def _calc_study(tmp_path, files):
    code = tmp_path / "code"
    code.mkdir()
    (code / "sim.py").write_text(CALC, encoding="utf-8")
    for name, text in files.items():
        (code / name).write_text(text, encoding="utf-8")
    return code


def test_a_reply_with_code_and_functions_is_judged_with_the_project_like_functions_only(tmp_path):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE})
    only_fn = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    both = script_reply.resolve(CALC, CALC_FN, CALC_CODE, code_dir=code, rel="sim.py")
    assert only_fn.why == ""
    assert both.why == ""
    assert "def calculate(x):" in both.code


def test_a_use_through_a_name_built_at_run_time_refuses(tmp_path):
    dyn = 'import importlib\n\nm = importlib.import_module("sim")\nvalue = getattr(m, "calcu" + "late")(1, 2)\n'
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "dyn.py": dyn})
    assert "dyn.py" in fr.project_files_using(code, "sim.py", {"calculate"})
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.code == "" and "dyn.py" in made.why


def test_a_use_through_a_module_object_with_a_computed_attribute_refuses(tmp_path):
    dyn = 'import sim as s\n\nname = "calcu" + "late"\nvalue = getattr(s, name)(1, 2)\n'
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "dyn.py": dyn})
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.code == "" and "dyn.py" in made.why


def test_a_reexport_under_another_name_refuses(tmp_path):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "bridge.py": "from sim import calculate as g\n",
                                  "far.py": "from bridge import g as h\n\nh(1, 2)\n"})
    assert "far.py" in fr.project_files_using(code, "sim.py", {"calculate"})
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.code == "" and "bridge.py" in made.why


def test_an_import_nobody_imports_from_is_the_end_of_the_chain_and_is_accepted(tmp_path):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "last.py": "from sim import calculate as g\n\ng(1)\n"})
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.why == ""


def test_a_file_that_cannot_be_read_refuses(tmp_path, monkeypatch):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "locked.py": "x = 1\n"})
    real = type(code).read_text

    def read_text(self, *a, **k):
        if self.name == "locked.py":
            raise PermissionError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(type(code), "read_text", read_text)
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.code == "" and "locked.py" in made.why


def test_a_file_that_is_not_valid_text_refuses(tmp_path):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE})
    (code / "bad.py").write_bytes(b"x = '\xff\xfe'\n")
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.code == "" and "bad.py" in made.why


FRAME = """import sim


class _Plan:
    def __getattr__(self, name):
        return getattr(sim, name)
"""


def test_fi_analysis_frame_reads_only_declared_names_so_a_helper_it_does_not_declare_is_not_a_use(tmp_path):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "experiment.py": FRAME})
    made = script_reply.resolve(CALC, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.why == ""


def test_fi_analysis_frame_looks_up_a_function_the_plan_declares_so_it_refuses(tmp_path):
    code = _calc_study(tmp_path, {"use.py": FITTING_USE, "experiment.py": FRAME})
    declared = CALC + "\nPLAN = {'by': 'calculate'}\n"
    (code / "sim.py").write_text(declared, encoding="utf-8")
    made = script_reply.resolve(declared, "", CALC_CODE, code_dir=code, rel="sim.py")
    assert made.code == "" and "looked up by name" in made.why  # already refused by the text rule
