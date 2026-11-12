"""Which functions the run reaches, read from the code (replay of suite r9k first_order_kinetics: the paper published, as a
"plotting defect" of its Figure 3, a code reader's finding about ``simulate.py``'s ``main()``, which the run never called;
the figure was drawn by ``experiment.py``). A finding about code the run never calls stays in FI's record, not the paper."""

from __future__ import annotations

import shutil
from pathlib import Path

from core import code_reach
from core import code_review
from core import quality_report

FIXTURE = Path(__file__).parent / "fixtures" / "code_reach_r9k"


def _quest(tmp_path: Path) -> Path:
    """The real quest's record, for the code as checked out (line endings change its bytes, so its hash is set again)."""
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    record = code_review.load(tmp_path)
    record["version"] = record["read_version"] = code_review.version(tmp_path / "code")
    code_review.save(tmp_path, record)
    return tmp_path


def test_r9k_s_unused_main_is_never_run_and_its_trial_function_is(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    never_run = code_reach.never_run_in(root)
    assert never_run("simulate.py", "main")
    assert not never_run("simulate.py", "run_trial")  # FI's harness calls it
    assert not never_run("kinetics_parameter_recovery/model.py", "fit_arrhenius_line")  # reached through run_trial
    assert not never_run("experiment.py", "plot_difference")  # the analysis draws the figure
    assert not never_run("simulate.py", "no_such_function") and not never_run("", "main")


def test_r9k_s_paper_and_quality_report(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    found = code_review.open_findings(root, root / "code")
    assert [v["fid"] for v in found["never_run"]] == ["F-560671f3"]
    assert [v["fid"] for v in found["flagged"]] == ["F-a3441c2a"]
    block = code_review.findings_block(root, root / "code")
    note = code_review.write_note(root, root / "code")
    assert "weighted_minus_unweighted" not in block and "weighted_minus_unweighted" not in note
    assert "diagonal matrix" in block and "diagonal matrix" in note
    rows = {r["id"]: r for r in quality_report.build(root)["code"]["issues"]}
    assert rows["F-560671f3"]["never_run"] is True and rows["F-a3441c2a"]["never_run"] is False


def _code(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_a_name_computed_at_run_time_makes_everything_reached(tmp_path: Path) -> None:
    code = _code(tmp_path, {"experiment.py": "import helpers\n\ndef main(obj):\n    getattr(obj, NAME)()\n\nmain(helpers)\n",
                            "helpers.py": "def unused():\n    return 1\n"})
    assert code_reach.reached(code, scripts=["experiment.py"]) is None
    assert not code_reach.never_run(code, "helpers.py", "unused", scripts=["experiment.py"])


def test_a_name_computed_on_an_imported_module_reaches_every_function_of_that_module_and_no_other(tmp_path: Path) -> None:
    code = _code(tmp_path, {"experiment.py": "import helpers\n\ndef main():\n    getattr(helpers, NAME)()\n\nmain()\n",
                            "helpers.py": "def one():\n    return 1\n\ndef two():\n    return 2\n",
                            "other.py": "def lone():\n    return 3\n"})
    done = code_reach.reached(code, scripts=["experiment.py"])
    assert done is not None and {("helpers.py", "one"), ("helpers.py", "two")} <= done
    assert code_reach.never_run(code, "other.py", "lone", scripts=["experiment.py"])
    assert not code_reach.never_run(code, "helpers.py", "two", scripts=["experiment.py"])


def _frame_quest(tmp_path: Path, derive: str, model: str) -> Path:
    """A quest with FI's own frame files as FI writes them, and a study's ``derive.py`` and model package."""
    from core import analysis_frame, skeleton

    files = {"experiment.py": analysis_frame.EXPERIMENT, "fi_frame.py": skeleton.RUNTIME, "fi_table.py": skeleton.TABLE_SOURCE,
             "fi_figure_calls.py": skeleton.FRAME_FIGURE_SOURCE, "derive.py": derive,
             "simulate.py": "def run_trial(cell):\n    return {}\n",
             "pkg/__init__.py": "", "pkg/model.py": model, **analysis_frame.carried_files()}
    return _code(tmp_path, files)


MODEL_NESTED = ("def aggregate_and_plot(rows):\n    def mean_interval(x):\n        return sum(x) / len(x)\n"
                "    return [mean_interval(r) for r in rows]\n\n\nclass Box:\n    def size(self):\n        return 1\n")


def test_a_quest_in_fi_s_frame_can_be_read_although_fi_s_own_files_compute_names(tmp_path: Path) -> None:
    code = _frame_quest(tmp_path, "def dose():\n    return 1.0\n\ndef also_unused_by_the_plan():\n    return 2.0\n", MODEL_NESTED)
    done = code_reach.reached(code, scripts=["experiment.py"], harness="simulate.py")
    assert done is not None, "FI's own getattr calls (experiment.py, fi_figures.py) do not make the answer 'cannot tell'"
    assert {("derive.py", "dose"), ("derive.py", "also_unused_by_the_plan")} <= done  # experiment.py names them by getattr
    assert ("simulate.py", "run_trial") in done
    assert ("pkg/model.py", "aggregate_and_plot") not in done


def test_a_function_named_as_outer_dot_inner_is_judged_by_the_function_it_sits_in(tmp_path: Path) -> None:
    code = _frame_quest(tmp_path, "def dose():\n    return 1.0\n", MODEL_NESTED)
    kw = {"scripts": ["experiment.py"], "harness": "simulate.py"}
    assert code_reach.never_run(code, "pkg/model.py", "aggregate_and_plot.mean_interval", **kw)
    assert code_reach.never_run(code, "pkg/model.py", "aggregate_and_plot", **kw)
    assert code_reach.never_run(code, "pkg/model.py", "Box.size", **kw)  # a method: its own name is judged
    assert not code_reach.never_run(code, "pkg/model.py", "no_such.thing", **kw)
    (code / "simulate.py").write_text("from pkg.model import aggregate_and_plot\n\ndef run_trial(cell):\n"
                                      "    return aggregate_and_plot([[1]])\n", encoding="utf-8")
    assert not code_reach.never_run(code, "pkg/model.py", "aggregate_and_plot.mean_interval", **kw)  # outer is reached
    assert not code_reach.never_run(code, "pkg/model.py", "aggregate_and_plot", **kw)


def test_a_dead_function_nested_in_a_function_that_runs_is_unreached(tmp_path: Path) -> None:
    model = ("def fit(rows):\n    def used(x):\n        return x\n\n    def dead(x):\n        return -x\n\n"
             "    return [used(r) for r in rows]\n")
    code = _frame_quest(tmp_path, "def dose():\n    return 1.0\n", model)
    (code / "simulate.py").write_text("from pkg.model import fit\n\ndef run_trial(cell):\n    return fit([1])\n",
                                      encoding="utf-8")
    kw = {"scripts": ["experiment.py"], "harness": "simulate.py"}
    assert not code_reach.never_run(code, "pkg/model.py", "fit.used", **kw)
    assert not code_reach.never_run(code, "pkg/model.py", "fit", **kw)
    assert code_reach.never_run(code, "pkg/model.py", "fit.dead", **kw)  # judged by the outer function it was kept


def test_a_nested_function_of_a_function_that_never_runs_is_unreached_although_another_function_has_its_name(
        tmp_path: Path) -> None:
    code = _code(tmp_path, {
        "experiment.py": "import model\n\nprint(model.score(3))\n",
        "model.py": ("def score(x):\n    return x\n\ndef never_called(rows):\n    def score(x):\n        return -x\n\n"
                     "    return [score(r) for r in rows]\n")})
    kw = {"scripts": ["experiment.py"]}
    assert code_reach.never_run(code, "model.py", "never_called", **kw)
    assert code_reach.never_run(code, "model.py", "never_called.score", **kw)  # a name of the same text is reached elsewhere
    assert not code_reach.never_run(code, "model.py", "score", **kw)  # the top-level one is the one the program calls


def test_a_computed_name_in_a_study_file_still_means_fi_cannot_tell_even_in_fi_s_frame(tmp_path: Path) -> None:
    code = _frame_quest(tmp_path, "def dose():\n    return 1.0\n", MODEL_NESTED + "\n\ndef pick(obj, n):\n    return getattr(obj, n)\n")
    (code / "simulate.py").write_text("from pkg.model import pick\n\ndef run_trial(cell):\n    return pick(cell, 'a')\n",
                                      encoding="utf-8")
    kw = {"scripts": ["experiment.py"], "harness": "simulate.py"}
    assert code_reach.reached(code, **kw) is None
    assert not code_reach.never_run(code, "pkg/model.py", "aggregate_and_plot", **kw)
    # a derive.py of the study that evaluates a computed name is the study's too
    (code / "simulate.py").write_text("def run_trial(cell):\n    return {}\n", encoding="utf-8")
    (code / "derive.py").write_text("def dose():\n    return __import__('math').pi\n", encoding="utf-8")
    assert code_reach.reached(code, **kw) is None


def test_a_study_module_called_like_one_of_fi_s_files_is_not_fi_s(tmp_path: Path) -> None:
    code = _frame_quest(tmp_path, "def dose():\n    return 1.0\n", MODEL_NESTED)
    # a function FI's analysis calls on its table module (``fi_table.load``), here the study's own
    (code / "fi_table.py").write_text("def load(obj, n):\n    return getattr(obj, n)\n", encoding="utf-8")
    assert code_reach.reached(code, scripts=["experiment.py"], harness="simulate.py") is None


def test_a_function_passed_on_as_a_value_or_called_on_its_module_is_reached(tmp_path: Path) -> None:
    code = _code(tmp_path, {
        "experiment.py": ("import helpers as h\nfrom helpers import cb\n\ndef main():\n    TABLE = {'a': cb}\n"
                          "    return h.used(), TABLE\n\nif __name__ == '__main__':\n    main()\n"),
        "helpers.py": ("def used():\n    return 1\n\ndef cb():\n    return 2\n\ndef unused():\n    return 3\n\n"
                       "if __name__ == '__main__':\n    unused()\n")})
    done = code_reach.reached(code, scripts=["experiment.py"])
    assert done is not None
    assert {("helpers.py", "used"), ("helpers.py", "cb"), ("experiment.py", "main")} <= done
    assert ("helpers.py", "unused") not in done  # only its own main block calls it, and FI never runs it as a program


def test_a_file_that_does_not_parse_makes_everything_reached(tmp_path: Path) -> None:
    code = _code(tmp_path, {"experiment.py": "def main(:\n", "helpers.py": "def unused():\n    return 1\n"})
    assert code_reach.reached(code, scripts=["experiment.py"]) is None


def test_a_function_a_package_re_exports_is_reached(tmp_path: Path) -> None:
    code = _code(tmp_path, {
        "experiment.py": "import pkg\nfrom pkg import other\n\nprint(pkg.compute_metric(), other())\n",
        "pkg/__init__.py": "from .sub import compute_metric\nfrom .more import *\n",
        "pkg/sub.py": "def compute_metric():\n    return 1\n\ndef unused():\n    return 2\n",
        "pkg/more.py": "def other():\n    return 3\n"})
    done = code_reach.reached(code, scripts=["experiment.py"])
    assert done is not None
    assert {("pkg/sub.py", "compute_metric"), ("pkg/more.py", "other")} <= done
    assert ("pkg/sub.py", "unused") not in done


def test_a_harness_entry_imported_into_the_simulation_is_reached(tmp_path: Path) -> None:
    code = _code(tmp_path, {"experiment.py": "print(1)\n", "simulate.py": "from sim_core import run_trial\n",
                            "sim_core.py": "def run_trial(cell, trial_id, seed):\n    return step()\n\ndef step():\n    return 1\n"})
    done = code_reach.reached(code, scripts=["experiment.py"], harness="simulate.py")
    assert done is not None and {("sim_core.py", "run_trial"), ("sim_core.py", "step")} <= done


def test_a_special_method_of_a_class_the_run_uses_is_never_judged_never_run(tmp_path: Path) -> None:
    code = _code(tmp_path, {"experiment.py": "from sim import Sim\n\ns = Sim(3)\nprint(len(s))\nwith s:\n    pass\n",
                            "sim.py": ("class Sim:\n    def __init__(self, n):\n        self.n = n\n\n    def __len__(self):\n"
                                       "        return self.n\n\n    def __enter__(self):\n        return self\n\n"
                                       "    def __exit__(self, *a):\n        return False\n\n    def never_called(self):\n"
                                       "        return 1\n")})
    kw = {"scripts": ["experiment.py"]}
    for name in ("__init__", "__len__", "__enter__", "__exit__"):
        assert not code_reach.never_run(code, "sim.py", f"Sim.{name}", **kw), name
        assert not code_reach.never_run(code, "sim.py", name, **kw), name
    assert code_reach.never_run(code, "sim.py", "Sim.never_called", **kw)


def test_a_name_computed_on_a_package_reaches_what_the_package_re_exports(tmp_path: Path) -> None:
    code = _code(tmp_path, {"experiment.py": "import pkg\n\ngetattr(pkg, 'x' + 'y')()\n",
                            "pkg/__init__.py": "from .impl import xy\nfrom .more import *\n",
                            "pkg/impl.py": "def xy():\n    return 1\n",
                            "pkg/more.py": "def starred():\n    return 2\n",
                            "pkg/lone.py": "def lone():\n    return 3\n"})
    kw = {"scripts": ["experiment.py"]}
    assert not code_reach.never_run(code, "pkg/impl.py", "xy", **kw)
    assert not code_reach.never_run(code, "pkg/more.py", "starred", **kw)
    assert code_reach.never_run(code, "pkg/lone.py", "lone", **kw)


_PKG = {"trial.py": "from pkg import model\n\ndef run(x):\n    return model.fit(x)\n",
        "pkg/__init__.py": "from . import model\n",
        "pkg/model.py": ("import math\nR = 8.0\n\ndef rate(a):\n    r = globals().get('R', 1.0)\n    return a / r\n\n"
                         "def fit(x):\n    return rate(x) + math.sqrt(x)\n\ndef orphan(x):\n    return x\n")}


def test_a_package_helper_that_reads_a_module_name_by_a_constant_key_can_still_be_followed(tmp_path: Path) -> None:
    code = _code(tmp_path, _PKG)
    done = code_reach.reached(code, scripts=[], start=[("trial.py", "run")])
    assert done is not None  # globals().get("R") names one variable, not a computed function
    assert ("pkg/model.py", "fit") in done and ("pkg/model.py", "rate") in done
    assert ("pkg/model.py", "orphan") not in done


def test_a_constant_key_into_globals_reaches_a_function_of_that_name(tmp_path: Path) -> None:
    files = dict(_PKG)
    files["pkg/model.py"] = ("def rate(a):\n    return globals()['hidden'](a)\n\ndef hidden(a):\n    return a\n\n"
                             "def fit(x):\n    return rate(x)\n\ndef orphan(x):\n    return x\n")
    code = _code(tmp_path, files)
    done = code_reach.reached(code, scripts=[], start=[("trial.py", "run")])
    assert done is not None and ("pkg/model.py", "hidden") in done and ("pkg/model.py", "orphan") not in done


def test_a_computed_key_into_globals_still_means_fi_cannot_tell(tmp_path: Path) -> None:
    bodies = ["return globals()[name](a)", "return globals().get(name)(a)", "g = globals()" + chr(10) + "    return g['x'](a)",
              "return vars()", "return eval('a')"]
    for i, body in enumerate(bodies):
        files = dict(_PKG)
        files["pkg/model.py"] = "def fit(a, name='x'):" + chr(10) + "    " + body + chr(10)
        code = _code(tmp_path / f"case{i}", files)
        assert code_reach.reached(code, scripts=[], start=[("trial.py", "run")]) is None, body


# ---- an attribute of FI's own objects read in FI's own files ------------------------------------------------------------------------
TRIAL_STUDY = ("from pkg.model import helper\n\n\ndef trial(cell, ctx):\n    return {'x': helper(cell)}\n\n\n"
               "def run_trial(cell, trial_id, seed):\n    return trial(cell, None)\n")
FIGURE_DERIVE = "def figure_f1(table, fig):\n    fig.line([1.0], [2.0])\n    return fig\n\n\ndef derive_x(table):\n    return 1.0\n"


def _trial_quest(tmp_path: Path, model: str = "def helper(cell):\n    return 1.0\n") -> Path:
    code = _frame_quest(tmp_path, FIGURE_DERIVE, model)
    (code / "simulate.py").write_text(TRIAL_STUDY, encoding="utf-8")
    return code


def test_an_attribute_fi_s_own_classes_define_does_not_reach_the_study_function_of_that_name(tmp_path: Path) -> None:
    # FI's frame reads ``ctx.trial`` (the trial's number); the study has a function ``trial``: a figure's function reaches FI's
    # figure code, which reads ctx.trial, and must not reach the study's trial and its helpers through that name
    code = _trial_quest(tmp_path)
    from_figure = code_reach.reached(code, scripts=[], start=[("derive.py", "figure_f1")])
    assert from_figure is not None
    assert ("simulate.py", "trial") not in from_figure and ("pkg/model.py", "helper") not in from_figure
    # the run still reaches them: FI's harness calls run_trial, which calls trial
    run = code_reach.reached(code, scripts=["experiment.py"], harness="simulate.py")
    assert run is not None and {("simulate.py", "trial"), ("pkg/model.py", "helper")} <= run


def test_the_names_fi_s_classes_define_are_read_from_fi_s_own_text(tmp_path: Path) -> None:
    import ast as _ast

    from core import skeleton

    mod = code_reach._Module("fi_frame.py", _ast.parse(skeleton.RUNTIME))
    attrs, classes = code_reach._fis_names([mod])
    assert {"trial", "seed", "results"} <= attrs and "Ctx" in classes
    assert "helper" not in attrs


STUDY_OBJECT = ("class Model:\n    def trial(self, cell):\n        return 1.0\n\n    def results(self, cell):\n        return 2.0\n\n\n"
                "def make():\n    return Model()\n")
FI_EXTRA = {
    "typed": "def _typed(cell):\n    import pkg.model as study\n    box = study.Model()\n    return box.trial(cell)\n",
    "made": "def _made(cell):\n    from pkg.model import make\n    box = make()\n    return box.results(cell)\n",
    "imported": "from pkg.model import Model as _Study\n\n\ndef _imported(cell):\n    return _Study.results(None, cell)\n",
    "fi_own": "def _fi_own(cell):\n    c = Ctx(seed=1, trial=0)\n    return c.trial\n",
    "untyped": "def _untyped(obj, cell):\n    return obj.results(cell)\n",
    "other": "def _other(obj):\n    return obj.make()\n",
}


def _with_fi_function(tmp_path: Path, which: str) -> set[tuple[str, str]]:
    """What the run reaches when FI's own frame file holds one more function (FI's own computed getattr reaches every
    function of FI's files, so the quest holds one such function at a time)."""
    code = _trial_quest(tmp_path / which, "def helper(cell):\n    return 1.0\n\n\n" + STUDY_OBJECT)
    (code / "fi_frame.py").write_text((code / "fi_frame.py").read_text(encoding="utf-8") + "\n\n" + FI_EXTRA[which],
                                      encoding="utf-8")
    assert "fi_frame.py" in code_reach._fis_files(code, ["fi_frame.py"])  # still FI's file with the extra function
    done = code_reach.reached(code, scripts=[], start=[("fi_frame.py", "_" + which)])
    assert done is not None
    return done


def test_a_study_object_whose_type_the_code_fixes_is_still_followed_in_fi_s_files(tmp_path: Path) -> None:
    # an object bound from a study class or from any call that is not FI's class, or a name imported from a study module:
    # any function of that name, as before
    assert ("pkg/model.py", "trial") in _with_fi_function(tmp_path, "typed")
    assert ("pkg/model.py", "results") in _with_fi_function(tmp_path, "made")
    assert ("pkg/model.py", "results") in _with_fi_function(tmp_path, "imported")
    # an object bound from FI's own class: FI's functions only
    own = _with_fi_function(tmp_path, "fi_own")
    assert ("simulate.py", "trial") not in own and ("pkg/model.py", "trial") not in own


def test_the_limit_an_object_of_unknown_type_is_judged_by_the_attribute_name(tmp_path: Path) -> None:
    # Said plainly in core/code_reach.py: a parameter's type is not fixed by the code, so a study object handed to FI's code as
    # a parameter whose method shares a name with an attribute of FI's classes (``results``) is not followed by that read.
    assert ("pkg/model.py", "results") not in _with_fi_function(tmp_path, "untyped")
    # a name FI's classes do not define is followed on any object, as before
    assert ("pkg/model.py", "make") in _with_fi_function(tmp_path, "other")


def test_in_a_study_file_an_attribute_still_reaches_any_function_of_that_name(tmp_path: Path) -> None:
    code = _trial_quest(tmp_path)
    (code / "pkg" / "runner.py").write_text("def go(ctx):\n    return ctx.trial\n", encoding="utf-8")
    got = code_reach.reached(code, scripts=[], start=[("pkg/runner.py", "go")])
    assert got is not None and ("simulate.py", "trial") in got
