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
    code = _code(tmp_path, {"experiment.py": "import helpers\n\ndef main():\n    getattr(helpers, NAME)()\n\nmain()\n",
                            "helpers.py": "def unused():\n    return 1\n"})
    assert code_reach.reached(code, scripts=["experiment.py"]) is None
    assert not code_reach.never_run(code, "helpers.py", "unused", scripts=["experiment.py"])


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
