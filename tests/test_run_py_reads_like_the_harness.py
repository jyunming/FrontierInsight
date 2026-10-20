"""code/run.py, the script a person runs to reproduce a run, reads what a simulation returns as FI's trial harness does
(core/in_run.py ``read_returned``): a value that does not exist is ``None`` with its reason under ``why_missing``, and a
result per value is read when the spec names the values a run compares inside it."""
import json
import math
import subprocess
import sys
from pathlib import Path

from core import code_project

PROTOCOL = {"grid": {"gap": [1, 2]}, "runs_per_setting": 2, "thresholds": {}}
SIMULATE = '''
def run_trial(cell, trial_id, seed):
    if cell["gap"] == 1:
        return {"width": None, "depth": 3.0, "why_missing": {"width": "no crossing in the range"}}
    return {"width": {"a": 1.0, "b": 2.0}, "depth": 3.0}
'''


def _evaluate(tmp_path: Path, cell: dict, **spec_extra):
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('x')\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    spec, out = tmp_path / "spec.json", tmp_path / "rows.json"
    spec.write_text(json.dumps({"entry": "run_trial", "cell": cell, "trials": [{"trial": 0, "seed": 7}],
                                **spec_extra}), encoding="utf-8")
    done = subprocess.run([sys.executable, "run.py", "--evaluate", str(spec), str(out)], cwd=root / "code",
                          capture_output=True, text=True, timeout=120)
    return json.loads(out.read_text(encoding="utf-8")), done


def test_an_evaluation_keeps_a_missing_value_and_its_reason(tmp_path):
    rows, done = _evaluate(tmp_path, {"gap": 1})
    (row,) = rows
    assert row["ok"] is True, done.stderr
    assert math.isnan(row["values"]["width"]) and row["values"]["depth"] == 3.0 and "why_missing" not in row["values"]
    assert row["why_missing"] == {"width": "no crossing in the range"}


def test_an_evaluation_reads_a_result_per_value_when_the_spec_names_them(tmp_path):
    levels = [{"load": "a"}, {"load": "b"}]
    rows, done = _evaluate(tmp_path, {"gap": 2}, levels=levels)
    (row,) = rows
    assert row["ok"] is True, done.stderr
    assert sorted(row["values"]["width"].values()) == [1.0, 2.0] and len(row["values"]["width"]) == 2
