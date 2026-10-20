"""The second look at a search's coarse scan in parts (tests/test_optimise_scan_parts.py): a part that leaves a value of
the scan unrun loses only the parts, not the search; what each evaluation goes through is read as strictly as a
measurement's; a run's numbers per value are not designs of their own; ``code/run.py`` scans what FI scanned."""
from __future__ import annotations

import json
import sys
import types
from typing import Any

from core import code_project
from core import optimisation_plan as op
from core import optimise_search as osearch
from tests.test_optimise_runner import BLOCK
from tests.test_optimise_scan_parts import PARTS, _parts


def test_a_part_that_leaves_a_value_unrun_loses_the_parts_not_the_search() -> None:
    one_part = _parts(sub_studies=[{"name": "spread", "vary": ["spread"], "hold": {"kind": "spread", "offset": 0.0}}])
    _, why = op.normalize(one_part)
    assert why and why.startswith("`protocol.optimisation.sub_studies`") and "would never be run" in why
    repaired, notes = op.repair(one_part)
    assert repaired is not None, notes
    assert "sub_studies" not in repaired and repaired["design_variables"], "the search stays; only its parts go"


def test_what_each_evaluation_goes_through_is_read_as_a_measurement_s() -> None:
    for bad, words in (({"case": [1, 1]}, "same value twice"), ({"case": [1, "a"]}, "mixes names and numbers"),
                       ({"mesh": [0.5, 0.25]}, "numerical setting")):
        _, why = op.normalize({**BLOCK, "in_each_run": bad})
        assert why and words in why and why.startswith("`protocol.optimisation.in_each_run"), (bad, why)
    block, why = op.normalize({**BLOCK, "in_each_run": {"case": ["a", "b"]}})
    assert why is None and block["in_each_run"] == {"case": ["a", "b"]}


def test_a_run_s_numbers_per_value_are_not_designs_of_their_own() -> None:
    block, why = op.normalize({**BLOCK, "in_each_run": {"case": [1, 2]}, "grid": {"x": [-1.0, 1.0]}})
    assert why is None
    cells = []
    for x in (-1.0, 1.0):
        cell = {"x": x, "y": 0.0, "scale": 1.0, "mesh": 0.5}
        cells.append({"cell": cell, "ok": True, "failed": 0, "metrics": {"f": {"values": [x * x]}}})
        cells += [{"cell": {**cell, "case": c}, "level": {"case": c}, "ok": True, "failed": 0,
                   "metrics": {"f_case": {"values": [c * x]}}} for c in (1, 2)]
    rows = osearch.scan_rows({"cells": cells}, block)
    assert [r["design"]["x"] for r in rows] == [-1.0, 1.0]


def test_code_run_py_scans_the_parts_and_gives_each_cell_what_it_goes_through(tmp_path, monkeypatch) -> None:
    block, why = op.normalize(_parts(in_each_run={"case": [1.0, 2.0]}))
    assert why is None, why
    namespace: dict[str, Any] = {"__name__": "fi_run_py", "__file__": str(tmp_path / "run.py")}
    exec(compile(code_project.RUN_SOURCE, "run.py", "exec"), namespace)  # noqa: S102 -- FI's own file
    (tmp_path / "raw").mkdir()
    namespace["WORK"] = tmp_path
    asked: list[dict[str, Any]] = []

    def fake_run_trials(study: dict[str, Any]) -> dict[str, Any]:
        asked.append(json.loads(json.dumps(study)))
        return {"schema": "fi.trials/v1", "cells": []}

    namespace["run_trials"] = fake_run_trials
    fake = types.ModuleType("fi_search")
    fake.scan_rows = lambda summary, blk: []
    fake.run_sync = lambda blk, evaluate, **kw: (kw["scan_step"](osearch.scan_grid(blk)), {"rows": [{"status": "ok"}]})[1]
    fake.best_design = lambda outcome, blk, **kw: {"says": "done"}
    fake.ledger_lines = lambda outcome, blk: []
    monkeypatch.setitem(sys.modules, "fi_search", fake)
    namespace["search"]({"optimisation": block, "entry": "run_cell"})
    assert len(asked) == 1
    study = asked[0]
    assert study["sub_studies"] == PARTS["sub_studies"]
    assert study["in_each_run"] == {"case": [1.0, 2.0]}
    assert "case" not in study["grid"], "what each run goes through is not a setting of the scan"
    # The settings run.py lists are the parts' settings, the same as FI's own scan.
    assert namespace["cells"](study["grid"], study["sub_studies"]) == osearch.grid_cells(osearch.scan_grid(block))
