"""A result the plan works out afterwards from the trials (derive.py's AFTERWARDS) has FI's own per-setting value as its only
value when no trial returned it: the table serves that value (``Table.set_derived``), so a figure line drawn from exactly
those numbers is named by the result instead of "computed value". A result a trial returned is never replaced. Made-up study:
a colony's ``spread`` worked out from its trials' ``growth`` at two sizes. No model is called."""
from __future__ import annotations

import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

from tests.test_frame_figure import _Host  # noqa: E402

CELLS = [{"cell": {"size": s}, "key": "size=%s" % s, "planned": 3, "ok": 3, "failed": 0,
          "metrics": {"growth": {"values": [1.0 * s, 2.0 * s, 3.0 * s]}}} for s in (1, 2)]
SPREAD = {"size=1": 0.25, "size=2": 0.5}  # FI's record of the afterwards result, one value per setting


def _table(records: dict[str, Any]):
    from core import frame_table, skeleton

    sys.modules["fi_table"] = frame_table
    ns: dict[str, Any] = {"fi_table": frame_table}
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    table = ns["Table"](CELLS, ["growth", "spread"])  # spread is declared (afterwards) and no trial returned it
    table.set_derived(records)
    return table


def _record(by_setting: dict[str, float]) -> dict[str, Any]:
    return {"status": "ok", "by_setting": dict(by_setting), "role": "result"}


def _drawn_what(records, ys, what="spread"):
    h = _Host(panels=1)
    h.table = _table(records)
    h.settings = ["size"]
    h.derived_ids = list(records)
    h.labels = {"spread": "Spread", "growth": "Growth", "size": "Size"}
    h.line(0, "fast", [1, 2], ys, what)
    return h.record()["panels"][0]["series"][0]["what"]


def test_a_result_worked_out_afterwards_is_read_from_fis_record():
    table = _table({"spread": _record(SPREAD)})
    assert table.mean("spread", "size=1") == 0.25 and table.mean("spread", "size=2") == 0.5


def test_a_line_of_exactly_fis_values_of_it_is_named_by_the_result():
    # before: the table refused FI's value of a declared id and read the trials, which hold nothing: "computed value"
    assert _drawn_what({"spread": _record(SPREAD)}, [0.25, 0.5]) == "spread"


def test_known_positive_other_numbers_under_its_name_stay_unnamed():
    assert _drawn_what({"spread": _record(SPREAD)}, [0.3, 0.5]) == "other"


def test_a_result_a_trial_returned_is_never_replaced_by_a_derived_record():
    table = _table({"growth": _record({"size=1": 99.0, "size=2": 99.0})})
    assert table.mean("growth", "size=1") == 2.0  # the trials' mean, not the record's 99
    assert _drawn_what({"growth": _record({"size=1": 99.0, "size=2": 99.0})}, [99.0, 99.0], "growth") == "other"
    assert _drawn_what({"growth": _record({"size=1": 99.0, "size=2": 99.0})}, [2.0, 4.0], "growth") == "growth"
