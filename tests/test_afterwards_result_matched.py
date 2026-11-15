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


def test_a_table_read_of_fis_derived_value_is_not_called_a_mean():
    # review: FI's one value of such a result at a setting is no mean of trials, so the line's name says no "Mean of"
    h = _Host(panels=1)
    h.table = _table({"spread": _record(SPREAD)})
    h.settings = ["size"]
    h.labels = {"spread": "Spread", "growth": "Growth", "size": "Size"}
    h.line(0, "fast", [1, 2], [0.25, 0.5], "spread")
    h.ref(0, at="spread", key="size=1")
    assert [r["label"] for r in h.record()["panels"][0]["refs"]] == ["Spread at Size = 1"]
    h.ref(0, at="growth", key="size=1")  # a declared result the trials returned keeps "Mean of"
    assert h.record()["panels"][0]["refs"][-1]["label"].startswith("Mean of Growth at Size = 1")


def test_the_run_that_only_draws_failed_figures_serves_fis_record_too(tmp_path):
    # review: FI_FIGURE_FALLBACK=only draws the plan's figure with the earlier run's record; the table must serve it there too
    from core import analysis_figures as af

    class _Spy:
        served = None

        def set_derived(self, records):
            self.served = records

    spy, records = _Spy(), {"spread": _record(SPREAD)}
    af.default_figure({"id": "f1", "file": "f1.png"}, spy, str(tmp_path), None, records)  # no shows: dropped, but served first
    assert spy.served is records
