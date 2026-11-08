"""FI counts, from its own record of the trials, how often a result's reported interval holds the true value the plan gives
(``known_value``), and prints it as ``<id>_coverage`` beside the result. It does not depend on any function of the study's:
a figure of the study's that fails does not take it away, and the figures that drew are still saved. Made-up topic (a colony);
no model is called."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.test_analysis_frame import KEYS, RECORD, _design, _quest, _run
from core import trial_runner

KNOWN = {"estimate": "growth", "interval": ["g_lo", "g_hi"], "level": 0.95, "true_value": "base"}


def _metrics(known: bool = True) -> list[dict]:
    return [{"id": "growth", "kind": "mean", "role": "result", **({"known_value": KNOWN} if known else {})},
            {"id": "alive", "kind": "proportion", "role": "result"}, {"id": "wobble", "kind": "mean", "role": "diagnostic"}]


def _record_with_intervals() -> dict:
    """The first setting: 4 trials, true value 2.0 (``base``); intervals hold it in 3 of them. Others hold it in all or none."""
    record = copy.deepcopy(RECORD)
    for i, c in enumerate(record["cells"]):
        n = len(c["metrics"]["growth"]["values"])
        lo = [1.0] * n
        hi = [3.0] * n
        if i == 0:
            hi[1] = 1.5  # the second trial's interval misses 2.0
        if i == 3:
            lo = [2.5] * n  # every trial misses
        c["metrics"]["g_lo"] = {"values": lo, "trials": list(range(n))}
        c["metrics"]["g_hi"] = {"values": hi, "trials": list(range(n))}
    return record


def test_the_coverage_of_a_result_with_a_known_value_is_printed_beside_it(tmp_path: Path) -> None:
    code, printed, _record, err = _run(_quest(tmp_path, _design(metrics=_metrics()), record=_record_with_intervals()))
    assert code == 0, err
    assert printed[KEYS[0]]["growth_coverage"] == pytest.approx(0.75)
    assert printed[KEYS[1]]["growth_coverage"] == pytest.approx(1.0)  # the setting with a failed trial counts the trials that ran
    assert printed[KEYS[3]]["growth_coverage"] == pytest.approx(0.0)


def test_a_result_with_no_known_value_prints_no_coverage(tmp_path: Path) -> None:
    code, printed, _record, err = _run(_quest(tmp_path, _design(metrics=_metrics(known=False)), record=_record_with_intervals()))
    assert code == 0, err
    assert not any("coverage" in k for row in printed.values() if isinstance(row, dict) for k in row)


def test_the_coverage_of_a_result_recorded_per_compared_value_is_printed_under_each_value(tmp_path: Path) -> None:
    record = _record_with_intervals()
    for c in list(record["cells"]):
        growth = c["metrics"].pop("growth")
        n = len(growth["values"])
        for mode, hi in (("a", 3.0), ("b", 1.5)):
            cell = {**c["cell"], "mode": mode}
            record["cells"].append({"cell": cell, "key": trial_runner.cell_key(cell), "level": {"mode": mode}, "planned": c["planned"],
                                    "ok": c["ok"], "failed": c["failed"],
                                    "metrics": {"growth": growth, "g_lo": {"values": [1.0] * n, "trials": growth["trials"]},
                                                "g_hi": {"values": [hi] * n, "trials": growth["trials"]}}})
    code, printed, _record, err = _run(_quest(tmp_path, _design(metrics=_metrics()), record=record))
    assert code == 0, err
    assert printed[KEYS[1] + ",mode=a"]["growth_coverage"] == pytest.approx(1.0)
    assert printed[KEYS[1] + ",mode=b"]["growth_coverage"] == pytest.approx(0.0)


def test_a_failing_figure_does_not_take_the_coverage_or_the_other_figure_away(tmp_path: Path) -> None:
    figures = [{"id": "fig1", "file": "one.png", "shows": ["growth"], "x": "rate", "series": "dose"},
               {"id": "fig2", "file": "two.png", "shows": ["growth"], "x": "rate", "series": "dose"}]
    bodies = ("def figure_fig1(fig, table):\n    raise ValueError('cannot read it')\n\n"
              "def figure_fig2(fig, table):\n    fig.axes[0].plot([1, 2], [table.mean('growth', k) for k in table.keys()][:2])\n")
    quest = _quest(tmp_path, _design(metrics=_metrics(), figures=figures), bodies, _record_with_intervals())
    code, printed, record, _err = _run(quest)
    assert code == 1  # a figure that really fails is still a failure, and is said
    status = {o["id"]: o["status"] for o in record["figures"]}
    assert status["fig1"] == "failed" and "cannot read it" in next(o["why"] for o in record["figures"] if o["id"] == "fig1")
    assert status["fig2"] == "ok" and (quest / "figures" / "two.png").is_file()
    assert printed[KEYS[0]]["growth_coverage"] == pytest.approx(0.75)


def test_a_result_of_the_study_under_the_coverage_name_keeps_its_own_value(tmp_path: Path) -> None:
    record = _record_with_intervals()
    for c in record["cells"]:
        n = len(c["metrics"]["growth"]["values"])
        c["metrics"]["growth_coverage"] = {"values": [0.5] * n, "trials": list(range(n))}
    metrics = _metrics() + [{"id": "growth_coverage", "kind": "mean", "role": "result"}]
    code, printed, _record, err = _run(_quest(tmp_path, _design(metrics=metrics), record=record))
    assert code == 0, err
    assert printed[KEYS[0]]["growth_coverage"] == pytest.approx(0.5)  # the study's own value, not FI's count 0.75
