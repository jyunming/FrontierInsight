"""A figure body is told how many panels FI makes for it, and a body that reaches past them is told the count (core/run_checks.py,
core/analysis_figures.py). Made-up topic (a colony); a body that indexes `fig.axes` once for each value of a setting, while the
plan declares one panel, is the recorded fault."""

from __future__ import annotations

import re
import tempfile
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

from core import analysis_figures as af
from core import analysis_frame
from core import run_checks

BODIES = ["figure_growth", "figure_split", "derive_rate"]


class _Table:
    pass


def _namespace(source: str, name: str) -> Any:
    ns: dict[str, Any] = {}
    exec(source, ns)
    return ns[name]


def test_the_request_states_the_panels_under_each_figure_body() -> None:
    reply = run_checks._frame_reply(False, BODIES, {"figure_growth": 1, "figure_split": 3})
    lines = {ln.split("`")[1]: ln for ln in reply.splitlines() if ln.startswith("   - `")}
    assert "ONE panel" in lines["def figure_growth(fig, table)"] and "fig.axes[0]" in lines["def figure_growth(fig, table)"]
    assert "exactly 3 panels" in lines["def figure_split(fig, table)"] and "fig.axes[2]" in lines["def figure_split(fig, table)"]
    assert "panel" not in lines["def derive_rate(table)"], "only a figure body is told about panels"


def test_without_the_panels_the_request_is_as_it_was() -> None:
    plain = run_checks._frame_reply(False, BODIES)
    with_panels = run_checks._frame_reply(False, BODIES, {"figure_growth": 2})
    assert plain == run_checks._frame_reply(False, BODIES, None) == run_checks._frame_reply(False, BODIES, {})
    stripped = re.sub(r" FI makes (ONE panel|exactly \d+ panels) for it:[^\n]*", "", with_panels)
    assert stripped == plain, "the panel statement is the only difference"


def test_the_panels_come_from_the_plans_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = {"figures": {"growth": {"panels": 1}, "split": {"panels": 2}, "other": {"panels": 3}}}
    monkeypatch.setattr(analysis_frame, "plan_of", lambda design: plan)
    mixin = run_checks.RunChecksMixin.__new__(run_checks.RunChecksMixin)
    assert mixin._figure_panels({"design": {}}, BODIES) == {"figure_growth": 1, "figure_split": 2}
    assert mixin._figure_panels({"design": {}}, []) is None

    def broken(design: Any) -> Any:
        raise KeyError("protocol")

    monkeypatch.setattr(analysis_frame, "plan_of", broken)
    assert mixin._figure_panels({"design": {}}, BODIES) is None, "a plan FI cannot read gives the request as before"


def test_a_body_that_indexes_past_the_panels_is_told_how_many_there_are() -> None:
    body = _namespace("def figure_growth(fig, table):\n    for i in range(3):\n        fig.axes[i].plot([1, 2], [1, 2])\n",
                      "figure_growth")
    out = af._draw_one({"id": "growth", "file": "growth.png", "panels": 1}, body, _Table(), tempfile.mkdtemp())
    assert out["status"] == "failed"
    assert "IndexError" in out["why"] and "FI made 1 panel(s)" in out["why"] and "0 to 0" in out["why"], out["why"]


def test_a_body_that_stays_within_the_panels_draws_as_before(tmp_path: Any) -> None:
    body = _namespace("def figure_split(fig, table):\n    for i in range(3):\n        fig.panel(i)\n"
                      "        fig.axes[i].plot([1, 2], [i, 2])\n    assert fig.axes[-1] is fig.axes[2] and len(fig.axes) == 3\n",
                      "figure_split")
    out = af._draw_one({"id": "split", "file": "split.png", "panels": 3}, body, _Table(), str(tmp_path))
    assert out["status"] == "ok", out
    assert (tmp_path / "split.png").exists()
    handle = af._figure_handle(af._new_figure(2)[2], {"id": "x"}, "x.png")
    with pytest.raises(IndexError):
        handle.axes[5]
    assert isinstance(handle.axes[0:1], list) and len(handle.axes[0:1]) == 1
