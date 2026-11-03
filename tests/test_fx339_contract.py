"""FI computes per-parameter coverage and bias and across-trial statistics from its own record of the trials, and holds the finding
test writer to the names the plan declares. Neutral made-up plan: two estimates with true values, a coverage and a correlation."""

from __future__ import annotations

import json
import math
import string
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import finding_repair, frame_table, skeleton

PROTOCOL = {
    "form": "contract", "grid": {"noise": [0.1, 0.2]}, "in_each_run": {"channel": [0, 1]}, "runs_per_setting": 20,
    "constants": {"truth_a": 2.0, "truth_b": 5.0},
    "metrics": [
        {"id": "est_a", "kind": "mean", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_a", "estimate": "est_a", "standard_error": "se_a", "interval": ["lo_a", "hi_a"], "level": 0.95}},
        {"id": "est_b", "kind": "mean", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_b", "estimate": "est_b", "standard_error": "se_b", "interval": ["lo_b", "hi_b"], "level": 0.95}},
        {"id": "cover", "kind": "proportion", "role": "physical_bound", "returned": "afterwards"},
        {"id": "corr_ab", "kind": "mean", "role": "result", "returned": "afterwards"}]}


def _metric(values: list[float], trials: list[int] | None = None) -> dict[str, Any]:
    return {"values": list(values), "trials": list(range(len(values))) if trials is None else list(trials)}


def _cell(noise: float, est_a: list[float], est_b: list[float], width: float = 1.0, trials: list[int] | None = None) -> dict[str, Any]:
    cell = {"noise": noise}
    m = {"est_a": _metric(est_a, trials), "est_b": _metric(est_b, trials),
         "lo_a": _metric([v - width for v in est_a], trials), "hi_a": _metric([v + width for v in est_a], trials),
         "lo_b": _metric([v - width for v in est_b], trials), "hi_b": _metric([v + width for v in est_b], trials)}
    return {"cell": cell, "key": f"noise={noise}", "planned": len(est_a), "ok": len(est_a), "failed": 0, "metrics": m}


def _table(cells: list[dict[str, Any]], declared: tuple[str, ...] = ("est_a", "est_b", "lo_a", "hi_a", "lo_b", "hi_b"), **kw: Any) -> Any:
    ns: dict[str, Any] = {"fi_table": frame_table}
    sys.modules["fi_table"] = frame_table
    exec(compile(skeleton.RUNTIME, "fi_frame.py", "exec"), ns)  # noqa: S102 -- FI's own runtime text
    known = {"est_a": {"interval": ["lo_a", "hi_a"], "truth": 2.0}, "est_b": {"interval": ["lo_b", "hi_b"], "truth": 5.0}}
    return ns["Table"](cells, list(declared), known=known, **kw)


# ---- A1: the finding test writer is given the declared names -------------------------------------------------------------------------

def _root(tmp_path: Path) -> Path:
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps({"protocol": PROTOCOL}), encoding="utf-8")
    return tmp_path


def test_the_declared_block_names_every_result_and_what_fi_counts(tmp_path: Path) -> None:
    text = finding_repair.declared_block(_root(tmp_path))
    for name in ("est_a", "est_b", "cover", "corr_ab", "channel", "est_a_coverage", "est_b_bias"):
        assert name in text, name
    assert finding_repair.declared_block(tmp_path / "nowhere") == "(none stated)"


def test_the_rendered_finding_test_request_holds_the_declared_names(tmp_path: Path) -> None:
    root = _root(tmp_path)
    code = root / "code"
    code.mkdir()
    (code / "sim.py").write_text("def run(x):\n    return x\n", encoding="utf-8")
    template = (Path(__file__).parent.parent / "agents" / "code_finding_test.md").read_text(encoding="utf-8")
    env = SimpleNamespace(root=root, code=code, package="", topic="a made-up topic", equations="[]", scrub=lambda t: t,
                          prompts={"code_finding_test": string.Template(template)})
    issue = {"id": "F-1", "file": "sim.py", "function": "run", "statement": "the coverage of each parameter is wrong"}
    prompt = finding_repair._test_prompt(env, issue, "sim.py", "")
    assert "`est_a`" in prompt and "`corr_ab`" in prompt and "may read only these names" in prompt
    assert "$declared" not in prompt


# ---- A2: bias and coverage per result, counted by FI ------------------------------------------------------------------------------------

def test_bias_is_the_mean_estimate_minus_the_true_value_and_known_lists_the_results() -> None:
    t = _table([_cell(0.1, [1.5, 2.5, 4.0], [5.0, 5.0, 5.0])])
    assert t.bias("est_a", "noise=0.1") == pytest.approx(8 / 3 - 2.0)
    assert t.bias("est_b", "noise=0.1") == pytest.approx(0.0)
    assert t.known() == ["est_a", "est_b"]
    assert t.coverage("est_a", "noise=0.1") == [1.0, 1.0, 0.0]
    with pytest.raises(type(t).__init__.__globals__["UnknownResult"]):
        t.bias("lo_a", "noise=0.1")


def test_the_printed_result_has_a_bias_beside_each_coverage() -> None:
    from core import analysis_frame

    src = analysis_frame.EXPERIMENT
    assert "_bias" in src and "_coverage" in src
