"""A plan cannot declare one result both as one each run returns and as one worked out after the runs.

A revision of a plan added results worked out after the runs (``derived``) to the results each run returns
(``metrics``) and left them in ``derived`` too. The analysis reads a derived name from its formula, so the trials' value
of that name was never read and the results left the paper. FI now refuses such a plan at the one reader of a protocol
(``core.plan.protocol_fault``), at the metric entry; every writer of the plan reads it, so the change that adds the name
is not taken and the model is told why.
"""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from core import plan
from core.engine import Engine

PROTOCOL = {
    "form": "contract", "grid": {"rate": [0.1, 0.2], "dose": [1, 2]}, "runs_per_setting": 3,
    "metrics": [{"id": "growth", "kind": "mean", "role": "result", "estimand": "the mean growth of a setting",
                 "unit": "one run"}],
    "derived": [{"id": "peak", "inputs": ["growth"], "role": "result"},
                {"id": "spread", "inputs": ["growth"], "role": "diagnostic", "formula": "growth / (rate * dose)"}],
}


def _with_metrics(*ids: str) -> dict:
    added = [{"id": i, "kind": "mean", "role": "result", "estimand": f"the mean {i} of a setting", "unit": "one run"}
             for i in ids]
    return {**PROTOCOL, "metrics": [*PROTOCOL["metrics"], *added]}


def test_a_result_in_both_lists_is_refused_at_its_metric_entry_with_the_reason() -> None:
    fixed, fault = plan.protocol_fault(_with_metrics("peak"))
    assert fixed is None and fault is not None
    assert fault.path == "protocol.metrics[2]" and fault.kind == "duplicate"
    assert "`peak`" in fault.reason and "protocol.derived" in fault.reason and "only one" in fault.reason


def test_the_same_plan_without_the_overlap_reads_as_before() -> None:
    fixed, fault = plan.protocol_fault(PROTOCOL)
    assert fault is None and fixed is not None
    fixed, fault = plan.protocol_fault(_with_metrics("height"))  # a new name that is not derived
    assert fault is None and [m["id"] for m in fixed["metrics"]] == ["growth", "height"]


def test_the_reading_leaves_out_each_metric_entry_that_is_derived_and_keeps_the_derived_results() -> None:
    reading = plan.read_protocol(_with_metrics("peak", "height", "spread"))
    assert reading.kept is not None
    assert [m["id"] for m in reading.kept["metrics"]] == ["growth", "height"]
    assert [d["id"] for d in reading.kept["derived"]] == ["peak", "spread"]
    assert reading.faults and all(f.kind == "duplicate" for f in reading.faults)


def test_a_revision_that_adds_a_derived_result_to_the_metrics_is_not_taken_and_the_model_is_told_why() -> None:
    block = yaml.safe_dump({"hypothesis": "growth rises with the rate", "protocol": PROTOCOL}, sort_keys=False)
    text = f"# Plan\n\n## The design\n\n```yaml\n{block}```\n"
    assert plan.raw_design_block(text)["protocol"]["metrics"] == PROTOCOL["metrics"]
    answer = {"protocol": {**_with_metrics("peak"), "ci_method": "percentile bootstrap over the runs of a setting"}}
    logs: list[str] = []
    fake = SimpleNamespace(_log=SimpleNamespace(warning=lambda m, *a: logs.append(m % a if a else m),
                                                info=lambda m, *a: logs.append(m % a if a else m)),
                           _test_run_returned=lambda *a, **k: None, _answer_refused=[])
    kept = Engine._keep_only_the_parts(fake, text, answer, ["metrics", "ci_method"])  # type: ignore[arg-type]
    assert kept is not None
    protocol = plan.raw_design_block(kept)["protocol"]
    assert [m["id"] for m in protocol["metrics"]] == ["growth"]  # the metrics change was not taken
    assert protocol["ci_method"] == "percentile bootstrap over the runs of a setting"  # the other asked part was
    assert any("`peak` is in `protocol.metrics`" in str(r) for r in fake._answer_refused)
