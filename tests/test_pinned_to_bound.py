"""A value reported on a declared bound while FI's record of that setting's trials lies beyond it is sent back.

The record, not the script's code, shows a pinned number: ``core.trial_runner.pinned_to_bound``. A value that only differs
from the record mean, a 0/1 quantity, a rounding, and a cap inside the trial (the record is already capped) are not acted on.
"""
from __future__ import annotations

from collections import Counter

from core import plausibility
from core.engine import _assertion_violations
from core.trial_runner import pinned_to_bound

PROTOCOL = {"grid": {"size": [10, 20]}, "runs_per_setting": 5,
            "metrics": [{"id": "score", "kind": "mean", "estimand": "mean score"}]}
ASSERT = [{"path": "score", "min": 0.0, "max": 1.0}]


def _record(per_setting: dict[str, list[float]]) -> dict[str, dict[str, Counter]]:
    return {key: {"score": Counter(repr(v) for v in values)} for key, values in per_setting.items()}


def _run(result: dict, per_setting: dict[str, list[float]], assertions=ASSERT):
    return pinned_to_bound(result, PROTOCOL, plausibility.parse_assertions({"result_assertions": assertions}),
                           _record(per_setting))


def _result(a: float, b: float) -> dict:
    return {"by_setting": {"size=10": {"score": a}, "size=20": {"score": b}}}


def test_planted_pinned_value_is_a_violation():
    found = _run(_result(1.0, 0.5), {"size=10": [1.2, 1.3, 1.24, 1.26, 1.2], "size=20": [0.5] * 5})
    assert [v.kind for v in found] == ["pinned"]
    text = found[0].describe()
    assert "size=10" in text and "on its bound 1" in text and "5 recorded trials" in text and "1.24" in text
    assert text.endswith("report the value the trials give")


def test_planted_pinned_value_is_sent_back_by_the_engine_check():
    design = {"result_assertions": ASSERT, "protocol": PROTOCOL}
    state = {"design": design, "result_json": _result(1.0, 0.5),
             "_trial_record": _record({"size=10": [1.2, 1.3, 1.24, 1.26, 1.2], "size=20": [0.5] * 5})}
    assert [v.kind for v in _assertion_violations(state)] == ["pinned"]
    state["_trial_record"] = _record({"size=10": [1.0] * 5, "size=20": [0.5] * 5})
    assert _assertion_violations(state) == []


def test_indicator_with_mean_one_is_honest():
    assert _run(_result(1.0, 0.5), {"size=10": [1.0] * 5, "size=20": [0.5] * 5}) == []


def test_a_value_that_differs_but_is_not_on_a_bound_is_left_alone():
    assert _run(_result(0.7, 0.5), {"size=10": [0.9] * 5, "size=20": [0.5] * 5}) == []


def test_rounding_of_the_record_mean_is_not_pinning():
    assert _run(_result(1.0, 0.5), {"size=10": [1.00004] * 5, "size=20": [0.5] * 5}) == []


def test_a_cap_inside_the_trial_is_not_visible_to_the_record():
    # the trials already returned the capped values: the record holds them, and the reported mean equals the record
    assert _run(_result(1.0, 0.5), {"size=10": [min(x, 1.0) for x in (1.2, 1.3, 1.24, 1.26, 1.2)], "size=20": [0.5] * 5}) == []


def test_lower_bound_is_pinned_the_same_way():
    found = _run(_result(0.5, 0.0), {"size=10": [0.5] * 5, "size=20": [-0.2, -0.1, -0.3, -0.2, -0.2]})
    assert [v.kind for v in found] == ["pinned"] and "size=20" in found[0].describe()


def test_no_record_or_no_assertion_says_nothing():
    assert pinned_to_bound(_result(1.0, 0.5), PROTOCOL, [], _record({"size=10": [1.3] * 5})) == []
    assert pinned_to_bound(_result(1.0, 0.5), PROTOCOL, plausibility.parse_assertions({"result_assertions": ASSERT}), None) == []


def test_a_value_on_a_bound_in_several_settings_is_said_in_the_paper(tmp_path):
    import json

    from core import paper_facts

    v = plausibility.Violation("rel", 1.0, plausibility.parse_assertions({"result_assertions": [
        {"path": "rel", "min": 0, "max": 1}]})[0], kind="at_bound", paths=("a.rel", "b.rel", "c.rel"))
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "edge_everywhere.json").write_text(
        json.dumps({"results": [{"text": v.disclosure(), "short": v.disclosure_short()}]}), encoding="utf-8")
    (item,) = [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.AT_BOUND]
    assert item.sentence.startswith("FI found that rel equals its bound 1 in 3 settings")
    paper = paper_facts.render(paper_facts.place_blocks("# T\n\n## Limitations\nOwn words.\n", tmp_path), tmp_path).text
    assert "rel equals its bound 1 in 3 settings" in paper
