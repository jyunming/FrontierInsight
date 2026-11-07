"""A value reported on a declared bound while FI's record of that setting's trials lies beyond it is sent back.

The record, not the script's code, shows a pinned number: ``core.trial_runner.pinned_to_bound``. A value that only differs
from the record mean, a 0/1 quantity, a rounding, and a cap inside the trial (the record is already capped) are not acted on.
"""
from __future__ import annotations

import json
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
    from core import disclosure

    row = {"text": v.disclosure(), "short": v.disclosure_short()}
    edge = tmp_path / ".fi" / "edge_everywhere.json"
    edge.write_text(json.dumps({"results": [row]}), encoding="utf-8")  # no stamp: not found for this run
    assert not [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.AT_BOUND]
    edge.write_text(json.dumps({"results": [row], "of": {**disclosure.edge_stamp(tmp_path), "ledger": "another"}}),
                    encoding="utf-8")  # found for another run
    assert not [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.AT_BOUND]
    edge.write_text(json.dumps({"results": [row], "of": disclosure.edge_stamp(tmp_path)}), encoding="utf-8")
    (item,) = [i for i in paper_facts.required_items(tmp_path) if i.kind == paper_facts.AT_BOUND]
    assert item.sentence.startswith("FI found that rel equals its bound 1 in 3 settings")
    paper = paper_facts.render(paper_facts.place_blocks("# T\n\n## Limitations\nOwn words.\n", tmp_path), tmp_path).text
    assert "rel equals its bound 1 in 3 settings" in paper


def _printed(text: str, record_values: list[float]):
    from core.trial_runner import parse_printed

    result = json.loads(text)
    return parse_printed(text, result), _record({"size=10": record_values, "size=20": [0.5] * len(record_values)})


def _pinned_by(text: str, values: list[float]):
    printed, record = _printed(text, values)
    return pinned_to_bound(printed, PROTOCOL, plausibility.parse_assertions({"result_assertions": ASSERT}), record)


def test_rounding_comes_from_the_printed_places():
    both = '{"by_setting": {"size=10": {"score": %s}, "size=20": {"score": 0.5}}}'
    assert _pinned_by(both % "1.0", [1.04] * 5) == []  # 1.04 printed to one place is 1.0
    found = _pinned_by(both % "1.000", [1.04] * 5)  # printed to three places, it is not
    assert [v.kind for v in found] == ["pinned"]


def test_text_that_is_not_the_result_is_not_used():
    from core.trial_runner import parse_printed

    assert parse_printed('{"a": 1.000}', {"a": 2.0}) == {"a": 2.0} and parse_printed("", {"a": 2.0}) == {"a": 2.0}


def test_a_median_under_its_own_key_is_not_the_mean():
    text = '{"by_setting": {"size=10": {"median": {"score": 1.0}}, "size=20": {"median": {"score": 0.5}}}}'
    assert _pinned_by(text, [1.0, 1.0, 1.0, 1.0, 2.0]) == []
    own = '{"by_setting": {"size=10": {"score": 1.0}, "size=20": {"score": 0.5}}}'
    assert [v.kind for v in _pinned_by(own, [1.0, 1.0, 1.0, 1.0, 2.0])] == ["pinned"]  # known positive: its own key


def test_a_reference_metric_is_not_held_to_the_record():
    protocol = {**PROTOCOL, "metrics": [{"id": "score", "kind": "mean", "role": "reference"}]}
    found = pinned_to_bound(_result(1.0, 0.5), protocol, plausibility.parse_assertions({"result_assertions": ASSERT}),
                            _record({"size=10": [1.3] * 5, "size=20": [0.5] * 5}))
    assert found == []


def test_only_the_record_of_this_run_is_used(tmp_path):
    import hashlib

    from core import trial_runner as tr

    values = {"score": 1.2}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
    (tmp_path / "raw").mkdir()
    (tmp_path / ".fi" / "trials").mkdir(parents=True)
    (tmp_path / "raw" / "ledger.jsonl").write_text(json.dumps(
        {"event": "trial", "status": "ok", "cell": "size=10", "trial": 0, "values_sha256": digest}) + "\n", encoding="utf-8")
    (tmp_path / "raw" / "trials.json").write_text("{}", encoding="utf-8")
    record = {"key": "k", "files": tr._file_hashes(tmp_path), "cells": [
        {"key": "size=10", "rows": [{"status": "ok", "trial": 0, "values": values}]}]}
    path = tmp_path / ".fi" / "trials" / "run.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert tr.record_of_this_run(tmp_path, False, "k") is None  # a run of a single script has no trial record of its own
    assert tr.record_of_this_run(tmp_path, True, "k")["size=10"]["score"]  # known positive
    (tmp_path / "raw" / "ledger.jsonl").write_text("changed\n", encoding="utf-8")
    assert tr.record_of_this_run(tmp_path, True, "k") is None  # the ledger is no longer the one the record was written beside


def test_a_record_with_another_runs_key_is_not_this_runs(tmp_path):
    import hashlib

    from core import trial_runner as tr

    values = {"score": 1.2}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()
    (tmp_path / "raw").mkdir()
    (tmp_path / ".fi" / "trials").mkdir(parents=True)
    (tmp_path / "raw" / "ledger.jsonl").write_text(json.dumps(
        {"event": "trial", "status": "ok", "cell": "size=10", "trial": 0, "values_sha256": digest}) + "\n", encoding="utf-8")
    (tmp_path / "raw" / "trials.json").write_text("{}", encoding="utf-8")
    record = {"key": "another", "files": tr._file_hashes(tmp_path), "cells": [
        {"key": "size=10", "rows": [{"status": "ok", "trial": 0, "values": values}]}]}
    (tmp_path / ".fi" / "trials" / "run.json").write_text(json.dumps(record), encoding="utf-8")
    # the ledger and summary match the record and it holds a trial; the key is not the one this run's trial runner computed
    assert tr.record_of_this_run(tmp_path, True, "another")["size=10"]["score"]  # known positive
    assert tr.record_of_this_run(tmp_path, True, "this run") is None
    assert tr.record_of_this_run(tmp_path, True, "") is None


def test_a_statistic_named_anywhere_on_the_path_is_not_the_mean_and_a_neutral_key_is():
    each = '{"%s": {"size=10": {"score": 1.0}, "size=20": {"score": 0.5}}}'
    for ancestor in ("median", "ci", "bootstrap_interval"):
        assert _pinned_by(each % ancestor, [1.0, 1.0, 1.0, 1.0, 2.0]) == [], ancestor
    neutral = '{"size=10": {"summary": {"score": 1.0}}, "size=20": {"summary": {"score": 0.5}}}'
    assert [v.kind for v in _pinned_by(neutral, [1.0, 1.0, 1.0, 1.0, 2.0])] == ["pinned"]


def test_the_printed_text_reaches_the_next_node_of_a_compiled_graph():
    import asyncio

    from langgraph.graph import END, START, StateGraph

    from core.engine import QuestState

    seen = {}

    def first(state):
        return {"result_json": {"a": 1.0}, "result_json_text": '{"a": 1.000}'}

    def second(state):
        seen["text"] = state.get("result_json_text")
        return {}

    graph = StateGraph(QuestState)
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    asyncio.run(graph.compile().ainvoke({}))
    assert seen["text"] == '{"a": 1.000}'
