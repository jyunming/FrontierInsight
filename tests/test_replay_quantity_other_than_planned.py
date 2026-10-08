"""Replay of a real quest whose paper reported a quantity under the plan's name while the code computed another one.

In tests/fixtures/replay/quantity_other_than_planned/ the plan asks for three spreads as a mean absolute error
(``plan_metrics.yaml``); a second model read the final code and said, for each, that it is computed as a standard deviation
(``code_review.json``, status ``different``) and the repairs did not change that. The run's results hold the standard
deviations under longer names than the plan's, each with an interval (``result_json.json``), and the paper quotes them as
the mean absolute error (``paper_excerpt.md``). No model is called."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from core import code_review as cr
from core import quality_report
from core import record_claims
from tests.test_code_review import _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "quantity_other_than_planned"
PLANNED = {m["id"] for m in yaml.safe_load((FIXTURE / "plan_metrics.yaml").read_text(encoding="utf-8"))}
HEADLINE_VALUE = 4824.8174867929165  # a value of one of them that the paper quotes as 4824.8


def _keys(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(str(k))
            out |= _keys(v)
    elif isinstance(node, list):
        for v in node:
            out |= _keys(v)
    return out


def _left_out_keys() -> set[str]:
    """The keys of the results that carry one of the plan's three quantities."""
    result = _result()
    return set().union(*(cr.named_in_results(name, result) for name in PLANNED))


def _result() -> dict[str, Any]:
    return json.loads((FIXTURE / "result_json.json").read_text(encoding="utf-8"))


def _values_of(node: Any) -> list[float]:
    if isinstance(node, dict):
        return [x for v in node.values() for x in _values_of(v)]
    if isinstance(node, list):
        return [x for v in node for x in _values_of(v)]
    return [float(node)] if isinstance(node, (int, float)) and not isinstance(node, bool) else []


def _quest(tmp_path: Path):
    """A quest whose code-review record is the real one's findings, for the code that is on disk."""
    engine = _engine(tmp_path)
    real = json.loads((FIXTURE / "code_review.json").read_text(encoding="utf-8"))
    assert real["final"] is True and real["converged"] is False
    record = dict(real)
    record["version"] = cr.version(engine.quest_root / "code")
    record["read_version"] = record["version"]
    cr.save(engine.quest_root, record)
    return engine


def _run(engine: Any, result_json: Any) -> Any:
    from core.engine import ExecutionResult

    out = ExecutionResult(0, "RESULT_JSON: " + json.dumps(result_json) + "\n", "", 1.0, False)
    engine._trial_mode = False  # the checks of the trial record are not this test's; the reader's findings are
    return asyncio.run(engine._results_match_run({"iteration": 0, "exec_reflect_iter": 0}, out, None, True))


def test_the_evidence_is_what_the_real_quest_held() -> None:
    plan_metrics = yaml.safe_load((FIXTURE / "plan_metrics.yaml").read_text(encoding="utf-8"))
    assert {m["id"] for m in plan_metrics} == PLANNED
    assert all("mean absolute" in m["estimand"].lower() for m in plan_metrics)
    real = json.loads((FIXTURE / "code_review.json").read_text(encoding="utf-8"))
    different = {p["name"] for p in real["problems"] if p.get("status") == "different" and p.get("kind") in cr.QUANTITY_KINDS}
    assert PLANNED <= different
    assert "4824.8" in (FIXTURE / "paper_excerpt.md").read_text(encoding="utf-8")
    assert "4824.8" in (FIXTURE / "paper_excerpt.md").read_text(encoding="utf-8")
    assert any(abs(v - HEADLINE_VALUE) < 1e-9 for v in _values_of(_result()))


def test_before_the_fix_nothing_in_the_results_was_the_plans_quantity_by_name() -> None:
    # The old reading was by exact name: only one of the three is a key of the results as the plan spells it, and its
    # interval is no key of that name, so two spreads and every interval could not be left out.
    held = [n for n in sorted(PLANNED) if cr.in_results(n, _result())]
    assert len(held) == 1 and held[0] in _keys(_result())


def test_the_plans_quantities_are_found_under_the_names_the_results_use() -> None:
    result = _result()
    keys = _keys(result)
    for name in PLANNED:
        found = cr.named_in_results(name, result)
        assert found <= keys and len(found) == 2, (name, found)  # the quantity and its interval
        assert any("bootstrap" in k for k in found) and any("bootstrap" not in k for k in found)
    # The plan's longest name is found under a shorter key of the results (it is a longer form of it).
    longest = max(PLANNED, key=len)
    assert any(len(k) < len(longest) for k in cr.named_in_results(longest, result))
    # What is not one of the three stays: a bias or a coverage is never taken for a spread.
    assert not any("bias" in k or "coverage" in k for k in _left_out_keys())


def test_the_open_findings_of_the_version_that_ran_name_the_quantities(tmp_path: Path) -> None:
    engine = _quest(tmp_path)
    found = cr.differing_quantities(engine.quest_root, engine.quest_root / "code")
    assert {f["name"] for f in found} == PLANNED and len(found) == 3
    assert all(f["status"] == cr.DIFFERENT for f in found)
    # Not for other code: the record is for the version that was read.
    (engine.quest_root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')  # changed\n", encoding="utf-8")
    assert cr.differing_quantities(engine.quest_root, engine.quest_root / "code") == []


def test_the_quantities_are_left_out_of_the_results_and_the_writer_is_told(tmp_path: Path) -> None:
    engine = _quest(tmp_path)
    _run(engine, _result())
    record = json.loads((engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    left = set(record["removed_quantities"])
    assert left == _left_out_keys() and len(left) == 6
    assert not any("bias" in k or "coverage" in k for k in left)
    assert any(abs(v - HEADLINE_VALUE) < 1e-9 for values in record["removed_values"].values() for v in values)
    assert {c["quantity"] for c in record["checks"]} == PLANNED
    note = engine._run_checks_note()
    assert all(f"`{name}`" in note for name in PLANNED) and "Do NOT report" in note and "limitations" in note
    # The analysis and the paper are given the results without them.
    stripped = engine._without_typed_results(_result())
    assert not (_keys(stripped) & left) and any("bias" in k for k in _keys(stripped))
    assert _keys(_result()) - left == _keys(stripped)


def test_the_paper_that_quotes_the_left_out_numbers_is_flagged(tmp_path: Path) -> None:
    engine = _quest(tmp_path)
    _run(engine, _result())
    paper = (FIXTURE / "paper_excerpt.md").read_text(encoding="utf-8")
    hits = engine._record_contradiction_hits(paper, {"iteration": 0})
    left = _left_out_keys()
    assert any("4824.8" in h and any(f"`{k}`" in h for k in left) for h in hits), hits
    assert all(h.startswith(record_claims.HIT + ":") for h in hits)
    # A paper that leaves them out, and says so, is not flagged; nor is a number that is not theirs.
    clean = ("The coverage stayed near 0.9467 in every setting. The spread of the estimates is left out of this paper: the "
             "code computes it differently from the plan's definition.")
    assert engine._record_contradiction_hits(clean, {"iteration": 0}) == []


def test_a_number_a_kept_quantity_also_holds_is_not_taken_for_a_left_out_one() -> None:
    result = {"a_spread": [3.25, 7.5], "a_other": 3.25}
    from core.result_checks import _values_left_out

    assert _values_left_out(result, {"a_spread"}) == {"a_spread": [7.5]}
    assert record_claims.quoted_left_out("We found 3.25 and 1.2.", {"a_spread": [3.25]}) != []  # no kept holder given here
    assert record_claims.quoted_left_out("We found 3.25.", {}) == []


def test_the_quality_report_says_what_was_left_out_and_why(tmp_path: Path) -> None:
    engine = _quest(tmp_path)
    _run(engine, _result())
    report = quality_report.build(engine.quest_root, running=False)
    assert {x["quantity"] for x in report["left_out_of_paper"]} == PLANNED
    text = quality_report.render(report)
    first = sorted(PLANNED)[0]
    assert "## Left out of the paper" in text and f"`{first}`" in text and "standard deviation" in text
    statuses = {p["status"] for s in report["stages"] if s["stage"] == "code" for p in s["problems"]["items"]
                if first in p["text"]}
    assert statuses == {"left out"}
