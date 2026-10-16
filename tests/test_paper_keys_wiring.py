"""FI fills the writer's result keys, finds the numbers the writer typed, sends them back once and then takes them out.

The engine test runs write -> review -> rewrite -> review with a fake model on the s2 kinetics trial record."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import frozen_protocol, paper_facts, paper_patch, paper_values, quality_report

FIXTURES = Path(__file__).parent / "fixtures" / "paper_values"
KEY_SD = "{{v:Ea_rel_bias:sd @ noise_sigma=0.01, n_points=all | %}}"
KEY_MISSING = "{{v:no_such_metric @ noise_sigma=all, n_points=all}}"
TYPED = "0.7777"
PROTOCOL = {
    "grid": {"noise_sigma": [0.005, 0.01, 0.02, 0.05, 0.1], "n_points": [5, 10, 20, 40]},
    "thresholds": {"nominal_coverage": 0.95},
    "constants": {"z": 1.96},
    "metrics": [{"name": "Ea_rel_bias"}, {"name": "coverage_Ea"}],
}


def _quest(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "s2_kinetics_trials.json", root / "raw" / "trials.json")
    (root / ".fi").mkdir(exist_ok=True)
    frozen_protocol.freeze(root, PROTOCOL, approved_by="test", source="test")
    return root


def _engine(root: Path, **engine: Any) -> Any:
    from tests.test_disclosure import _engine as make

    eng = make(root)
    for key, value in engine.items():
        setattr(eng.config.engine, key, value)
    eng.config.topic = "Estimate an activation energy at 1.96 sigma."
    return eng


def test_a_raw_key_reaching_the_fact_renderer_never_becomes_a_fact_marker() -> None:
    out = paper_facts.render("A value {{v:x @ a=1}} here.", "/nonexistent-quest-root").text
    assert "fi:fact-v" not in out and "fact-v" not in out


def test_the_marker_round_trips_to_the_writers_key_and_to_plain_text() -> None:
    marked = f"SD <!-- fi:v {KEY_SD} -->0.8244%<!-- /fi:v --> here."
    assert paper_facts.to_placeholders(marked) == f"SD {KEY_SD} here."
    assert paper_facts.without_marks(marked) == "SD 0.8244% here."


def test_fi_fills_a_key_from_the_record_and_a_typed_value_is_a_hit(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    rec = paper_values.records_of(root, PROTOCOL)
    filled, _values = paper_values.fill_values(f"SD was {KEY_SD}, bias {TYPED}.", rec, None)
    assert "0.8244%" in filled and "fi:v" in filled
    hits = paper_facts.check(f"SD was {KEY_SD}, bias {TYPED}.", root, topic="t", keyed=True, protocol=PROTOCOL)
    assert any(h.startswith("paper_record_mismatch") and TYPED in h for h in hits)
    assert not any("0.8244" in h for h in hits)


def test_a_key_fi_cannot_fill_is_a_hit_and_reads_not_recorded(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    rendered = paper_facts.render(f"It was {KEY_MISSING}.", root, records=paper_values.records_of(root, PROTOCOL)).text
    assert "not recorded" in rendered
    hits = paper_facts.check(rendered, root, topic="t", keyed=True, protocol=PROTOCOL)
    assert any("paper_record_mismatch" in h and "no_such_metric" in h for h in hits), hits


def test_an_older_paper_is_not_held_to_the_rule(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    hits = paper_facts.check(f"The bias was {TYPED}.", root, topic="t", keyed=False, protocol=PROTOCOL)
    assert not any(TYPED in h for h in hits)


def test_the_passage_editor_takes_a_quoted_number_hit_and_not_an_unquoted_one() -> None:
    paper = "# T\n\n## Results\nThe bias was 0.7777 at noise 0.01. Another sentence stands.\n\n## Discussion\nOk.\n"
    quoted = ("paper_record_mismatch: you typed 0.7777, which is not a number of the plan or the topic. "
              "— “The bias was 0.7777 at noise 0.01.”")
    passages, reason = paper_patch.plan_passages(paper, len(paper), hits=[("paper_record_mismatch", quoted)], claims=[],
                                                 captions=[])
    assert not reason and passages and passages[0].located
    _none, reason = paper_patch.plan_passages(
        paper, len(paper), hits=[("paper_record_mismatch", "paper_record_mismatch: a further 3 typed number(s) elsewhere.")],
        claims=[], captions=[])
    assert reason


def test_keep_check_records_what_was_removed_and_why_a_paper_is_not_held(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    paper_facts.keep_check(root, [], "p", removed=[("0.7777", "The bias was 0.7777.")], not_filled="")
    kept = paper_facts.kept_check(root)
    assert kept["removed_numbers"]
    paper_facts.keep_check(root, [], "p", not_filled="older_paper")
    assert paper_facts.kept_check(root)["not_filled_by_fi"] == "older_paper"
    (root / "QUALITY.md").unlink(missing_ok=True)


def test_the_writer_prompts_hold_the_sheet_and_the_rule(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    block = paper_values.writer_block(paper_values.records_of(root, PROTOCOL))
    assert "{{v:" in block and "Ea_rel_bias" in block
    assert "not available" in block.lower()
    assert paper_values.writer_block(None) == ""
    agents = Path(__file__).parent.parent / "agents"
    assert "$values_sheet" in (agents / "write_patch.md").read_text(encoding="utf-8")


def test_fi_fills_the_keys_sends_a_typed_result_back_once_and_then_takes_it_out(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    eng = _engine(root, max_iterations=2)
    calls: list[tuple[str, str]] = []
    paper = (f"# Kinetics\n\n## Abstract\nWe study the bias.\n\n## Methods\nThe run took {{{{run_time}}}}.\n\n## Results\nThe SD of the bias was {KEY_SD} at noise 0.01, "
             f"and the mean bias was {TYPED} in the main grid. The nominal level is 95%.\n\n## Discussion\nIt is small.\n")

    async def chat(prompt: str, *, node: Any = None) -> str:
        calls.append((str(node), prompt))
        if node == "write":
            return paper
        if node == "write.patch":
            return json.dumps([{"find": "in the main grid", "replace": "in the main grid again"}])
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = chat  # type: ignore[assignment]
    state: dict[str, Any] = {"topic": eng.config.topic, "title": "t", "iteration": 0,
                             "exec_result": {"returncode": 0, "duration_s": 3.0}}
    # Round 0: the writer is given the sheet, FI fills the key, the typed result is sent back (a round is left).
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    assert out["paper_keys"] is True
    assert any(n == "write" and "Results you can place" in p for n, p in calls)
    state.update(out)
    values = json.loads((root / ".fi" / "paper_values.json").read_text(encoding="utf-8"))
    assert KEY_SD[4:-2] in json.dumps(values) and "0.8244%" in json.dumps(values)
    shown = Path(out["paper_md"]).read_text(encoding="utf-8")
    assert "0.8244%" in paper_facts.without_marks(shown) and TYPED in shown
    review = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    hits = [h for h in review["review"]["must_flag_hits"] if h.startswith("paper_record_mismatch")]
    assert hits and any(TYPED in h for h in hits)
    assert review["iteration"] == 1 and "[number removed]" not in Path(out["paper_md"]).read_text(encoding="utf-8")
    state.update(review)
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    # Round 1: the rewrite is an edit of the sentence; the writer keeps the number; no round is left: FI takes it out.
    state.update(asyncio.run(eng._node_write(state)))  # type: ignore[arg-type]
    assert any(n == "write.patch" and TYPED in p and "Results you can place" in p for n, p in calls)
    review = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    final = Path(state["paper_md"]).read_text(encoding="utf-8")
    assert "[number removed]" in final and TYPED not in final and "{{v:" not in paper_facts.without_marks(final)
    assert "0.8244%" in paper_facts.without_marks(final)
    kept = paper_facts.kept_check(root)
    assert any(r["number"] == TYPED for r in kept["removed_numbers"])
    state.update(review)
    assert eng._route_after_review(state) not in ("re_execute", "stuck")  # type: ignore[arg-type]
    report = quality_report.render(quality_report.build(root))
    assert "Numbers taken out of the paper" in report and TYPED in report
