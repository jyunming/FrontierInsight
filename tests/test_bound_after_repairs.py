"""What happens to a value beyond its declared bound once the repairs have run out (core/result_checks.py ``bound_breaks``).

A value beyond the bound the plan flags as physical (``range`` with ``physical: true``) is left out of the results, and the
paper says the part was not done; a value beyond a bound the plan only expected is kept, printed with FI's mark beside each
number FI fills in for it, in its table's title, the Limitations and QUALITY.md. A value on a bound, and a quantity that is
missing, are not changed. The study never stops for it and no model is asked. Neutral made-up topics only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core import paper_facts as pf
from core import paper_values as pv
from core import plausibility as pl
from core import quality_report as qr
from core import result_checks as rc
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine


def _cfg(tmp_path: Path, max_iter: int = 2) -> Config:
    return Config(
        topic="bounds after repairs", title="bounds",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=max_iter, ideate_reflect=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )


def _design(physical: bool) -> dict:
    """A made-up study of a catalyst bed: ``yield_frac`` is a fraction (a physical bound of 0 to 1 when ``physical``), and
    ``drift_rate`` has a range the plan only expected."""
    return {
        "hypothesis": "h",
        "result_assertions": [{"path": "yield_frac", "min": 0.0, "max": 0.9, "reason": "plan's guess"},
                              {"path": "drift_rate", "min": 0.0, "max": 5.0, "reason": "plan's guess"}],
        "protocol": {"metrics": [
            {"id": "yield_frac", "label": "Bed yield", **({"range": {"min": 0.0, "max": 1.0, "physical": True}}
                                                          if physical else {})},
            {"id": "drift_rate", "label": "Drift rate"}]},
    }


def _state(design: dict, result: dict, *, iters: int = 2) -> dict:
    return {"exec_result": {"returncode": 0, "stdout_tail": "yield_frac printed\nother line", "stderr_tail": ""},
            "result_json": result, "code": "print('ok')", "design": design, "exec_reflect_iter": iters}


def _engine(tmp_path: Path, max_iter: int = 2) -> Engine:
    eng = Engine(_cfg(tmp_path, max_iter))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value="{}")})()
    return eng


MARK = "(unconfirmed)"  # FI's mark beside a printed value (the words of core/result_checks.py BOUND_MARK)


def _kept(quest_root: Path) -> dict:
    """The kept quantities as FI's record holds them (``kept_bound_breaks`` of needs/RESULT_CHECK.json)."""
    path = quest_root / "needs" / "RESULT_CHECK.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {r["quantity"]: r for r in record.get("kept_bound_breaks") or []}


RESULT = {"bed=1": {"yield_frac": 1.4, "drift_rate": 7.5}, "bed=2": {"yield_frac": 0.5, "drift_rate": 2.0}}


# --- rule 1: a physical bound broken -> left out ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_value_beyond_its_physical_bound_is_left_out_when_the_repairs_run_out(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    patch = await eng._node_execute_reflect(_state(_design(physical=True), RESULT))
    gone = rc.left_out_by_record(eng.quest_root)
    assert "yield_frac" in gone and "breaks its physical bound (0 to 1)" in gone["yield_frac"]
    assert "drift_rate" not in gone  # a bound the plan only expected: kept
    # the analysis no longer reads it, from the results or from the run's output
    assert "yield_frac" not in json.dumps(patch["result_json"]) and "drift_rate" in json.dumps(patch["result_json"])
    assert "yield_frac printed" not in patch["exec_result"]["stdout_tail"]
    # the paper's Limitations say the part was not done, in plain words
    items = {i.kind: i.sentence for i in pf.required_items(eng.quest_root)}
    assert "is not reported: a value of it is outside the physical bound the plan gives it (0 to 1)" in items[pf.LEFT_OUT]
    # the writer is told not to use it, and no key fills it
    assert "Do NOT report, describe or quote any value for `yield_frac`" in eng._result_checks_note()
    assert eng._client.chat.await_count == 0  # no model call


@pytest.mark.asyncio
async def test_a_value_only_on_its_physical_bound_is_kept_not_left_out(tmp_path: Path) -> None:
    """1.0 is beyond the plan's guess (0.9) but on the physical bound (1): a value on a bound is not left out."""
    eng = _engine(tmp_path)
    await eng._node_execute_reflect(_state(_design(physical=True), {"bed=1": {"yield_frac": 1.0}, "bed=2": {"yield_frac": 0.4}}))
    assert "yield_frac" not in rc.left_out_by_record(eng.quest_root)
    assert "yield_frac" in _kept(eng.quest_root)


# --- rule 2: a bound the plan only expected -> kept and marked ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_value_beyond_an_expected_bound_is_kept_and_recorded(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    patch = await eng._node_execute_reflect(_state(_design(physical=False), RESULT))
    assert patch == {}  # nothing leaves the results
    assert rc.left_out_by_record(eng.quest_root) == {}
    kept = _kept(eng.quest_root)
    assert set(kept) == {"yield_frac", "drift_rate"}
    assert kept["drift_rate"]["value"] == 7.5 and kept["drift_rate"]["setting"] == "bed=1"
    assert "is outside the range the plan expected (0 to 5)" in kept["drift_rate"]["note"]
    # the Limitations say it, with the range and the furthest value
    sentences = [i.sentence for i in pf.required_items(eng.quest_root)]
    assert any("outside the range the plan expected for it (0 to 5); the furthest is 7.5" in s for s in sentences)
    # one QUALITY.md line per quantity
    report = qr.build(eng.quest_root)
    assert {x["quantity"] for x in report["kept_outside_range"]} == {"yield_frac", "drift_rate"}
    text = qr.render(report)
    assert "## Kept outside the plan's expected range" in text and "`drift_rate`: `drift_rate` is outside" in text


def _paper_quest(tmp_path: Path, kept: bool) -> Path:
    cells = [{"cell": {"bed": b}, "key": f"bed={b}",
              "metrics": {"drift_rate": {"values": [v, v + 0.1, v - 0.1]}, "load": {"values": [1.0, 1.1, 0.9]}}}
             for b, v in ((1, 7.5), (2, 2.0))]
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "needs").mkdir()
    if kept:
        row = {"quantity": "drift_rate", "min": 0.0, "max": 5.0, "unit": "", "value": 7.5, "setting": "bed=1", "count": 1,
               "bound": "0 to 5", "note": "`drift_rate` is outside the range the plan expected (0 to 5): 7.5 at bed=1"}
        (tmp_path / "needs" / "RESULT_CHECK.json").write_text(json.dumps({"kept_bound_breaks": [row]}), encoding="utf-8")
    return tmp_path


DRAFT = ("The drift is {{v:drift_rate:mean_ci @ bed=1}} at the first bed. The load is {{v:load @ bed=1}}.\n\n"
         "- range: {{v:drift_rate @ bed=each | range}}\n")


def test_a_number_of_a_kept_quantity_is_printed_as_it_is_with_no_mark_and_a_round_trip_keeps_it(tmp_path: Path) -> None:
    q = _paper_quest(tmp_path, kept=True)
    rec = pv.records_of(q, None)
    once = pf.render(DRAFT, q, records=rec).text
    reader = pf.without_marks(once)
    assert MARK not in reader  # the reader's text carries no mark; the bound is in the records and the quality report
    assert "The load is 1.000" in reader
    again = pf.render(pf.to_placeholders(once), q, records=pv.records_of(q, None)).text
    assert again == once
    assert pv.verify_marks(once, rec).repaired == []  # FI's own check of its values agrees
    assert MARK not in pf.to_placeholders(once)  # the writer gets the key back, never the mark


def test_a_quantity_inside_its_bound_renders_byte_identical(tmp_path: Path) -> None:
    """Known positive: with no kept break the paper is what it was."""
    plain = pf.render(DRAFT, _paper_quest(tmp_path / "a", kept=False), records=pv.records_of(tmp_path / "a", None)).text
    assert MARK not in plain
    other = tmp_path / "b"
    _paper_quest(other, kept=False)
    (other / "needs" / "RESULT_CHECK.json").write_text(json.dumps({"kept_bound_breaks": []}), encoding="utf-8")
    assert pf.render(DRAFT, other, records=pv.records_of(other, None)).text == plain


def test_the_table_of_a_kept_quantity_carries_no_mark_in_its_title_or_caption(tmp_path: Path) -> None:
    from core import results_by_setting as rbs

    q = _paper_quest(tmp_path, kept=True)
    protocol = {"metrics": [{"id": "drift_rate", "label": "Drift rate"}, {"id": "load", "label": "Load"}]}
    rbs.write(q, protocol)
    block = (q / rbs.PAPER_BLOCK).read_text(encoding="utf-8")
    assert "Drift rate" in block, "the quantity's table is still there"
    for text in (block, (q / "paper" / "results_by_setting.md").read_text(encoding="utf-8")):
        assert MARK not in text and "outside the range the plan expected" not in text


# --- rules 3 and 4, and the decision itself -----------------------------------------------------------------------


def _v(path: str, value: float, lo: float, hi: float, kind: str = "out_of_range") -> pl.Violation:
    return pl.Violation(path, value, pl.Assertion(path.rsplit(".", 1)[-1], lo, hi), kind=kind)


def test_the_decision_is_per_quantity_and_ignores_missing_and_on_bound_findings() -> None:
    design = _design(physical=True)
    breaks = rc.bound_breaks([_v("bed=1.yield_frac", 1.4, 0, 0.9), _v("bed=2.yield_frac", 0.95, 0, 0.9),
                              _v("bed=1.drift_rate", 7.5, 0, 5), _v("bed=2.drift_rate", 6.0, 0, 5),
                              _v("load", 0.0, 0, 1, kind="missing"), _v("bed=3.drift_rate", 5.0, 0, 5, kind="at_bound")],
                             design)
    by = {b.quantity: b for b in breaks}
    assert set(by) == {"yield_frac", "drift_rate"}
    assert by["yield_frac"].leave_out and by["yield_frac"].count == 1 and by["yield_frac"].value == 1.4
    assert not by["drift_rate"].leave_out and by["drift_rate"].count == 2 and by["drift_rate"].value == 7.5
    assert rc.bound_breaks([_v("load", 0.0, 0, 1, kind="missing")], design) == []  # a missing quantity is unchanged


@pytest.mark.asyncio
async def test_a_value_inside_its_bound_records_nothing(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    patch = await eng._node_execute_reflect(_state(_design(physical=True), {"bed=1": {"yield_frac": 0.5, "drift_rate": 1.0}}))
    assert patch == {} and not (eng.quest_root / "needs" / "RESULT_CHECK.json").exists()


@pytest.mark.asyncio
async def test_with_repairs_left_an_out_of_range_value_is_still_sent_back(tmp_path: Path) -> None:
    """Known positive of the check itself: nothing changes while a repair is left."""
    eng = _engine(tmp_path, max_iter=3)
    state = _state(_design(physical=True), RESULT, iters=0)
    assert eng._route_after_execute_reflect(state) == "retry"


def test_a_value_on_a_cap_of_the_script_is_said_once_per_quantity_in_the_run_record() -> None:
    from core.engine_run import _said_on_bounds

    a = pl.Assertion("share", 0.0, 1.0)
    warned = [pl.Violation(f"g={i}.share", 1.0, a, kind="clamped") for i in range(3)]
    said = _said_on_bounds(warned)
    assert len(said) == 1 and "the script caps values at 1" in said[0]["text"] and "3 values of `share`" in said[0]["text"]


# --- what the review of the first version found -------------------------------------------------------------------


def test_no_number_of_a_kept_quantity_carries_a_mark(tmp_path: Path) -> None:
    """Only a number computed from the quantity's values is unconfirmed: its trial count, the plan's reference for it and the
    setting where it peaks are not."""
    q = _paper_quest(tmp_path, kept=True)
    rec = pv.records_of(q, None)
    for key in ("{{v:drift_rate:n @ bed=1}}", "{{v:drift_rate @ bed=each | argmax}}"):
        got = pv.resolve(key, rec)
        assert got.filled and MARK not in got.text, key
    assert MARK not in pv.resolve("{{v:drift_rate:sd @ bed=1}}", rec).text  # nor a spread of its values


def test_a_figure_axis_fi_could_not_tie_to_the_quantity_does_not_drop_the_figure(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / ".fi" / "figure_records").mkdir(parents=True)
    for stem, x in (("tied", {"id": "yield_frac", "kind": "result"}),
                    ("untied", {"id": "yield_frac", "kind": "result", "label": "computed x", "neutral": True}),
                    ("other", {"id": "load", "kind": "result"})):
        (tmp_path / "figures" / f"{stem}.png").write_bytes(b"png")
        (tmp_path / ".fi" / "figure_records" / f"{stem}.calls.json").write_text(json.dumps(
            {"panels": [{"panel": 1, "x": x, "y": {"what": "drift_rate"}, "series": [{"id": "drift_rate"}]}]}), encoding="utf-8")
    assert rc.figures_on_axis(tmp_path, {"yield_frac"}) == ["tied.png"]


def test_a_figure_whose_shared_y_axis_draws_the_quantity_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / ".fi" / "figure_records").mkdir(parents=True)
    x = {"id": "bed", "kind": "setting"}
    panels = {  # FI joins the ids of two results drawn on one y axis with " / "
        "shared": {"y": {"what": "load / yield_frac"}, "series": [{"id": "load", "what": "load"}]},
        "by_series": {"y": {"what": "load"}, "series": [{"id": "curve_b", "what": "yield_frac"}]},
        "untied": {"y": {"what": "other / load"}, "series": [{"id": "curve_c", "what": "yield_frac", "neutral": True}]},
        "apart": {"y": {"what": "load / drift_rate"}, "series": [{"id": "load", "what": "load"}]},
    }
    for stem, p in panels.items():
        (tmp_path / "figures" / f"{stem}.png").write_bytes(b"png")
        (tmp_path / ".fi" / "figure_records" / f"{stem}.calls.json").write_text(json.dumps(
            {"panels": [{"panel": 1, "x": x, **p}]}), encoding="utf-8")
    assert rc.figures_on_axis(tmp_path, {"yield_frac"}) == ["by_series.png", "shared.png"]


def test_a_figure_with_a_reference_line_at_the_quantity_is_dropped(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / ".fi" / "figure_records").mkdir(parents=True)
    refs = {  # a line read from the table keeps the result it is drawn at; a plan reference is named by its own id
        "table_line": [{"id": "refline1", "axis": "y", "value": 1.2, "source": "table", "at": "yield_frac"}],
        "plan_line": [{"id": "yield_frac", "axis": "y", "value": 1.0, "source": "plan"}],
        "other_line": [{"id": "refline1", "axis": "y", "value": 2.0, "source": "table", "at": "load"}],
    }
    for stem, r in refs.items():
        (tmp_path / "figures" / f"{stem}.png").write_bytes(b"png")
        (tmp_path / ".fi" / "figure_records" / f"{stem}.calls.json").write_text(json.dumps({"panels": [
            {"panel": 1, "x": {"id": "bed"}, "y": {"what": "load"}, "series": [{"id": "load", "what": "load"}], "refs": r}]}),
            encoding="utf-8")
    assert rc.figures_on_axis(tmp_path, {"yield_frac"}) == ["plan_line.png", "table_line.png"]


def test_each_left_out_quantity_is_given_the_note_of_its_own_check(tmp_path: Path) -> None:
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / "RESULT_CHECK.json").write_text(json.dumps({
        "removed_quantities": ["drift_rate", "load", "spare"],
        "checks": [{"kind": "a", "note": "drift note", "quantity": "drift_rate", "keys": ["drift_rate"]},
                   {"kind": "b", "note": "load note", "quantity": "load", "keys": ["load"]}],
        "note": "drift note; load note"}), encoding="utf-8")
    gone = rc.left_out_by_record(tmp_path)
    assert gone == {"drift_rate": "drift note", "load": "load note", "spare": "drift note; load note"}


@pytest.mark.asyncio
async def test_a_repair_that_gives_up_leaves_out_a_physical_break_as_well(tmp_path: Path) -> None:
    eng = _engine(tmp_path, max_iter=3)
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value='{"give_up_reason": "cannot see the fault"}')})()
    state = _state(_design(physical=True), RESULT, iters=0)
    patch = await eng._node_execute_reflect(state)
    assert patch["exec_give_up_reason"] == "cannot see the fault"
    assert "yield_frac" in rc.left_out_by_record(eng.quest_root)
    assert "yield_frac" not in json.dumps(patch["result_json"])


@pytest.mark.asyncio
async def test_each_seeds_results_lose_a_left_out_quantity_too(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    state = {**_state(_design(physical=True), RESULT),
             "result_json_replicates": [{"_seed": 0, **RESULT}, {"_seed": 1, **RESULT}]}
    patch = await eng._node_execute_reflect(state)
    assert "yield_frac" not in json.dumps(patch["result_json_replicates"])
    assert "drift_rate" in json.dumps(patch["result_json_replicates"])


@pytest.mark.asyncio
async def test_the_per_trial_lists_of_a_left_out_quantity_leave_the_results_too(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    result = {k: {**row, "yield_frac_values": [row["yield_frac"]] * 3, "yield_frac_pair_id": [0, 1, 2],
                  "drift_rate_values": [row["drift_rate"]] * 3} for k, row in RESULT.items()}
    patch = await eng._node_execute_reflect(_state(_design(physical=True), result))
    shown = json.dumps(patch["result_json"])
    assert "yield_frac" not in shown  # its trials and their ids are its values too
    assert "drift_rate_values" in shown  # a kept quantity keeps its trials
