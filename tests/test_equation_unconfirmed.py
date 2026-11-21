"""An equation whose test still fails after the code was repaired and the plan was asked about its worked example does not
stop the study: it goes on with that equation not confirmed, and every quantity it may reach is marked (unconfirmed).

Neutral model: a mass on a damped spring. The equation's text is one FI cannot read (a condition after the formula), so it
cannot tell a wrong example from a wrong function; the plan is asked once and keeps the example."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.plan_parts_helpers import checks_reply, is_scoped_ask

from core import equation_tests as eqt, paper_values, plan
from tests.test_equation_gate import _engine

FORMULA = "Q = m * w / c for c > 0"
RIGHT_EXAMPLE = {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c"}
SLIP_EXAMPLE = {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w * c"}  # a slip in the example
RIGHT_CODE = "# E1\ndef quality(m, w, c):\n    return m * w / c\n"
WRONG_CODE = "# E1\ndef quality(m, w, c):\n    return w / c\n"  # the normalisation m dropped
SIMULATE = "from spring import model\n\n\ndef run_cell(cell):\n    return {'q': model.quality(2.0, 3.0, cell['c'])}\n"
METRICS = [{"id": "q_mean", "kind": "mean"}, {"id": "q_sd", "kind": "mean"}]


def _protocol(example: dict[str, Any]) -> dict[str, Any]:
    return {"grid": {"c": [0.5, 1.0]}, "metrics": METRICS, "model": {"summary": "a damped spring", "equations": [
        {"id": "E1", "formula": FORMULA, "role": "generates", "source": "derivation", "derivation": "energy ratio",
         "example": example}]}}


class _Plan:
    """The plan keeps its example when asked; a request to repair the function gets the function as it is."""

    def __init__(self, code: str) -> None:
        self.code, self.asked, self.repairs = code, 0, 0

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if is_scoped_ask(prompt):
            self.asked += 1
            from tests.plan_parts_helpers import plan_shown

            return checks_reply(plan_shown(prompt), prompt)
        self.repairs += 1
        return "```python\n" + self.code + "```"

    async def aclose(self) -> None:
        return None


def _setup(tmp_path: Path, example: dict[str, Any], code: str, *, rounds: int = 1):
    engine = _engine(tmp_path)
    engine.config.engine.exec_reflect_max_iterations = rounds
    protocol = _protocol(example)
    text = plan.render(engine.config.topic, {}, {"hypothesis": "h", "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    root = engine.quest_root / "code"
    (root / "spring").mkdir(parents=True, exist_ok=True)
    (root / "spring" / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (root / "spring" / "model.py").write_text(code, encoding="utf-8")
    (root / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    model = _Plan(code)
    engine._client = model
    return engine, model, {"design": {"protocol": protocol}}


def _record(engine) -> dict[str, Any]:  # noqa: ANN001
    return json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_wrong_example_the_plan_keeps_stops_the_study_only_until_the_repairs_are_spent(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, SLIP_EXAMPLE, RIGHT_CODE, rounds=2)
    for _ in range(2):  # the whole script is repaired between the runs, as before
        patch = await engine._equation_gate(state, None, None)
        assert patch is not None and patch["exec_result"]["returncode"] == 1
    assert await engine._equation_gate(state, None, None) is None, "the repairs are spent: the study goes on"
    record = _record(engine)
    held = record["unconfirmed"]["E1"]
    assert held["marked"] is True and held["results"] == ["q_mean", "q_sd"]
    assert any("equation E1 is not confirmed before the run" in n for n in record["not_tested"])
    assert "E1" not in record["tested"]
    asked = json.loads((engine.fi_dir / "examples_rederived.json").read_text(encoding="utf-8"))["asked"]
    assert len(asked) == 1 and model.asked >= 1, "the plan was asked about the example (once per equation)"
    marks = eqt.unconfirmed_marks(engine.quest_root)
    assert set(marks) == {"q_mean", "q_sd"} and marks["q_mean"]["unconfirmed_by"] == ["E1"]
    assert set(paper_values.bound_break_names(_protocol(SLIP_EXAMPLE), marks)) >= {"q_mean", "q_sd"}


@pytest.mark.asyncio
async def test_a_wrong_function_with_a_right_example_never_reaches_the_paper_unmarked(tmp_path: Path) -> None:
    """The known positive: the check still fires, and what the wrong function may reach is marked."""
    engine, _model, state = _setup(tmp_path, RIGHT_EXAMPLE, WRONG_CODE, rounds=1)
    assert await engine._equation_gate(state, None, None) is not None  # stops, as before, while repairs remain
    assert await engine._equation_gate(state, None, None) is None
    record = _record(engine)
    assert record["failed"][0]["id"] == "E1" and record["failed"][0]["status"] == "mismatch", "the failure is recorded"
    assert set(eqt.unconfirmed_marks(engine.quest_root)) == {"q_mean", "q_sd"}


@pytest.mark.asyncio
async def test_an_example_the_equation_as_written_gives_too_still_stops_the_study(tmp_path: Path) -> None:
    """FI computed both and they agree: the code is wrong, whatever the repairs (the readable-equation case)."""
    readable = {"id": "E1", "formula": "Q = m * w / c", "role": "generates", "source": "derivation", "derivation": "d",
                "example": RIGHT_EXAMPLE}
    engine, _model, state = _setup(tmp_path, RIGHT_EXAMPLE, WRONG_CODE, rounds=1)
    protocol = _protocol(RIGHT_EXAMPLE)
    protocol["model"]["equations"] = [readable]
    state = {"design": {"protocol": protocol}}
    text = plan.render(engine.config.topic, {}, {"hypothesis": "h", "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    for _ in range(4):
        assert await engine._equation_gate(state, None, None) is not None
    assert not _record(engine).get("unconfirmed")


@pytest.mark.asyncio
async def test_a_function_that_cannot_be_called_never_goes_on(tmp_path: Path) -> None:
    broken = "# E1\ndef quality(m, w):\n    return m * w\n"  # the parameter c is missing: the test cannot call it
    engine, _model, state = _setup(tmp_path, RIGHT_EXAMPLE, broken, rounds=1)
    for _ in range(4):
        assert await engine._equation_gate(state, None, None) is not None
    assert not _record(engine).get("unconfirmed")


@pytest.mark.asyncio
async def test_a_held_equation_is_not_repaired_or_asked_again_and_a_pass_clears_the_mark(tmp_path: Path) -> None:
    engine, model, state = _setup(tmp_path, SLIP_EXAMPLE, RIGHT_CODE, rounds=1)
    assert await engine._equation_gate(state, None, None) is not None
    assert await engine._equation_gate(state, None, None) is None
    calls = (model.asked, model.repairs)
    assert await engine._equation_gate(state, None, None) is None, "a later run (the rewrite, the second version) goes on too"
    assert (model.asked, model.repairs) == calls, "nothing is repaired or asked about it again"
    assert set(eqt.unconfirmed_marks(engine.quest_root)) == {"q_mean", "q_sd"}
    # the example is put right (the plan, or a person, changed it): the test passes and the mark is gone
    protocol = _protocol(RIGHT_EXAMPLE)
    text = plan.render(engine.config.topic, {}, {"hypothesis": "h", "protocol": protocol})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    assert await engine._equation_gate({"design": {"protocol": protocol}}, None, None) is None
    assert eqt.unconfirmed_marks(engine.quest_root) == {} and not _record(engine).get("unconfirmed")


@pytest.mark.asyncio
async def test_a_function_that_is_never_run_marks_no_result(tmp_path: Path) -> None:
    engine, _model, state = _setup(tmp_path, SLIP_EXAMPLE, RIGHT_CODE, rounds=1)
    (engine.quest_root / "code" / "simulate.py").write_text(
        "from spring import model\n\n\ndef run_cell(cell):\n    return {'q': 1.0}\n", encoding="utf-8")
    assert await engine._equation_gate(state, None, None) is not None
    assert await engine._equation_gate(state, None, None) is None
    held = _record(engine)["unconfirmed"]["E1"]
    assert held["marked"] is False and held["results"] == []
    assert eqt.unconfirmed_marks(engine.quest_root) == {}


# --- everything the mark must reach: the note under the title, the figures, QUALITY.md ----------------------------------------


def _held_record(root: Path, results: list[str]) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "equation_tests.json").write_text(json.dumps({
        "equations": {"E1": {"state": "ok", "why": ""}},
        "failed": [{"id": "E1", "function": "quality", "status": "mismatch", "reason": "got 3, expected 12"}],
        "repairs": {"E1": 2}, "tested": [],
        "not_tested": ["equation E1 is not confirmed before the run: its test still failed (got 3, expected 12)"],
        "unconfirmed": {"E1": {"formula": FORMULA, "function": "quality", "marked": True, "results": results}}}), encoding="utf-8")


def test_a_note_under_the_title_says_a_hand_written_number_is_not_marked_and_clears_with_the_mark(tmp_path: Path) -> None:
    """A number the writer typed cannot be placed on a result, so FI's own note says the paper may hold one unmarked."""
    from core import engine_base as eb

    _held_record(tmp_path, ["q_mean"])
    paper = "# A study\n\n## Abstract\nThe mean q was 12.4.\n"
    once = eb._mark_unconfirmed_results(paper, tmp_path)
    assert once.index("fi:marks-note") < once.index("## Abstract") and "(unconfirmed)" in once and "`q_mean`" in once
    assert "written by hand" in once and eb._mark_unconfirmed_results(once, tmp_path) == once, "placed once, the same again"
    (tmp_path / ".fi" / "equation_tests.json").write_text(json.dumps({"unconfirmed": {}}), encoding="utf-8")
    assert eb._mark_unconfirmed_results(once, tmp_path).count("fi:marks-note") == 0, "the mark is gone, so is the note"


def test_a_quantity_kept_out_of_range_gets_the_same_note_and_figure_marks(tmp_path: Path) -> None:
    """The same hole the existing out-of-range mark had: one place now says which quantities and figures carry the mark."""
    from core import result_checks as rc

    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / rc.RECORD).write_text(json.dumps({rc.KEPT_BREAKS: [{"quantity": "q_mean", "note": "n"}]}), encoding="utf-8")
    assert set(rc.marked_quantities(tmp_path)) == {"q_mean"} and "`q_mean`" in rc.marked_note(tmp_path)


def test_figures_that_may_draw_a_marked_quantity_are_marked_by_fis_record_of_the_drawing(tmp_path: Path) -> None:
    from core import result_checks as rc

    _held_record(tmp_path, ["q_mean"])
    (tmp_path / "figures").mkdir()
    (tmp_path / ".fi" / "figure_records").mkdir()
    records = {"tied": {"y": {"what": "q_mean"}, "series": [{"id": "q_mean", "what": "q_mean"}]},
               "apart": {"y": {"what": "load"}, "series": [{"id": "load", "what": "load"}]},
               "untied": {"y": {"what": "load"}, "series": [{"id": "s", "what": "load", "neutral": True}]}}
    for stem, p in records.items():
        (tmp_path / "figures" / f"{stem}.png").write_bytes(b"png")
        (tmp_path / ".fi" / "figure_records" / f"{stem}.calls.json").write_text(
            json.dumps({"panels": [{"panel": 1, "x": {"id": "c", "kind": "setting"}, **p}]}), encoding="utf-8")
    (tmp_path / "figures" / "norecord.png").write_bytes(b"png")  # FI cannot tell what it draws: marked, never presented bare
    assert rc.marked_figures(tmp_path) == ["norecord.png", "tied.png", "untied.png"]
    (tmp_path / ".fi" / "equation_tests.json").write_text("{}", encoding="utf-8")
    assert rc.marked_figures(tmp_path) == []


def test_quality_says_the_study_went_on_with_the_equation_not_confirmed(tmp_path: Path) -> None:
    from core import quality_report as qr

    _held_record(tmp_path, ["q_mean"])
    rows = qr._checks_problems(qr._Quest(tmp_path))
    text = " | ".join(r["text"] for r in rows)
    assert "equation E1 could not be confirmed before the run" in text and "went on" in text
    assert "has no test" not in text and "equation equation" not in text
    assert [r["status"] for r in rows] == ["went on, not confirmed"], "one row, not 'not fixed' and not a second sentence"


def test_the_caption_of_a_figure_that_may_draw_a_marked_quantity_carries_the_mark(tmp_path: Path) -> None:
    from core import engine_base as eb

    _held_record(tmp_path, ["q_mean"])
    (tmp_path / "figures").mkdir()
    (tmp_path / ".fi" / "figure_records").mkdir()
    for stem, what in (("fig_q", "q_mean"), ("fig_load", "load")):
        (tmp_path / "figures" / f"{stem}.png").write_bytes(b"png")
        (tmp_path / ".fi" / "figure_records" / f"{stem}.calls.json").write_text(json.dumps({"panels": [{
            "panel": 1, "x": {"id": "c", "kind": "setting"}, "y": {"what": what}, "series": [{"id": what, "what": what}]}]}),
            encoding="utf-8")
    state = {"quest_root": str(tmp_path), "figures": ["fig_q.png", "fig_load.png"], "figure_records": {}}
    text, placed, _n = eb._place_figures("# T\n\n## Results\nSee Figure 1 and Figure 2.\n", state, root=tmp_path)
    assert set(placed) == {"fig_q.png", "fig_load.png"}
    by_file = {line.split("figures/")[1].rstrip(")"): line for line in text.split("\n") if "](figures/" in line}
    assert "(unconfirmed)" in by_file["fig_q.png"] and "(unconfirmed)" not in by_file["fig_load.png"]
