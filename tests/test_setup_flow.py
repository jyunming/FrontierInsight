"""FI's set-up figure (``core/setup_flow.py``): drawn by FI from its own records at the start of the write stage, with no
model call, numbered after the run's figures with FI's caption, and never read as a result figure."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core import frozen_protocol, quality_report, setup_flow
from core.engine import _figure_list_for_prompt, _figure_numbers, _place_figures, _take_out_unpointed

# A made-up study: how a catalyst's yield depends on temperature and pressure.
GRID_PLAN = {
    "grid": {"temperature_C": [20, 40, 60, 80, 100], "pressure_bar": [1, 5], "stir_rpm": [300]},
    "runs_per_setting": 3,
    "constants": {"volume_L": 2.5, "note": "a long free-text note that is not a number"},
    "labels": {"temperature_C": "Temperature $T$ (C)", "pressure_bar": "Pressure $p$ (bar)",
               "stir_rpm": "Stirring rate (rpm)", "yield_pct": "Yield $Y$ (%)"},
    "metrics": [{"id": "yield_pct", "label": "Yield", "role": "result", "family": "output"},
                {"id": "runtime_s", "label": "Run time", "role": "diagnostic"}],
    "model": {"steps": [{"id": "S1", "name": "rate law", "uses": ["E1"]}, {"id": "S2", "name": "mass balance"}],
              "equations": [{"id": "E1", "formula": "r = k * c"}]},
}
SEARCH_PLAN = {
    "grid": None, "sub_studies": None,
    "optimisation": {
        "objective": {"quantity": "cost_eur", "direction": "minimise", "unit": "EUR"},
        "design_variables": [{"name": "wall_mm", "low": 1, "high": 8, "unit": "mm"},
                             {"name": "fin_count", "low": 2, "high": 20, "unit": "dimensionless"}],
        "fixed": {"ambient_C": 25},
        "constraints": [{"quantity": "peak_C", "limit": "<= 90"}],
    },
    "labels": {"wall_mm": "Wall thickness (mm)", "fin_count": "Number of fins", "peak_C": "Peak temperature (C)",
               "cost_eur": "Material cost"},
    "metrics": [{"id": "cost_eur", "label": "Material cost", "role": "result"}],
}
RESULT = "yield_curve.png"
STATE = {"figures": [RESULT], "figure_records": {RESULT: {"axes": [{"ylabel": "Yield", "series": [
    {"label": "p = 1", "min": 1.0, "max": 9.0, "shows": "yes"}]}]}}}


def _quest(tmp_path: Path, protocol: dict | None, steps: dict | None = None) -> Path:
    root = tmp_path / "quest"
    (root / "figures").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "figures" / RESULT).write_bytes(b"\x89PNG run figure")
    if protocol is not None:
        frozen_protocol.freeze(root, protocol, approved_by="test", source="test")
    if steps is not None:
        (root / ".fi" / "function_steps.json").write_text(json.dumps(steps), encoding="utf-8")
    return root


def _texts(record: dict) -> str:
    return "\n".join(" | ".join([b["name"], *b["entries"]]) for b in record["boxes"])


# ---- the figure: what it says, from the records ----------------------------------------------------------------------

def test_a_grid_plan_draws_inputs_steps_and_results_in_the_plans_words(tmp_path: Path) -> None:
    root = _quest(tmp_path, GRID_PLAN)
    record = setup_flow.draw(root, [RESULT])
    assert record["drawn"] and (root / "figures" / setup_flow.FILE).is_file()
    assert [b["column"] for b in record["boxes"]] == ["inputs", "inputs", "computation", "computation", "computation",
                                                       "results"]
    text = _texts(record)
    assert "Temperature $T$ (C): 20 to 100 (5 values)" in text and "Pressure $p$ (bar): 1, 5" in text
    assert "Stirring rate (rpm): 300" in text and "Runs per setting: 3" in text and "volume_L = 2.5" in text
    assert "S1 rate law" in text and "uses E1" in text and "S2 mass balance" in text
    assert "Yield" in text and "Run time" not in text, "a diagnostic is not a result the study reports"
    assert "free-text note" not in text
    assert all(b["fits"] for b in record["boxes"]), "every text lies inside its box"
    assert setup_flow.BODY_PT >= 8 and setup_flow.TITLE_PT >= 8 and setup_flow.WIDTH_IN <= 6.3


def test_a_design_search_with_no_grid_draws_its_variables_and_objective(tmp_path: Path) -> None:
    root = _quest(tmp_path, SEARCH_PLAN)
    record = setup_flow.draw(root, [])
    assert record["drawn"] and record["shape"] == "search"
    text = _texts(record)
    assert "Wall thickness (mm): 1 to 8" in text and "Number of fins: 2 to 20" in text
    assert "ambient_C: 25" in text
    assert "minimise Material cost (EUR)" in text and "Peak temperature (C) <= 90" in text
    assert "computation" not in {b["column"] for b in record["boxes"]}, "no step card: no middle column"
    assert "design variables it searches over" in record["caption"]


def test_without_function_steps_the_plans_steps_are_drawn_and_a_failed_check_is_dashed(tmp_path: Path) -> None:
    plain = setup_flow.draw(_quest(tmp_path / "a", GRID_PLAN), [])
    assert plain["drawn"] and not any(b["dashed"] for b in plain["boxes"])
    steps = {"status": "partial", "functions": {"rate_law": {"equation": "S1", "status": "failed"},
                                                 "eq_e1": {"equation": "E1", "status": "done"}}}
    marked = setup_flow.draw(_quest(tmp_path / "b", GRID_PLAN, steps), [])
    dashed = [b for b in marked["boxes"] if b["dashed"]]
    assert [b["name"] for b in dashed] == ["S1 rate law"]
    assert "dashed box is a step whose own check did not pass" in marked["caption"]


def test_a_long_list_is_cut_with_and_more_and_the_caption_says_so(tmp_path: Path) -> None:
    plan = {**GRID_PLAN, "metrics": [{"id": f"m{i}", "label": f"Measure {i}", "role": "result"} for i in range(9)]}
    record = setup_flow.draw(_quest(tmp_path, plan), [])
    assert "and 6 more results" in _texts(record)
    assert 'ends with "more"' in record["caption"]
    assert not any(ch.isdigit() for ch in record["caption"]), "the caption types no number"


def test_a_label_matplotlib_cannot_draw_as_math_is_drawn_as_words(tmp_path: Path) -> None:
    plan = {**GRID_PLAN, "labels": {**GRID_PLAN["labels"], "temperature_C": r"Temperature $\notacommand{T$ (C)"}}
    record = setup_flow.draw(_quest(tmp_path, plan), [])
    assert record["drawn"] and all(b["fits"] for b in record["boxes"])


# ---- too thin, or a fault: nothing drawn, one QUALITY.md line, never raised -------------------------------------------

@pytest.mark.parametrize("protocol, why", [
    (None, "no frozen plan"),
    ({"metrics": [{"id": "y", "label": "Y", "role": "result"}]}, "no setting the study varies"),
    ({"grid": {"a": [1, 2]}, "metrics": [{"id": "y", "role": "diagnostic"}]}, "no result the study reports"),
])
def test_records_too_thin_draw_nothing_and_quality_says_why(tmp_path: Path, protocol: dict | None, why: str) -> None:
    root = _quest(tmp_path, protocol)
    setup_flow.draw_quietly(root, [RESULT])
    assert not (root / "figures" / setup_flow.FILE).exists()
    assert setup_flow.shown(root) is None
    assert why in (setup_flow.quality_line(root) or "")
    rows = quality_report._paper_problems(quality_report._Quest(root))
    assert [r["status"] for r in rows if "set-up figure" in r["text"] and why in r["text"]] == ["disclosed"], rows
    assert "set-up figure" in quality_report.render(quality_report.build(root, running=False))
    numbers = _figure_numbers({**STATE, "quest_root": str(root)})
    assert numbers == {RESULT: 1}


def test_a_fault_while_drawing_never_raises_and_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, GRID_PLAN)
    setup_flow.draw_quietly(root, [])
    assert setup_flow.shown(root) == setup_flow.FILE

    def broken(flow, png):  # noqa: ANN001
        raise RuntimeError("no font")

    monkeypatch.setattr(setup_flow, "render", broken)
    setup_flow.draw_quietly(root, [])
    assert setup_flow.shown(root) is None and not (root / "figures" / setup_flow.FILE).exists()
    assert "could not draw it (RuntimeError)" in (setup_flow.quality_line(root) or "")


def test_a_drawing_for_an_earlier_plan_or_a_run_figure_of_that_name_is_not_fis(tmp_path: Path) -> None:
    root = _quest(tmp_path, GRID_PLAN)
    setup_flow.draw(root, [])
    assert setup_flow.shown(root) == setup_flow.FILE
    # the run drew a file of the same name: not FI's drawing (its record's digest), so it is a result like any other
    (root / "figures" / setup_flow.FILE).write_bytes(b"\x89PNG a run's own drawing")
    assert setup_flow.shown(root) is None
    assert setup_flow.without(root, [RESULT, setup_flow.FILE]) == [RESULT, setup_flow.FILE]
    record = setup_flow.draw(root, [setup_flow.FILE])
    assert not record["drawn"] and "the run drew a figure of its own" in record["why"]
    # the plan changed after FI drew it: the drawing is not shown for the new plan
    other = _quest(tmp_path / "b", GRID_PLAN)
    setup_flow.draw(other, [])
    (other / "needs" / "FROZEN_PROTOCOL.json").write_text(
        (other / "needs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8").replace("rate law", "rate rule"),
        encoding="utf-8")
    assert setup_flow.shown(other) is None


# ---- the paper, the poster and the slides -------------------------------------------------------------------------------

DRAFT = ("# T\n\n## Method\n\nFigure 2 shows how the study is set up.\n\n## Results\n\n"
         "Figure 1 shows the yield rising with temperature.\n\n## References\n\n1. A cited paper.\n")


def test_numbered_after_the_run_figures_placed_with_fis_caption_and_listed_for_the_writer(tmp_path: Path) -> None:
    root = _quest(tmp_path, GRID_PLAN)
    setup_flow.draw(root, [RESULT])
    numbers = _figure_numbers({**STATE, "quest_root": str(root)})
    assert numbers == {RESULT: 1, setup_flow.FILE: 2}, "the run's figures keep the numbers the analysis used"
    assert _figure_numbers(STATE) == {RESULT: 1}, "without the quest's root nothing changes"
    paper, placed, _ = _place_figures(DRAFT, STATE, root=root)
    assert placed == [RESULT, setup_flow.FILE]
    assert f"![**Figure 2.** How the study is set up, drawn by FI from the study's frozen plan" in paper
    assert paper.count(f"](figures/{setup_flow.FILE})") == 1
    listing = _figure_list_for_prompt(STATE, root=root)
    assert f"- Figure 2: figures/{setup_flow.FILE}" in listing and "not a result" in listing


def test_the_write_node_draws_it_and_places_it_without_a_model_call_for_it(tmp_path: Path) -> None:
    from tests.test_fi_places_figures import _engine

    root = _quest(tmp_path, GRID_PLAN)
    eng = _engine(root)
    (root / "paper").mkdir()
    calls = []

    async def fake_chat(prompt, *, node=None, temperature=None):  # noqa: ANN001
        calls.append(node)
        return DRAFT

    eng._chat = fake_chat  # type: ignore[assignment]
    eng._writing_skills_block = lambda state: ""  # type: ignore[assignment]
    eng._resolve_write_persona = lambda state: ""  # type: ignore[assignment]
    patch = asyncio.run(eng._node_write({"topic": "t", "title": "t", "iteration": 0, **STATE}))  # type: ignore[arg-type]
    written = Path(patch["paper_md"]).read_text(encoding="utf-8")
    assert (root / "figures" / setup_flow.FILE).is_file()
    assert "![**Figure 2.** How the study is set up" in written and "![**Figure 1.** Yield." in written
    assert "figures" not in patch, "the run's own figure list is left as the run drew it"
    assert all(n not in ("setup_flow", "figures") for n in calls)


def test_poster_and_slides_take_it_with_the_papers_number(tmp_path: Path) -> None:
    from generation._paper_figures import paper_figure_set

    root = _quest(tmp_path, GRID_PLAN)
    setup_flow.draw(root, [RESULT])
    paper, _, _ = _place_figures(DRAFT, STATE, root=root)
    files = sorted(p.name for p in (root / "figures").iterdir())
    shown, numbers = paper_figure_set(paper, files, dict(STATE), root)
    assert setup_flow.FILE in shown and numbers[setup_flow.FILE] == 2
    # a study none of whose figures has FI's record (a quest with no simulation figures) still numbers it
    shown, numbers = paper_figure_set(paper, files, {"figures": [RESULT]}, root)
    assert numbers.get(setup_flow.FILE) == 2


def test_a_sentence_naming_only_the_setup_figure_is_kept_by_the_pointer_rule(tmp_path: Path) -> None:
    from tests.test_fi_places_figures import _panel, _record

    root = _quest(tmp_path, GRID_PLAN)
    setup_flow.draw(root, [RESULT])
    (root / ".fi" / "figure_records").mkdir()
    (root / ".fi" / "figure_records" / "yield_curve.calls.json").write_text(json.dumps(_record(_panel())),
                                                                            encoding="utf-8")
    text = "Figure 2 shows how the study is set up. Figure 1 shows a yield of nine percent.\n"
    out, taken = _take_out_unpointed(text, STATE, root=root)
    assert "Figure 2 shows how the study is set up." in out
    assert "nine percent" not in out and taken, "the rule still takes out a sentence about a checked result figure"


# ---- the checks that judge result figures ---------------------------------------------------------------------------------

def test_the_checks_of_the_runs_figures_never_count_it(tmp_path: Path) -> None:
    from core.result_checks import ResultChecksMixin

    root = _quest(tmp_path, GRID_PLAN)
    setup_flow.draw(root, [RESULT])

    class _Checks(ResultChecksMixin):
        quest_root = root
        fi_dir = root / ".fi"

    assert _Checks()._saved_figures() == [RESULT], "a planned figure the run did not draw is not hidden by FI's figure"
    assert setup_flow.without(root, [RESULT, setup_flow.FILE]) == [RESULT]
