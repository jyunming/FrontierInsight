"""General limits of the plan's own schema, met by a real failed run (a topic with tuple-valued and name-valued settings):
FI's reader refused every plan the model wrote, the model repeated the same answer, and the quest still ran its analysis
and a redesign after it had already decided that nothing would run. Neutral names throughout."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core import metric_spec as ms
from core import optimisation_plan as op
from core import plan
from core import run_manifest as rm
from tests.test_code_review import _engine
from tests.test_optimise_runner import BLOCK


# 1. a setting that is a tuple -------------------------------------------------------------------------------------------


def test_a_grid_axis_may_take_tuples_of_numbers_each_one_setting() -> None:
    fixed, why = plan.normalize_protocol({"grid": {"pair": [[0.5, 0.8], [0.6, 0.9]], "n": [1, 2]}})
    assert fixed is not None, why
    assert fixed["grid"]["pair"] == [[0.5, 0.8], [0.6, 0.9]]


def test_tuples_of_different_lengths_or_mixed_with_single_values_are_still_refused() -> None:
    _, why = plan.normalize_protocol({"grid": {"pair": [[0.5, 0.8], [0.6]]}})
    assert why and "pair" in why and "same number" in why
    _, why = plan.normalize_protocol({"grid": {"pair": [[0.5, 0.8], 3]}})
    assert why and "mixes lists" in why
    _, why = plan.normalize_protocol({"grid": {"pair": [[0.5, "a"], [0.6, 0.9]]}})
    assert why and "pair" in why


def test_the_run_manifest_reads_a_tuple_setting_through_the_cell_key_the_runner_writes() -> None:
    from core import trial_runner

    grid = {"pair": [[0.5, 0.8], [0.6, 0.9]], "n": [1, 2]}
    cells = trial_runner.cells(grid)
    keys = {trial_runner.cell_key(c) for c in cells}
    canon = {rm._canonicalize_cell(k, grid)[0] for k in keys}
    assert None not in canon and canon == rm._all_cells(grid), "every cell the runner names is one the protocol has"
    ledger = [{"cell": k, "trial": 1, "status": "ok"} for k in sorted(keys)]
    manifest, problems = rm.manifest_from_ledger({"grid": grid}, ledger) if hasattr(rm, "manifest_from_ledger") else (None, [])
    if manifest is not None:
        assert not problems
        assert [[0.5, 0.8], [0.6, 0.9]] == manifest["realized_grid"]["pair"]
    assert rm._same([0.5, 0.8], (0.5, 0.8)) and not rm._same([0.5, 0.8], [0.5, 0.9])


# 2. a list that mixes names and numbers -----------------------------------------------------------------------------------


def test_a_list_that_mixes_names_and_numbers_is_refused_with_how_to_write_it() -> None:
    # a numeric sweep with a stray piece of text must not silently become names (the numeric checks would be lost)
    fixed, why = plan.normalize_protocol({"grid": {"temp": [290, 300, "310K"]}})
    assert fixed is None and "mixes names and numbers" in why and "in quotes" in why
    # a tuple setting's numbers count as numbers the plan fixes (the topic's numbers are found in it)
    from core import protocol_check as pc
    assert (0.8, "grid pair") in pc.protocol_numbers({"grid": {"pair": [[0.5, 0.8], [0.6, 0.9]]}})
    # all-number and all-name lists are as before
    fixed, _ = plan.normalize_protocol({"grid": {"a": [1, 2.5], "b": [" x ", "y"]}})
    assert fixed["grid"] == {"a": [1, 2.5], "b": ["x", "y"]}
    _, why = plan.normalize_protocol({"grid": {"a": []}})
    assert why and "non-empty" in why
    _, why = plan.normalize_protocol({"grid": {"a": [True, 1]}})
    assert why


# 3. a plain measured value per run ---------------------------------------------------------------------------------------


def test_a_metric_that_is_one_measured_value_per_run_is_a_mean_and_the_refusal_says_so() -> None:
    spec = {"id": "width", "estimand": "the width", "unit": "one run", "kind": "value"}
    clean, why = ms.normalize([spec])
    assert clean is not None, why
    assert clean[0]["kind"] == "mean"
    _, why = ms.normalize([{**spec, "kind": "count"}])
    assert why and "mean" in why and "proportion" in why and "value" in why


# 4. a name-valued setting the study compares across ---------------------------------------------------------------------


def test_a_scan_over_names_needs_no_entry_under_fixed_and_the_search_does_not_change_it() -> None:
    fixed, why = op.normalize({**BLOCK, "grid": {"shape": ["round", "square"], "x": [-1.0, 1.0]}})
    assert fixed is not None, why
    assert fixed["fixed"]["shape"] == "round", "held at the first name, so the search never treats it as a variable"
    assert "shape" not in {v["name"] for v in fixed["design_variables"]}
    # what was held fixed before stays; a name the plan already holds is not changed
    again, why = op.normalize({**BLOCK, "fixed": {"shape": "square"}, "grid": {"shape": ["round", "square"]}})
    assert again is not None, why
    assert again["fixed"]["shape"] == "square"


def test_a_name_valued_scan_that_looks_like_a_misspelt_design_variable_is_still_refused() -> None:
    block = {**BLOCK, "design_variables": [*BLOCK["design_variables"],
                                           {"name": "solver_type", "kind": "choice", "values": ["rk4", "euler"]}],
             "baseline": {"values": {"x": 0.0, "y": 0.0, "solver_type": "rk4"}, "source": "the design in use"}}
    for typo in ("solver", "solvr_type"):
        _, why = op.normalize({**block, "grid": {typo: ["rk4", "euler"]}})
        assert why and typo in why, f"{typo} must be refused so the typo is seen"
    # a short design-variable name does not make every axis containing its letter look like a typo
    fixed, why = op.normalize({**BLOCK, "grid": {"family": ["a", "b"]}})
    assert fixed is not None and fixed["fixed"]["family"] == "a", why


def test_a_scan_over_numbers_that_are_neither_a_variable_nor_fixed_is_still_refused() -> None:
    _, why = op.normalize({**BLOCK, "grid": {"unlisted": [1.0, 2.0]}})
    assert why and "unlisted" in why and "optimisation.fixed" in why


# 5. the same refusal twice stops the asking ----------------------------------------------------------------------------


def test_the_design_step_stops_asking_when_the_model_repeats_an_answer_that_was_refused(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path)
    calls: list[str] = []

    async def chat(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps({"protocol": {"grid": {"a": [1, 2]}}})

    engine._chat = chat  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    monkeypatch.setattr("core.engine._plan.normalize_protocol", lambda p: (None, "`protocol.grid.a` is not readable"))
    monkeypatch.setattr("core.engine._topic_req.merge", lambda old, new, gaps: {"protocol": {"grid": {}}})
    design = {"protocol": {"grid": {"b": [1]}}}
    gaps = [{"sentence": "the topic asks for a"}]
    result = asyncio.run(engine._ask_design_for_the_topic({}, design, {}, gaps, iteration=1))
    assert result == design
    assert len(calls) == 2, "the second answer got the same refusal: no third call"
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "same answer that could not be used twice" in log


def test_the_plan_step_stops_asking_when_the_model_repeats_an_answer_that_was_refused(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    plan.plan_path(engine.quest_root).parent.mkdir(parents=True, exist_ok=True)
    plan.plan_path(engine.quest_root).write_text("# plan\n", encoding="utf-8")
    asked: list[str] = []

    async def rewrite(ask, path, **kwargs):
        asked.append(ask)
        raise ValueError("`protocol.grid.a` is not readable; plan.md is unchanged")

    engine._rewrite_plan = rewrite  # type: ignore[method-assign]
    engine._design_from_plan = lambda: ({"a": 1}, "sha2")  # type: ignore[method-assign]
    engine._progress = lambda *a, **k: None  # type: ignore[method-assign]
    asyncio.run(engine._ask_plan_to_complete(
        {}, "sha", lacks=lambda design: ["a is missing"], request=lambda design, text: "ask",
        keep=lambda before, revised: revised, file="repeat_asked.json", what="settings", short="it", note="n"))
    assert len(asked) == 2, "the same refusal twice: no more calls"
    assert "same answer that could not be used for it twice" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


# 6. a stop decided before anything ran ends the quest ----------------------------------------------------------------


def test_a_stop_for_the_topic_goes_straight_to_the_end_without_analysis_or_redesign(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    state = {"exec_result": {"topic_unmet": "the plan leaves out a", "returncode": 0}, "result_json": {}}
    assert engine._route_after_execute_reflect(state) == "stuck"
    assert engine._route_after_execute_reflect({"exec_result": {"returncode": 0}, "result_json": {"x": 1}}) == "proceed"
    verdict = engine._no_results_verdict({**state})
    assert verdict and verdict["stuck"] is True and verdict["stuck_reason"] == "topic_unmet"
    assert "leaves out a" in verdict["rationale"]


def test_the_graph_sends_a_topic_stop_from_the_reflect_step_to_the_gate() -> None:
    import inspect

    from core.engine import Engine

    source = inspect.getsource(Engine)
    assert '"proceed": "improve", "stuck": "evidence_gate"' in source
