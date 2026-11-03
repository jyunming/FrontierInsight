"""The shown-fault gate places a fault from FI's records (the plan's contract, the record of the trials, the record of the
analysis, the code's own call graph) instead of from the words of the finding. Neutral made-up study: two estimates with a
true value, a coverage and a correlation worked out afterwards, one figure, and a few helpers. No model is called; the
re-run of a finding's test is replaced by "still fails" (what the gate decides from that is what these tests are about)."""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from core import code_reach, code_review, shown_reach
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.result_checks import _values_left_out

PROTOCOL = {
    "form": "contract", "grid": {"noise": [0.1, 0.2]}, "runs_per_setting": 20, "constants": {"truth_a": 2.0, "truth_b": 5.0},
    "metrics": [
        {"id": "est_a", "kind": "mean", "estimand": "the first quantity", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_a", "estimate": "est_a", "standard_error": "se_a", "interval": ["lo_a", "hi_a"], "level": 0.95}},
        {"id": "est_b", "kind": "mean", "estimand": "the second quantity", "unit": "one trial", "role": "result", "returned": "each_run",
         "known_value": {"true_value": "truth_b", "estimate": "est_b", "standard_error": "se_b", "interval": ["lo_b", "hi_b"], "level": 0.95}},
        {"id": "cover", "kind": "proportion", "estimand": "coverage", "unit": "one setting", "role": "physical_bound", "returned": "afterwards"},
        {"id": "corr_ab", "kind": "mean", "estimand": "correlation", "unit": "one setting", "role": "result", "returned": "afterwards"}]}

FILES = {
    "model.py": ("def helper_for_cover(x):\n    return x\n\n\ndef helper_for_figure(x):\n    return x\n\n\n"
                 "def helper_for_trial(x):\n    return x\n\n\ndef helper_unused(x):\n    return x\n"),
    "derive.py": ("import model\n\n\ndef derive_cover(table):\n    return model.helper_for_cover(1)\n\n\n"
                  "def derive_corr_ab(table):\n    return 0.5\n\n\ndef figure_fig1(fig, table):\n    return model.helper_for_figure(1)\n"),
    "simulate.py": ("import model\n\n\ndef trial(cell, ctx):\n    return {'est_a': model.helper_for_trial(1), 'est_b': 2.0, 'se_a': 1.0, "
                    "'lo_a': 0.0, 'hi_a': 3.0, 'extra_t': 7.0}\n"),
}
#: What the run printed: the trial-level numbers by setting; the derived `corr_ab` and `cover` are not here.
RESULT = {"noise=0.1": {"est_a": 2.1, "est_b": 5.2, "se_a": 1.0, "lo_a": 0.0, "hi_a": 3.0},
          "noise=0.2": {"est_a": 2.3, "est_b": 5.4, "se_a": 1.1, "lo_a": 0.1, "hi_a": 3.1}}


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(topic="a made-up study", title="made up", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)

    async def fails(self: Any, rel: str) -> Any:  # the test is run again by the gate: here it still fails by its assertion
        return types.SimpleNamespace(status="fail", crash_in_program=False, kind="assertion", detail="", place="", at="")

    Engine._rerun_finding_test = fails  # type: ignore[method-assign,assignment]
    return engine


def _quest(engine: Engine) -> Path:
    from core import frozen_protocol

    root = engine.quest_root
    for rel, text in FILES.items():
        (root / "code" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "code" / rel).write_text(text, encoding="utf-8")
    (root / "needs").mkdir(parents=True, exist_ok=True)
    record = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": PROTOCOL, "sha256": frozen_protocol.sha256(PROTOCOL)}
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(record), encoding="utf-8")
    (root / "raw").mkdir(exist_ok=True)
    metrics = {k: {"values": [1.0], "trials": [0]} for k in ("est_a", "est_b", "se_a", "lo_a", "hi_a", "extra_t")}
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [{"cell": {"noise": 0.1}, "metrics": metrics}]}), encoding="utf-8")
    derived = {"schema": "fi.derived/v1", "complete": True, "not_computed": [], "bodies": {},
               "derived": {"corr_ab": {"status": "ok", "by": "derive_corr_ab", "inputs": ["est_a"],
                                       "by_setting": {"noise=0.1": 0.41, "noise=0.2": 0.43}},
                           "cover": {"status": "ok", "by": "derive_cover", "inputs": [], "by_setting": {"noise=0.1": 0.94}}},
               "figures": [{"id": "fig1", "file": "fig1_made_up.png", "status": "ok"}]}
    (root / "data" / "results").mkdir(parents=True, exist_ok=True)
    (root / "data" / "results" / "derived.json").write_text(json.dumps(derived), encoding="utf-8")
    (root / "figures").mkdir(exist_ok=True)
    (root / "figures" / "fig1_made_up.png").write_bytes(b"png")
    return root


def _run(engine: Engine, test_src: str, *, file: str = "derive.py", function: str = "", quantity: str = "cover",
         headline: tuple[str, ...] = ("est_a",), fid: str = "F-1", line: int = 0) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The gate's verdict for one finding whose test source is ``test_src`` (the real gate code; only the test's re-run is
    replaced)."""
    import asyncio

    root = _quest(engine)
    rel = f".fi/finding_tests/{fid}_1.py"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(test_src, encoding="utf-8")
    issue = {"id": fid, "file": file, "function": function, "quantity": quantity, "kind": "quantity", "changes_result": True,
             "demonstrated": True, "test": rel, "text": f"the {quantity} is wrong", "outcome": "not repaired",
             "failure_kind": "assertion", "readings": [{"n": 1, "status": "open"}], "severity": "high", "first_seen": 1}
    if line:
        issue["assertion_at"] = f"{fid}_1.py:{line}"
    record = {"issues": {fid: issue}, "read_problems": [{"id": "L1", "kind": "logic", "status": "different", "name": quantity,
                                                         "quantities": [quantity], "fid": fid, "text": "x"}]}
    state = {"design": None, "result_json": RESULT, "exec_result": {"returncode": 0}}
    out = asyncio.run(engine._shown_fault_check([issue], record, RESULT, list(headline), state))
    return out, record


DERIVE_COVER = "import derive\n\n\ndef test_x():\n    assert derive.derive_cover(None) == 0.9, 'wrong'\n"
DERIVE_EST = "import derive\n\n\ndef test_x():\n    assert derive.derive_est_a(None) == 0.9, 'wrong'\n"
DERIVE_CORR = "import derive\n\n\ndef test_x():\n    assert derive.derive_corr_ab(None) == 0.9, 'wrong'\n"


def _removed(engine: Engine) -> dict[str, Any]:
    return json.loads((engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))


# ---- a result worked out afterwards ----------------------------------------------------------------------------------------------
def test_a_fault_in_a_derived_result_leaves_that_result_out_and_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    out, record = _run(engine, DERIVE_CORR, quantity="corr_ab")
    assert out is None  # on the old gate: stopped, "FI cannot find which of the results it changes" (not in the printed results)
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and gate["keys"] == ["corr_ab"]
    kept = _removed(engine)
    assert "corr_ab" in kept["removed_quantities"]


def test_the_numbers_of_a_left_out_derived_result_are_kept_for_the_check_of_the_paper(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _run(engine, DERIVE_CORR, quantity="corr_ab")
    assert sorted(_removed(engine)["removed_values"]["corr_ab"]) == [0.41, 0.43]  # the printed results hold none of them


def test_values_left_out_looks_in_the_record_of_the_analysis_too() -> None:
    derived = {"derived": {"x": {"status": "ok", "by_setting": {"s": 3.5}, "value": 4.5},
                           "y": {"status": "ok", "by_setting": {"s": 9.5}}, "z": {"status": "failed", "by_setting": {"s": 8.5}}}}
    assert _values_left_out({"a": 1.0}, {"x"}, derived) == {"x": [3.5, 4.5]}
    assert _values_left_out({"a": 1.0, "b": 3.5}, {"x"}, derived) == {"x": [4.5]}  # 3.5 is also held by a result that stays
    assert _values_left_out({"a": 1.0}, {"x"}) == {}


def test_a_fault_in_a_derived_result_the_main_result_is_worked_out_from_still_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    out, _ = _run(engine, DERIVE_CORR, quantity="corr_ab", headline=("corr_ab",))
    assert out and out["stuck_reason"] == "shown_fault_open"
    # corr_ab is worked out from est_a (FI's record of the analysis): a fault in derive_est_a reaches it
    out, _ = _run(_engine(tmp_path / "b"), DERIVE_EST, quantity="est_a", headline=("corr_ab",))
    assert out and "corr_ab" in out["rationale"]
    out, _ = _run(_engine(tmp_path / "c"), DERIVE_COVER, quantity="cover", headline=("est_a",))
    assert out is None  # cover is not the main result, and est_a is not worked out from it


# ---- a figure ---------------------------------------------------------------------------------------------------------------------
def test_a_fault_in_a_figure_takes_that_figure_out_and_states_it(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    src = "import derive\n\n\ndef test_x():\n    assert derive.figure_fig1(None, None) == 1, 'wrong'\n"
    out, record = _run(engine, src, function="figure_fig1", quantity="figone")
    assert out is None  # on the old gate: stopped
    assert not (engine.quest_root / "figures" / "fig1_made_up.png").exists()
    assert _removed(engine)["removed_figures"] == ["fig1_made_up.png"]
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"


# ---- a test that calls the trial -------------------------------------------------------------------------------------------------
def _trial_test(read: str) -> str:
    return ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert r['" + read + "'] == 1.0, 'wrong'\n")


def test_a_trial_value_that_is_not_the_main_result_is_left_out_and_the_main_result_is_not_touched(tmp_path: Path) -> None:
    out, record = _run(_engine(tmp_path), _trial_test("est_b"), file="simulate.py", function="trial", quantity="est_b")
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"][0] == "est_b"


@pytest.mark.parametrize("name", ["est_a", "se_a", "lo_a", "hi_a"])
def test_the_main_result_and_its_standard_error_and_interval_ends_stop(tmp_path: Path, name: str) -> None:
    out, _ = _run(_engine(tmp_path), _trial_test(name), file="simulate.py", function="trial", quantity=name)
    assert out and out["stuck_reason"] == "shown_fault_open"


def test_a_returned_name_the_plan_does_not_declare_stops_when_the_trial_also_returns_the_main_result(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path), _trial_test("extra_t"), file="simulate.py", function="trial", quantity="extra_t")
    assert out and "does not declare" in out["rationale"]
    out, record = _run(_engine(tmp_path / "b"), _trial_test("extra_t"), file="simulate.py", function="trial", quantity="extra_t",
                       headline=("corr_ab",))
    assert out is None  # the main result is worked out afterwards, in another function: a plan item not done
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


def test_a_value_fault_with_a_probe_for_an_absent_name_still_stops(tmp_path: Path) -> None:
    src = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert r['est_a'] == 1.0, 'wrong'\n"
           "    assert r.get('est_a_error', 0) == 0\n")
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a")
    assert out and out["stuck_reason"] == "shown_fault_open"


# ---- an assertion that fails only because the names it looks up are not returned ----------------------------------------------------
ABSENT_SRC = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n"
              "    got = r.get('est_a_signed_error', r.get('signed_error_est_a'))\n"
              "    assert got == r['est_a'] - 2.0, 'wrong'\n")


def test_an_assertion_that_looks_a_value_up_only_under_names_the_trial_never_returns_shows_a_part_missing(tmp_path: Path) -> None:
    out, record = _run(_engine(tmp_path), ABSENT_SRC, file="simulate.py", function="trial", quantity="est_a", line=7)
    assert out is None  # on the old gate: stopped, est_a is the main result and the test reads it
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


def test_the_same_assertion_without_the_failing_line_recorded_is_judged_by_the_names_it_reads(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path), ABSENT_SRC, file="simulate.py", function="trial", quantity="est_a")
    assert out and out["stuck_reason"] == "shown_fault_open"


def test_a_failing_assertion_on_a_returned_value_is_not_explained_by_absence(tmp_path: Path) -> None:
    src = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    got = r.get('est_a_signed_error', 0.0)\n"
           "    assert r['est_a'] == 1.0, 'wrong'\n    assert got == 0.0\n")
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a", line=7)
    assert out and out["stuck_reason"] == "shown_fault_open"


def test_absence_is_read_from_the_names_in_the_record_of_the_trials() -> None:
    absent = lambda pieces: pieces == ["gone"]  # noqa: E731
    src = "def test_x(r):\n    got = r.get('gone', None)\n    assert got == 1.0\n"
    assert shown_reach.explained_by_absence(src, 3, absent)
    assert not shown_reach.explained_by_absence(src, 3, lambda pieces: False)
    assert not shown_reach.explained_by_absence(src, 0, absent)


# ---- a helper ----------------------------------------------------------------------------------------------------------------------
def test_a_helper_only_a_derived_result_reaches_leaves_that_result_out(tmp_path: Path) -> None:
    src = "from model import helper_for_cover\n\n\ndef test_x():\n    assert helper_for_cover(1) == 2, 'wrong'\n"
    out, record = _run(_engine(tmp_path), src, file="model.py", function="helper_for_cover", quantity="cover")
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"][0] == "cover"


def test_a_helper_no_result_reaches_changes_no_result(tmp_path: Path) -> None:
    src = "from model import helper_unused\n\n\ndef test_x():\n    assert helper_unused(1) == 2, 'wrong'\n"
    out, record = _run(_engine(tmp_path), src, file="model.py", function="helper_unused", quantity="est_a")
    assert out is None
    assert "at_gate" not in record["issues"]["F-1"]


def test_a_helper_the_trial_reaches_stops_when_the_trial_returns_the_main_result(tmp_path: Path) -> None:
    src = "from model import helper_for_trial\n\n\ndef test_x():\n    assert helper_for_trial(1) == 2, 'wrong'\n"
    out, _ = _run(_engine(tmp_path), src, file="model.py", function="helper_for_trial", quantity="est_a")
    assert out and out["stuck_reason"] == "shown_fault_open"


def test_a_helper_only_a_figure_reaches_takes_that_figure_out(tmp_path: Path) -> None:
    src = "from model import helper_for_figure\n\n\ndef test_x():\n    assert helper_for_figure(1) == 2, 'wrong'\n"
    engine = _engine(tmp_path)
    out, _ = _run(engine, src, file="model.py", function="helper_for_figure", quantity="figone")
    assert out is None
    assert not (engine.quest_root / "figures" / "fig1_made_up.png").exists()


# ---- a test that only checks that something is there, or reads text FI wrote -------------------------------------------------------
@pytest.mark.parametrize("assertion", [
    "assert len(r['lo_a']) == len(r['est_a'])",
    "assert r['lo_a'].size == 5",
    "assert r.get('est_a_error') is not None",
    "assert all(r.get(k) is not None for k in ('est_a', 'cover'))",
    "assert 'cover' in r",
])
def test_a_test_that_only_checks_that_something_is_there_shows_a_part_missing_not_a_wrong_number(tmp_path: Path, assertion: str) -> None:
    src = f"import simulate\n\n\ndef test_x():\n    r = simulate.trial({{}}, None)\n    {assertion}\n"
    out, record = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a")
    assert out is None  # even for the main result: no value is shown wrong
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


@pytest.mark.parametrize("assertion,structural", [
    ("assert len(r) == 3", True), ("assert r['a'] == r['b']", False), ("assert r['a'] == 0.5", False),
    ("assert len(r['a']) == len(r['b']) and r['c'] is None", True), ("assert abs(r['a'] - 1) < 1e-6", False),
    ("assert len(r['a']) == len(r['b'])\n    assert r['a'][0] == 1.0", False),
    ("assert np.isclose(r['a'], 1.0)", False), ("assert r['a'] is not None and r['a'] > 0", False),
])
def test_what_counts_as_an_assertion_about_what_is_there(assertion: str, structural: bool) -> None:
    src = f"import numpy as np\n\n\ndef test_x():\n    r = {{}}\n    {assertion}\n"
    assert shown_reach.read_test(src).structural is structural  # type: ignore[union-attr]


def test_a_failing_assertion_on_text_fi_wrote_from_the_plan_is_kept_in_the_report_and_stops_nothing(tmp_path: Path) -> None:
    src = "import derive\n\n\ndef test_x():\n    assert derive.ROLES['cover'] == 'result', 'wrong role'\n"
    engine = _engine(tmp_path)
    out, record = _run(engine, src, quantity="cover")
    assert out is None
    saved = record["issues"]["F-1"]
    assert saved["changes_result"] is False and saved["about_plan_text"] is True
    assert not (engine.quest_root / "needs" / "RESULT_CHECK.json").exists()  # nothing is left out of the paper


# ---- the call graph ---------------------------------------------------------------------------------------------------------------
def test_the_call_graph_can_start_from_one_function(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    root = _quest(engine)
    got = code_reach.reached(root / "code", scripts=[], harness=None, start=[("derive.py", "derive_cover")])
    assert got is not None and ("model.py", "helper_for_cover") in got
    assert ("model.py", "helper_for_figure") not in got and ("model.py", "helper_unused") not in got


def test_a_computed_getattr_on_the_table_fi_hands_a_body_does_not_make_every_function_reached(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    root = _quest(engine)
    (root / "code" / "derive.py").write_text(
        FILES["derive.py"] + "\n\ndef pick(table, name):\n    return getattr(table, name, None)\n\n\n"
        "def derive_pick(table):\n    return pick(table, 'mean')\n", encoding="utf-8")
    got = code_reach.reached(root / "code", scripts=[], harness=None, start=[("derive.py", "derive_pick")])
    assert got is not None and ("model.py", "helper_unused") not in got
    (root / "code" / "derive.py").write_text(
        FILES["derive.py"] + "\n\ndef pick(thing, name):\n    return getattr(thing, name, None)\n\n\n"
        "def derive_pick(table):\n    return pick(table, 'mean')\n", encoding="utf-8")
    assert code_reach.reached(root / "code", scripts=[], harness=None, start=[("derive.py", "derive_pick")]) is None


# ---- the main result named by the plan --------------------------------------------------------------------------------------------
def test_the_main_result_is_the_declared_id_the_plans_words_name() -> None:
    ids = ["fdr_bh", "power", "any_false", "est_a"]
    assert code_review.headline_ids(["realised fdr_bh (mean over runs)"], ids) == {"fdr_bh"}
    assert code_review.headline_ids(["est_a"], ids) == {"est_a"}
    assert code_review.headline_ids(["something else"], ids) == set()
    assert code_review.headline_ids(["the estimated quantity of interest"], ["q1"], {"q1": "estimated quantity"}) == {"q1"}
