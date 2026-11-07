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

    async def fails(rel: str) -> Any:  # the test is run again by the gate: here it still fails by its assertion
        return types.SimpleNamespace(status="fail", crash_in_program=False, kind="assertion", detail="", place="", at="")

    async def no_reruns(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:  # what these tests judge is the placement alone
        return {}

    engine._rerun_finding_test = fails  # type: ignore[method-assign,assignment]  # this one engine only, never the class
    engine._rerun_shown = no_reruns  # type: ignore[method-assign,assignment]
    return engine


def _quest(engine: Engine, files: dict[str, str] | None = None) -> Path:
    from core import frozen_protocol

    root = engine.quest_root
    for rel, text in {**FILES, **(files or {})}.items():
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
         headline: tuple[str, ...] = ("est_a",), fid: str = "F-1", line: int = 0,
         files: dict[str, str] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The gate's verdict for one finding whose test source is ``test_src`` (the real gate code; only the test's re-run is
    replaced)."""
    import asyncio

    root = _quest(engine, files)
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


# ---- an assertion that looks up names the trial never returned is judged by the values it reads --------------------------------------
ABSENT_SRC = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n"
              "    got = r.get('est_a_signed_error', r.get('signed_error_est_a'))\n"
              "    assert got == r['est_a'] - 2.0, 'wrong'\n")


def test_a_lookup_under_names_the_trial_never_returns_does_not_hide_the_value_the_assertion_reads(tmp_path: Path) -> None:
    # Once such a test was a "part missing" because its lookups were absent; est_a, the main result, is read on the other side.
    out, _ = _run(_engine(tmp_path), ABSENT_SRC, file="simulate.py", function="trial", quantity="signed_error_est_a", line=7)
    assert _stops(out)


def test_a_value_fault_next_to_an_absent_lookup_in_an_and_is_not_a_part_missing(tmp_path: Path) -> None:
    src = TRIAL + "    assert r['est_a'] == 2.0 and r.get('zz_none') is not None\n"
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="bias of est", line=6)
    assert _stops(out)  # the quantity is the reader's words, not an id: the number is placed from the keys the test reads


def test_an_absent_name_with_a_default_that_is_the_value_compared_is_not_a_part_missing(tmp_path: Path) -> None:
    src = TRIAL + "    assert r['est_a'] == r.get('zz_truth', 2.0)\n"
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a_hat", line=6)
    assert _stops(out)


def test_a_name_only_the_absent_lookup_reads_is_a_part_not_done_when_nothing_else_is_read(tmp_path: Path) -> None:
    src = TRIAL + "    assert r.get('zz_none') is not None\n"
    out, record = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="zz_none")
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


def test_a_presence_check_is_placed_by_the_keys_it_reads_whatever_the_reader_called_the_quantity(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    out, record = _run(engine, TRIAL + "    assert 'est_b' in r\n", file="simulate.py", function="trial", quantity="estimate b")
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"  # on the earlier gate: "not computed", est_b stayed
    assert "est_b" in _removed(engine)["removed_quantities"]


def test_a_failing_assertion_on_a_returned_value_is_still_a_stop_next_to_an_absent_default(tmp_path: Path) -> None:
    src = ("import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    got = r.get('est_a_signed_error', 0.0)\n"
           "    assert r['est_a'] == 1.0, 'wrong'\n    assert got == 0.0\n")
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a", line=7)
    assert _stops(out)


def test_a_derived_result_used_by_name_without_a_call_is_what_another_is_worked_out_from(tmp_path: Path) -> None:
    for body in ("    f = derive_cover\n    return f(table) * 2\n", "    return {'c': derive_cover}['c'](table) * 2\n"):
        files = {"derive.py": "def derive_cover(table):\n    return 1\n\n\ndef derive_corr_ab(table):\n" + body}
        engine = _engine(tmp_path / str(abs(hash(body))))
        out, _ = _run(engine, "import derive\n\n\ndef test_x():\n    assert derive.derive_cover(None) == 0.9, 'wrong'\n",
                      quantity="cover", files=files)
        assert out is None
        assert set(_removed(engine)["removed_quantities"]) == {"corr_ab", "cover"}  # on the earlier gate: corr_ab stayed


def test_a_figure_function_passed_on_or_aliased_is_still_used_by_the_result(tmp_path: Path) -> None:
    for body in ("    g = figure_fig1\n    return g(None, table)\n", "    return apply(figure_fig1, table)\n"):
        files = {"derive.py": "def figure_fig1(fig, table):\n    return 3\n\n\ndef derive_cover(table):\n" + body}
        engine = _engine(tmp_path / str(abs(hash(body))))
        out, _ = _run(engine, "import derive\n\n\ndef test_x():\n    assert derive.figure_fig1(None, None) == 1, 'wrong'\n",
                      quantity="fig1", files=files)
        assert out is None
        assert "cover" in _removed(engine)["removed_quantities"]


def test_a_function_the_study_defines_is_never_a_plain_call_whatever_its_name(tmp_path: Path) -> None:
    files = {"model.py": FILES["model.py"] + "\n\ndef count(x):\n    return x\n",
             "simulate.py": "import model\n\n\ndef trial(cell, ctx):\n    return {'est_a': model.count(1), 'est_b': 2.0}\n"}
    src = "import derive\nfrom model import count\n\n\ndef test_x():\n    assert count(derive.ROLES['cover']) == 2, 'wrong'\n"
    out, _ = _run(_engine(tmp_path), src, file="model.py", function="count", quantity="est_a", files=files)
    assert _stops(out)  # on the earlier gate: kept in the quality report, as only about the plan


def test_eval_is_not_a_plain_call(tmp_path: Path) -> None:
    src = "import derive\nfrom model import helper_for_trial\n\n\ndef test_x():\n    assert eval(derive.ROLES['cover']) == 2, 'wrong'\n"
    out, record = _run(_engine(tmp_path), src, file="model.py", function="helper_for_trial", quantity="est_a")
    assert "about_plan_text" not in record["issues"]["F-1"]


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


# ---- a name the test looks for that nothing holds, and a test that reads text FI wrote ---------------------------------------------
@pytest.mark.parametrize("assertion", [
    "assert r.get('est_a_error') is not None",
    "assert 'cover_error' in r",
])
def test_a_test_that_looks_for_a_quantity_nothing_holds_is_a_part_not_done(tmp_path: Path, assertion: str) -> None:
    src = f"import simulate\n\n\ndef test_x():\n    r = simulate.trial({{}}, None)\n    {assertion}\n"
    out, record = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a_error")
    assert out is None  # the run holds no value for it: nothing to leave out
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "not_computed"


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


# ---- round 2 of the review: each fix, with the input that went wrong ----------------------------------------------------------------
TRIAL = "import simulate\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n"


def _stops(out: dict[str, Any] | None) -> bool:
    return bool(out and out.get("stuck_reason") == "shown_fault_open")


def test_a_part_a_test_looks_for_that_the_run_holds_a_value_for_is_left_out_not_kept(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    out, record = _run(engine, TRIAL + "    assert 'est_b' in r\n", file="simulate.py", function="trial", quantity="est_b")
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and "incomplete" not in gate and "rerun" not in gate  # on the old gate: "not computed"
    kept = _removed(engine)
    assert "est_b" in kept["removed_quantities"]
    said = " ".join(c.get("say", "") + c.get("note", "") for c in kept["checks"])
    assert "did not establish whether the value is wrong or missing" in said and "wrongly" not in said
    assert "missing or incomplete" not in said


def test_a_part_a_test_looks_for_that_is_the_main_result_stops_the_paper(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path), TRIAL + "    assert 'est_a' in r\n", file="simulate.py", function="trial", quantity="est_a")
    assert _stops(out)


def test_a_derived_result_a_test_looks_for_is_left_out_with_what_is_worked_out_from_it(tmp_path: Path) -> None:
    files = {"derive.py": FILES["derive.py"].replace("return 0.5", "return table.derived('cover') * 2")}
    engine = _engine(tmp_path)
    out, record = _run(engine, "import derive\n\n\ndef test_x():\n    assert derive.derive_cover(None) is not None\n",
                       quantity="cover", files=files)
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["outcome"] == "left_out"
    assert set(_removed(engine)["removed_quantities"]) >= {"cover", "corr_ab"}


def test_the_words_said_of_a_failing_test_with_no_rerun_do_not_say_the_code_computes_it_wrongly(tmp_path: Path) -> None:
    from core import disclosure

    engine = _engine(tmp_path)
    _run(engine, TRIAL + "    assert 'est_b' in r\n", file="simulate.py", function="trial", quantity="est_b")
    text = disclosure._code_faults_sentence(engine.quest_root)
    assert "did not establish whether the value is wrong or missing" in text and "wrongly" not in text


def test_a_lookup_in_a_dictionary_the_test_built_is_not_a_name_the_trial_failed_to_return(tmp_path: Path) -> None:
    src = TRIAL + "    truth = {'theta': 2.0}\n    assert r['est_a'] == truth['theta'], 'wrong'\n"
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a", line=7)
    assert _stops(out)  # on the earlier gate: delivered, as a part not done


def test_a_test_that_calls_the_trial_and_a_derived_result_places_the_fault_by_both(tmp_path: Path) -> None:
    src = "import simulate, derive\n\n\ndef test_x():\n    r = simulate.trial({}, None)\n    assert r['est_a'] == 2.0, 'wrong'\n    derive.derive_cover(None)\n"
    out, _ = _run(_engine(tmp_path), src, file="simulate.py", function="trial", quantity="est_a", line=6)
    assert _stops(out)  # on the earlier gate: only cover was left out and est_a stayed


HELPER = "from model import helper_for_trial\n\n\ndef test_x():\n    assert helper_for_trial(1) == 2, 'wrong'\n"


def test_a_helper_only_experiment_py_reaches_for_the_trial_is_not_lost(tmp_path: Path) -> None:
    files = {"simulate.py": "", "experiment.py": "import model\n\n\ndef run_trial(cell, trial_id, seed):\n    return {'est_a': model.helper_for_trial(1)}\n"}
    out, _ = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files=files)
    assert _stops(out)  # on the earlier gate: "no function reaches it", delivered


def test_a_helper_only_the_oracle_reaches_is_not_lost(tmp_path: Path) -> None:
    files = {"simulate.py": "import model\n\n\ndef trial(cell, ctx):\n    return {'est_a': 1.0}\n\n\ndef oracle():\n    return {'truth_a': model.helper_for_trial(1)}\n"}
    out, _ = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files=files)
    assert _stops(out)


def test_when_no_entry_of_the_study_is_found_a_helper_is_judged_as_before(tmp_path: Path) -> None:
    files = {"simulate.py": "", "derive.py": ""}
    out, _ = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files=files)
    assert _stops(out)  # nothing was asked of the call graph: it is not "reached by no result"


def test_a_derived_result_another_derived_result_calls_is_what_it_is_worked_out_from(tmp_path: Path) -> None:
    files = {"derive.py": "def derive_cover(table):\n    return 1\n\n\ndef derive_corr_ab(table):\n    return derive_cover(table) * 2\n"}
    src = "import derive\n\n\ndef test_x():\n    assert derive.derive_cover(None) == 0.9, 'wrong'\n"
    out, _ = _run(_engine(tmp_path), src, quantity="cover", headline=("corr_ab",), files=files)
    assert _stops(out)


def test_a_derived_result_named_by_a_module_constant_is_read_too(tmp_path: Path) -> None:
    files = {"derive.py": "SOURCE = 'cover'\n\n\ndef derive_cover(table):\n    return 1\n\n\ndef derive_corr_ab(table):\n    return table.derived(SOURCE) * 2\n"}
    src = "import derive\n\n\ndef test_x():\n    assert derive.derive_cover(None) == 0.9, 'wrong'\n"
    out, _ = _run(_engine(tmp_path), src, quantity="cover", headline=("corr_ab",), files=files)
    assert _stops(out)


def test_what_is_worked_out_from_a_left_out_result_is_left_out_with_it(tmp_path: Path) -> None:
    files = {"derive.py": "def derive_cover(table):\n    return 1\n\n\ndef derive_corr_ab(table):\n    return table.derived('cover') * 2\n"}
    engine = _engine(tmp_path)
    out, _ = _run(engine, "import derive\n\n\ndef test_x():\n    assert derive.derive_cover(None) == 0.9, 'wrong'\n", quantity="cover", files=files)
    assert out is None
    assert set(_removed(engine)["removed_quantities"]) == {"corr_ab", "cover"}  # on the earlier gate: corr_ab stayed


def test_an_undeclared_returned_name_a_derived_result_reads_reaches_that_result(tmp_path: Path) -> None:
    files = {"derive.py": "def derive_cover(table):\n    return 1\n\n\ndef derive_corr_ab(table):\n    return table.col('extra_t').mean()\n"}
    out, _ = _run(_engine(tmp_path), TRIAL + "    assert r['extra_t'] == 7.0, 'wrong'\n", file="simulate.py", function="trial",
                  quantity="extra_t", headline=("corr_ab",), files=files)
    assert _stops(out)  # on the earlier gate: delivered, corr_ab is built on extra_t


def test_a_presence_check_of_the_main_derived_result_stops(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path), "import derive\n\n\ndef test_x():\n    r = derive.derive_corr_ab(None)\n    assert r is not None\n",
                  quantity="corr_ab", headline=("corr_ab",))
    assert _stops(out)


def test_a_count_compared_with_a_count_is_not_a_presence_check(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path), "import derive\n\n\ndef test_x():\n    r = derive.derive_corr_ab(None)\n    assert len(r) == 2\n",
                  quantity="corr_ab", headline=("corr_ab",))
    assert _stops(out)


def test_a_check_that_a_trial_value_is_a_float_on_the_main_result_stops(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path), TRIAL + "    assert isinstance(r['est_a'], float)\n", file="simulate.py", function="trial", quantity="est_a")
    assert _stops(out)


def test_a_test_that_reads_the_frame_and_calls_the_study_is_not_only_about_the_plan(tmp_path: Path) -> None:
    src = "import derive\nfrom model import helper_for_trial\n\n\ndef test_x():\n    assert helper_for_trial(derive.ROLES['cover']) == 2, 'wrong'\n"
    out, _ = _run(_engine(tmp_path), src, file="model.py", function="helper_for_trial", quantity="est_a")
    assert _stops(out)  # on the earlier gate: kept in the quality report, the helper's fault ignored


def test_a_figure_function_a_result_uses_is_a_helper_of_that_result(tmp_path: Path) -> None:
    files = {"derive.py": "def figure_fig1(fig, table):\n    return 3\n\n\ndef derive_cover(table):\n    return figure_fig1(None, table) * 2\n\n\ndef derive_corr_ab(table):\n    return 0.5\n"}
    engine = _engine(tmp_path)
    out, _ = _run(engine, "import derive\n\n\ndef test_x():\n    assert derive.figure_fig1(None, None) == 1, 'wrong'\n",
                  quantity="fig1", headline=("corr_ab",), files=files)
    assert out is None
    kept = _removed(engine)
    assert "cover" in kept["removed_quantities"] and kept["removed_figures"]  # on the earlier gate: only the figure


def test_a_record_that_cannot_be_read_for_the_keys_does_not_end_the_gate_in_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self: Any, ids: Any, result: Any) -> Any:
        raise RuntimeError("unreadable")

    monkeypatch.setattr(shown_reach.Reach, "printed_keys", broken)
    out, _ = _run(_engine(tmp_path), DERIVE_COVER, quantity="cover")
    assert out is None or _stops(out)  # judged as before, not an exception


def test_a_finding_about_text_fi_wrote_from_the_plan_is_told_in_the_disclosure_and_the_quality_report(tmp_path: Path) -> None:
    from core import disclosure

    engine = _engine(tmp_path)
    src = "import derive\n\n\ndef test_x():\n    assert derive.ROLES['cover'] == 'result', 'wrong role'\n"
    _run(engine, src, quantity="cover")
    assert "text FI wrote from the plan" in disclosure._code_faults_sentence(engine.quest_root)


# ---- a test that names a declared result by a literal key and calls no function -------------------------------------------------------
READS_KEY = "def test_x():\n    rows = {}\n    assert rows['est_b'] > 0, 'wrong'\n"
READS_OTHER = "def test_x():\n    rows = {}\n    assert rows['nothing_declared'] > 0, 'wrong'\n"


def test_a_test_that_reads_a_declared_key_and_calls_no_function_leaves_that_result_out(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    out, record = _run(engine, READS_KEY, quantity="est_b", headline=("est_a",))
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and "est_b" in gate["keys"]
    from core import frozen_protocol

    root = engine.quest_root
    reach = shown_reach.Reach(root, frozen_protocol.protocol_of(root), ["est_a"], [])
    issue = {"id": "F-1", "test": ".fi/finding_tests/F-1_1.py", "quantity": "est_b", "function": ""}
    placed = reach.effect({"issues": {"F-1": issue}}, issue, RESULT)  # placed from the key the test names, not from the finding's words
    assert placed is not None and placed.left_out == ["est_b"] and not placed.stop


def test_a_literal_key_that_is_the_main_result_stops_and_an_undeclared_key_places_nothing(tmp_path: Path) -> None:
    out, _ = _run(_engine(tmp_path / "a"), READS_KEY, quantity="est_b", headline=("est_b",))
    assert out and "main result" in out["rationale"]
    engine = _engine(tmp_path / "b")
    _run(engine, READS_OTHER, quantity="est_b", headline=("est_a",))
    from core import frozen_protocol

    root = engine.quest_root
    reach = shown_reach.Reach(root, frozen_protocol.protocol_of(root), ["est_a"], [])
    issue = {"id": "F-1", "test": ".fi/finding_tests/F-1_1.py", "quantity": "est_b", "function": ""}
    assert reach.effect({"issues": {"F-1": issue}}, issue, RESULT) is None


# ---- a helper of the trial is tied to the results whose value is worked out with it -----------------------------------------------------
TRIAL_KEYS = ("import model\n\n\ndef trial(cell, ctx):\n    out = {}\n    x = model.helper_for_trial(1)\n    y = x * 2\n"
              "    out['est_b'] = y\n    out['est_a'] = 1.0\n    out['se_a'] = 0.5\n    return out\n")


def test_a_helper_only_one_result_of_the_trial_is_worked_out_with_leaves_that_result_out(tmp_path: Path) -> None:
    # the helper feeds est_b through two assignments; the main result est_a does not use it. The call graph alone (the trial reaches
    # it) ties it to every result of the trial; the trial's own code says which value is worked out with it.
    out, record = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a",
                       files={"simulate.py": TRIAL_KEYS})
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"][0] == "est_b"
    assert "est_a" not in record["issues"]["F-1"]["at_gate"]["keys"]


def test_a_helper_the_main_result_is_worked_out_with_through_the_trial_still_stops(tmp_path: Path) -> None:
    src = TRIAL_KEYS.replace("out['est_a'] = 1.0", "out['est_a'] = x + 1.0")
    out, _ = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files={"simulate.py": src})
    assert _stops(out)


def test_a_helper_that_decides_by_a_condition_which_result_is_returned_reaches_what_the_condition_guards(tmp_path: Path) -> None:
    src = ("import model\n\n\ndef trial(cell, ctx):\n    out = {}\n    if model.helper_for_trial(1) > 0:\n        out['est_a'] = 1.0\n"
           "    out['est_b'] = 2.0\n    return out\n")
    out, _ = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files={"simulate.py": src})
    assert _stops(out)


def test_a_trial_whose_keys_are_computed_keeps_the_helper_tied_to_every_result_of_the_trial(tmp_path: Path) -> None:
    src = ("import model\n\n\ndef trial(cell, ctx):\n    out = {}\n    for k in ('est_a', 'est_b'):\n        out[k] = model.helper_for_trial(1)\n"
           "    return out\n")
    out, _ = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files={"simulate.py": src})
    assert _stops(out)


def test_a_helper_reached_through_another_helper_of_the_trial_is_tied_to_the_results_that_use_that_helper(tmp_path: Path) -> None:
    files = {"simulate.py": ("import model\n\n\ndef _inner(x):\n    return model.helper_for_trial(x)\n\n\ndef trial(cell, ctx):\n"
                             "    out = {}\n    out['est_b'] = _inner(3)\n    out['est_a'] = 1.0\n    return out\n")}
    out, record = _run(_engine(tmp_path), HELPER, file="model.py", function="helper_for_trial", quantity="est_a", files=files)
    assert out is None
    assert record["issues"]["F-1"]["at_gate"]["keys"][0] == "est_b"
