"""The shown-fault gate places a fault from FI's records (the plan's contract, the record of the trials, the record of the
analysis, the code's own call graph) instead of from the words of the finding. Neutral made-up study: two estimates with a
true value, a coverage and a correlation worked out afterwards, one figure, and a few helpers. No model is called; the
re-run of a finding's test is replaced by "still fails" (what the gate decides from that is what these tests are about)."""
from __future__ import annotations

import json
from pathlib import Path


from core import code_reach, code_review
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


def test_values_left_out_looks_in_the_record_of_the_analysis_too() -> None:
    derived = {"derived": {"x": {"status": "ok", "by_setting": {"s": 3.5}, "value": 4.5},
                           "y": {"status": "ok", "by_setting": {"s": 9.5}}, "z": {"status": "failed", "by_setting": {"s": 8.5}}}}
    assert _values_left_out({"a": 1.0}, {"x"}, derived) == {"x": [3.5, 4.5]}
    assert _values_left_out({"a": 1.0, "b": 3.5}, {"x"}, derived) == {"x": [4.5]}  # 3.5 is also held by a result that stays
    assert _values_left_out({"a": 1.0}, {"x"}) == {}


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
