"""The 2026-09-26 re-audit's P1 findings: paired designs join trials by id, an exploration stays preliminary, the
environment is the one the experiment ran on, and the trace hashes the evidence records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from core import evidence, metric_spec
from core.engine import Engine
from core.interview import answers_to_yaml
from tests.test_evidence import ON, _quest, _state
from tests.test_interview_roundtrip import _full_answers

PAIRED = {
    "metrics": [{"id": "score", "estimand": "mean paired score difference", "kind": "mean", "unit": "trial",
                 "paired": True}],
    "contrasts": [{"metric": "score", "a": "a", "b": "b"}],
}


def _replicates(with_ids: bool) -> list[dict]:
    def one(values: list[float]) -> dict:
        return {"score_values": values, **({"score_pair_id": [0, 1, 2]} if with_ids else {})}

    return [{"_seed": s, "by_method": {"a": one([1.0, 2.0, 3.0]), "b": one([0.5, 1.5, 2.5])}} for s in range(3)]


def test_a_paired_contrast_joined_by_position_is_a_gap_and_one_joined_by_id_is_not() -> None:
    by_position = metric_spec.statistics(_replicates(False), PAIRED)
    assert by_position["contrasts"][0]["paired_id_verified"] is False
    gaps = metric_spec.coverage_gaps(PAIRED, _replicates(False), by_position)
    assert any("by their position" in g for g in gaps)
    by_id = metric_spec.statistics(_replicates(True), PAIRED)
    assert not any("by their position" in g for g in metric_spec.coverage_gaps(PAIRED, _replicates(True), by_id))


def test_research_asks_for_the_pair_ids_a_paired_metric_is_printed_without() -> None:
    no_ids = {"by_method": {"a": {"score_values": [1.0, 2.0]}, "b": {"score_values": [0.5, 1.5]}}}
    (found,) = metric_spec.missing_pair_ids(PAIRED, no_ids)
    assert "`score_pair_id`" in found and "FI_TRIALS" in found
    with_ids = {"by_method": {"a": {"score_values": [1.0, 2.0], "score_pair_id": [0, 1]},
                              "b": {"score_values": [0.5, 1.5], "score_pair_id": [0, 1]}}}
    assert metric_spec.missing_pair_ids(PAIRED, with_ids) == []
    assert metric_spec.missing_pair_ids({"metrics": [{**PAIRED["metrics"][0], "paired": False}]}, no_ids) == []
    # One setting's ids do not stand for another's; a setting run once has nothing to pair.
    one_side = {"by_method": {"a": with_ids["by_method"]["a"], "b": no_ids["by_method"]["b"]}}
    assert len(metric_spec.missing_pair_ids(PAIRED, one_side)) == 1
    once = {"by_method": {"a": {"score_values": [1.0]}, "b": {"score_values": [0.5]}}}
    assert metric_spec.missing_pair_ids(PAIRED, once) == []


def test_a_paired_design_gives_each_trial_number_one_seed_across_the_settings(tmp_path: Path) -> None:
    from core.trial_runner import _plan

    grid = {"method": ["a", "b"]}
    folder = tmp_path / "specs"
    folder.mkdir()
    kw = dict(runs_per_setting=3, base_seed=7, deterministic=False, folder=folder, out_name="o{index}.jsonl")
    paired = _plan(tmp_path, "code/simulate.py", grid, paired=True, **kw)
    assert [t["seed"] for t in paired[0]["trials"]] == [t["seed"] for t in paired[1]["trials"]]
    assert len({t["seed"] for t in paired[0]["trials"]}) == 3
    unpaired = _plan(tmp_path, "code/simulate.py", grid, **kw)
    assert [t["seed"] for t in unpaired[0]["trials"]] != [t["seed"] for t in unpaired[1]["trials"]]


def test_fis_summary_names_the_trial_behind_every_value() -> None:
    from core.trial_runner import CellRun, _summary

    run = CellRun(key="R0=1.5", cell={"R0": 1.5}, planned=3, returncode=0, timed_out=False, stderr="")
    run.rows = [{"trial": 0, "status": "ok", "values": {"x": 1.0}}, {"trial": 1, "status": "failed"},
                {"trial": 2, "status": "ok", "values": {"x": 3.0}}]
    metric = _summary([run])["cells"][0]["metrics"]["x"]
    assert metric["values"] == [1.0, 3.0] and metric["trials"] == [0, 2]


def test_an_exploration_is_kept_and_is_never_publication_ready(tmp_path: Path) -> None:
    yaml_text = answers_to_yaml(replace(_full_answers(), result_use="explore"), frontend="cli")
    assert 'result_use: "explore"' in yaml_text
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    assert evidence.assess(root, _state(), settings=ON)["status"] == "publication_ready"
    explored = evidence.assess(root, _state(), settings={**ON, "result_use": "explore"})
    assert explored["status"] != "publication_ready"
    assert any("set up to explore" in g for g in explored["gaps"])


def test_the_vscode_interview_writes_the_result_use_too() -> None:
    ts = (Path(__file__).resolve().parent.parent / "vscode-frontier-insight" / "src" / "interview-core.ts").read_text(
        encoding="utf-8")
    assert "lines.push(`result_use: \"${answers.result_use}\"`);" in ts


def test_a_package_list_that_could_not_be_made_is_a_gap_even_in_its_own_environment(tmp_path: Path) -> None:
    root = _quest(tmp_path, protocol_status="ok", oracle_status="ok")
    (root / "needs" / "ENVIRONMENT.json").write_text(json.dumps({
        "isolated": True, "shared_interpreter": False, "system_site_packages": False,
        "packages_error": "pip freeze failed",
    }), encoding="utf-8")
    record = evidence.assess(root, _state(), settings=ON)
    assert record["status"] != "publication_ready"
    assert any("could not be listed" in g for g in record["all_gaps"]["protocol_runtime_matched"])


def test_the_trace_hashes_the_records_the_evidence_is_read_from() -> None:
    watched = set(Engine._AUDIT_WATCHED)
    for rel in (".fi/approved_plan.json", "needs/receipts/evidence_gate.json", "needs/receipts/claim_check.json",
                "needs/receipts/design_audit.json", "needs/ENVIRONMENT.json", "raw/ledger.jsonl", "raw/trials.json",
                ".fi/trials/run.json"):
        assert rel in watched, rel
