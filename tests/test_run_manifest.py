"""What a simulation says it did (core/run_manifest.py) against the frozen protocol, and the strict two-script contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import run_manifest as rm
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

PROTOCOL = {"grid": {"R0": [0.9, 1.5, 3.0]}, "runs_per_setting": 300, "thresholds": {"outbreak": 0.1}}


def _manifest(**over: Any) -> dict[str, Any]:
    cells = {f"R0={r}": 300 for r in (0.9, 1.5, 3.0)}
    base = {
        "schema": rm.SCHEMA, "realized_grid": {"R0": [0.9, 1.5, 3.0]}, "attempted_per_cell": dict(cells),
        "successful_per_cell": dict(cells), "failed_trials": [], "thresholds_used": {"outbreak": 0.1},
    }
    return {**base, **over}


# --- reading and comparing ---------------------------------------------------------------------------------------------


def test_a_manifest_that_says_what_the_protocol_fixed_has_no_difference() -> None:
    assert rm.problems(PROTOCOL, _manifest()) == []


# --- the claimed trial count against the real per-trial values a mean metric reports -------------------------------


_MEAN_PROTOCOL = {**PROTOCOL, "metrics": [{"id": "final_size", "kind": "mean", "estimand": "x", "unit": "y"}]}


def _result_json_with(n_per_cell: int) -> dict[str, Any]:
    """A real result_json shaped like the audit's own bypass example: attempted_per_cell can claim
    anything, but this is what the analysis actually had to work with."""
    return {"by_R0": {str(r): {"final_size_values": [0.1] * n_per_cell} for r in (0.9, 1.5, 3.0)}}


def test_a_manifest_reports_300_but_the_analysis_only_has_10_real_values_is_caught() -> None:
    """The audit's own concrete bypass: a script that ran 10 trials writes attempted_per_cell=300 by
    copying the protocol. Without a real result to check against this passed; with it, it does not."""
    found = rm.problems(_MEAN_PROTOCOL, _manifest(), result_json=_result_json_with(10))
    assert any("not backed by the data the analysis actually used" in f for f in found), found


def test_a_manifest_backed_by_enough_real_values_is_not_flagged() -> None:
    found = rm.problems(_MEAN_PROTOCOL, _manifest(), result_json=_result_json_with(300))
    assert found == []


def test_the_value_count_check_is_a_noop_without_a_mean_metric_or_a_result(tmp_path: Path) -> None:  # noqa: ARG001
    # No result_json at all: unchanged from before this check existed.
    assert rm.problems(_MEAN_PROTOCOL, _manifest()) == []
    # A result_json, but the protocol names no mean-kind metric: nothing to check against.
    assert rm.problems(PROTOCOL, _manifest(), result_json=_result_json_with(10)) == []


_GIVEN_PROTOCOL = {**PROTOCOL, "metrics": [
    {"id": "prob_major", "kind": "proportion", "estimand": "P(major | R0)", "unit": "run"},
    {"id": "final_size", "kind": "mean", "estimand": "E[final size | major]", "unit": "run", "given": "prob_major"},
]}


def _conditional(majors: dict[float, int], values_per_major: float = 1.0, total: int = 300) -> dict[str, Any]:
    return {"by_R0": {str(r): {
        "prob_major": k / total, "prob_major_count": k, "prob_major_total": total,
        "final_size_values": [0.6] * int(k * values_per_major),
    } for r, k in majors.items()}}


def test_a_mean_over_a_declared_subset_is_backed_by_that_subset_not_by_every_trial() -> None:
    """A live quest's final size over major outbreaks had 1022 values from 2700 runs, one per major outbreak, and this
    check read the other 1678 runs as missing. Declared with `given`, it is checked against the outbreaks counted."""
    majors = {0.9: 3, 1.5: 100, 3.0: 200}
    assert rm.problems(_GIVEN_PROTOCOL, _manifest(), result_json=_conditional(majors)) == []
    undeclared = {**PROTOCOL, "metrics": [_GIVEN_PROTOCOL["metrics"][1] | {"given": None}]}
    undeclared["metrics"][0].pop("given")
    found = rm.problems(undeclared, _manifest(), result_json=_conditional(majors))
    assert len(found) == 1 and "says so with `given`" in found[0], "without the declaration it is still flagged, and says how"


def test_a_declared_subset_still_catches_fabricated_values_and_missing_counts() -> None:
    majors = {0.9: 3, 1.5: 100, 3.0: 200}
    short = rm.problems(_GIVEN_PROTOCOL, _manifest(), result_json=_conditional(majors, values_per_major=0.1))
    assert len(short) == 1 and "a mean over the trials `prob_major` counts (303)" in short[0]
    # The proportion's own total must still account for the trials the manifest claims.
    thin = rm.problems(_GIVEN_PROTOCOL, _manifest(), result_json=_conditional(majors, total=10))
    assert len(thin) == 1 and "counts only 30 trial(s) in its `prob_major_total`" in thin[0]
    # No count for the subset at all: the analysis's to print.
    no_count = {"by_R0": {str(r): {"final_size_values": [0.6] * 5} for r in (0.9, 1.5, 3.0)}}
    assert rm.analysis_output_problems(_GIVEN_PROTOCOL, _manifest(), no_count) == rm.problems(
        _GIVEN_PROTOCOL, _manifest(), result_json=no_count)
    assert "prints no `prob_major_count`" in rm.analysis_output_problems(_GIVEN_PROTOCOL, _manifest(), no_count)[0]


def test_given_must_name_a_declared_proportion_and_sit_on_a_mean() -> None:
    from core import metric_spec as ms

    prop = {"id": "p", "kind": "proportion", "estimand": "P", "unit": "run"}
    mean = {"id": "m", "kind": "mean", "estimand": "E", "unit": "run", "given": "p"}
    assert ms.normalize([prop, mean])[0] is not None
    assert ms.normalize([mean])[0] is None and ms.normalize([{**prop, "given": "m"}, mean])[0] is None
    assert ms.normalize([prop, {**mean, "given": "m"}])[0] is None
    # In a draft, a mean declared before the proportion it is given is still kept.
    kept, notes = ms.repair([mean, prop])
    assert {m["id"] for m in kept} == {"p", "m"} and notes == []


def test_an_analysis_that_lists_no_values_is_the_analysis_to_fix_not_the_simulation() -> None:
    """A live kimi-k3 quest's experiment.py printed no `<metric>_values` at all, while simulate.py had saved every final
    size on disk; the difference sent simulate.py back, and its rewrite broke the oracle checks it had passed. No list at
    all is the analysis's; a list far shorter than the claim is still the audit's fabrication case, the simulation's."""
    no_list = {"by_R0": {str(r): {"final_size_mean": 0.3} for r in (0.9, 1.5, 3.0)}}
    found = rm.problems(_MEAN_PROTOCOL, _manifest(), result_json=no_list)
    assert len(found) == 1 and "has no `final_size_values` list at all" in found[0], found
    assert rm.analysis_output_problems(_MEAN_PROTOCOL, _manifest(), no_list) == found
    short = _result_json_with(10)
    assert rm.problems(_MEAN_PROTOCOL, _manifest(), result_json=short)
    assert rm.analysis_output_problems(_MEAN_PROTOCOL, _manifest(), short) == [], "10 real values against 300 claimed stays the simulation's"
    assert rm.analysis_output_problems(_MEAN_PROTOCOL, None, no_list) == []
    assert "experiment.py (not simulate.py)" in rm.analysis_directive(found)


def test_a_declared_failure_rate_is_not_mistaken_for_fabrication() -> None:
    """A design with a genuine ~15% solver-divergence rate, correctly excluding failed trials from
    its value arrays, must not trip the value-count check -- the baseline is trials the manifest
    says SUCCEEDED (255 here), not the protocol's raw 300, and 255 real values back that exactly."""
    cells = {"R0=0.9": 255, "R0=1.5": 300, "R0=3.0": 300}
    manifest = _manifest(successful_per_cell=cells, failed_trials=[
        {"cell": "R0=0.9", "trial": i, "reason": "solver diverged"} for i in range(45)
    ])
    result_json = {"by_R0": {
        "0.9": {"final_size_values": [0.1] * 255}, "1.5": {"final_size_values": [0.1] * 300},
        "3.0": {"final_size_values": [0.1] * 300},
    }}
    found = rm.problems({**_MEAN_PROTOCOL, "failure_policy": "excluded from the pooled estimate"}, manifest, result_json=result_json)
    assert found == []


def test_reading_says_why_a_manifest_cannot_be_used(tmp_path: Path) -> None:
    assert rm.read(tmp_path)[0] is None and "was not written" in rm.read(tmp_path)[1]
    (tmp_path / rm.NAME).write_text("{not json", encoding="utf-8")
    assert "could not be read as JSON" in rm.read(tmp_path)[1]
    (tmp_path / rm.NAME).write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    assert "does not say" in rm.read(tmp_path)[1]
    (tmp_path / rm.NAME).write_text(json.dumps(_manifest()), encoding="utf-8")
    assert rm.read(tmp_path) == (_manifest(), "")
    assert rm.problems(PROTOCOL, None, "no manifest here") == ["no manifest here"]


def test_a_grid_the_run_swept_differently_is_named_value_by_value() -> None:
    found = rm.problems(PROTOCOL, _manifest(realized_grid={"R0": [0.9, 3.0, 4.5]}))
    assert len(found) == 1 and "leaves out [1.5]" in found[0] and "adds [4.5]" in found[0]
    assert "reports no realized values" in rm.problems(PROTOCOL, _manifest(realized_grid={}))[0]


def test_a_run_count_that_differs_in_any_setting_is_a_shortfall() -> None:
    short = _manifest(attempted_per_cell={"R0=0.9": 300, "R0=1.5": 30, "R0=3.0": 300}, successful_per_cell={"R0=0.9": 300, "R0=1.5": 30, "R0=3.0": 300})
    found = rm.problems(PROTOCOL, short)
    assert any("ran another number (R0=1.5: 30)" in f for f in found)
    assert any("no attempted trials per cell" in f for f in rm.problems(PROTOCOL, _manifest(attempted_per_cell={})))
    fewer_cells = _manifest(attempted_per_cell={"R0=0.9": 300}, successful_per_cell={"R0=0.9": 300})
    assert any("grid has 3 setting(s) exactly" in f for f in rm.problems(PROTOCOL, fewer_cells))


def test_a_failed_trial_must_be_listed_and_the_protocol_must_say_how_failures_are_treated() -> None:
    ok = {"R0=0.9": 300, "R0=1.5": 297, "R0=3.0": 300}
    lost = _manifest(successful_per_cell=ok)
    found = rm.problems(PROTOCOL, lost)
    assert any("3 trial(s) did not complete, and the run lists 0" in f for f in found)
    listed = _manifest(successful_per_cell=ok, failed_trials=[{"cell": "R0=1.5", "trial": i, "reason": "solver"} for i in range(3)])
    # a listed failure is not a difference; a protocol with no policy for it is said in the plan and named by the evidence level
    assert rm.problems(PROTOCOL, listed) == [] and rm.problems({**PROTOCOL, "failure_policy": "counted as failures"}, listed) == []
    assert rm.failure_count(listed) == 3 and rm.failure_count(_manifest()) == 0 and rm.failure_count(None) == 0
    impossible = _manifest(successful_per_cell={"R0=0.9": 301, "R0=1.5": 300, "R0=3.0": 300})
    assert any("reports 301 completed" in f for f in rm.problems(PROTOCOL, impossible))


def test_a_threshold_the_run_did_not_use_is_named() -> None:
    assert "used 0.05" in rm.problems(PROTOCOL, _manifest(thresholds_used={"outbreak": 0.05}))[0]
    assert "reports none" in rm.problems(PROTOCOL, _manifest(thresholds_used={}))[0]


def test_only_a_protocol_that_fixes_something_can_be_compared() -> None:
    assert rm.checkable(PROTOCOL) and rm.checkable({"runs_per_setting": 3}) and rm.checkable({"thresholds": {"a": 1}})
    assert not rm.checkable({"oracles": [{"name": "x"}]}) and not rm.checkable(None) and not rm.checkable({})


def test_the_contract_lint_names_a_simulation_that_publishes_or_an_analysis_that_simulates() -> None:
    clean = {"simulate.py": "import numpy as np\nnp.save('x', 1)\n", "experiment.py": "import json\nprint('RESULT_JSON: {}')\n"}
    assert rm.split_lint(clean) == []
    found = rm.split_lint({"simulate.py": "print('RESULT_JSON: {}')\nplt.savefig('a.png')", "experiment.py": "from simulate import run\n"})
    assert len(found) == 3 and any("imports or runs simulate.py" in f for f in found)
    assert rm.split_lint({"experiment.py": "import subprocess\nsubprocess.run(['python', 'simulate.py'])"})


# --- through the real graph ---------------------------------------------------------------------------------------------

_SIM = """\
import json, os, pathlib
R0_LIST = [0.9, 1.5, 3.0]
NUM_RUNS = 300
OUTBREAK_THRESHOLD = 0.1
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
raw.mkdir(parents=True, exist_ok=True)
attempted, done = {}, {}
for r0 in R0_LIST:
    key = f"R0={r0}"
    attempted[key] = done[key] = 0
    for i in range(__RUNS__):
        attempted[key] += 1
        done[key] += 1
(raw / "outcomes.json").write_text(json.dumps(done))
(raw / "run_manifest.json").write_text(json.dumps({
    "schema": "fi.run-manifest/v1", "realized_grid": {"R0": R0_LIST}, "attempted_per_cell": attempted,
    "successful_per_cell": done, "failed_trials": [], "thresholds_used": {"outbreak": OUTBREAK_THRESHOLD},
}))
"""
SIM_OK = _SIM.replace("__RUNS__", "NUM_RUNS")
SIM_SHORT = _SIM.replace("__RUNS__", "30")  # the constant says 300 and the loop runs 30: no lint finds it
SIM_NO_MANIFEST = SIM_OK.split("(raw / \"run_manifest.json\")")[0]

# Runs 10 real trials per cell, appends a real trial_ledger.jsonl line for each one, then writes a
# run_manifest.json that LIES about it (claims 300 attempted/successful) -- the audit's own bypass,
# through the real graph: does the engine trust the ledger's 10 real lines, or the lying summary's 300?
SIM_LEDGER_LIES_IN_SUMMARY = """\
import json, os, pathlib
R0_LIST = [0.9, 1.5, 3.0]
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
raw.mkdir(parents=True, exist_ok=True)
ledger = open(raw / "trial_ledger.jsonl", "a")
done = {}
for r0 in R0_LIST:
    key = f"R0={r0}"
    done[key] = 0
    for i in range(10):  # really only 10 trials per cell
        ledger.write(json.dumps({"cell": key, "trial": i, "status": "ok"}) + "\\n")
        done[key] += 1
ledger.close()
(raw / "outcomes.json").write_text(json.dumps(done))
lie = {r0: 300 for r0 in [f"R0={r}" for r in R0_LIST]}
(raw / "run_manifest.json").write_text(json.dumps({
    "schema": "fi.run-manifest/v1", "realized_grid": {"R0": R0_LIST}, "attempted_per_cell": lie,
    "successful_per_cell": lie, "failed_trials": [], "thresholds_used": {"outbreak": OUTBREAK_THRESHOLD},
}))
""".replace("raw.mkdir(parents=True, exist_ok=True)\n", "raw.mkdir(parents=True, exist_ok=True)\nOUTBREAK_THRESHOLD = 0.1\n")
ANALYSIS = """\
import json, os, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
done = json.loads((raw / "outcomes.json").read_text())
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({'score': 0.987, 'cells': len(done)}))
"""


def _reply(simulate: str, analysis: str = ANALYSIS) -> str:
    return f"```python\n# file: simulate.py\n{simulate}\n```\n```python\n# file: experiment.py\n{analysis}\n```\nDEPS: matplotlib\n"


def _cfg(tmp_path: Path, *, split: bool | str = True, engine: dict[str, Any] | None = None, execution: dict[str, Any] | None = None) -> Config:
    return Config(
        topic="smoke topic for the run manifest", title="manifest-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(**{"max_iterations": 1, "review_loop": False, "auto_accept_on_pass": True, "execute_replicates": 1,
                               "pilot_run": False, "oracle_check": "off", **(engine or {})}),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=split, **(execution or {})),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    )


def _fake(calls: list[str], *, implement: str, repair: str | None = None, prompts: list[str] | None = None):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if prompts is not None:
            prompts.append(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            return implement
        if kind == "ExecuteReflect" and repair is not None:
            return json.dumps({"code": repair, "deps": [], "patch_summary": "runs the protocol's design"})
        return _fake_response_for(prompt)

    return fake_chat


def _record(engine: Engine) -> dict[str, Any]:
    return json.loads((engine.quest_root / "needs" / "RUN_MANIFEST_CHECK.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_a_simulation_on_the_older_contract_is_checked_and_marked_as_its_own_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_OK)))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert _record(engine)["status"] == "self_reported" and "ExecuteReflect" not in calls
    assert (engine.quest_root / "raw" / "seed0" / "run_manifest.json").is_file()


@pytest.mark.asyncio
async def test_a_lying_run_manifest_is_overruled_by_the_real_ledger_through_the_real_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not a unit test of manifest_from_ledger() in isolation -- this exercises
    Engine._run_manifest_problems itself: a script that ran 10 real trials per cell (a real
    trial_ledger.jsonl says so) but writes run_manifest.json claiming 300 (the audit's own bypass)
    must be caught by the actual engine wiring, not just by the pure function it calls."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_LEDGER_LIES_IN_SUMMARY)))
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))  # warn: inspect the record without a repair loop
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    record = _record(engine)
    assert record["status"] == "warned"
    assert any("ran another number" in p and "R0=0.9: 10" in p for p in record["problems"]), record["problems"]
    ledger_path = engine.quest_root / "raw" / "seed0" / "trial_ledger.jsonl"
    assert ledger_path.is_file() and len(ledger_path.read_text(encoding="utf-8").splitlines()) == 30


@pytest.mark.asyncio
async def test_a_loop_that_runs_fewer_trials_than_its_constant_says_is_sent_back_and_the_repair_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The static protocol check reads NUM_RUNS = 300 and is satisfied; only the run itself says it did 30."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_SHORT), repair=SIM_OK))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("ExecuteReflect") == 1
    protocol_record = json.loads((engine.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert protocol_record["status"] == "ok" and protocol_record["differences"] == []
    record = _record(engine)
    assert record["status"] == "self_reported"
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[run_manifest] the run differs from the frozen protocol (1)" in log and "ran another number" in log
    assert "sending simulate.py back (1 of 1)" in log
    assert "NUM_RUNS" in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")


ANALYSIS_WITH_VALUES = ANALYSIS.replace(
    "'cells': len(done)}", "'cells': len(done), 'final_size_values': [0.1] * sum(done.values())}",
)


@pytest.mark.asyncio
async def test_an_analysis_that_prints_no_values_is_sent_back_alone_and_the_simulation_is_left_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real graph: the simulation did everything the protocol fixed; only the analysis left out the per-trial
    values of a mean metric. experiment.py is the script repaired (against the raw files already on disk), and
    simulate.py is not touched -- a live quest's rewrite of it broke the oracle checks it had passed."""
    calls: list[str] = []
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        prompts.append(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = _MEAN_PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(SIM_OK, ANALYSIS)
        if kind == "ExecuteReflect":
            return json.dumps({"code": ANALYSIS_WITH_VALUES, "deps": [], "patch_summary": "lists the values"})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and calls.count("ExecuteReflect") == 1
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "has no `final_size_values` list at all" in log and "sending experiment.py back (1 of 1)" in log
    reflect = next(p for p in prompts if _classify(p) == "ExecuteReflect")
    assert "Rewrite experiment.py (not simulate.py)" in reflect
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8").strip() == SIM_OK.strip()
    assert "final_size_values" in (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    assert _record(engine)["status"] == "self_reported"


@pytest.mark.asyncio
async def test_a_simulation_that_still_differs_after_the_repair_stops_the_quest_and_a_fixed_one_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_SHORT), repair=SIM_SHORT))
    cfg = _cfg(tmp_path)
    first = Engine(cfg)
    await first.run()
    log = (first.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[manifest] paused" in log and "[FI] paused for the two-script contract" in log and "paused for clarify" not in log
    assert not (first.fi_dir / "clarify_questions.json").exists()
    descriptor = json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "manifest" and descriptor["interaction"] == "supply"
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "ran another number" in text and "engine.run_manifest_check: warn" in text
    assert _record(first)["status"] == "stopped"
    assert not (first.quest_root / "paper" / "paper.md").exists()

    (first.quest_root / "code" / "simulate.py").write_text(SIM_OK, encoding="utf-8")
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None and _record(second)["status"] == "self_reported"


@pytest.mark.asyncio
async def test_a_simulation_that_writes_no_manifest_is_a_difference_of_its_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_NO_MANIFEST), repair=SIM_NO_MANIFEST))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    assert "run_manifest.json was not written" in (engine.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_warn_records_the_difference_and_goes_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_SHORT)))
    engine = Engine(_cfg(tmp_path, engine={"run_manifest_check": "warn"}))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and "ExecuteReflect" not in calls
    record = _record(engine)
    assert record["status"] == "warned" and any("ran another number" in p for p in record["problems"])


@pytest.mark.asyncio
async def test_a_quest_that_runs_as_one_script_says_it_has_no_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    one_script = json.dumps({"code": "R0_LIST = [0.9, 1.5, 3.0]\nNUM_RUNS = 300\nOUTBREAK_THRESHOLD = 0.1\n" + ANALYSIS.replace("raw = pathlib.Path(os.environ[\"FI_RAW_DIR\"])\ndone = json.loads((raw / \"outcomes.json\").read_text())", "done = {}"), "deps": ["matplotlib"]})
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=one_script))
    engine = Engine(_cfg(tmp_path, split=False))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert _record(engine)["status"] == "single_script"
    evidence = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert any("writes no run manifest" in g for g in evidence["all_gaps"].get("protocol_runtime_matched", []))


@pytest.mark.asyncio
async def test_a_reply_without_both_scripts_runs_as_one_script_by_default_and_stops_when_the_contract_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    one = json.dumps({"code": "R0_LIST = [0.9, 1.5, 3.0]\nNUM_RUNS = 300\nOUTBREAK_THRESHOLD = 0.1\n" + ANALYSIS.replace('raw = pathlib.Path(os.environ["FI_RAW_DIR"])\ndone = json.loads((raw / "outcomes.json").read_text())', "done = {}"), "deps": ["matplotlib"]})
    loose = Engine(_cfg(tmp_path / "loose"))
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=one))
    artifacts = await loose.run()
    assert artifacts.paper_md is not None
    assert "this quest runs as ONE script" in (loose.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")

    strict = Engine(_cfg(tmp_path / "strict", execution={"split_failure": "block"}))
    await strict.run()
    log = (strict.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[split] paused" in log and "[FI] paused for the two-script contract" in log and "this quest runs as ONE script" not in log
    descriptor = json.loads((strict.fi_dir / "pause.json").read_text(encoding="utf-8"))
    assert descriptor["kind"] == "split"
    assert "did not hold both scripts" in (strict.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert not (strict.quest_root / "paper" / "paper.md").exists()


@pytest.mark.asyncio
async def test_scripts_that_break_the_contract_are_named_and_stop_a_strict_quest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    publishing = SIM_OK + "\nprint('RESULT_JSON: {}')\n"
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=_reply(publishing)))
    warned = Engine(_cfg(tmp_path / "warn"))
    await warned.run()
    assert "simulate.py prints or names RESULT_JSON" in (warned.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")

    strict = Engine(_cfg(tmp_path / "strict", execution={"split_failure": "block"}))
    await strict.run()
    assert "simulate.py prints or names RESULT_JSON" in (strict.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert not (strict.quest_root / "paper" / "paper.md").exists()


@pytest.mark.asyncio
async def test_the_two_script_directive_asks_for_the_trial_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=_reply(SIM_TRIAL, ANALYSIS_TRIAL), prompts=prompts))
    await Engine(_cfg(tmp_path)).run()
    implement = [p for p in prompts if "run_trial(cell: dict, trial_id: int, seed: int)" in p]
    assert implement and "FI_TRIALS" in implement[0] and "fi.run-manifest/v1" not in implement[0]


def test_the_pages_label_the_new_pauses() -> None:
    root = Path(__file__).resolve().parent.parent
    for page in ("quest.html", "index.html"):
        text = (root / "web" / "static" / page).read_text(encoding="utf-8")
        for kind in ("amendment", "manifest", "split"):
            assert f"{kind}: '" in text, (page, kind)


@pytest.mark.asyncio
async def test_the_other_seeds_are_checked_and_a_replicate_that_was_never_run_is_not_a_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three seeds of a simulation that draws no random number: seeds 0 and 1 agree, the study is taken as deterministic and
    seed 2 is never run, so it has no manifest and that is not a difference. A replicate that ran and wrote none is."""
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=_reply(SIM_OK)))
    engine = Engine(_cfg(tmp_path / "det", engine={"execute_replicates": 3}))
    await engine.run()
    assert _record(engine)["status"] == "self_reported" and not (engine.quest_root / "raw" / "seed2").exists()

    liar = SIM_OK.replace("(raw / \"run_manifest.json\").write_text", "(raw / (\"run_manifest.json\" if os.environ.get(\"FI_REPLICATE_SEED\", \"0\") == \"0\" else \"other.json\")).write_text")
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=_reply(liar)))
    other = Engine(_cfg(tmp_path / "seeds", engine={"execute_replicates": 2}))
    await other.run()
    record = _record(other)
    assert record["status"] == "self_reported", "the other seeds are compared only when the first seed's record is FI's own"


def test_an_axis_the_protocol_does_not_list_is_a_difference_that_needs_an_amendment() -> None:
    """Sweeping an axis the protocol never fixed is a wider study than the one that was frozen, and it must be reached
    through an amendment, not accepted silently because every listed axis still looks fine."""
    swept_more = _manifest(
        realized_grid={"R0": [0.9, 1.5, 3.0], "N": [100, 1000, 5000]},
        attempted_per_cell={f"R0={r},N={n}": 300 for r in (0.9, 1.5, 3.0) for n in (100, 1000, 5000)},
        successful_per_cell={f"R0={r},N={n}": 300 for r in (0.9, 1.5, 3.0) for n in (100, 1000, 5000)},
    )
    found = rm.problems(PROTOCOL, swept_more)
    assert any("sweeps N" in f and "amendment" in f for f in found)


def test_a_cell_key_that_names_a_value_or_axis_outside_the_grid_is_a_difference() -> None:
    """The bypass a real quest could hit: cell keys that do not exactly match the protocol's own Cartesian product still
    passed before, because the checker only compared axis VALUES (via realized_grid) and a raw cell COUNT, never the cells
    themselves."""
    wrong_keys = _manifest(attempted_per_cell={"foo": 300, "bar": 300, "baz": 300}, successful_per_cell={"foo": 300, "bar": 300, "baz": 300})
    found = rm.problems(PROTOCOL, wrong_keys)
    assert any("does not look like" in f for f in found)

    bad_value = _manifest(
        attempted_per_cell={"R0=0.9": 300, "R0=1.5": 300, "R0=99": 300}, successful_per_cell={"R0=0.9": 300, "R0=1.5": 300, "R0=99": 300},
    )
    found = rm.problems(PROTOCOL, bad_value)
    assert any("R0=99" in f and "the protocol's grid for R0 is" in f for f in found)

    grid2 = {**PROTOCOL, "grid": {"R0": [0.9, 1.5], "N": [10, 20]}}
    missing_axis = _manifest(
        realized_grid={"R0": [0.9, 1.5], "N": [10, 20]},
        attempted_per_cell={"R0=0.9": 300, "R0=1.5": 300}, successful_per_cell={"R0=0.9": 300, "R0=1.5": 300},
    )
    found = rm.problems(grid2, missing_axis)
    assert any("is missing N" in f for f in found)

    # Number formatting drift (a float re-serialized) is not a real difference: the value still matches the protocol's.
    formatted = _manifest(
        attempted_per_cell={"R0=0.9": 300, "R0=1.5000000001": 300, "R0=3.0": 300},
        successful_per_cell={"R0=0.9": 300, "R0=1.5000000001": 300, "R0=3.0": 300},
    )
    assert rm.problems(PROTOCOL, formatted) == []


def test_two_cell_keys_for_the_same_setting_is_a_difference() -> None:
    dup = _manifest(
        attempted_per_cell={"R0=0.9": 150, "R0=0.900000000001": 150, "R0=1.5": 300, "R0=3.0": 300},
        successful_per_cell={"R0=0.9": 150, "R0=0.900000000001": 150, "R0=1.5": 300, "R0=3.0": 300},
    )
    found = rm.problems(PROTOCOL, dup)
    assert any("more than one cell key names the same setting" in f for f in found)


def test_a_failed_trial_needs_a_valid_cell_a_trial_id_and_a_reason() -> None:
    ok = {"R0=0.9": 300, "R0=1.5": 297, "R0=3.0": 300}
    listed = [
        {"cell": "R0=1.5", "trial": 0, "reason": ""}, {"cell": "nope", "trial": 1, "reason": "solver"},
        {"cell": "R0=1.5", "reason": "solver"},
    ]
    found = rm.problems(PROTOCOL, _manifest(successful_per_cell=ok, failed_trials=listed))
    assert any("has no `reason`" in f for f in found)
    assert any("names no valid cell" in f for f in found)
    assert any("has no `trial` id" in f for f in found)

    dup_trial = [{"cell": "R0=1.5", "trial": 0, "reason": "solver"}, {"cell": "R0=1.5", "trial": 0, "reason": "solver again"}, {"cell": "R0=1.5", "trial": 1, "reason": "solver"}]
    found = rm.problems(PROTOCOL, _manifest(successful_per_cell=ok, failed_trials=dup_trial))
    assert any("same cell and trial id twice" in f for f in found)


def test_a_protocol_with_runs_but_no_failure_policy_is_told_so_before_the_freeze() -> None:
    from core import protocol_check as pc

    assert "does not say how a trial that fails" in pc.failure_notes({"runs_per_setting": 300})[0]
    assert pc.failure_notes({"runs_per_setting": 300, "failure_policy": "counted as failures"}) == []
    assert pc.failure_notes({"grid": {"a": [1]}}) == [] and pc.failure_notes(None) == []


@pytest.mark.asyncio
async def test_when_the_repair_gives_up_the_rejected_runs_numbers_and_figures_do_not_reach_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run the manifest rejected has a RESULT_JSON and figures on disk; if nothing repairs it they must not become a paper."""
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(SIM_SHORT)
        if kind == "ExecuteReflect":
            return json.dumps({"give_up_reason": "the simulation cannot be made to match the protocol"})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.raw_state["result_json"] == {}, "the 30-trial run's numbers were kept"
    assert not list((engine.quest_root / "figures").glob("*.png")), "the 30-trial run's figure was kept"
    assert _record(engine)["status"] == "repairing"


@pytest.mark.asyncio
async def test_a_deterministic_study_that_runs_as_one_script_needs_no_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = {"grid": {"dt": [0.1, 0.01]}}
    script = "DT_LIST = [0.1, 0.01]\n" + ANALYSIS.replace('raw = pathlib.Path(os.environ["FI_RAW_DIR"])\ndone = json.loads((raw / "outcomes.json").read_text())', "done = {}")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            body["method"] = "integrate the ODE with RK4 and compare step sizes"
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": script, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, split="auto"))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and _record(engine)["status"] == "not_applicable"
    evidence = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert not any("run manifest" in g or "writes no run manifest" in g for g in evidence["all_gaps"].get("protocol_runtime_matched", []))


def test_a_comment_that_names_result_json_is_not_a_simulation_that_publishes() -> None:
    assert rm.split_lint({"simulate.py": "# results are printed by the analysis, never RESULT_JSON here\nx = 1"}) == []
    assert rm.split_lint({"experiment.py": "import subprocess\nsubprocess.run(['python', 'simulate.py'])"})


# --- the per-trial ledger: the engine counts real rows instead of trusting a self-reported summary -----------------


def _ledger_rows(n_per_cell: int, *, cells: tuple[float, ...] = (0.9, 1.5, 3.0)) -> list[dict[str, Any]]:
    return [
        {"cell": f"R0={r}", "trial": t, "status": "ok"}
        for r in cells for t in range(n_per_cell)
    ]


def test_reading_a_ledger_that_is_not_there_says_why(tmp_path: Path) -> None:
    rows, why = rm.read_ledger(tmp_path)
    assert rows is None and "was not written" in why


def test_reading_a_ledger_parses_one_json_object_per_line(tmp_path: Path) -> None:
    (tmp_path / rm.LEDGER_NAME).write_text(
        '{"cell": "R0=0.9", "trial": 0, "status": "ok"}\n\n{"cell": "R0=0.9", "trial": 1, "status": "ok"}\n',
        encoding="utf-8",
    )
    rows, why = rm.read_ledger(tmp_path)
    assert why == "" and len(rows) == 2 and rows[1]["trial"] == 1


def test_reading_a_ledger_with_a_bad_line_fails_with_the_line_number(tmp_path: Path) -> None:
    (tmp_path / rm.LEDGER_NAME).write_text('{"cell": "R0=0.9", "trial": 0, "status": "ok"}\nnot json\n', encoding="utf-8")
    rows, why = rm.read_ledger(tmp_path)
    assert rows is None and "line 2" in why


def test_a_lying_summary_is_ignored_once_a_real_ledger_backs_the_counts() -> None:
    """The audit's own bypass, closed: the script can still write ANY attempted_per_cell it likes into
    run_manifest.json, but manifest_from_ledger never reads that field at all -- it counts real ledger rows."""
    ledger = _ledger_rows(300)
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, ledger, thresholds_used={"outbreak": 0.1})
    assert problems_ == []
    assert derived["attempted_per_cell"] == {"R0=0.9": 300, "R0=1.5": 300, "R0=3.0": 300}
    assert rm.problems(PROTOCOL, derived) == []


def test_a_script_that_ran_10_and_claims_300_is_caught_because_only_10_rows_exist() -> None:
    """The concrete audit example: attempted_per_cell=300 in run_manifest.json is simply never consulted;
    the ledger only has 10 real rows per cell, so that is what attempted_per_cell derives to."""
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, _ledger_rows(10))
    assert problems_ == []
    found = rm.problems(PROTOCOL, derived)
    assert any("ran another number" in f for f in found), found


def test_a_duplicate_trial_id_is_named_not_silently_merged() -> None:
    ledger = _ledger_rows(2) + [{"cell": "R0=0.9", "trial": 0, "status": "ok"}]  # replays trial 0
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, ledger)
    assert any("more than one ledger row claims" in p and "trial 0" in p for p in problems_), problems_
    # The replayed row does not count twice.
    assert derived["attempted_per_cell"]["R0=0.9"] == 2


def test_a_replayed_trial_id_under_a_different_json_type_is_still_caught() -> None:
    """A script could try to dodge the duplicate check by resubmitting the same trial as a string
    the second time (`0` then `"0"`) -- these must be recognised as the same trial, not two."""
    ledger = _ledger_rows(2) + [{"cell": "R0=0.9", "trial": "0", "status": "ok"}]
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, ledger)
    assert any("more than one ledger row claims" in p for p in problems_), problems_
    assert derived["attempted_per_cell"]["R0=0.9"] == 2


def test_a_ledger_row_outside_the_protocols_grid_is_named_not_folded_in() -> None:
    ledger = _ledger_rows(300) + [{"cell": "R0=99", "trial": 0, "status": "ok"}]
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, ledger)
    assert any("R0=99" in p for p in problems_), problems_
    assert "R0=99" not in derived["attempted_per_cell"]


def test_a_dropped_trial_with_no_failure_entry_is_a_shortfall() -> None:
    """A trial the script silently never ran (not even recorded as failed) shows up as a plain count
    shortfall against runs_per_setting, the same as any other missing trial."""
    rows = _ledger_rows(300)
    del rows[0]  # one R0=0.9 trial never appended a ledger line at all
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, rows)
    assert problems_ == []
    found = rm.problems(PROTOCOL, derived)
    assert any("R0=0.9: 299" in f for f in found), found


def test_a_ledger_row_with_no_recognised_status_is_named() -> None:
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, [{"cell": "R0=0.9", "trial": 0, "status": "maybe"}])
    assert any("no `status`" in p for p in problems_), problems_
    assert derived["attempted_per_cell"] == {}


def test_a_failed_ledger_row_is_counted_attempted_but_not_successful() -> None:
    rows = _ledger_rows(300, cells=(1.5, 3.0)) + _ledger_rows(299, cells=(0.9,))
    rows += [{"cell": "R0=0.9", "trial": 299, "status": "failed", "reason": "solver diverged"}]
    derived, problems_ = rm.manifest_from_ledger(PROTOCOL, rows, thresholds_used={"outbreak": 0.1})
    assert problems_ == []
    assert derived["attempted_per_cell"]["R0=0.9"] == 300
    assert derived["successful_per_cell"]["R0=0.9"] == 299
    assert derived["failed_trials"] == [{"cell": "R0=0.9", "trial": 299, "reason": "solver diverged"}]
    assert rm.problems({**PROTOCOL, "failure_policy": "excluded from the pooled estimate"}, derived) == []


def test_a_count_repeated_outside_its_setting_is_not_added_twice() -> None:
    """A live run printed 933 + 96 = 1,029 trials: the top level repeated one setting's count beside the per-setting
    ones. Counts are taken beside the values they back."""
    result = {
        "p_major_count": 96,  # a headline copy of one setting's count, with no values beside it
        "by_R0": {
            "1.5": {"p_major_count": 96, "p_major_total": 300, "final_size_values": [0.5] * 96},
            "3.0": {"p_major_count": 837, "p_major_total": 300, "final_size_values": [0.9] * 837},
        },
    }
    assert rm._total_counts_reported(result, "p_major_count", beside="final_size_values") == 933
    assert rm._total_counts_reported(result, "p_major_count") == 1029, "without a pairing, every count as before"


# --- the trial contract: FI runs the trials and keeps their record ------------------------------------------------------

SIM_TRIAL = """\
import random

def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    return {"outbreak": 1.0 if rng.random() < cell["R0"] / 4 else 0.0, "final_size": 100.0 * cell["R0"] + rng.random()}
"""
ANALYSIS_TRIAL = """\
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
out = {}
for c in data["cells"]:
    m = c["metrics"]["final_size"]
    out[c["key"]] = {"final_size_values": m["values"], "final_size_count": m["count"]}
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({'score': 0.987, 'by_cell': out}))
"""


@pytest.mark.asyncio
async def test_a_simulation_on_the_trial_contract_is_run_by_fi_and_fis_record_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the real graph: simulate.py only defines run_trial; FI runs 300 trials in each of the three settings, one
    process per setting, writes the ledger itself, and the checker counts FI's rows. The simulation cannot write the
    record: it writes no file at all."""
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_TRIAL, ANALYSIS_TRIAL)))
    engine = Engine(_cfg(tmp_path, engine={"execute_replicates": 3}))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and "ExecuteReflect" not in calls
    assert _record(engine)["status"] == "ok"
    ledger = [json.loads(line) for line in (engine.quest_root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    trials = [e for e in ledger if e["event"] == "trial"]
    assert len(trials) == 900 and {e["cell"] for e in trials} == {"R0=0.9", "R0=1.5", "R0=3.0"}
    assert len({e["seed"] for e in trials}) == 900, "every trial its own seed"
    assert not (engine.quest_root / "raw" / "seed1").exists(), "the trials are the replicates: no second whole run"
    evidence = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert not any("own statement" in g for gaps in evidence.get("all_gaps", {}).values() for g in gaps)
    # The one result holds every trial: it is what the intervals and the metric statistics are computed from.
    state = artifacts.raw_state
    assert state["result_json_trials"] is True and len(state["result_json_replicates"]) == 1
    values = state["result_json_replicates"][0]["by_cell"]["R0=1.5"]["final_size_values"]
    assert len(values) == 300


@pytest.mark.asyncio
async def test_the_older_contract_under_research_is_sent_back_for_the_trial_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Research needs FI's own record: a simulation that runs its own loop is sent back once, asked for run_trial, and the
    repaired one runs through FI."""
    calls: list[str] = []
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat",
                        _fake(calls, implement=_reply(SIM_OK), repair=SIM_TRIAL, prompts=prompts))
    cfg = _cfg(tmp_path).model_copy(update={"rigor_profile": "research"})
    engine = Engine(cfg)
    await engine.run()
    reflect = [p for p in prompts if _classify(p) == "ExecuteReflect"]
    assert reflect and "run_trial(cell, trial_id, seed)" in reflect[0]
    assert "def run_trial" in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert (engine.quest_root / "raw" / "ledger.jsonl").is_file()


@pytest.mark.asyncio
async def test_the_oracle_of_the_trial_contract_is_its_own_function(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FI calls simulate.py's oracle() and judges its value against the protocol: no FI_ORACLE run, no ORACLE_JSON."""
    protocol = {**PROTOCOL, "oracles": [{"name": "half", "check": "a known case", "expected": 0.5, "tolerance": 0.01}]}
    sim = SIM_TRIAL + "\n\ndef oracle():\n    return {\"half\": 0.5004}\n"

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(sim, ANALYSIS_TRIAL)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_cfg(tmp_path, engine={"oracle_check": "block"}))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    record = json.loads((engine.quest_root / "needs" / "ORACLE_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok", record


# --- the trial contract on a cluster: one job-array task per setting ----------------------------------------------------

SUBMIT_FAKE = """\
import json, os, pathlib, subprocess, sys

cluster = pathlib.Path(os.environ["FAKE_CLUSTER_DIR"])
tasks = json.loads(pathlib.Path(os.environ["FI_TASKS"]).read_text(encoding="utf-8"))
job = pathlib.Path("job"); job.mkdir(exist_ok=True)
state = job / "state.json"
if not state.exists():
    state.write_text(json.dumps({"id": "A1", "n": tasks["count"]}))
    (cluster / "submitted").write_text(str(tasks["count"]))
    print("RESULT_JSON: " + json.dumps({"fi_job": {"status": "pending", "id": "A1", "note": "queued", "poll_s": 5}}))
elif not (cluster / "done").exists():
    print("RESULT_JSON: " + json.dumps({"fi_job": {"status": "pending", "id": "A1", "note": "running"}}))
else:
    # The array ran: each task ran FI's harness for its setting (here, on this machine).
    if not (cluster / "ran").exists():
        for t in tasks["tasks"]:
            subprocess.run([sys.executable, *t["argv"]], check=False)
        (cluster / "ran").write_text("1")
    print("RESULT_JSON: " + json.dumps({"fi_job": {"status": "done", "id": "A1"}}))
"""


def _reply3(simulate: str, analysis: str, submit: str) -> str:
    return _reply(simulate, analysis).replace(
        "DEPS: matplotlib", f"```python\n# file: submit.py\n{submit}\n```\nDEPS: matplotlib")


@pytest.mark.asyncio
async def test_on_a_cluster_fi_runs_the_trials_as_a_job_array_and_keeps_their_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real HPC is not exercised: the cluster is a folder, and the 'array' runs FI's tasks on this machine once the test
    says the job is done. What is tested is FI's half: the tasks are FI's, the quest waits on the pending job, and the
    record of what ran is written by FI from each task's results."""
    cluster = tmp_path / "cluster"
    cluster.mkdir()
    monkeypatch.setenv("FAKE_CLUSTER_DIR", str(cluster))
    calls: list[str] = []
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(
        calls, implement=_reply3(SIM_TRIAL, ANALYSIS_TRIAL, SUBMIT_FAKE), prompts=prompts))
    cfg = _cfg(tmp_path, execution={"background_jobs": True})
    first = Engine(cfg)
    await first.run()
    tasks = json.loads((first.quest_root / "job" / "fi" / "tasks.json").read_text(encoding="utf-8"))
    assert tasks["count"] == 3 and [t["setting"] for t in tasks["tasks"]] == ["R0=0.9", "R0=1.5", "R0=3.0"]
    assert (cluster / "submitted").read_text() == "3"
    assert json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "results"
    assert not (first.quest_root / "raw" / "ledger.jsonl").exists(), "nothing is recorded before the job is done"
    assert any("# file: submit.py" in p for p in prompts), "the code-writing step is told to write submit.py"

    (cluster / "done").write_text("1")
    second = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await second.run()
    assert artifacts.paper_md is not None and "ExecuteReflect" not in calls
    trials = [json.loads(line) for line in (second.quest_root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    trials = [e for e in trials if e["event"] == "trial"]
    assert len(trials) == 900 and all(e["status"] == "ok" for e in trials)
    assert _record(second)["status"] == "ok"


def test_a_cluster_task_that_reported_nothing_is_a_failed_setting_with_why(tmp_path: Path) -> None:
    from core import trial_runner

    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM_TRIAL, encoding="utf-8")
    record = trial_runner.prepare_cluster(root, "code/simulate.py", {"R0": [1.5, 3.0]}, runs_per_setting=4,
                                          base_seed=0, deterministic=False, key="k1")
    assert trial_runner.prepare_cluster(root, "code/simulate.py", {"R0": [1.5]}, runs_per_setting=1, base_seed=0,
                                        deterministic=False, key="k1") == record, "kept while the key is the same"
    import subprocess
    import sys

    first = json.loads((root / "job" / "fi" / "tasks.json").read_text(encoding="utf-8"))["tasks"][0]
    subprocess.run([sys.executable, *first["argv"]], cwd=root, check=True)  # only the first task ran
    run = trial_runner.collect_cluster(root, record)
    assert run.ok_trials == 4 and run.failed_trials == 4
    assert all("reported no result" in r["reason"] for r in run.cells[1].rows)


def test_a_new_plan_or_a_new_job_starts_from_no_results_and_a_task_can_run_again(tmp_path: Path) -> None:
    import subprocess
    import sys

    from core import trial_runner

    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM_TRIAL, encoding="utf-8")
    (root / "job").mkdir()
    (root / "job" / "state.json").write_text('{"id": "OLD"}', encoding="utf-8")
    record = trial_runner.prepare_cluster(root, "code/simulate.py", {"R0": [1.5]}, runs_per_setting=2, base_seed=0,
                                          deterministic=False, key="k1")
    assert not (root / "job" / "state.json").exists(), "a new plan: the old job's state goes, so submit.py submits again"
    assert (root / "job" / "state.previous.json").read_text(encoding="utf-8") == '{"id": "OLD"}', "kept, not deleted"
    task = json.loads((root / "job" / "fi" / "tasks.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["argv"][2].endswith("-k1.jsonl")
    for _ in range(2):  # the scheduler ran the task twice (a requeue): the spec is still there the second time
        subprocess.run([sys.executable, *task["argv"]], cwd=root, check=True)
    assert trial_runner.collect_cluster(root, record).ok_trials == 2
    # submit.py reports another job id (the last one failed and it submitted again): the old results are not read.
    trial_runner._note_job(root, record, {"id": "J1"})
    assert trial_runner.collect_cluster(root, record).ok_trials == 2, "the first job FI sees keeps what it wrote"
    trial_runner._note_job(root, record, {"id": "J2"})
    run = trial_runner.collect_cluster(root, record)
    assert run.ok_trials == 0 and all("reported no result" in r["reason"] for r in run.cells[0].rows)
