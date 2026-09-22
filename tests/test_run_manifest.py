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
async def test_a_simulation_whose_manifest_matches_the_protocol_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, implement=_reply(SIM_OK)))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    assert _record(engine)["status"] == "ok" and "ExecuteReflect" not in calls
    assert (engine.quest_root / "raw" / "seed0" / "run_manifest.json").is_file()


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
    assert record["status"] == "ok"
    log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "[run_manifest] the run differs from the frozen protocol (1)" in log and "ran another number" in log
    assert "sending simulate.py back (1 of 1)" in log
    assert "NUM_RUNS" in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")


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
    assert artifacts.paper_md is not None and _record(second)["status"] == "ok"


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
async def test_the_two_script_directive_asks_for_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=_reply(SIM_OK), prompts=prompts))
    await Engine(_cfg(tmp_path)).run()
    implement = [p for p in prompts if "run_manifest.json" in p]
    assert implement and "fi.run-manifest/v1" in implement[0] and "attempted_per_cell" in implement[0]


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
    assert _record(engine)["status"] == "ok" and not (engine.quest_root / "raw" / "seed2").exists()

    liar = SIM_OK.replace("(raw / \"run_manifest.json\").write_text", "(raw / (\"run_manifest.json\" if os.environ.get(\"FI_REPLICATE_SEED\", \"0\") == \"0\" else \"other.json\")).write_text")
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake([], implement=_reply(liar)))
    other = Engine(_cfg(tmp_path / "seeds", engine={"execute_replicates": 2}))
    await other.run()
    record = _record(other)
    assert record["status"] == "differs_in_replicates" and "1" in record["seeds"]
    assert "was not written" in record["seeds"]["1"][0]


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
