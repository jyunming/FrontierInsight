"""``execution.split_analysis`` through the engine, with real scripts run by the real executor.

The model is faked (canned replies); the scripts are real Python run in subprocesses, so what is
checked is what a quest does: the simulation runs once per seed, an analysis that fails or is rewritten
runs again against the raw files already on disk, and the simulation runs again only when its own script
changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from core import split_run
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine

SIMULATE = """\
import json, os, pathlib
seed = int(os.environ.get("FI_REPLICATE_SEED", "0"))
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
raw.mkdir(parents=True, exist_ok=True)
with open("simulated.log", "a") as fh:
    fh.write(f"{seed}\\n")
(raw / "values.json").write_text(json.dumps({"seed": seed, "values": [seed + 1.0, seed + 2.0, seed + 3.0]}))
"""

ANALYSIS = """\
import json, os, pathlib
raw = pathlib.Path(os.environ["FI_RAW_DIR"])
data = json.loads((raw / "values.json").read_text())
if pathlib.Path("analysis_should_fail").exists():
    raise RuntimeError("the analysis broke")
print("RESULT_JSON: " + json.dumps({"mean": sum(data["values"]) / len(data["values"])}))
"""


def _config(tmp_path: Path, *, split: bool = True, replicates: int = 1, **execution) -> Config:
    return Config(
        topic="split analysis", title="split",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, execute_replicates=replicates, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=split, **execution),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=["paper_md"], output_dir=tmp_path / "out"),
    )


def _engine(tmp_path: Path, **kw) -> Engine:
    eng = Engine(_config(tmp_path, **kw))
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(exist_ok=True)
    eng.executor.install = AsyncMock(return_value=type("IR", (), {"returncode": 0, "stderr": ""})())  # type: ignore[method-assign]
    return eng


def _write(eng: Engine, *, simulate: str = SIMULATE, analysis: str = ANALYSIS) -> None:
    (eng.quest_root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    (eng.quest_root / "code" / "experiment.py").write_text(analysis, encoding="utf-8")


def _simulated(eng: Engine) -> list[str]:
    log = eng.quest_root / "simulated.log"
    return log.read_text(encoding="utf-8").split() if log.is_file() else []


def _faked_model(eng: Engine, replies: list[str]) -> list[str]:
    """Answer the engine's next model calls with ``replies``; returns the prompts it was sent."""
    prompts: list[str] = []
    client = MagicMock()
    client.last_usage, client.last_model = None, "test-model"
    client.endpoint = MagicMock()
    client.endpoint.provider_name = "openai"
    queue = list(replies)

    async def chat(messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"] if isinstance(messages, list) else "")
        return queue.pop(0) if queue else "{}"

    client.chat = chat
    eng._client = client
    return prompts


# --- the configuration --------------------------------------------------------


def test_the_switch_is_auto_by_default_and_needs_no_folder() -> None:
    execution = ExecutionConfig()
    assert execution.split_analysis == "auto" and execution.raw_dir == ""
    assert ExecutionConfig(split_analysis=True).split_analysis is True
    assert ExecutionConfig(split_analysis=False).split_analysis is False
    assert ExecutionConfig(raw_dir="somewhere").raw_dir == "somewhere", "auto may keep its raw files elsewhere"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"raw_dir": "somewhere", "split_analysis": False}, "only means something with execution.split_analysis"),
        ({"split_analysis": True, "background_jobs": True}, "cannot be combined with execution.background_jobs"),
        ({"split_analysis": True, "sandbox": "docker", "raw_dir": str(Path.cwd())}, "must be relative"),
    ],
)
def test_combinations_that_cannot_work_are_refused_at_load(kwargs: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ExecutionConfig(**kwargs)


def test_a_relative_folder_is_fine_with_docker_and_an_absolute_one_with_venv(tmp_path: Path) -> None:
    ExecutionConfig(split_analysis=True, sandbox="docker", raw_dir="scratch/raw")
    ExecutionConfig(split_analysis=True, sandbox="venv", raw_dir=str(tmp_path / "big-disk"))


# --- auto: a stochastic design keeps its raw results ----------------------------


@pytest.mark.parametrize("design, expected", [
    ({"hypothesis": "h", "protocol": {"runs_per_setting": 300}}, True),
    ({"hypothesis": "h", "protocol": {"runs_per_setting": 1}}, False),
    ({"hypothesis": "h", "method": "Gillespie simulation of an SIR epidemic"}, True),
    ({"hypothesis": "A Monte Carlo study of variance"}, True),
    ({"hypothesis": "h", "method": "stochastic differential equation"}, True),
    ({"hypothesis": "h", "method": "integrate the ODE with RK4 and compare step sizes"}, False),
    ({"hypothesis": "h", "method": "a deterministic simulation of heat flow"}, False),
    ({"hypothesis": "h", "variables": {"independent": ["random walk length"]}}, True),
    (None, False),
])
def test_a_design_counts_as_stochastic_by_its_runs_or_its_words(design, expected) -> None:
    from core.split_run import design_is_stochastic

    assert design_is_stochastic(design) is expected


def _engine_with(tmp_path: Path, **execution) -> Engine:
    cfg = Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, **execution),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    return Engine(cfg)


def test_auto_splits_a_stochastic_design_and_not_a_deterministic_one(tmp_path: Path) -> None:
    eng = _engine_with(tmp_path)
    stochastic = {"design": {"hypothesis": "h", "protocol": {"runs_per_setting": 300}}}
    assert eng._split_on(stochastic) is True
    assert eng._split_on({"design": {"hypothesis": "h", "method": "RK4 step sizes"}}) is False
    assert eng._split_on({}) is False
    assert eng._split_block(stochastic) != ""
    assert eng._split_block({}) == ""


def test_auto_never_splits_a_background_job_a_study_with_no_experiment_or_a_data_only_run(tmp_path: Path) -> None:
    stochastic = {"design": {"hypothesis": "h", "protocol": {"runs_per_setting": 300}}}
    assert _engine_with(tmp_path / "a", background_jobs=True)._split_on(stochastic) is False
    eng = _engine_with(tmp_path / "b")
    assert eng._split_on({**stochastic, "no_simulation_resolved": True}) is False
    assert eng._split_on({**stochastic, "survey_mode_resolved": True}) is False
    eng.config.engine.analyze_local_first = True
    assert eng._split_on(stochastic) is False


def test_true_and_false_decide_for_every_quest(tmp_path: Path) -> None:
    stochastic = {"design": {"hypothesis": "h", "protocol": {"runs_per_setting": 300}}}
    assert _engine_with(tmp_path / "t", split_analysis=True)._split_on({}) is True
    assert _engine_with(tmp_path / "f", split_analysis=False)._split_on(stochastic) is False


# --- the code-writing step ----------------------------------------------------


def _reply(sim: str, ana: str, deps: str = "DEPS: numpy") -> str:
    return f"```python\n# file: simulate.py\n{sim}\n```\n```python\n# file: experiment.py\n{ana}\n```\n{deps}\n"


STATE = {"topic": "t", "title": "x", "design": {"hypothesis": "h", "dependencies": ["scipy"]}, "clarify_answers": {}}


@pytest.mark.asyncio
async def test_the_prompt_asks_for_two_scripts_only_when_the_switch_is_on(tmp_path: Path) -> None:
    on, off = _engine(tmp_path / "on"), _engine(tmp_path / "off", split=False)
    prompt_on, prompt_off = _faked_model(on, [_reply(SIMULATE, ANALYSIS)]), _faked_model(off, ["```python\nprint(1)\n```\nDEPS: numpy"])
    await on._node_implement(STATE)
    await off._node_implement(STATE)
    assert "The experiment is two scripts" in prompt_on[0] and "# file: simulate.py" in prompt_on[0]
    assert "two scripts" not in prompt_off[0] and "FI_RAW_DIR" not in prompt_off[0]


@pytest.mark.asyncio
async def test_a_reply_with_both_scripts_writes_both_and_state_keeps_the_analysis(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _faked_model(eng, [_reply(SIMULATE, ANALYSIS, "DEPS: numpy, matplotlib")])
    out = await eng._node_implement(STATE)
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8").strip() == SIMULATE.strip()
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8").strip() == ANALYSIS.strip()
    assert out["code"].strip() == ANALYSIS.strip()  # the script that prints RESULT_JSON
    assert set(out["deps"]) == {"numpy", "matplotlib", "scipy"}  # read after the last block, plus the design's


@pytest.mark.asyncio
async def test_a_reply_without_both_is_asked_for_again_and_then_run_as_one_script(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    one = "```python\nprint('RESULT_JSON: {}')\n```\nDEPS: numpy"
    prompts = _faked_model(eng, [one, one])
    out = await eng._node_implement(STATE)
    assert len(prompts) >= 2 and "did not hold both scripts" in prompts[1]
    assert not (eng.quest_root / "code" / "simulate.py").exists()
    assert "RESULT_JSON" in out["code"]


@pytest.mark.asyncio
async def test_a_second_reply_can_make_up_for_the_first(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _faked_model(eng, ["```python\nprint(1)\n```\nDEPS: numpy", _reply(SIMULATE, ANALYSIS)])
    await eng._node_implement(STATE)
    assert (eng.quest_root / "code" / "simulate.py").is_file()


@pytest.mark.asyncio
async def test_a_simulation_given_back_unchanged_is_left_exactly_as_it_was(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    simulate = eng.quest_root / "code" / "simulate.py"
    simulate.write_bytes(SIMULATE.encode("utf-8"))
    before = split_run.sha256_of(simulate)
    # the same script with other line endings and trailing spaces, as a model gives it back
    same = SIMULATE.replace("\n", "  \r\n")
    _faked_model(eng, [_reply(same, ANALYSIS + "\n# analysis rewritten\n")])
    await eng._node_implement(STATE)
    assert split_run.sha256_of(simulate) == before
    assert "# analysis rewritten" in (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_leftover_simulation_does_not_stay_beside_a_one_script_quest(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (eng.quest_root / "code" / "simulate.py").write_text("# old\n", encoding="utf-8")
    one = "```python\nprint('RESULT_JSON: {}')\n```\nDEPS: numpy"
    _faked_model(eng, [one, one])
    await eng._node_implement(STATE)
    assert not (eng.quest_root / "code" / "simulate.py").exists()


@pytest.mark.asyncio
async def test_the_seed_is_checked_in_the_simulation_not_the_analysis(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, replicates=3)
    seen: list[str] = []

    async def spy(self, state, code_path, code, deps):  # noqa: ANN001
        seen.append(Path(code_path).name)
        return code, deps

    monkeypatch.setattr(Engine, "_repair_ignored_replicate_seed", spy)
    _faked_model(eng, [_reply(SIMULATE, ANALYSIS)])
    await eng._node_implement(STATE)
    assert seen == ["simulate.py"]


# --- running it, for real -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_quest_simulates_once_per_seed_and_reruns_only_what_changed(tmp_path: Path) -> None:
    eng = _engine(tmp_path, replicates=2)
    _write(eng)
    out = await eng._node_execute({"deps": []})
    assert out["exec_result"]["returncode"] == 0 and out["exec_result"]["failed_script"] is None
    assert out["result_json"] == {"mean": 2.0}
    assert [r["_seed"] for r in out["result_json_replicates"]] == [0, 1]  # the replicate index; the seeds are 0 and 1,000,000
    assert _simulated(eng) == ["0", "1000000"]  # once per seed
    assert (eng.quest_root / "raw" / "seed0" / "manifest.json").is_file()
    assert (eng.quest_root / "raw" / "seed1" / "values.json").is_file()

    # run again with nothing changed: the analysis alone, twice (both seeds)
    again = await eng._node_execute({"deps": []})
    assert again["result_json"] == out["result_json"]
    assert _simulated(eng) == ["0", "1000000"]

    # an analysis that fails: the raw files are kept, and it is the analysis that is named
    (eng.quest_root / "analysis_should_fail").write_text("", encoding="utf-8")
    failed = await eng._node_execute({"deps": []})
    assert failed["exec_result"]["returncode"] != 0
    assert failed["exec_result"]["failed_script"] == "experiment.py"
    assert "the analysis broke" in failed["exec_result"]["stderr_tail"]
    assert _simulated(eng) == ["0", "1000000"]

    # ... and once it is repaired, it runs against the files already there
    (eng.quest_root / "analysis_should_fail").unlink()
    repaired = await eng._node_execute({"deps": []})
    assert repaired["result_json"] == {"mean": 2.0} and _simulated(eng) == ["0", "1000000"]

    # a rewritten simulation runs again, for every seed
    (eng.quest_root / "code" / "simulate.py").write_text(SIMULATE + "\n# rewritten\n", encoding="utf-8")
    await eng._node_execute({"deps": []})
    assert _simulated(eng) == ["0", "1000000", "0", "1000000"]


@pytest.mark.asyncio
async def test_a_failing_simulation_is_named_and_the_analysis_is_not_run(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _write(eng, simulate="import sys\nprint('no such data', file=sys.stderr)\nsys.exit(4)\n")
    out = await eng._node_execute({"deps": []})
    assert out["exec_result"]["returncode"] == 4
    assert out["exec_result"]["failed_script"] == "simulate.py"
    assert "no such data" in out["exec_result"]["stderr_tail"]
    assert out["result_json"] == {}


@pytest.mark.asyncio
async def test_a_one_script_quest_is_unchanged_by_the_switch_being_on(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    (eng.quest_root / "code" / "experiment.py").write_text(
        'import json\nprint("RESULT_JSON: " + json.dumps({"x": 1.0}))\n', encoding="utf-8",
    )
    out = await eng._node_execute({"deps": []})
    assert out["result_json"] == {"x": 1.0} and out["exec_result"]["failed_script"] is None
    assert not (eng.quest_root / "raw").exists()


@pytest.mark.asyncio
async def test_the_switch_off_runs_one_script_and_records_no_raw_files(tmp_path: Path) -> None:
    eng = _engine(tmp_path, split=False)
    (eng.quest_root / "code" / "experiment.py").write_text(
        'import json\nprint("RESULT_JSON: " + json.dumps({"x": 1.0}))\n', encoding="utf-8",
    )
    out = await eng._node_execute({"deps": []})
    assert out["result_json"] == {"x": 1.0} and out["exec_result"]["failed_script"] is None
    assert not (eng.quest_root / "raw").exists()


@pytest.mark.asyncio
async def test_a_configured_raw_folder_is_where_the_files_go(tmp_path: Path) -> None:
    eng = _engine(tmp_path, raw_dir=str(tmp_path / "big-disk"))
    _write(eng)
    await eng._node_execute({"deps": []})
    assert (tmp_path / "big-disk" / "seed0" / "values.json").is_file()
    assert not (eng.quest_root / "raw").exists()


@pytest.mark.asyncio
async def test_there_is_no_pilot_pass_for_two_scripts(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.engine.pilot_run = True
    _write(eng)
    await eng._node_execute({"deps": []})
    assert _simulated(eng) == ["0"]  # the pilot would have simulated a second time


# --- the repair ---------------------------------------------------------------


def _state_after(out: dict) -> dict:
    return {"deps": [], "design": {}, "clarify_answers": {}, **out}


@pytest.mark.asyncio
async def test_a_crashed_analysis_is_repaired_alone_and_runs_against_the_raw_files(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _write(eng)
    (eng.quest_root / "analysis_should_fail").write_text("", encoding="utf-8")
    failed = await eng._node_execute({"deps": []})
    fixed = ANALYSIS.replace('raise RuntimeError("the analysis broke")', "pass")
    prompts = _faked_model(eng, [json.dumps({"code": fixed, "deps": [], "patch_summary": "no raise"})])
    simulate_before = (eng.quest_root / "code" / "simulate.py").read_bytes()
    patch = await eng._node_execute_reflect(_state_after(failed))
    assert "this script is experiment.py, the analysis half" in prompts[0]
    assert "values.json" in prompts[0]  # the repair is told what is on disk
    assert patch["code"] == fixed and patch["exec_patch_pending"] is True
    assert (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == fixed
    assert (eng.quest_root / "code" / "simulate.py").read_bytes() == simulate_before
    (eng.quest_root / "analysis_should_fail").unlink()
    out = await eng._node_execute({"deps": []})
    assert out["result_json"] == {"mean": 2.0} and _simulated(eng) == ["0"]


@pytest.mark.asyncio
async def test_a_crashed_simulation_is_repaired_in_its_own_script_and_runs_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _write(eng, simulate="raise SystemExit('the simulation broke')\n")
    failed = await eng._node_execute({"deps": []})
    assert failed["exec_result"]["failed_script"] == "simulate.py"
    prompts = _faked_model(eng, [json.dumps({"code": SIMULATE, "deps": [], "patch_summary": "works now"})])
    analysis_before = (eng.quest_root / "code" / "experiment.py").read_bytes()
    patch = await eng._node_execute_reflect(_state_after({**failed, "code": ANALYSIS}))
    assert "this script is simulate.py, the simulation half" in prompts[0]
    assert "the simulation broke" in prompts[0]
    assert "code" not in patch  # the state's script is the analysis, which was not touched
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIMULATE
    assert (eng.quest_root / "code" / "experiment.py").read_bytes() == analysis_before
    out = await eng._node_execute({"deps": []})
    assert out["result_json"] == {"mean": 2.0} and _simulated(eng) == ["0"]


# --- the other readers of the scripts -----------------------------------------


def test_the_topic_is_checked_against_the_simulation_too(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.topic = "Compare R0 in {0.9, 1.5, 3.0} with 300 runs"
    _write(eng, simulate="R0S = [0.9, 1.5]\nRUNS = 300\n", analysis="print('RESULT_JSON: {\"a\": 1}')\n")
    notes = eng._goal_coverage_notes({"topic": eng.config.topic, "result_json": {"a": 1}, "figures": []})
    assert any("3.0" in note or "3" in note for note in notes), notes  # the missing 3.0 is found in simulate.py


def test_the_critique_reads_both_scripts(tmp_path: Path) -> None:
    from core import critique

    quest = tmp_path / "quest"
    (quest / "code").mkdir(parents=True)
    (quest / "code" / "simulate.py").write_text("# the simulation body\n", encoding="utf-8")
    (quest / "code" / "experiment.py").write_text("# the analysis body\n", encoding="utf-8")
    (quest / "paper.md").write_text("# paper\n", encoding="utf-8")
    artifacts = critique._load_quest_artifacts(quest)
    assert "the simulation body" in artifacts.code and "the analysis body" in artifacts.code
    assert artifacts.code.index("the simulation body") < artifacts.code.index("the analysis body")


def test_the_switch_is_documented_where_a_person_looks() -> None:
    repo = Path(__file__).resolve().parent.parent
    for rel in ("docs/features.md", "docs/capabilities.md", "docs/USAGE.md", "docs/recipes.md", "vscode-frontier-insight/README.md"):
        assert "split_analysis" in (repo / rel).read_text(encoding="utf-8"), rel
