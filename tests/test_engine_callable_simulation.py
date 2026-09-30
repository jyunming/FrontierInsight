"""Every simulation is a function FI can call itself, deterministic or random.

Under ``execution.split_analysis: auto`` (the default) a quest that runs a simulation keeps it in ``code/simulate.py``
as ``run_cell`` (deterministic) or ``run_trial`` (random), and its analysis in ``code/experiment.py``. FI then calls the
simulation on each oracle's case itself, which is what ``independently_validated`` needs; before, a deterministic
design got one script and could never reach that level. ``split_analysis: false`` keeps one script and gives it up.

The plan's model lists its equations (E1, E2 ...); each one whose role is ``generates`` is marked in the simulation
(``# E1``). A missing label is a warning and a gap below ``independently_validated``; under ``rigor_profile: research``
the quest stops before the run until the label is there. Only the label is read, never the mathematics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import evidence, oracle_check as oc, run_manifest
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

MODEL = {
    "summary": "classical RK4 on the linear ODE y' = -y, y(0) = 1",
    "assumptions": ["a fixed step"],
    "holds_for": "dt up to 0.1",
    "equations": [
        {"id": "E1", "formula": "y' = -y", "role": "generates", "source": "derivation",
         "derivation": "y(t) = exp(-t) solves y' = -y with y(0) = 1, so y(1) = exp(-1) = 0.3679"},
        {"id": "E2", "formula": "err = |y_N - exp(-1)|", "role": "analyses", "source": "derivation",
         "derivation": "the error at t = 1 is err = |y_N - exp(-1)| with exp(-1) = 0.3679"},
    ],
}
ORACLE = {"name": "rk4 error at dt 0.1", "kind": "special_case", "check": "error of RK4 at t = 1 against exp(-1)",
          "expected": 0.0, "tolerance": 1e-5, "reference": "E1", "case": {"dt": 0.1}, "measure": "error"}
PROTOCOL = {"grid": {"dt": [0.1, 0.05]}, "oracles": [ORACLE], "model": MODEL}

SIMULATE = """\
import math


def _f(y):  # E1: y' = -y
    return -y


def _rk4(y, h):
    k1 = _f(y); k2 = _f(y + h * k1 / 2); k3 = _f(y + h * k2 / 2); k4 = _f(y + h * k3)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def run_cell(cell):
    dt = cell["dt"]
    y = 1.0
    for _ in range(int(round(1.0 / dt))):
        y = _rk4(y, dt)
    return {"error": abs(y - math.exp(-1.0))}
"""
UNLABELLED = SIMULATE.replace("  # E1: y' = -y", "")
ANALYSIS = """\
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
errors = {c["key"]: c["metrics"]["error"]["values"][0] for c in data["cells"]}
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot(range(len(errors)), list(errors.values())); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({'max_error': max(errors.values())}))
"""
ONE_SCRIPT = """\
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1], [0, 1]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: {"max_error": 0.0}')
"""


def _reply(simulate: str) -> str:
    return (f"```python\n# file: simulate.py\n{simulate}\n```\n```python\n# file: experiment.py\n{ANALYSIS}\n```\n"
            "DEPS: matplotlib\n")


def _cfg(tmp_path: Path, **execution: Any) -> Config:
    return Config(
        topic="the error of RK4 on y' = -y at two step sizes", title="engine-callable",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False),
        # An isolated venv: a shared interpreter is a gap of its own at the protocol level.
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, shared_interpreter=False, system_site_packages=False,
                                  **execution),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    )


def _fake(calls: list[str], simulate: str):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        calls.append(kind)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["method"] = "integrate y' = -y to t = 1 with classical RK4 at each step size"
            body["protocol"] = PROTOCOL
            return json.dumps(body)
        if kind == "Implementation":
            # The two-script layout is asked for only when the quest keeps two scripts.
            return _reply(simulate) if "# file: simulate.py" in prompt else json.dumps(
                {"code": ONE_SCRIPT, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    return fake_chat


def _json(engine: Engine, rel: str) -> Any:
    return json.loads((engine.quest_root / rel).read_text(encoding="utf-8"))


# --- the auto rule ---------------------------------------------------------------------------------------------------


def _engine(tmp_path: Path, **execution: Any) -> Engine:
    return Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", **execution),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def test_auto_gives_a_deterministic_design_the_engine_callable_layout(tmp_path: Path) -> None:
    deterministic = {"design": {"hypothesis": "RK4 is fourth order", "method": "RK4 at several step sizes",
                                "protocol": {"grid": {"dt": [0.1, 0.05]}}}}
    assert _engine(tmp_path / "a")._split_on(deterministic) is True
    assert _engine(tmp_path / "b")._split_on({"design": {"method": "Monte Carlo"}}) is True
    # The no-simulation path and a survey have no simulation to call.
    assert _engine(tmp_path / "c")._split_on({**deterministic, "no_simulation_resolved": True}) is False
    assert _engine(tmp_path / "d")._split_on({**deterministic, "survey_mode_resolved": True}) is False


def test_split_analysis_false_still_keeps_one_script(tmp_path: Path) -> None:
    assert _engine(tmp_path, split_analysis=False)._split_on({"design": {"method": "Monte Carlo"}}) is False


# --- the labels --------------------------------------------------------------------------------------------------------


def test_a_generates_equation_is_found_by_its_label_in_a_comment_or_a_docstring() -> None:
    protocol = {"model": MODEL}
    assert oc.generating_equations(protocol) == ["E1"], "an equation that analyses the results is not the simulation's"
    assert oc.unlabelled_equations(protocol, SIMULATE) == []
    assert oc.unlabelled_equations(protocol, UNLABELLED) == ["E1"]
    assert oc.unlabelled_equations(protocol, 'def f(y):\n    """Implements e1."""\n    return -y\n') == [], "any case"
    assert oc.unlabelled_equations(protocol, "# E1\ndef f(y):\n    return -y\n") == []
    # A name or a string in the code is not a label; nor is a longer id.
    assert oc.unlabelled_equations(protocol, "E1 = 3\nlabel = 'E1'\n# E12 and E1.5\n") == ["E1"]
    assert oc.unlabelled_equations({"model": {"summary": "x"}}, "") == [], "no equations, nothing to label"
    three = {"model": {"equations": [{"id": f"E{i}", "role": "generates"} for i in (1, 2, 3)]}}
    assert oc.unlabelled_equations(three, "def f():  # E1-E3\n    pass\n") == [], "a range labels each id in it"
    assert oc.unlabelled_equations(three, "def f():  # E1 to 2\n    pass\n") == ["E3"]
    assert oc.unlabelled_equations(three, '"""Implements E1, E2 and E3."""\nx = 1\n') == ["E1", "E2", "E3"], \
        "the module docstring marks no code"
    assert oc.unlabelled_equations(protocol, {"simulate.py": "x = 1\n", "model.py": "def f(y):  # E1\n    return -y\n"}) == [], \
        "a label in a helper module counts"
    gaps = oc.label_gaps(protocol, UNLABELLED, "code/simulate.py")
    assert len(gaps) == 1 and "code/simulate.py" in gaps[0] and "E1" in gaps[0] and "`# E1`" in gaps[0]


def test_a_missing_label_is_a_gap_below_independently_validated(tmp_path: Path) -> None:
    from tests.test_evidence import ON, _quest, _state

    ok = {"protocol_status": "ok", "oracle_status": "ok"}
    assert evidence.assess(_quest(tmp_path / "a", **ok), _state(), settings=ON)["status"] == "publication_ready"
    got = evidence.assess(_quest(tmp_path / "b", **ok), _state(), settings=ON,
                          equation_label_gaps=["code/simulate.py does not mark where it implements equation E1"])
    assert got["status"] == "protocol_runtime_matched", "the level below is kept"
    assert any("E1" in g for g in got["gaps"]), got["gaps"]


def test_under_research_a_missing_label_stops_before_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "simulate.py").write_text(UNLABELLED, encoding="utf-8")
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    stops: list[dict[str, Any]] = []
    monkeypatch.setattr(engine, "_pause_for_human", lambda **kw: stops.append(kw))
    assert engine._check_equation_labels({}) and stops == [], "outside research: a warning, no stop"
    assert "does not mark where it implements equation E1" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    monkeypatch.setattr(engine.config, "rigor_profile", "research")
    engine._check_equation_labels({})
    assert len(stops) == 1 and stops[0]["kind"] == "equation_labels" and stops[0]["payload"]["contract_stage"]
    assert any("`# E1`" in step for step in stops[0]["steps"])
    # A person adds the label and resumes: the script on disk is read again and nothing stops.
    (engine.quest_root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    assert engine._check_equation_labels({}) == [] and len(stops) == 1


def test_a_simulate_py_left_beside_a_one_script_quest_is_not_what_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path, split_analysis=False)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(SIMULATE, encoding="utf-8")  # labelled, but not what runs
    (code / "experiment.py").write_text(ONE_SCRIPT, encoding="utf-8")
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    gaps = engine._equation_label_gaps({}, PROTOCOL)
    assert len(gaps) == 1 and "code/experiment.py" in gaps[0]


@pytest.mark.asyncio
async def test_missing_labels_are_asked_for_once_and_kept_only_when_the_code_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    sim = code / "simulate.py"
    monkeypatch.setattr(engine, "_protocol_block", lambda state: PROTOCOL)
    asked: list[str] = []

    async def changes_the_code(prompt, **kw):  # noqa: ANN001
        asked.append(prompt)
        return "```python\n" + UNLABELLED.replace("return -y", "return -2 * y  # E1") + "\n```"

    sim.write_text(UNLABELLED, encoding="utf-8")
    monkeypatch.setattr(engine, "_chat", changes_the_code)
    assert await engine._label_equations({}) is None and sim.read_text(encoding="utf-8") == UNLABELLED
    assert len(asked) == 1 and "E1" in asked[0] and "y' = -y" in asked[0]

    async def only_comments(prompt, **kw):  # noqa: ANN001
        return "```python\n" + SIMULATE + "\n```"

    monkeypatch.setattr(engine, "_chat", only_comments)
    assert await engine._label_equations({}) == sim and oc.unlabelled_equations(PROTOCOL, sim.read_text(encoding="utf-8")) == []

    async def never(prompt, **kw):  # noqa: ANN001
        raise AssertionError("labels already there: nothing to ask")

    monkeypatch.setattr(engine, "_chat", never)
    assert await engine._label_equations({}) is None


@pytest.mark.asyncio
async def test_a_cluster_quest_that_still_has_one_script_does_not_run_its_driver_for_the_oracles(tmp_path: Path) -> None:
    """Under auto a cluster quest asks for two scripts, but one begun before (or whose reply held one) has only its job
    driver: running it for the oracle pre-check would submit the job."""
    engine = _engine(tmp_path, background_jobs=True)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "experiment.py").write_text(ONE_SCRIPT, encoding="utf-8")

    class NoRun:
        async def execute(self, *a, **kw):  # noqa: ANN002, ANN003
            raise AssertionError("the job driver must not run for the oracle pre-check")

    engine.executor = NoRun()
    assert engine._split_on({"design": {"protocol": PROTOCOL}}) is True
    assert await engine._oracle_gate({"design": {"protocol": PROTOCOL}}, "python", None,
                                     engine.quest_root / "code" / "experiment.py") is None


# --- through the real graph --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deterministic_quest_under_auto_is_run_by_fi_and_reaches_independently_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, SIMULATE))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert (engine.quest_root / "code" / "simulate.py").is_file(), "a deterministic design keeps two scripts"
    oracle = _json(engine, "needs/ORACLE_CHECK.json")
    assert oracle["status"] == "ok" and oracle.get("contract") == "trial", oracle
    assert [j["measured_by"] for j in oracle["attempts"][-1]["judged"]] == ["engine"], "FI ran the oracle's case itself"
    trials = [json.loads(line) for line in (engine.quest_root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    trials = [t for t in trials if t["event"] == "trial"]
    assert len(trials) == 2 and all(t["seed"] is None for t in trials), "one call per setting, no seed"
    state = artifacts.raw_state
    assert not state.get("result_json_trials") and not state.get("result_json_replicates"), \
        "no replicate statistics are computed for a deterministic run"
    record = _json(engine, "needs/EVIDENCE.json")
    assert record["levels"]["independently_validated"] is True, record["all_gaps"]


@pytest.mark.asyncio
async def test_a_simulation_that_does_not_label_its_equation_is_warned_and_not_independently_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(calls, UNLABELLED))
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "code/simulate.py does not mark where it implements equation E1" in log
    record = _json(engine, "needs/EVIDENCE.json")
    assert record["levels"]["independently_validated"] is False
    assert any("E1" in g for g in record["all_gaps"]["independently_validated"]), record["all_gaps"]
    assert _json(engine, "needs/ORACLE_CHECK.json")["status"] == "ok", "the oracle itself passed: the gap is the label"


def _one_trial_per_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulate: str = SIMULATE) -> Engine:
    """An engine after FI ran a run_cell simulation once per setting, under a protocol that asks for 3 runs."""
    engine = _engine(tmp_path)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    (engine.quest_root / "raw").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "raw" / "ledger.jsonl").write_text("".join(
        json.dumps({"event": "trial", "cell": f"dt={dt}", "trial": 0, "status": "ok"}) + "\n" for dt in (0.1, 0.05)
    ), encoding="utf-8")
    monkeypatch.setattr(engine, "_protocol_block", lambda state: {"grid": {"dt": [0.1, 0.05]}, "runs_per_setting": 3})
    engine._trial_mode, engine._trial_entries = True, {"run_cell"}
    return engine


@pytest.mark.asyncio
async def test_a_run_cell_is_held_to_one_run_per_setting_only_when_fi_finds_no_randomness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine's own wiring: FI's ledger holds one trial per setting and the protocol asks for 3. A run_cell with no
    source of random numbers in its code, whose setting returns the same numbers when called again, matches; one whose
    code names a source, whose second call differs, or that could not be called again is told why and to define
    run_trial, and the count still differs."""
    from core import trial_runner
    from core.execution import ExecutionResult

    ok = ExecutionResult(0, 'RESULT_JSON: {"x": 1}', "", 0.1, False)
    again: list[tuple[bool | None, str]] = []

    async def fake_again(*a, **kw):  # noqa: ANN002, ANN003
        return again.pop(0)

    monkeypatch.setattr(trial_runner, "run_cell_again", fake_again)

    engine = _one_trial_per_setting(tmp_path / "same", monkeypatch)
    record = engine.quest_root / trial_runner.RUN_RECORD
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text('{"key": "k", "cells": []}', encoding="utf-8")
    again.append((True, ""))
    found = await engine._run_cell_randomness({}, "python", None)
    assert found == ("", "none in its code, and settings run a second time returned the same numbers", "")
    assert engine._run_manifest_problems({}, True, ok, found) == ("ok", [])
    assert engine._manifest_note.startswith("the protocol asks for 3 runs per setting, but FI found no randomness")
    assert "calling a setting again to check" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    # The same trials (an analysis repaired on them): the answer is kept, nothing is called again.
    assert await engine._run_cell_randomness({}, "python", None) == found and again == []
    record.write_text('{"key": "other", "cells": []}', encoding="utf-8")  # new trials: asked again
    again.append((True, ""))
    assert await engine._run_cell_randomness({}, "python", None) == found and again == []

    for why, reason, fix in (
        ((False, "the setting dt=0.1 returned other numbers (error: 1 then 2)"), "returned other numbers",
         "make run_cell return the same numbers on every call"),
        ((None, "did not report"), "could not call it a second time", "can be called on its own for one setting"),
    ):
        engine = _one_trial_per_setting(tmp_path / reason[:5], monkeypatch)
        again.append(why)
        found = await engine._run_cell_randomness({}, "python", None)
        status, problems = engine._run_manifest_problems({}, True, ok, found)
        assert status == "differs" and any(reason in p and "define run_trial" in p and fix in p for p in problems), problems
        assert any("ran another number" in p for p in problems) and engine._manifest_note == ""

    # A source of random numbers in the code, or in a module it imports, or named in a docstring: no second call is
    # needed, and what was found is named so that it can be taken out.
    for index, (simulate, named) in enumerate((("import random\n" + SIMULATE, "`random` in simulate.py"),
                                               ("from sampler import draw\n" + SIMULATE, "in sampler.py"),
                                               ('"""Deterministic: no seed."""\n' + SIMULATE, "`seed` in simulate.py"))):
        engine = _one_trial_per_setting(tmp_path / f"code{index}", monkeypatch, simulate)
        (engine.quest_root / "code" / "sampler.py").write_text(
            "import numpy as np\n\ndef draw():\n    return np.random.default_rng().random()\n", encoding="utf-8")
        found = await engine._run_cell_randomness({}, "python", None)
        assert named in found[0] and "take that out of the code" in found[2] and again == [], found
        status, problems = engine._run_manifest_problems({}, True, ok, found)
        assert status == "differs" and any(named in p for p in problems), problems
    # Not asked at all (None): held to the protocol's count.
    engine = _one_trial_per_setting(tmp_path / "none", monkeypatch)
    assert engine._run_manifest_problems({}, True, ok)[0] == "differs"
    # A simulation that defines run_trial is run as trials: the count is held as fixed, and nothing is called again.
    engine = _one_trial_per_setting(tmp_path / "trial", monkeypatch)
    engine._trial_entries = {"run_trial", "run_cell"}
    assert await engine._run_cell_randomness({}, "python", None) == ("", "", "")
    status, problems = engine._run_manifest_problems({}, True, ok, ("", "", ""))
    assert status == "differs" and any("ran another number" in p for p in problems) and engine._manifest_note == ""
    # No entries known (a stale or missing record of what simulate.py defines): not relaxed.
    engine = _one_trial_per_setting(tmp_path / "unknown", monkeypatch)
    engine._trial_entries = set()
    assert engine._run_manifest_problems({}, True, ok, ("", "", ""))[0] == "differs"


@pytest.mark.asyncio
async def test_run_cell_again_compares_a_second_call_with_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import trial_runner

    record = tmp_path / trial_runner.RUN_RECORD
    assert await trial_runner.run_cell_again(None, "python", tmp_path, "code/simulate.py", timeout_s=5) == (
        None, "FI's run record could not be read")
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"cells": [
        {"key": "dt=0.1", "cell": {"dt": 0.1}, "rows": [{"trial": 0, "status": "failed"}]},
        {"key": "dt=0.05", "cell": {"dt": 0.05},
         "rows": [{"trial": 0, "status": "ok", "values": {"error": 0.25, "gone": float("nan")}, "duration_s": 2.0}]},
        {"key": "dt=0.02", "cell": {"dt": 0.02},
         "rows": [{"trial": 0, "status": "ok", "values": {"error": 1e-17}, "duration_s": 1.0}]},
    ]}), encoding="utf-8")
    replies: list[tuple[dict[str, float] | None, str]] = []
    cells: list[dict[str, Any]] = []

    async def fake_case(*a, cell, **kw):  # noqa: ANN002, ANN003
        cells.append(cell)
        return replies.pop(0)

    monkeypatch.setattr(trial_runner, "run_case", fake_case)
    # Rounding noise and two NaNs are the same numbers; the fastest setting first, then the next while the budget lasts.
    replies += [({"error": 1.3e-17}, ""), ({"error": 0.25, "gone": float("nan")}, "")]
    assert await trial_runner.run_cell_again(None, "python", tmp_path, "code/simulate.py", timeout_s=100) == (True, "")
    assert cells == [{"dt": 0.02}, {"dt": 0.05}] and replies == []
    # A short budget: only the fastest setting is called again.
    cells.clear()
    replies.append(({"error": 1e-17}, ""))
    assert await trial_runner.run_cell_again(None, "python", tmp_path, "code/simulate.py", timeout_s=20) == (True, "")
    assert cells == [{"dt": 0.02}]
    replies += [({"error": 1e-17}, ""), ({"error": 0.5, "gone": float("nan")}, "")]
    same, why = await trial_runner.run_cell_again(None, "python", tmp_path, "code/simulate.py", timeout_s=100)
    assert same is False and "dt=0.05" in why and "error: 0.25 then 0.5" in why
    # A start that did not reach the code is tried once more; a second failure is reported.
    replies += [(None, "run_cell() did not report (exit code 1)"), ({"error": 1e-17}, ""),
                ({"error": 0.25, "gone": float("nan")}, "")]
    assert await trial_runner.run_cell_again(None, "python", tmp_path, "code/simulate.py", timeout_s=100) == (True, "")
    replies.append((None, "run_cell() ran out of time"))
    assert await trial_runner.run_cell_again(None, "python", tmp_path, "code/simulate.py", timeout_s=100) == (
        None, "run_cell() ran out of time")


HIDDEN_CLOCK = SIMULATE.replace("import math", "import math\nimport time").replace(
    'return {"error": abs(y - math.exp(-1.0))}', 'return {"error": abs(y - math.exp(-1.0)), "t": time.perf_counter_ns()}')


@pytest.mark.asyncio
async def test_a_run_cell_whose_second_call_differs_is_not_held_to_one_run_through_the_real_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing in the code names a random source, but the numbers change from call to call: FI calls a setting again,
    sees other numbers, and holds the run to the protocol's 3 runs per setting."""
    calls: list[str] = []
    fake = _fake(calls, HIDDEN_CLOCK)

    async def with_repeats(self, messages, **kw):  # noqa: ANN001
        reply = await fake(self, messages, **kw)
        if _classify(messages[-1]["content"]) == "Experiment Design":
            body = json.loads(reply)
            body["protocol"] = {**PROTOCOL, "runs_per_setting": 3}
            return json.dumps(body)
        return reply

    monkeypatch.setattr("core.engine.LLMClient.chat", with_repeats)
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    manifest = _json(engine, "needs/RUN_MANIFEST_CHECK.json")
    assert manifest["status"] in ("stopped", "repairing"), manifest
    assert any("returned other numbers" in p and "define run_trial" in p for p in manifest["problems"]), manifest
    assert "note" not in manifest


@pytest.mark.asyncio
async def test_a_deterministic_quest_whose_protocol_asks_for_repeats_runs_each_setting_once_and_is_not_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A protocol with `runs_per_setting: 3` for a simulation that draws no random numbers: repeating an identical
    calculation returns the same numbers, so FI runs each setting once, and the run check counts one run per setting
    as what the protocol asks, not as a difference to send simulate.py back for."""
    calls: list[str] = []
    prompts: list[str] = []
    fake = _fake(calls, SIMULATE)

    async def with_repeats(self, messages, **kw):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        reply = await fake(self, messages, **kw)
        if _classify(messages[-1]["content"]) == "Experiment Design":
            body = json.loads(reply)
            body["protocol"] = {**PROTOCOL, "runs_per_setting": 3}
            return json.dumps(body)
        return reply

    monkeypatch.setattr("core.engine.LLMClient.chat", with_repeats)
    engine = Engine(_cfg(tmp_path))
    await engine.run()
    trials = [json.loads(line) for line in (engine.quest_root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([t for t in trials if t["event"] == "trial"]) == 2, "one call per setting"
    manifest = _json(engine, "needs/RUN_MANIFEST_CHECK.json")
    assert manifest["status"] == "ok", manifest
    said = ("the protocol asks for 3 runs per setting, but FI found no randomness in the simulation (none in its code, "
            "and settings run a second time returned the same numbers), so each setting ran once")
    assert said in manifest.get("note", ""), manifest
    assert said in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    plan = (engine.quest_root / "plan.md").read_text(encoding="utf-8")
    assert "The protocol asks for 3 runs per setting, but the design's description names nothing random" in plan
    assert "but no target precision" not in plan, "not asked for a precision its repeats would buy"
    assert any("[FI NOTE] The protocol asks for 3 runs per setting" in p and "Report one run per setting" in p
               for p in prompts), "the analysis is told each setting ran once"
    told = {_classify(p) for p in prompts if "Report one run per setting" in p}
    assert len(told) >= 2, f"the paper's writer is told too: {told}"
    assert "ExecuteReflect" not in calls
    record = _json(engine, "needs/EVIDENCE.json")
    assert record["levels"]["protocol_runtime_matched"] is True, record["all_gaps"]
