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

from core import evidence, oracle_check as oc
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
