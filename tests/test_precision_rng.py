"""Precision from a target, random streams that are not shared by accident, a grid dense enough for its claim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import plan, protocol_check as pc, stats
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_EXPERIMENT_CODE, _FAKE_RESPONSES, _classify, _fake_response_for

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "protocol_runs"


def _real(name: str) -> str:
    return (FIXTURES / f"sir_{name}_experiment.py").read_text(encoding="utf-8")


# --- the arithmetic ------------------------------------------------------------------------------------------------------


def test_the_trials_a_target_needs_and_the_width_the_trials_give() -> None:
    assert stats.trials_for_half_width(0.03) == 1068
    assert stats.trials_for_half_width(0.05) == 385
    assert stats.trials_for_half_width(0.05, p=0.1) == 139
    for bad in (0, 1, -0.1, 1.5, True, "0.03", None):
        assert stats.trials_for_half_width(bad) is None  # type: ignore[arg-type]
    assert stats.wilson_half_width(450, 900) == pytest.approx(0.0326, abs=1e-4)
    assert stats.wilson_half_width(5, 3) is None


def test_the_target_in_the_protocol_is_checked_for_shape() -> None:
    ok, _ = plan.normalize_design({"hypothesis": "h", "protocol": {"precision": {"target_half_width": 0.03, "metric": "p"}}})
    assert ok["protocol"]["precision"]["target_half_width"] == 0.03
    for bad in ({"precision": 0.03}, {"precision": {}}, {"precision": {"target_half_width": 2}},
                {"precision": {"target_half_width": "small"}}, {"precision": {"target_half_width": 0.03, "reason": 5}}):
        design, error = plan.normalize_design({"hypothesis": "h", "protocol": bad})
        assert design is None and "precision" in (error or "")


# --- the plan says what the target needs -------------------------------------------------------------------------------


def test_a_target_the_planned_trials_cannot_reach_is_said_in_the_plan() -> None:
    proto = {"runs_per_setting": 300, "precision": {"target_half_width": 0.03}}
    (note,) = pc.precision_notes(proto, 3)
    assert "needs about 1068 trials" in note and "300 runs over 3 seed(s) give 900" in note and "about ±0.033" in note
    assert "Raise the runs to about 356 per seed" in note
    assert pc.precision_notes({**proto, "runs_per_setting": 400}, 3) == [], "1200 trials reach ±0.03"
    assert pc.precision_notes({**proto, "precision": {"target_half_width": 0.05}}, 3) == []


def test_runs_with_no_target_are_told_what_they_give() -> None:
    (note,) = pc.precision_notes({"runs_per_setting": 300}, 3)
    assert "no target precision" in note and "±0.033" in note and "target_half_width" in note
    assert pc.precision_notes({"grid": {"R0": [1.0]}}, 3) == [] and pc.precision_notes(None, 3) == []
    assert pc.precision_notes({"runs_per_setting": 1}, 3) == []


@pytest.mark.parametrize("design, expected", [
    ({"hypothesis": "the error converges as N grows", "protocol": {"grid": {"N": [100, 1000, 5000]}}}, True),
    ({"hypothesis": "h", "expected_outcome": "a phase transition near R0 = 1", "protocol": {"grid": {"R0": [0.9, 1.5, 3.0]}}}, True),
    ({"hypothesis": "the error converges", "protocol": {"grid": {"N": [10, 20, 40, 80, 160]}}}, False),
    ({"hypothesis": "model A beats model B", "protocol": {"grid": {"N": [100, 1000]}}}, False),
    ({"hypothesis": "converges", "protocol": {"grid": {"N": [100]}}}, False),
    ({"hypothesis": "converges"}, False),
    (None, False),
])
def test_a_thin_grid_is_said_only_when_the_design_claims_something_about_the_parameter(design: Any, expected: bool) -> None:
    notes = pc.grid_notes(design)
    assert bool(notes) is expected
    if expected:
        assert "fewer than five values" in notes[0]


# --- streams that restart on every call ----------------------------------------------------------------------------------


def test_the_run_that_rebuilt_its_stream_for_every_setting_is_the_only_one_of_the_three_flagged() -> None:
    assert pc.rng_reuse(_real("fr1")) == [] and pc.rng_reuse(_real("fr2")) == []
    assert pc.rng_reuse(_real("fr3")) == [("run_stochastic_experiment", 78)]
    found = pc.check({"seed_policy": "one independent stream per setting and run"}, {"experiment.py": _real("fr3")})
    assert [(m.kind, m.name) for m in found] == [("rng", "run_stochastic_experiment")]
    assert "SeedSequence" in found[0].message() and "line 78" in found[0].message()


REUSED = """
import os
import numpy as np
def cell(R0, N):
    seed = int(os.environ.get("FI_REPLICATE_SEED", 0))
    rng = np.random.default_rng(seed)
    return rng.random(N).mean()
for R0 in (0.9, 1.5):
    for N in (100, 1000):
        cell(R0, N)
"""


def test_a_stream_built_from_the_seed_alone_inside_a_function_called_for_every_setting_is_reported() -> None:
    assert pc.rng_reuse(REUSED) == [("cell", 6)]
    assert pc.rng_reuse(REUSED.replace("default_rng(seed)", "RandomState(seed)")) == [("cell", 6)]
    assert pc.rng_reuse(REUSED.replace("rng = np.random.default_rng(seed)", "np.random.seed(seed)")) == [("cell", 6)]


@pytest.mark.parametrize("code", [
    REUSED.replace("default_rng(seed)", "default_rng([seed, R0, N])"),                    # varies with the arguments
    REUSED.replace("default_rng(seed)", "default_rng(np.random.SeedSequence([seed, N]))"),
    REUSED.replace("    seed = int", "    base = R0 * N\n    seed = int(base) + int"),        # a local computed from them
    "import numpy as np\ndef one(N):\n    return np.random.default_rng(0).random(N)\none(5)\n",    # called once
    "import numpy as np\ndef make():\n    return np.random.default_rng(0)\nfor _ in range(3):\n    make()\n",  # no arguments
    "import numpy as np\ndef cell(rng, N):\n    return rng.random(N)\nrng = np.random.default_rng(1)\nfor N in (1, 2):\n    cell(rng, N)\n",
    "def broken(:\n",
])
def test_a_stream_that_varies_with_the_setting_or_is_built_once_is_not_reported(code: str) -> None:
    assert pc.rng_reuse(code) == []


def test_common_random_numbers_declared_in_the_protocol_are_not_reported() -> None:
    assert pc.check({"seed_policy": "independent streams"}, {"e.py": REUSED})[0].kind == "rng"
    for policy in ("common random numbers across settings, analysed as paired", "CRN", "shared stream by design", "coupled runs"):
        assert pc.check({"seed_policy": policy}, {"e.py": REUSED}) == [], policy
    assert pc.check({}, {"e.py": REUSED})[0].kind == "rng", "no stated policy means independent streams"


# --- through the real graph and a real checkpoint -------------------------------------------------------------------------

_PROTOCOL = {"runs_per_setting": 300, "seed_policy": "one independent stream per setting and run",
             "precision": {"target_half_width": 0.01, "metric": "outbreak_probability"}}
_COUNTS = """\
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
seed = int(os.environ.get("FI_REPLICATE_SEED", "0"))
k = 100 - (seed // 1000000) % 3
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({"p": k / 300, "p_count": k, "p_total": 300}))
"""


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="smoke topic for precision", title="precision-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=3,
                            pilot_run=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _fake(prompts: list[str], *, implement: str, repair: str | None = None, protocol: dict[str, Any] = _PROTOCOL):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": implement, "deps": ["matplotlib"]})
        if kind == "ExecuteReflect" and "experiment protocol that was fixed in the plan" in prompt:
            prompts.append("ProtocolRepair")
            return json.dumps({"code": repair, "deps": [], "patch_summary": "derived the stream from the setting"})
        if kind == "Analysis":
            prompts.append(prompt)
        return _fake_response_for(prompt)

    return fake_chat


@pytest.mark.asyncio
async def test_the_analysis_is_told_when_the_target_precision_was_not_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(seen, implement=_COUNTS))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    analysis = next(p for p in seen if p != "ProtocolRepair")
    assert '"ci_method": "wilson_pooled_counts"' in analysis and '"ci_half_width": 0.03' in analysis
    assert '"target_half_width": 0.01' in analysis and '"precision_reached": false' in analysis
    assert "was not reached for 1 of the probabilities: p (±0.03" in analysis
    text = plan.plan_path(engine.quest_root).read_text(encoding="utf-8")
    assert "needs about 9604 trials" in text, "the plan said so before any compute was spent"


_REUSING = """\
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def cell(R0):
    rng = np.random.default_rng(int(os.environ.get("FI_REPLICATE_SEED", "0")))
    return float(rng.random())

vals = [cell(r) for r in (0.9, 1.5, 3.0)]
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({"score": vals[0]}))
"""
_FIXED = _REUSING.replace('rng = np.random.default_rng(int(os.environ.get("FI_REPLICATE_SEED", "0")))',
                          'rng = np.random.default_rng([int(os.environ.get("FI_REPLICATE_SEED", "0")), int(R0 * 100)])')


@pytest.mark.asyncio
async def test_a_script_that_restarts_its_stream_for_every_setting_is_sent_back_and_the_repaired_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    protocol = {"runs_per_setting": 300, "seed_policy": "independent streams"}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(seen, implement=_REUSING, repair=_FIXED, protocol=protocol))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None and seen.count("ProtocolRepair") == 1
    assert "int(R0 * 100)" in (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    record = json.loads((engine.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok" and "restarts the same stream" not in record["attempts"][-1]["differences"]
    assert "share their random numbers" in record["attempts"][0]["differences"][0]


def test_the_plan_directive_asks_for_a_target_a_dense_grid_and_a_stream_policy() -> None:
    from core.engine import _PLAN_DIRECTIVE

    for words in ('"precision"', "0.96/h^2", "at least five values", "its own stream", "common random numbers"):
        assert words in _PLAN_DIRECTIVE, words
    assert "not reached" in (Path(__file__).resolve().parent.parent / "agents" / "analyze.md").read_text(encoding="utf-8")
