"""FI's own behaviour checks wired into the run (core/run_checks.py::_probe_gate), RECORD ONLY: called right after the pilot,
one reading of the plan per plan version, the metric names the simulation returns, nothing repaired or stopped, the verdicts
written down (record, needs/PROBES.json, QUALITY.md) and said in the paper's limitations. Fake model, fake executor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import probes_run as PR
from core import quality_report as qr
from core import run_checks
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.run_estimate import Probe
from tests.test_probes_run import FakeExecutor

GRID = {"dt": [0.4, 0.2, 0.1, 0.05], "mode": ["a", "b"]}
PROTOCOL = {"grid": GRID, "metrics": [{"id": "err_pct"}, {"id": "rate"}, {"id": "k_covered_1"}], "thresholds": {}}
PLAN = "# Plan\n\n## In short\n\nA smaller step dt gives a smaller error. Mode b is more accurate than mode a.\n"
SIM = "def run_trial(cell, trial, seed):\n    return {'err': 1.0}\n"


def noise(seed: int) -> float:
    return ((seed % 1000) / 1000.0 - 0.5) * 0.02


def good(cell: dict[str, Any], seed: int) -> dict[str, float]:
    return {"err": cell["dt"] ** 2 * 3 + noise(seed) + (0.0 if cell["mode"] == "b" else 0.5), "rate": 5.0 + noise(seed),
            "k_covered_1": 1.0}


def ignores_dt(cell: dict[str, Any], seed: int) -> dict[str, float]:
    return {"err": 1.0, "rate": 5.0 + noise(seed), "k_covered_1": 1.0}


class _Client:
    def __init__(self, reply: str) -> None:
        self.reply, self.asked = reply, []
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.reply


def _reply(*items: dict[str, Any]) -> str:
    return json.dumps({"relations": list(items)})


SENS = {"id": "R1", "kind": "sensitivity", "setting": "dt", "metric": "err_pct", "strength": "core", "from_plan": "a smaller step dt gives a smaller error"}


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: str, fn: Any = good) -> tuple[Engine, _Client, FakeExecutor]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    eng = Engine(Config(
        topic="a neutral topic", title="probes", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=3600),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    client = _Client(reply)
    eng._client = client  # type: ignore[assignment]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (eng.quest_root / "plan.md").write_text(PLAN, encoding="utf-8")
    eng._protocol_block = lambda state: PROTOCOL  # type: ignore[method-assign]
    ex = FakeExecutor(fn)
    eng.executor = ex  # type: ignore[assignment]
    return eng, client, ex


def _probes(**trial: float) -> list[Probe]:
    trial = trial or {"err": 1.0, "rate": 5.0, "k_covered_1": 1.0}
    return [Probe(axis=None, index=None, cell={"dt": 0.1, "mode": "a"}, seconds=0.1, trials=[dict(trial), dict(trial)])]


async def _gate(eng: Engine, grid: dict[str, Any] | None = None, probes: list[Probe] | None = None,
                state: dict[str, Any] | None = None) -> Any:
    return await eng._probe_gate(state or {"iteration": 0}, None, "python", None, GRID if grid is None else grid,
                                 _probes() if probes is None else probes, {}, 3600.0)


def _record(eng: Engine) -> dict[str, Any]:
    return json.loads((eng.quest_root / PR.RECORD).read_text(encoding="utf-8"))


def _mine(eng: Engine) -> dict[str, Any]:
    return _record(eng)["results"][PR.code_sha(eng.quest_root)]


def _log(eng: Engine) -> str:
    return (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")


def test_the_constants_start_off_and_the_gate_follows_the_pilot_in_the_sizing_loop() -> None:
    assert run_checks.STOP_ENABLED is False and run_checks.REPAIR_ENABLED is False
    src = (Path(run_checks.__file__)).read_text(encoding="utf-8")
    assert src.index("await self._pilot_gate(") < src.index("await self._probe_gate(") < src.index("est = _estimate.estimate(")


@pytest.mark.asyncio
async def test_a_good_simulation_is_read_once_checked_and_recorded_with_the_names_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, client, ex = _engine(tmp_path, monkeypatch, _reply(SENS))
    assert await _gate(eng) is None
    assert len(client.asked) == 1
    prompt = client.asked[0]
    # the reader is told the names the simulation returns, with the plan's own name next to the one it stands for
    assert "`err` (the plan's `err_pct`)" in prompt and "`rate`" in prompt and "`dt`" in prompt
    assert "A smaller step dt gives a smaller error" in prompt
    res = _mine(eng)
    assert res["R1"]["verdict"] == "held" and res["saturation"]["verdict"] == "held"
    # the relation named the plan's metric id; FI ran the returned name (the alias reached probes_run)
    assert not any("does not return" in v["says"] for v in res.values())
    assert json.loads((eng.quest_root / PR.SUMMARY).read_text(encoding="utf-8"))["held"]
    assert "FI's own behaviour checks (recorded, not yet acting): 1 held, 0 could not be checked, 0 did not hold" in _log(eng)
    assert eng._run_checks_note() == ""
    assert not (eng.quest_root / ".fi" / "probes_run").exists()
    # within about a tenth of the run's time limit
    assert ex.calls and sum(1 for _ in ex.calls) <= 20


@pytest.mark.asyncio
async def test_a_broken_core_relation_is_recorded_and_said_but_nothing_is_repaired_or_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, client, _ex = _engine(tmp_path, monkeypatch, _reply(SENS), fn=ignores_dt)

    def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("record only: nothing goes back and nothing stops")

    monkeypatch.setattr(eng, "_code_review_failed_run", never, raising=False)
    monkeypatch.setattr(eng, "_stop_before_run", never, raising=False)
    assert await _gate(eng) is None
    r = _mine(eng)["R1"]
    assert r["verdict"] == "broken" and r["strength"] == "core" and "exactly the same" in r["says"]
    plain = json.loads((eng.quest_root / PR.SUMMARY).read_text(encoding="utf-8"))
    assert [b["id"] for b in plain["broken"]] == ["R1"]
    note = eng._run_checks_note()
    assert "FI's own check that `dt` changes `err_pct` did not hold" in note and "limitations" in note
    assert "FI's own check that `dt` changes `err_pct` did not hold" in _log(eng)
    # QUALITY.md says it, in plain words
    quality = qr.render(qr.build(eng.quest_root))
    assert "FI's own behaviour checks (recorded, not yet acting)" in quality and "exactly the same" in quality


@pytest.mark.asyncio
async def test_the_repair_and_stop_switches_do_what_the_design_said_when_they_are_turned_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, _client_, _ex = _engine(tmp_path, monkeypatch, _reply(SENS), fn=ignores_dt)
    sent: list[str] = []
    monkeypatch.setattr(eng, "_code_review_failed_run", lambda text, script=None: sent.append(text) or {"sent_back": True}, raising=False)
    monkeypatch.setattr(run_checks, "REPAIR_ENABLED", True)
    assert await _gate(eng) == {"sent_back": True}
    assert "did not hold" in sent[0] and "exactly the same" in sent[0] and _record(eng)["repairs"] == 1
    # repairs used up and stop on: the honest stop
    monkeypatch.setattr(run_checks, "STOP_ENABLED", True)
    stopped = await _gate(eng, state={"iteration": 0, "exec_reflect_iter": 99})
    assert stopped is not None and "cannot_repair" in stopped["exec_result"]


@pytest.mark.asyncio
async def test_a_resume_never_asks_again_and_the_number_of_readings_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, client, ex = _engine(tmp_path, monkeypatch, _reply(SENS))
    await _gate(eng)
    cells = len(ex.calls)
    await _gate(eng)
    assert len(client.asked) == 1 and len(ex.calls) == cells, "the same plan and code: nothing asked, nothing run"
    # killed during the reading: the count was written before the call, the answer never came
    record = _record(eng)
    record["relations"], record["results"] = [], {}
    (eng.quest_root / PR.RECORD).write_text(json.dumps(record), encoding="utf-8")
    await _gate(eng)
    assert len(client.asked) == 1
    # a new plan version asks again, up to the bound
    for n in range(run_checks.PROBE_MAX_ASKS + 2):
        (eng.quest_root / "plan.md").write_text(PLAN + f"\nrevision {n}\n", encoding="utf-8")
        await _gate(eng)
    assert len(client.asked) == run_checks.PROBE_MAX_ASKS
    assert _record(eng)["asked"]["calls"] == run_checks.PROBE_MAX_ASKS


@pytest.mark.asyncio
async def test_a_reply_that_names_an_unknown_setting_or_metric_is_dropped_and_a_failed_reading_stops_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, client, _ex = _engine(tmp_path, monkeypatch, _reply({**SENS, "id": "R2", "metric": "made_up"}, {**SENS, "id": "R3", "setting": "zzz"}))
    assert await _gate(eng) is None
    rec = _record(eng)
    assert rec["relations"] == [] and len(rec["dropped"]) == 2

    async def boom(messages: list[dict[str, str]], **kw: Any) -> str:
        raise RuntimeError("provider down")

    eng2, client2, _ = _engine(tmp_path / "two", monkeypatch, "")
    client2.chat = boom  # type: ignore[method-assign]
    assert await _gate(eng2) is None
    assert _record(eng2)["asked"]["calls"] == 1 and "the call failed" in _record(eng2)["dropped"][0]["reason"]


@pytest.mark.asyncio
async def test_a_quest_with_no_grid_or_no_metrics_or_nothing_returned_skips_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, client, ex = _engine(tmp_path, monkeypatch, _reply(SENS))
    assert await _gate(eng, grid={}) is None
    assert await _gate(eng, probes=[Probe(axis=None, index=None, cell={}, seconds=0.1, trials=[])]) is None
    eng._protocol_block = lambda state: {"grid": GRID, "metrics": []}  # type: ignore[method-assign]
    assert await _gate(eng) is None
    assert client.asked == [] and ex.calls == [] and not (eng.quest_root / PR.RECORD).exists()
    assert "behaviour check" not in _log(eng)


@pytest.mark.asyncio
async def test_a_coverage_that_is_all_ones_at_a_few_draws_is_not_called_broken_on_good_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the r6t kinetics case: k_covered_1 is 8 of 8 at every setting on good code, and the protocol reports it
    eng, _client_, _ex = _engine(tmp_path, monkeypatch, _reply(SENS))
    assert await _gate(eng) is None
    res = _mine(eng)
    assert not any(k.startswith("saturation:") and v["verdict"] == "broken" for k, v in res.items())
    assert res["saturation"]["verdict"] == "held"


CMP = {"id": "R2", "kind": "compare", "metric": "err_pct", "a": {"mode": "a"}, "b": {"mode": "b"}, "larger": "a", "strength": "core",
       "from_plan": "mode b is more accurate than mode a"}


@pytest.mark.asyncio
async def test_a_comparison_of_two_methods_is_run_through_the_gate_and_judged_by_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, client, _ex = _engine(tmp_path, monkeypatch, _reply(CMP))
    assert await _gate(eng) is None
    assert "`compare`" in client.asked[0]
    assert _mine(eng)["R2"]["verdict"] == "held"

    def reversed_methods(cell: dict[str, Any], seed: int) -> dict[str, float]:
        out = good(cell, seed)
        out["err"] = out["err"] + (1.0 if cell["mode"] == "b" else -1.0)   # the wrong method now looks better
        return out

    eng2, _c, _e = _engine(tmp_path / "two", monkeypatch, _reply(CMP), fn=reversed_methods)
    assert await _gate(eng2) is None
    r = _mine(eng2)["R2"]
    assert r["verdict"] == "broken" and "smaller" in r["says"]
    assert "FI's own check that `err_pct` is larger in one case than in the other" in eng2._run_checks_note()
