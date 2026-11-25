"""Fleet validation: two Engines run concurrently with a fake LLM.

Verifies (a) N parallel engines can coexist in one process, (b) each gets
a distinct quest_id and quest_root, (c) per-quest venvs do not collide,
(d) figures and paper.md are written for both quests independently.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import Engine


_FAKE_EXPERIMENT_CODE = """\
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
plt.figure()
plt.plot(list(range(3)), [i * i for i in range(3)])
plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: {"score": 0.5}')
"""


_FAKE_RESPONSES = {
    "Experiment Design": json.dumps({
        "hypothesis": "fleet hypothesis",
        "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
        "method": "plot",
        "expected_outcome": "monotonic",
        "figures_planned": ["result.png"],
        "dependencies": ["matplotlib"],
    }),
    "Implementation": json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib"]}),
    "Analysis": json.dumps({
        "summary": "Curve plotted.",
        "key_findings": ["y grows with x"],
        "claims_supported": [],
        "claims_unsupported": [],
        "limitations": [],
    }),
    "Writing": "# fleet-fake\n\nResult.\n\n![r](figures/result.png)\n",
    "Review": json.dumps({
        "verdict": "accept",
        "score": 4,
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
        "blocking": "",
    }),
}


def _fake_response_for(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0]
    for tag, resp in _FAKE_RESPONSES.items():
        if tag in head:
            return resp
    return "{}"


def _make_cfg(tmp_path: Path, slug: str) -> Config:
    return Config(
        topic=f"fleet smoke {slug}",
        title=f"fleet-{slug}",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            # Default gate is "after_review"; auto-accept-on-pass
            # finalises the clean-verdict happy path without a callback.
            auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
async def test_two_engines_run_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    cfg_a = _make_cfg(tmp_path, "alpha")
    cfg_b = _make_cfg(tmp_path, "bravo")
    engine_a = Engine(cfg_a)
    engine_b = Engine(cfg_b)

    assert engine_a.quest_id != engine_b.quest_id
    assert engine_a.quest_root != engine_b.quest_root

    art_a, art_b = await asyncio.gather(engine_a.run(), engine_b.run())

    for art in (art_a, art_b):
        assert art.paper_md is not None and art.paper_md.exists()
        assert art.figures_dir is not None
        assert (art.figures_dir / "result.png").exists()
        assert art.raw_state.get("review", {}).get("verdict") == "accept"

    # Distinct on-disk artifacts.
    assert art_a.quest_root != art_b.quest_root
    assert art_a.paper_md != art_b.paper_md


# --- run_fleet runner unit tests (isolation + memory-cap deadlock, PR-3) ---


@pytest.mark.asyncio
async def test_run_fleet_isolates_construction_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quest whose Engine construction fails must NOT abort the fleet — the
    sibling quests still complete, and run_fleet returns a non-zero exit."""
    import launch as fi_launch

    good = _make_cfg(tmp_path, "good")
    bad = _make_cfg(tmp_path, "bad")
    ran: list[Config] = []

    class _FakeEngine:
        def __init__(self, cfg, **kw):  # noqa: ANN001
            if cfg is bad:
                raise ValueError("invalid provider config")
            self.quest_id = "good-quest"

    async def fake_run_one(cfg, **kw):  # noqa: ANN001
        ran.append(cfg)
        return {"quest_id": "good-quest"}

    monkeypatch.setattr(fi_launch, "Engine", _FakeEngine)
    monkeypatch.setattr(fi_launch, "run_one", fake_run_one)

    rc = await fi_launch.run_fleet(
        [good, bad], supervisor=fi_launch.ProxySupervisor(),
        max_concurrent=2, memory_cap_mb=None, profile=False,
    )
    assert good in ran          # the good quest ran — not orphaned
    assert rc == 1              # the bad quest is counted as a failure


@pytest.mark.asyncio
async def test_run_fleet_disables_unsatisfiable_memory_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """A memory cap below the baseline RSS would deadlock the fleet (nothing
    runs to free memory). It must be disabled with a warning, not hung on."""
    import launch as fi_launch

    good = _make_cfg(tmp_path, "g")

    class _FakeEngine:
        def __init__(self, cfg, **kw):  # noqa: ANN001
            self.quest_id = "g"

    async def fake_run_one(cfg, **kw):  # noqa: ANN001
        return {"quest_id": "g"}

    monkeypatch.setattr(fi_launch, "Engine", _FakeEngine)
    monkeypatch.setattr(fi_launch, "run_one", fake_run_one)
    monkeypatch.setattr(fi_launch, "_rss_mb", lambda: 2000.0)  # baseline > cap

    rc = await fi_launch.run_fleet(
        [good], supervisor=fi_launch.ProxySupervisor(),
        max_concurrent=1, memory_cap_mb=100, profile=False,
    )
    assert rc == 0  # completed (did not deadlock)
    assert "disabling the cap" in capsys.readouterr().err


# --- the quests of a fleet share the cores (execution.settings_at_once: auto) ---


@pytest.mark.asyncio
@pytest.mark.parametrize(("quests", "max_concurrent", "together"), [(3, 4, 3), (3, 2, 2), (1, 4, 1)])
async def test_run_fleet_tells_every_engine_how_many_quests_run_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quests: int, max_concurrent: int, together: int,
) -> None:
    """Every Engine of a fleet is told how many quests run at the same time: the fleet's size, at most
    --max-concurrent (the quests that wait for a slot do not compute meanwhile)."""
    import launch as fi_launch

    told: list[object] = []

    class _RecordingEngine:
        def __init__(self, cfg, **kw):  # noqa: ANN001
            told.append(kw.get("fleet"))
            self.quest_id = cfg.title

    async def fake_run_one(cfg, **kw):  # noqa: ANN001
        return {"quest_id": cfg.title}

    monkeypatch.setattr(fi_launch, "Engine", _RecordingEngine)
    monkeypatch.setattr(fi_launch, "run_one", fake_run_one)

    rc = await fi_launch.run_fleet(
        [_make_cfg(tmp_path, f"q{i}") for i in range(quests)], supervisor=fi_launch.ProxySupervisor(),
        max_concurrent=max_concurrent, memory_cap_mb=None, profile=False,
    )
    assert rc == 0
    assert told == [together] * quests


def test_auto_settings_at_once_is_the_quests_share_of_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``auto`` gives a quest of a fleet of 3 on 16 cores 5 settings at once (a quest alone keeps all 16), a number
    written in the config is kept as written, and the run log says the share in one line only in a fleet."""
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    alone = Engine(_make_cfg(tmp_path, "alone"))
    shared = Engine(_make_cfg(tmp_path, "shared"), fleet=3)
    written_cfg = _make_cfg(tmp_path, "written")
    written_cfg.execution.settings_at_once = 4
    written = Engine(written_cfg, fleet=3)
    assert alone._settings_at_once() == 16
    assert shared._settings_at_once() == 5
    assert written._settings_at_once() == 4
    assert Engine(_make_cfg(tmp_path, "many"), fleet=40)._settings_at_once() == 1

    def run_log(engine: Engine) -> str:
        for h in engine._log.handlers:
            h.flush()
        p = engine.fi_dir / "run.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""

    assert "[fleet]" not in run_log(alone)
    assert ("[fleet] 3 quests run at the same time: this one runs up to 5 settings of its study at the same time, its "
            "share of the computer's 16 cores") in run_log(shared)
    assert "up to 4 settings of its study at the same time, as execution.settings_at_once says" in run_log(written)


# --- every place that runs the study's settings uses the quest's share ---


class _Handed(Exception):
    """Raised by a stand-in runner once it has been handed its number, so the step stops there."""


def _split_engine(tmp_path: Path, simulate: str) -> Engine:
    """A quest of a fleet of 3 on a 12-core computer with two scripts written and nothing to install."""
    from unittest.mock import AsyncMock

    cfg = Config(topic="toy study", title="toy", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", pilot_run=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"))
    assert cfg.execution.settings_at_once == "auto"
    engine = Engine(cfg, fleet=3)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    engine.executor.install = AsyncMock(return_value=type("IR", (), {"returncode": 0, "stderr": ""})())  # type: ignore
    (engine.quest_root / "code" / "simulate.py").write_text(simulate, encoding="utf-8")
    (engine.quest_root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    return engine


async def _handed_at_once(engine: Engine, protocol: dict[str, object]) -> list[int]:
    engine._protocol_block = lambda state: protocol  # type: ignore[method-assign]
    state: dict[str, object] = {"deps": [], "design": {"protocol": protocol}, "iteration": 0, "exec_reflect_iter": 0}
    with pytest.raises(_Handed):
        await engine._node_execute_steps(state)  # type: ignore[arg-type]
    return handed


handed: list[int] = []


@pytest.mark.asyncio
async def test_the_run_step_hands_the_trials_runner_the_quests_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run step of a study FI runs trial by trial (core/engine_run.py, TrialsRunner): a quest of a fleet of 3 on 12
    cores runs 4 settings at once, not 12."""
    from core import trial_runner

    class _Runner:
        def __init__(self, *a, at_once: int = 1, **kw):  # noqa: ANN001, ANN002, ANN003
            handed.append(at_once)
            raise _Handed

    handed.clear()
    monkeypatch.setattr(trial_runner, "TrialsRunner", _Runner)
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    engine = _split_engine(tmp_path, "def run_trial(cell, seed):\n    return {'y': 1.0}\n")
    assert await _handed_at_once(engine, {"grid": {"x": [1.0, 2.0]}}) == [4]


@pytest.mark.asyncio
async def test_the_run_step_hands_the_search_the_quests_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run step of a search for the best design (core/engine_run.py, OptimisationRunner): 4 at once on 12 cores
    for a quest of a fleet of 3."""
    from core import optimise
    from tests.test_optimise_runner import SIMULATE, _protocol

    class _Runner:
        def __init__(self, *a, at_once: int = 1, **kw):  # noqa: ANN001, ANN002, ANN003
            handed.append(at_once)
            raise _Handed

    handed.clear()
    monkeypatch.setattr(optimise, "OptimisationRunner", _Runner)
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    engine = _split_engine(tmp_path, SIMULATE.replace("__COUNT__", str(tmp_path / "count.txt"))
                           .replace("__CWD__", str(tmp_path / "cwd.txt")))
    protocol = _protocol(evaluation_budget={"starts": 1, "per_start": 6}, grid={"x": [-1.0, 1.0]})
    assert await _handed_at_once(engine, protocol) == [4]


@pytest.mark.asyncio
async def test_a_round_of_improvement_runs_the_trials_at_the_quests_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A round of improvement measuring the trials (core/engine_improve.py::_improve_measure): 4 at once on 12 cores for
    a quest of a fleet of 3."""
    from core import trial_runner

    async def fake_run_trials(*a, at_once: int = 1, **kw):  # noqa: ANN001, ANN002, ANN003
        handed.append(at_once)
        raise _Handed

    handed.clear()
    monkeypatch.setattr(trial_runner, "run_trials", fake_run_trials)
    monkeypatch.setattr("os.cpu_count", lambda: 12)
    engine = _split_engine(tmp_path, "def run_trial(cell, seed):\n    return {'y': 1.0}\n")
    _rows, problems = await engine._improve_measure({"grid": {"x": [1.0, 2.0]}}, [], {"y"})
    assert handed == [4]
    assert any("the trials could not be run" in p for p in problems)


# --- the threads of a fleet's processes: each process its share of the quest's share of the cores ---


def _sharing_executor(cores: int | None):
    """Runs a command in this interpreter's environment, keeping the environment each process was given."""
    from tests.test_optimise_runner import LocalExecutor

    class _Sharing(LocalExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.cores = cores
            self.envs: list[dict[str, str]] = []

        async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001
            self.envs.append(dict(env or {}))
            return await super().execute(cmd, cwd=cwd, timeout_s=timeout_s, env=env)

    return _Sharing()


def _trial_quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(
        "def run_trial(cell, seed):\n    return {'y': float(cell['x'])}\n\n\n"
        "def run_cell(cell):\n    return {'y': float(cell['x'])}\n", encoding="utf-8")
    return root


def test_an_engine_of_a_fleet_gives_its_executor_its_share_of_the_cores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quest of a fleet of 2 on 16 cores: its processes share 8 cores; a quest alone does not limit them."""
    from core import execution

    monkeypatch.setattr("os.cpu_count", lambda: 16)
    assert execution.quest_cores(Engine(_make_cfg(tmp_path, "two"), fleet=2).executor) == 8
    assert execution.quest_cores(Engine(_make_cfg(tmp_path, "one")).executor) is None


@pytest.mark.parametrize(("cores", "threads"), [(8, "1"), (None, "2")])
def test_the_trials_of_a_quest_in_a_fleet_start_their_share_of_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cores: int | None, threads: str,
) -> None:
    """8 settings at once of a quest whose processes share 8 of 16 cores: 1 thread each (2 quests x 8 x 1 = 16); alone,
    2 each (8 x 2 = 16)."""
    import sys

    from core import trial_runner

    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    root = _trial_quest(tmp_path)
    executor = _sharing_executor(cores)
    run = asyncio.run(trial_runner.run_trials(
        executor, sys.executable, root, "code/simulate.py", {"x": [1.0, 2.0]}, runs_per_setting=1, base_seed=0,
        deterministic=True, timeout_s=120, env=dict(os.environ), at_once=8))
    assert run.ok_trials == 2
    assert executor.envs and {e.get("OMP_NUM_THREADS") for e in executor.envs} == {threads}


def test_the_check_of_settings_sharing_a_process_runs_on_the_quests_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check whether settings may share a process (core/trial_runner.py::measure_grouping), 2 at once on a quest's
    8 of 16 cores: 4 threads each, not 8."""
    import sys

    from core import trial_runner

    monkeypatch.setattr("os.cpu_count", lambda: 16)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    root = _trial_quest(tmp_path)
    executor = _sharing_executor(8)
    asyncio.run(trial_runner.measure_grouping(
        executor, sys.executable, root, "code/simulate.py", {"x": [1.0, 2.0, 3.0, 4.0]}, runs_per_setting=1,
        base_seed=0, deterministic=True, timeout_s=120, per_process=2, cost=None,
        env=dict(os.environ), at_once=2))
    assert executor.envs and {e.get("OMP_NUM_THREADS") for e in executor.envs} == {"4"}


def test_a_search_s_designs_at_once_start_the_quests_share_of_threads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A search's designs run together (core/optimise.py::EvaluationPool), 8 at once on a quest's 8 of 16 cores: 1
    thread each."""
    from core import optimise

    monkeypatch.setattr("os.cpu_count", lambda: 16)
    root = _trial_quest(tmp_path)
    pool = optimise.EvaluationPool(_sharing_executor(8), "python", root, "code/simulate.py", "run_trial", at_once=8,
                                   runs=1, base_seed=0, deadline=1e12, env={"PATH": "p"}, thresholds=None)
    assert pool.env is not None and pool.env["OMP_NUM_THREADS"] == "1"


def test_the_time_estimate_times_settings_at_once_on_the_quests_share(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timing of how much slower settings run together (core/run_estimate.py::slowdown_at_once) gives each copy the
    same threads the run gives it: 8 copies on a quest's 8 of 16 cores, 1 thread each."""
    import sys

    from core import run_estimate as est

    seen: list[str | None] = []

    async def timed(*a, name, env, **k):  # noqa: ANN001, ANN003
        seen.append((env or {}).get("OMP_NUM_THREADS"))
        return est.Probe(None, None, {}, 1.0), ""

    monkeypatch.setattr(est, "_time_one", timed)
    monkeypatch.setattr("os.cpu_count", lambda: 16)
    factor, _ = asyncio.run(est.slowdown_at_once(_sharing_executor(8), sys.executable, Path("."), "m.py", {}, at_once=8,
                                                 runs=1, deterministic=True, timeout_s=10, env={"PATH": "p"},
                                                 thresholds={}))
    assert factor == 1.0 and seen == [None] + ["1"] * 8
