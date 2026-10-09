"""Replay of a real stop: a study of 104 settings x 1000 trials ran for about two hours before FI found that two of the
protocol's metrics had never been a number (not-a-number in every trial that returned them: "0 with a value and 52000 with
none"), and then stopped with its repairs spent. The pilot that already times a few real trials now reads what they
returned, so the same fault is found in minutes and goes to the existing repair of the simulation; a metric that is
undefined at some settings and has a value at others is not a finding. The evidence is the real per-trial record of that
quest (tests/fixtures/replay/pilot_undefined_metric/: the first two trials of twenty real settings). No model is called and
no simulation is run: a stand-in executor answers each trial from the record. The names in these tests are read from the
record, not written here."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import quality_report, run_estimate as est, trial_runner
from tests.test_plan_optimisation import _engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "pilot_undefined_metric"


def _record() -> dict[str, Any]:
    return json.loads((FIXTURE / "pilot_trials.json").read_text(encoding="utf-8"))


def _nan_cells(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in record["cells"] if any(math.isnan(v) for t in c["trials"] for v in t.values())]


def _facts(record: dict[str, Any]) -> SimpleNamespace:
    """What the record shows, by name: the metrics that are never a number, the setting (name and value) at which they are
    returned, and the setting with the most values with its base value."""
    nan = _nan_cells(record)
    broken = sorted({m for c in nan for t in c["trials"] for m, v in t.items() if math.isnan(v)})
    big = max(record["grid"], key=lambda a: len(record["grid"][a]))
    shared = [(a, v) for a, v in nan[0]["cell"].items() if a != big and all(c["cell"][a] == v for c in nan)
              and not all(c["cell"][a] == v for c in record["cells"])]
    axis, value = shared[0]
    return SimpleNamespace(broken=broken, axis=axis, value=value, big=big, base=record["grid"][big][len(record["grid"][big]) // 2],
                           nan=nan, where=f"{axis} = {value}")


def _honest(record: dict[str, Any]) -> dict[str, Any]:
    """The same record where the undefined metrics have values at one of the settings that return them (as they would where
    something is produced) and stay undefined at the others."""
    out = copy.deepcopy(record)
    facts = _facts(record)
    target = _nan_cells(out)[0]
    for i, trial in enumerate(target["trials"]):
        for k, name in enumerate(facts.broken):
            trial[name] = 2.4 + 0.7 * k + 0.2 * i
    # the proportions of an honest record are sampled: one is a rare event that stays 0 in a small pilot, the others vary
    proportions = [m["id"] for m in record["metrics"] if m["kind"] == "proportion"]
    rare = next((m for m in proportions if any(m in t for c in out["cells"] for t in c["trials"])), None)
    for c in out["cells"]:
        for i, trial in enumerate(c["trials"]):
            for m in proportions:
                if m in trial:
                    trial[m] = 0.0 if m == rare else float(i % 2)
    return out


def _key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(sorted(cell.items()))


class Answers:
    """An executor that answers each trial from the record instead of running a simulation."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.by_cell = {_key(c["cell"]): c["trials"] for c in record["cells"]}
        self.calls = 0

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: Any = None) -> Any:
        self.calls += 1
        spec_path, out_path = Path(cwd) / cmd[2], Path(cwd) / cmd[3]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        trials = self.by_cell[_key(spec["cell"])]
        rows = [{"nonce": spec["nonce"], "trial": t["trial"], "seed": t.get("seed"), "status": "ok", "duration_s": 0.01,
                 "values": trials[t["trial"] % len(trials)], "warnings": []} for t in spec["trials"]]
        out_path.write_text("\n".join(json.dumps(r, allow_nan=True) for r in rows) + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, timed_out=False, stdout="", stderr="")

    def python_path(self, root: Path) -> str:
        return "python"


def _kinds(record: dict[str, Any]) -> dict[str, str]:
    return {m["id"]: m["kind"] for m in record["metrics"]}


def _observations(record: dict[str, Any], cells: list[dict[str, Any]] | None = None) -> list[tuple[dict[str, Any], dict[str, float]]]:
    wanted = {_key(c) for c in cells} if cells is not None else None
    return [(dict(item["cell"]), dict(t)) for item in record["cells"] if wanted is None or _key(item["cell"]) in wanted
            for t in item["trials"]]


def test_the_evidence_is_what_the_run_log_says() -> None:
    log = (FIXTURE / "run_log_excerpt.txt").read_text(encoding="utf-8")
    assert "0 with a value and 52000 with none" in log and "takes about 1 h 28 min" in log
    record = _record()
    facts = _facts(record)
    assert len(facts.broken) == 2 and facts.nan
    assert all(math.isnan(t[m]) for c in facts.nan for t in c["trials"] for m in facts.broken)
    others = [c for c in record["cells"] if c not in facts.nan]
    assert all(m not in t for c in others for t in c["trials"] for m in facts.broken), \
        "the other settings do not return these metrics at all"
    assert "the code does not compute the study's main quantity" in (FIXTURE / "STUCK.json").read_text(encoding="utf-8")


def test_the_pilot_reads_the_real_trials_and_names_the_metric_the_settings_and_the_values() -> None:
    record = _record()
    facts = _facts(record)
    pilot = est.pilot_problems(_observations(record), _kinds(record), tried_all="; FI also tried every value of every setting")
    found = {p.metric: p for p in pilot.problems}
    assert set(facts.broken) <= set(found) and all(found[m].kind == "undefined" for m in facts.broken)
    # this simulation also gave the same outcome in every trial for its proportions: found by the stuck-proportion rule
    assert {p.kind for p in pilot.problems} == {"undefined", "stuck"}
    first = facts.broken[0]
    text = found[first].says
    assert text.startswith(f"the metric `{first}` was not a number in any of the 8 pilot trials, at any of the 4 settings tried")
    assert facts.where in text and "The values that came back: nan" in text
    assert "may be undefined at the settings it does not apply to" in text
    assert pilot.trials == 40 and pilot.settings == 20 and not pilot.unsure
    message = est.directive(pilot)
    assert message.startswith("FI ran 40 trials of the simulation at 20 settings") and f"`{facts.broken[1]}`" in message
    assert "expected" not in message.lower(), "the repair is never given an expected value"


def test_a_metric_that_has_a_value_somewhere_is_not_a_finding_though_it_is_undefined_elsewhere() -> None:
    record = _honest(_record())
    facts = _facts(_record())
    pilot = est.pilot_problems(_observations(record), _kinds(record))
    assert pilot.problems == [], [p.says for p in pilot.problems]
    assert facts.broken[0] in pilot.checked
    name = facts.broken[0]
    values = [t[name] for c in record["cells"] for t in c["trials"] if name in t]
    assert any(math.isnan(v) for v in values) and any(not math.isnan(v) for v in values)


def test_nothing_is_claimed_about_a_subset_the_pilot_could_not_all_try() -> None:
    record = _record()
    facts = _facts(record)
    ran = [c["cell"] for c in record["cells"] if c["cell"][facts.big] == facts.base][:2]
    pilot = est.pilot_problems(_observations(record, ran), _kinds(record), unsure=set(facts.broken))
    assert pilot.problems == [] and pilot.unsure == facts.broken


def test_the_other_findings_are_a_constant_mean_and_an_output_that_never_changes() -> None:
    cells = [{"a": a} for a in (1, 2, 3)]
    moving = [({**c}, {"x": float(c["a"]) + 0.1 * t, "m": 5.0}) for c in cells for t in range(2)]
    pilot = est.pilot_problems(moving, {"x": "mean", "m": "mean"})
    assert [(p.metric, p.kind) for p in pilot.problems] == [("m", "constant")]
    assert "exactly 5" in pilot.problems[0].says and "never responds" in pilot.problems[0].says
    # a proportion that is 0 everywhere in a small pilot is a rare event, not a fault; nor is a metric the plan expects constant
    rare = [({**c}, {"x": float(c["a"]) + 0.1 * t, "p": 0.0}) for c in cells for t in range(2)]
    assert est.pilot_problems(rare, {"x": "mean", "p": "proportion"}).problems == []
    assert est.pilot_problems(moving, {"x": "mean", "m": "mean"}, exempt={"m"}).problems == []
    # every number the same at every setting: the existing flat check
    flat = [({**c}, {"x": 1.0, "m": 5.0}) for c in cells for _ in range(2)]
    assert [p.kind for p in est.pilot_problems(flat, {"x": "mean"}).problems] == ["flat"]


@pytest.mark.asyncio
async def test_the_pilot_tries_the_small_settings_in_every_combination_before_it_says_a_metric_never_had_a_value(
    tmp_path: Path,
) -> None:
    record = _record()
    facts = _facts(record)
    ex = Answers(record)
    quest = tmp_path / "quest"
    quest.mkdir()
    grid = record["grid"]
    probes, why = await est.measure(ex, "python", quest, quest / "simulate.py", grid, runs=1000, deterministic=False,
                                    limit_s=8 * 3600, budget_s=600)
    assert probes is not None and why == ""
    assert len(probes) == 6 and all(len(p.trials) == 2 for p in probes), "the base and the first and last of each setting"
    assert [c for c, _v in est.observations(probes) if c[facts.axis] == facts.value] != []
    # the six settings alone cannot say whether a metric applies only to some combinations of the small settings
    todo, whole = est.confirm_cells(grid, [p.cell for p in probes])
    assert whole and len(todo) == 14, "the 4 other combinations of the small settings and the 10 other values of the largest"
    assert len([c for c in todo if c[facts.big] != facts.base]) == 10
    result = await est.pilot(ex, "python", quest, quest / "simulate.py", grid, probes, _kinds(record), deterministic=False,
                             budget_s=600, allowance_s=60)
    assert set(facts.broken) <= {p.metric for p in result.problems} and result.settings == 20 and result.trials == 26
    assert ("FI also tried every value of every setting and every combination of the settings that take few values "
            "(20 settings in all)") in next(p for p in result.problems if p.kind == "undefined").says
    assert not (quest / "raw").exists(), "the pilot trials are not results"


@pytest.mark.asyncio
async def test_a_subset_that_cannot_all_be_tried_is_never_a_stop(tmp_path: Path) -> None:
    record = _record()
    facts = _facts(record)
    ex = Answers(record)
    quest = tmp_path / "quest"
    quest.mkdir()
    probes, _why = await est.measure(ex, "python", quest, quest / "simulate.py", record["grid"], runs=1000,
                                     deterministic=False, limit_s=8 * 3600, budget_s=600)
    result = await est.pilot(ex, "python", quest, quest / "simulate.py", record["grid"], probes, _kinds(record),
                             deterministic=False, budget_s=1, allowance_s=60)  # no time to try the combinations
    assert result.problems == [] and set(result.unsure) == set(facts.broken)


def _engine_and_runner(tmp_path: Path, record: dict[str, Any]) -> tuple[Any, Any, Answers]:
    eng = _engine(tmp_path, [])
    eng.config.execution.timeout_s = 8 * 3600  # the quest's own limit
    protocol = {"grid": record["grid"], "runs_per_setting": 1000, "metrics": record["metrics"]}
    eng._protocol_block = lambda state: protocol  # type: ignore[method-assign]
    code = eng.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text("def run_trial(cell, trial, seed):\n    return {'y': 1.0}\n", encoding="utf-8")
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    ex = Answers(record)
    eng.executor = ex  # type: ignore[assignment]
    runner = trial_runner.TrialsRunner(
        ex, quest_root=eng.quest_root, protocol=lambda: protocol, deterministic=False,  # type: ignore[arg-type]
        simulate=code / "simulate.py", analysis=code / "experiment.py", log=eng._log)
    return eng, runner, ex


@pytest.mark.asyncio
async def test_the_pilot_sends_the_simulation_back_with_the_exact_finding_and_then_ends_in_the_honest_stop(
    tmp_path: Path,
) -> None:
    record = _record()
    facts = _facts(record)
    first, second = facts.broken
    eng, runner, ex = _engine_and_runner(tmp_path, record)
    state: dict[str, Any] = {"iteration": 0, "exec_reflect_iter": 0}
    patch = await eng._size_the_run(state, runner, "python", None)
    assert patch is not None
    result = patch["exec_result"]
    assert result["failed_script"] == "simulate.py" and result["directive"] is True
    said = result["stderr_tail"]
    assert f"the metric `{first}` was not a number in any of the 5 pilot trials, at any of the 4 settings tried" in said
    assert facts.where in said and "The values that came back: nan" in said and f"`{second}`" in said
    assert eng._route_after_execute_reflect({"exec_result": result, "exec_reflect_iter": 0}) == "retry"
    assert ex.calls <= 24, "found in a couple of dozen trials, not after the study"
    pilot = est.read_pilot_record(eng.quest_root)
    assert pilot["outcome"] == "repairing" and pilot["repairs"] == 1 and pilot["problems"]
    assert not (eng.quest_root / "raw").exists() and est.read_record(eng.quest_root).get("fits") is None
    # the repairs are spent: the honest stop, with the same finding, in the same few trials
    state["exec_reflect_iter"] = eng.config.engine.exec_reflect_max_iterations
    stop = await eng._size_the_run(state, runner, "python", None)
    assert stop is not None and stop["exec_result"]["cannot_repair"].startswith("a small trial run before the study")
    assert f"`{first}`" in stop["exec_result"]["cannot_repair"] and ex.calls <= 48
    verdict = eng._no_results_verdict({**state, "exec_result": stop["exec_result"], "result_json": {}})
    assert verdict is not None and verdict["stuck_reason"] == "cannot_repair"
    final = est.read_pilot_record(eng.quest_root)
    assert final["outcome"] == "stopped" and final["repairs"] == 1
    # QUALITY.md says what the pilot found and how many repairs it caused
    report = quality_report.build(eng.quest_root, running=False)
    assert report["pilot"]["outcome"] == "stopped" and report["pilot"]["repairs"] == 1
    shown = quality_report.render(report)
    run_items = [p for s in report["stages"] if s["stage"] == "run" for p in s["problems"]["items"]]
    assert any(p["found_by"] == "FI's pilot check" and p["status"] == "not fixed" for p in run_items)
    assert "Pilot check:" in shown and "Repairs it caused: 1." in shown


@pytest.mark.asyncio
async def test_an_honest_record_with_some_undefined_values_goes_on_and_is_recorded(tmp_path: Path) -> None:
    record = _honest(_record())
    eng, runner, ex = _engine_and_runner(tmp_path, record)
    patch = await eng._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner, "python", None)
    assert patch is None, "no false stop"
    pilot = est.read_pilot_record(eng.quest_root)
    assert pilot["outcome"] == "passed" and pilot["repairs"] == 0 and pilot["problems"] == []
    report = quality_report.build(eng.quest_root, running=False)
    assert report["pilot"]["says"].endswith("nothing was wrong") and report["pilot"]["repairs"] == 0
    # a resume with the same simulation does not read it again
    before = ex.calls
    assert await eng._size_the_run({"iteration": 0, "exec_reflect_iter": 0}, runner, "python", None) is None
    assert ex.calls == before


@pytest.mark.asyncio
async def test_the_confirmation_is_made_once_per_version_of_the_code_and_counts_against_the_sizing_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _honest(_record())
    eng, runner, ex = _engine_and_runner(tmp_path, record)
    seen: list[float] = []
    real = est.pilot

    async def watching(*a: Any, **kw: Any) -> Any:
        seen.append(float(kw["budget_s"]))
        return await real(*a, **kw)

    monkeypatch.setattr(est, "pilot", watching)
    state = {"iteration": 0, "exec_reflect_iter": 0}
    assert await eng._size_the_run(state, runner, "python", None) is None
    assert len(seen) == 1 and eng._pilot_spent > 0
    # the run is made smaller (the same code): it is timed again, but the correctness reading is reused, not made again
    protocol = eng._protocol_block(state)
    protocol["runs_per_setting"] = 500
    before = ex.calls
    assert await eng._size_the_run(state, runner, "python", None) is None
    assert len(seen) == 1, "the confirmation is not run again for the same code"
    assert ex.calls - before <= 6, "only the timing trials ran"
    # a changed version of the code is read again, with what is left of the sizing's budget (the time already spent is deducted)
    code = eng.quest_root / "code" / "simulate.py"
    code.write_text(code.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    grid, runs, thresholds = eng._grid_and_runs(state)
    probes, _why = await est.measure(ex, "python", eng.quest_root, runner.simulate, grid, runs=runs, deterministic=False,
                                     limit_s=8 * 3600, budget_s=600)
    eng._pilot_spent = 100.0
    assert await eng._pilot_gate(state, runner, "python", None, grid, probes, thresholds, 8 * 3600, 600.0) is None
    assert len(seen) == 2 and seen[1] == 300.0 - 100.0, seen
    assert est.read_pilot_record(eng.quest_root)["code_key"].split("|")[0]


@pytest.mark.asyncio
async def test_a_finding_for_unchanged_code_is_reused_without_new_trials(tmp_path: Path) -> None:
    eng, runner, ex = _engine_and_runner(tmp_path, _record())
    state: dict[str, Any] = {"iteration": 0, "exec_reflect_iter": 0}
    first = await eng._size_the_run(state, runner, "python", None)
    assert first is not None
    before = ex.calls
    second = await eng._size_the_run(state, runner, "python", None)  # the repair changed nothing
    assert second is not None and est.read_pilot_record(eng.quest_root)["repairs"] == 2
    assert ex.calls - before <= 6, "the extra trials are not run again for the same code"
