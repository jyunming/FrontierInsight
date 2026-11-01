"""A figure of the study's that failed after its repairs belongs to the analysis, not to the run: FI decides once per run
whether its results were produced (``exec_result["results_produced"]``: exit code 0, or exit code 1 with only a figure failed in
FI's own record), and every reader that means "the run produced its results" uses that fact. A run that crashed without
results is still no result. No real model is called."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core import analysis_frame as frame
from core import evidence, phased
from tests.test_confirm_one_shot import _TRIAL, _code, _env
from tests.test_phased import _config


def _record(root: Path, *, figure: str = "failed", result: str = "ok", schema: str = frame.DERIVED_SCHEMA, **more) -> None:
    doc = {"schema": schema, "derived": {"r1": {"role": "result", "by": "derive_r1", "status": result, "why": "x"}},
           "figures": [{"id": "fig2", "status": figure, "file": "fig2.png"}, {"id": "fig1", "status": "ok", "file": "fig1.png"}],
           "fi_faults": [], **more}
    path = root / frame.DERIVED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _engine(tmp_path: Path, *, phased_on: bool):
    from core.engine import Engine

    cfg = _config(tmp_path / "out", phased_on=phased_on)
    return Engine(cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"improve_rounds": 2})}))


def _confirm(engine, exec_result: dict, result_json: dict) -> dict:
    root = engine.quest_root
    _code(root, _TRIAL)
    phased.prepare(root, engine.quest_id)
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0}}
    phased.seed_env(root, _env(0, 0))
    assert engine._route_after_evidence_gate(gate) == "confirm"
    phased.seed_env(root, _env(0, 0))
    engine._route_after_evidence_gate({**gate, "result_json": result_json, "exec_result": exec_result})
    return phased.load(root)


class _Ran:
    def __init__(self, returncode: int, printed: bool = True) -> None:
        self.returncode = returncode
        self.stdout = 'RESULT_JSON: {"a": 1}\n' if printed else ""
        self.stderr = ""


def _fis_experiment(engine, on: bool = True) -> None:
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "experiment.py").write_text(frame.EXPERIMENT_HEAD + "\n" if on else "print('hand written')\n", encoding="utf-8")


# ---- confirm and improve ----------------------------------------------------------------------------------------------


def test_a_failed_figure_does_not_wipe_out_the_confirmation_of_the_results(tmp_path: Path) -> None:
    record = _confirm(_engine(tmp_path, phased_on=True), {"returncode": 1, "results_produced": True}, {"a": 2})
    assert phased.status(record) == phased.CONFIRMED, phased.status(record)


def test_a_run_without_results_confirms_nothing(tmp_path: Path) -> None:
    record = _confirm(_engine(tmp_path, phased_on=True), {"returncode": 1, "results_produced": False}, {"a": 2})
    assert phased.status(record) != phased.CONFIRMED
    # an exec_result with no such fact: the return code decides
    record = _confirm(_engine(tmp_path / "again", phased_on=True), {"returncode": 1}, {"a": 2})
    assert phased.status(record) != phased.CONFIRMED


def test_a_failed_figure_does_not_skip_the_improve_step_but_a_crash_does(tmp_path: Path) -> None:
    engine = _engine(tmp_path, phased_on=False)
    state = {"result_json": {"a": 2}, "exec_result": {"returncode": 1, "results_produced": True},
             "exec_give_up_reason": "the figure could not be drawn"}
    assert not engine._run_has_no_result(state)
    assert engine._improve_skip(state) != "the run did not produce a result"
    assert engine._run_has_no_result({**state, "exec_result": {"returncode": 1, "results_produced": False}})
    assert engine._run_has_no_result({**state, "exec_result": {"returncode": 1}})
    assert engine._run_has_no_result({**state, "result_json": {}})
    # a clean run that gave up (not over a figure) is still no result
    assert engine._run_has_no_result({**state, "exec_result": {"returncode": 0, "results_produced": True}})


def test_only_figures_failed_reads_fis_record(tmp_path: Path) -> None:
    _record(tmp_path, figure="refused")
    rec = frame.read_record(tmp_path)
    assert frame.only_figures_failed(rec, 1)
    assert not frame.only_figures_failed(rec, 0) and not frame.only_figures_failed(rec, 2)
    assert not frame.only_figures_failed({**rec, "fi_faults": ["FI's own"]}, 1)
    assert not frame.only_figures_failed({**rec, "from_earlier_run": True}, 1)
    _record(tmp_path, figure="ok")
    assert not frame.only_figures_failed(frame.read_record(tmp_path), 1)


def test_run_produced_is_one_rule_for_the_run_and_the_clean_rerun(tmp_path: Path) -> None:
    _record(tmp_path, figure="failed")
    rec = frame.read_record(tmp_path)
    assert frame.run_produced(0, False, {}, False)
    assert frame.run_produced(1, True, rec, True)
    assert not frame.run_produced(1, False, rec, True) and not frame.run_produced(1, True, rec, False)
    assert not frame.run_produced(2, True, rec, True) and not frame.run_produced(1, True, {}, True)


# ---- the fact itself --------------------------------------------------------------------------------------------------


def test_run_produced_is_decided_from_fis_record_and_only_for_fis_own_analysis(tmp_path: Path) -> None:
    engine = _engine(tmp_path, phased_on=False)
    _record(engine.quest_root, figure="failed")
    _fis_experiment(engine)
    assert engine._run_produced(_Ran(0))
    assert engine._run_produced(_Ran(1))  # only a figure failed
    assert not engine._run_produced(_Ran(2))  # FI's exit code for a crash or its own fault, although the record was written
    assert not engine._run_produced(_Ran(1, printed=False))  # no RESULT line
    _record(engine.quest_root, figure="ok")
    assert not engine._run_produced(_Ran(1)), "exit 1 with a printed result and no failed figure in the record is a crash"
    _record(engine.quest_root, figure="failed", result="failed")
    assert not engine._run_produced(_Ran(1))
    _record(engine.quest_root, figure="failed")
    _fis_experiment(engine, on=False)  # a hand-written experiment.py is never figures-only
    assert not engine._run_produced(_Ran(1))


def test_evidence_executed_reads_the_fact(tmp_path: Path) -> None:
    def executed(exec_result: dict) -> bool:
        rec = evidence.assess(tmp_path, {"result_json": {"a": 1}, "exec_result": exec_result})
        return not any(lvl["gaps"] for lvl in rec["ladder"] if lvl["level"] == "executed")

    assert executed({"returncode": 1, "results_produced": True})
    assert not executed({"returncode": 1, "results_produced": False})
    assert not executed({"returncode": 1})
    assert executed({"returncode": 0})


def test_the_manifest_and_the_results_check_treat_a_figure_only_failure_as_a_run(tmp_path: Path, monkeypatch) -> None:
    engine = _engine(tmp_path, phased_on=False)
    _record(engine.quest_root, figure="failed")
    _fis_experiment(engine)
    monkeypatch.setattr(engine, "_protocol_block", lambda state: {"x": 1})
    monkeypatch.setattr("core.run_manifest.checkable", lambda protocol: True)
    state = {"design": {}}
    assert engine._run_manifest_problems(state, True, _Ran(2), None)[0] == "not_run"
    try:
        status = engine._run_manifest_problems(state, True, _Ran(1), None)[0]
    except Exception:  # noqa: BLE001 - it went past the "did not run" line into the comparison, which a bare fixture cannot make
        status = "compared"
    assert status != "not_run"
    seen: list[str] = []
    monkeypatch.setattr("core.engine._extract_result_json", lambda text: seen.append(text) or {})
    asyncio.run(engine._results_match_run(state, _Ran(1), None, True))
    assert seen, "the results check read the results of the figure-only run"


def test_a_figure_only_failure_gets_every_check_of_the_run(tmp_path: Path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_split_analysis import SIMULATE, _engine as split_engine, _write

    analysis = (
        "import json, os, pathlib, sys\n"
        "raw = pathlib.Path(os.environ['FI_RAW_DIR'])\n"
        "data = json.loads((raw / 'values.json').read_text())\n"
        "doc = {'schema': 'fi.derived/v1', 'derived': {}, 'fi_faults': [],\n"
        "       'figures': [{'id': 'fig2', 'status': 'failed', 'file': 'f.png'}]}\n"
        "os.makedirs('data/results', exist_ok=True)\n"
        "json.dump(doc, open('data/results/derived.json', 'w'))\n"
        "print('RESULT_JSON: ' + json.dumps({'mean': sum(data['values'])}))\n"
        "sys.exit(CODE)\n")

    def run(sub: str, code: int, *, fis: bool = True):
        eng = split_engine(tmp_path / sub, replicates=2)
        _write(eng, simulate=SIMULATE, analysis=analysis.replace("CODE", str(code)))
        monkeypatch.setattr(Engine, "_analysis_on", lambda self: fis)
        # the repairs are spent, so this run ends the loop (while they are not, the run is provisional)
        return eng, asyncio.run(eng._node_execute({"deps": [], "exec_reflect_iter": eng.config.engine.exec_reflect_max_iterations}))

    eng, out = run("fig", 1)
    er = out["exec_result"]
    assert er["returncode"] == 1 and er["results_produced"] is True
    raw = eng.quest_root / "data" / "results" / "raw"
    assert (raw / "seed0" / "values.json").is_file() and (raw / "seed1" / "values.json").is_file(), \
        "the run was accepted (its data is kept) and its replicates ran"
    assert out["result_json"] == {"mean": 6.0}
    # the same output with exit code 2 (a crash after derived.json was written), and from a hand-written analysis
    eng2, out2 = run("crash", 2)
    assert out2["exec_result"]["results_produced"] is False
    assert not (eng2.quest_root / "data" / "results" / "raw").exists()
    eng3, out3 = run("hand", 1, fis=False)
    assert out3["exec_result"]["results_produced"] is False
    assert not (eng3.quest_root / "data" / "results" / "raw").exists()


# ---- while the figure's repair is coming, the work of a final run waits; the stored fact is right; no false trial stop ----


def _figure_run(tmp_path: Path, monkeypatch, sub: str, code: int, state: dict, *, trial_fault: bool = False):
    from core.engine import Engine
    from core.split_run import SplitRunner
    from tests.test_split_analysis import SIMULATE, _engine as split_engine, _write

    analysis = (
        "import json, os, pathlib, sys\n"
        "raw = pathlib.Path(os.environ['FI_RAW_DIR'])\n"
        "data = json.loads((raw / 'values.json').read_text())\n"
        "doc = {'schema': 'fi.derived/v1', 'derived': {}, 'fi_faults': [],\n"
        "       'figures': [{'id': 'fig2', 'status': 'failed', 'file': 'f.png'}]}\n"
        "os.makedirs('data/results', exist_ok=True)\n"
        "json.dump(doc, open('data/results/derived.json', 'w'))\n"
        "print('RESULT_JSON: ' + json.dumps({'mean': sum(data['values'])}))\n"
        "sys.exit(CODE)\n")
    eng = split_engine(tmp_path / sub, replicates=2)
    _write(eng, simulate=SIMULATE, analysis=analysis.replace("CODE", str(code)))
    monkeypatch.setattr(Engine, "_analysis_on", lambda self: True)
    if trial_fault:
        monkeypatch.setattr(SplitRunner, "trial_fault", {"action": "stop", "say": "settings were left out"}, raising=False)
    return eng, asyncio.run(eng._node_execute({"deps": [], **state}))


def test_the_replicates_and_run_checks_wait_for_the_run_that_ends_the_repairs(tmp_path: Path, monkeypatch) -> None:
    from core.engine import Engine

    # repairs left: the figure's repair replaces this run, so no replicate, no saved run data (hidden, refinement and dependence
    # checks are gated with them), and the router agrees (it retries)
    eng, out = _figure_run(tmp_path, monkeypatch, "left", 1, {"exec_reflect_iter": 0})
    assert out["exec_result"]["results_produced"] is True and out["result_json"] == {"mean": 6.0}
    assert eng._repair_budget_left({"exec_reflect_iter": 0})
    assert not (eng.quest_root / "data" / "results" / "raw").exists()
    assert not (eng.quest_root / "data" / "results" / "raw" / "seed1").exists()
    simulated = (eng.quest_root / "simulated.log").read_text(encoding="utf-8").split()
    assert simulated == ["0"], f"the replicates ran: {simulated}"
    state = {**out, "exec_reflect_iter": 0, "design": {}}
    assert eng._route_after_execute_reflect(state) == "retry"
    # the repairs spent: this run ends the loop, so it gets all of that work
    spent = eng.config.engine.exec_reflect_max_iterations
    eng2, out2 = _figure_run(tmp_path, monkeypatch, "spent", 1, {"exec_reflect_iter": spent})
    assert not eng2._repair_budget_left({"exec_reflect_iter": spent})
    raw = eng2.quest_root / "data" / "results" / "raw"
    assert (raw / "seed0" / "values.json").is_file() and (raw / "seed1" / "values.json").is_file()
    assert eng2._route_after_execute_reflect({**out2, "exec_reflect_iter": spent, "design": {}}) == "proceed"
    assert Engine._repair_budget_left  # one method, asked by the router and by the node


def test_a_figure_only_run_is_not_stopped_by_the_trial_checks_and_a_crash_still_is(tmp_path: Path, monkeypatch) -> None:
    _eng, out = _figure_run(tmp_path, monkeypatch, "fig", 1, {"exec_reflect_iter": 99}, trial_fault=True)
    assert out["exec_result"]["results_produced"] is True
    assert out["exec_result"]["trial_checks"] == {}
    _eng2, crashed = _figure_run(tmp_path, monkeypatch, "crash", 2, {"exec_reflect_iter": 99}, trial_fault=True)
    assert crashed["exec_result"]["results_produced"] is False
    assert crashed["exec_result"]["trial_checks"].get("say")


def test_the_stored_fact_is_false_for_a_result_a_check_replaced(tmp_path: Path, monkeypatch) -> None:
    from core.engine import Engine

    async def sent_back(self, state, result, failed_script, split):  # noqa: ANN001
        from core.execution import ExecutionResult

        return ExecutionResult(1, "", "[FI] sent back", result.duration_s, False), "experiment.py"

    monkeypatch.setattr(Engine, "_results_match_run", sent_back)
    _eng, out = _figure_run(tmp_path, monkeypatch, "replaced", 1, {"exec_reflect_iter": 99})
    assert out["exec_result"]["results_produced"] is False


# ---- a figure body no repair is left for is drawn from the plan's own description instead of being lost ----


def _analysis_run(quest: Path, *, last_run: bool) -> tuple[int, dict, dict]:
    """FI's experiment.py in a fresh Python, with or without the flag the engine sets on the run that ends the repairs."""
    import os
    import subprocess
    import sys

    from tests.test_analysis_frame import NO_WINDOW

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FI_ROOT", "FI_TRIALS", "FI_FIGURE_FALLBACK")}
    if last_run:
        env["FI_FIGURE_FALLBACK"] = "1"
    done = subprocess.run([sys.executable, "experiment.py"], cwd=quest / "code", env=env, capture_output=True, text=True,
                          timeout=300, creationflags=NO_WINDOW)
    printed = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT_JSON: ")]
    derived = json.loads((quest / frame.DERIVED_PATH).read_text(encoding="utf-8"))
    return done.returncode, json.loads(printed[-1][len("RESULT_JSON: "):]) if printed else {}, derived


_RAISES = "def figure_fig1(fig, table):\n    raise ValueError('the body reads an id the plan does not declare')\n"
_DRAWS = ("def figure_fig1(fig, table):\n    fig.axes[0].plot([1, 2], [table.mean('growth', k) for k in table.keys()][:2])\n")


def test_failed_body_gets_default_figure_after_repair_budget(tmp_path: Path) -> None:
    from tests.test_analysis_frame import _design, _quest

    quest = _quest(tmp_path, _design(), _RAISES)
    rc, printed, derived = _analysis_run(quest, last_run=True)
    fig = next(o for o in derived["figures"] if o["id"] == "fig1")
    assert rc == 0, derived.get("fi_faults")
    assert fig["status"] == "ok" and fig["fallback"] is True
    assert (quest / "figures" / "growth.png").stat().st_size > 0
    # the results are the same ones a run whose body drew its figure printed
    good = _quest(tmp_path / "good", _design(), _DRAWS)
    assert _analysis_run(good, last_run=True)[1] == printed


def test_a_failed_body_is_still_a_failure_while_repairs_are_left(tmp_path: Path) -> None:
    from tests.test_analysis_frame import _design, _quest

    quest = _quest(tmp_path, _design(), _RAISES)
    rc, _printed, derived = _analysis_run(quest, last_run=False)
    fig = next(o for o in derived["figures"] if o["id"] == "fig1")
    assert rc == 1 and fig["status"] == "failed" and "fallback" not in fig
    assert not (quest / "figures" / "growth.png").exists()


def test_ok_body_is_not_replaced(tmp_path: Path) -> None:
    from tests.test_analysis_frame import _design, _quest

    quest = _quest(tmp_path, _design(), _DRAWS)
    rc, _printed, derived = _analysis_run(quest, last_run=True)
    fig = next(o for o in derived["figures"] if o["id"] == "fig1")
    assert rc == 0 and fig["status"] == "ok" and "fallback" not in fig


def test_undeclared_shows_figure_is_dropped_with_quality_row(tmp_path: Path) -> None:
    from tests.test_analysis_frame import _design, _quest

    design = _design()
    design["protocol"]["figures"] = [{"id": "fig1", "file": "growth.png", "shows": ["growth"]}]  # no x: nothing to draw in its place
    quest = _quest(tmp_path, design, _RAISES)
    rc, _printed, derived = _analysis_run(quest, last_run=True)
    fig = next(o for o in derived["figures"] if o["id"] == "fig1")
    assert rc == 0 and fig["status"] == "dropped" and fig["why"] and not (quest / "figures" / "growth.png").exists()
    # the engine says it in QUALITY.md's record (a row that stops nothing) and marks a drawn one in its record
    eng = _engine(tmp_path / "eng", phased_on=False)
    (eng.quest_root / frame.DERIVED_PATH).parent.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / frame.DERIVED_PATH).write_text(json.dumps(derived), encoding="utf-8")
    records: dict = {}
    eng._figure_fallbacks(records)
    rows = eng._cannot_repair_problems()
    assert len(rows) == 1 and "fig1" in rows[0]["text"] and "could not be drawn from the run's results" in rows[0]["text"]
    from core import stop_power

    assert not stop_power.may_stop(rows[0]["source"])
    drawn = {"schema": derived["schema"], "figures": [{"id": "fig1", "status": "ok", "file": "growth.png", "fallback": True}]}
    (eng.quest_root / frame.DERIVED_PATH).write_text(json.dumps(drawn), encoding="utf-8")
    records = {"growth.png": {"axes": []}, "other.png": {"axes": []}}
    eng._figure_fallbacks(records)
    assert records["growth.png"]["fallback"] is True and "fallback" not in records["other.png"]
    assert eng._cannot_repair_problems() == []  # the run drew it: the earlier run's row is gone


def test_the_fallback_flag_is_set_only_on_the_run_that_ends_the_repairs(tmp_path: Path, monkeypatch) -> None:
    from core.engine import Engine
    from tests.test_split_analysis import SIMULATE, _engine as split_engine, _write

    analysis = ("import json, os\nprint('RESULT_JSON: ' + json.dumps({'flag': os.environ.get('FI_FIGURE_FALLBACK', 'unset')}))\n")
    monkeypatch.setattr(Engine, "_analysis_on", lambda self: True)
    flags = {}
    for sub, used, more in (("left", 0, {}), ("spent", 99, {}), ("gave_up", 0, {"exec_give_up_reason": "the model gave up"})):
        eng = split_engine(tmp_path / sub, replicates=2)
        _write(eng, simulate=SIMULATE, analysis=analysis)
        flags[sub] = asyncio.run(eng._node_execute({"deps": [], "exec_reflect_iter": used, **more}))["result_json"]["flag"]
    assert flags == {"left": "unset", "spent": "1", "gave_up": "1"}


# ---- the repairs end by a give-up: the figure that failed is still drawn, from the results already on disk ----


def _give_up_quest(tmp_path: Path, monkeypatch, *, random_body: bool = False):
    """An engine whose quest holds FI's analysis, a record of the trials and one run (no flag) in which fig1 raised and fig2 drew."""
    import subprocess
    import sys
    import types

    from tests.test_analysis_frame import NO_WINDOW, RECORD, _design
    from tests.test_analysis_frame import _derived  # noqa: F401

    design = _design([_derived("peak")] if random_body else None)
    design["protocol"]["figures"].append({"id": "fig2", "file": "other.png", "shows": ["growth"], "x": "rate"})
    eng = _engine(tmp_path, phased_on=False)
    root = eng.quest_root
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "trials.json").write_text(json.dumps(RECORD), encoding="utf-8")
    frame.ensure_files(root / "code", design)
    with (root / "code" / "derive.py").open("a", encoding="utf-8") as f:
        if random_body:
            f.write("\n\ndef derive_peak(table):\n    import random\n    return random.random()\n")
        f.write("\n\n" + _RAISES + "\n\ndef figure_fig2(fig, table):\n    fig.axes[0].plot([1, 2], [table.mean('growth', k) for k in table.keys()][:2])\n")
    calls: list[dict] = []
    outputs: list[str] = []

    async def execute(cmd, cwd=None, timeout_s=None, env=None, **_kw):
        calls.append(dict(env or {}))
        outputs.append("")
        done = subprocess.run(cmd, cwd=root / "code", env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"} | (env or {}),
                              capture_output=True, text=True, timeout=300, creationflags=NO_WINDOW)
        outputs[-1] = done.stdout
        return types.SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)

    monkeypatch.setattr(eng.executor, "execute", execute)
    monkeypatch.setattr(eng.executor, "python_path", lambda _root: sys.executable)
    first = subprocess.run([sys.executable, "experiment.py"], cwd=root / "code", capture_output=True, text=True, timeout=300,
                           creationflags=NO_WINDOW)
    assert first.returncode == 1
    eng.first_stdout, eng.only_outputs = first.stdout, outputs
    return eng, root, calls


import os  # noqa: E402


def test_a_give_up_still_draws_the_failed_figure_from_the_results_on_disk_and_leaves_the_drawn_one(tmp_path: Path, monkeypatch) -> None:
    eng, root, calls = _give_up_quest(tmp_path, monkeypatch)
    other = (root / "figures" / "other.png").read_bytes()
    state = {"exec_result": {"returncode": 1, "results_produced": True},
             "figure_records": {"other.png": {"axes": []}, "growth.png": {"axes": []}}}
    out = asyncio.run(eng._figures_after_give_up(state))
    assert calls and calls[0]["FI_FIGURE_FALLBACK"] == "only"
    assert (root / "figures" / "growth.png").stat().st_size > 0
    assert (root / "figures" / "other.png").read_bytes() == other  # the figure that drew fine is not touched
    assert out["figures"] == ["growth.png", "other.png"]
    assert out["figure_records"]["growth.png"]["fallback"] is True and "fallback" not in out["figure_records"]["other.png"]
    assert eng._cannot_repair_problems() == []  # a drawn fallback is said in run.log only
    record = frame.read_record(root)
    assert {o["id"]: o["status"] for o in record["figures"]} == {"fig1": "ok", "fig2": "ok"}
    assert "fallback" not in next(o for o in record["figures"] if o["id"] == "fig2")


def test_a_give_up_runs_nothing_when_no_figure_failed_or_the_run_has_no_results(tmp_path: Path, monkeypatch) -> None:
    eng, root, calls = _give_up_quest(tmp_path, monkeypatch)
    assert asyncio.run(eng._figures_after_give_up({"exec_result": {"returncode": 1, "results_produced": False}})) == {}
    assert not calls and not (root / "figures" / "growth.png").exists()
    asyncio.run(eng._figures_after_give_up({"exec_result": {"returncode": 1, "results_produced": True}}))
    again = asyncio.run(eng._figures_after_give_up({"exec_result": {"returncode": 0, "results_produced": True}}))
    assert again == {} and len(calls) == 1  # nothing failed any more: no second run


def test_the_run_that_draws_the_failed_figures_leaves_every_derived_value_and_result_as_checked(tmp_path: Path, monkeypatch) -> None:
    eng, root, _calls = _give_up_quest(tmp_path, monkeypatch, random_body=True)
    before = frame.read_record(root)
    assert before["derived"]["peak"]["status"] == "ok"
    out = asyncio.run(eng._figures_after_give_up({"exec_result": {"returncode": 1, "results_produced": True},
                                                  "result_json": {"peak": 1}, "figure_records": {}}))
    after = frame.read_record(root)
    assert (root / "figures" / "growth.png").stat().st_size > 0 and "fallback" in str(after["figures"])
    # the study's function (a random number) was not called again: the record of every derived value is byte for byte the same
    assert json.dumps(after["derived"], sort_keys=True) == json.dumps(before["derived"], sort_keys=True)
    assert after["not_computed"] == before["not_computed"] and after["bodies"] == before["bodies"]
    assert "RESULT_JSON" not in eng.only_outputs[0] and "RESULT_JSON" in eng.first_stdout
    assert set(out) == {"figures", "figure_records"}  # the state's result_json is not in what comes back


def test_a_result_check_give_up_still_draws_the_failed_figure(tmp_path: Path, monkeypatch) -> None:
    eng, root, _calls = _give_up_quest(tmp_path, monkeypatch)
    state = {"exec_result": {"returncode": 1, "results_produced": False, "result_check_sent_back": True},
             "figure_records": {"growth.png": {"axes": []}, "other.png": {"axes": []}}}
    patch = asyncio.run(eng._result_check_gave_up_with_figures(state))
    assert patch["exec_patch_pending"] is True  # the run is made once more, as before
    assert patch["figure_records"]["growth.png"]["fallback"] is True and "fallback" not in patch["figure_records"]["other.png"]
    assert (root / "figures" / "growth.png").stat().st_size > 0
    plain = {"exec_result": {"returncode": 1, "results_produced": True}}
    assert asyncio.run(eng._result_check_gave_up_with_figures(plain)) is None  # not a run the results check sent back
