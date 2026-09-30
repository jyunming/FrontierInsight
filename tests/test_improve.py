"""The improve loop and the ratchet (core/improve.py, ``Engine._node_improve``).

A toy simulation integrates y' = -y to t = 1 with Euler's method; its checks of correctness are FI's own measurements of
the error against exp(-1) (and, where a test needs a second one, of a bias the code must not add). A fake model answers
the improve prompt with one edit; FI runs the simulation itself (the harness, on this interpreter) and judges. No real
model is called.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import code_project, criteria as cr, frozen_protocol as fp, improve, plan
from core import oracle_check as _oracle
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine

HAS_GIT = shutil.which("git") is not None

SIM = '''import math

BIAS = 0.0


def step(y, dt):
    return y + dt * (-y)


def run_cell(cell):
    dt = cell["dt"]
    y = 1.0
    for _ in range(int(round(1 / dt))):
        y = step(y, dt)
    print("error estimate", 0.3)
    return {"error": abs(y - math.exp(-1)), "drift": abs(BIAS), "y1": y}
'''
EULER = "    return y + dt * (-y)\n"
RK4 = ("    k1 = -y\n    k2 = -(y + dt * k1 / 2)\n    k3 = -(y + dt * k2 / 2)\n    k4 = -(y + dt * k3)\n"
       "    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6\n")
HEUN = "    return y * (1 - dt + dt * dt / 2)\n"
RK3 = "    return y * (1 - dt + dt * dt / 2 - dt ** 3 / 6)\n"
SAME = "    return y - dt * y\n"  # a different expression, the same number
ANALYSIS = '''import json, os
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
out = {c["key"]: c["metrics"]["y1"]["values"] for c in data["cells"]}
print("RESULT_JSON: " + json.dumps({"headline_value": 987654.321, "by_cell": out}))
'''

DECAY = {"name": "decay error", "kind": "special_case", "check": "error at t=1 of y'=-y against exp(-1)",
         "expected": 0.0, "tolerance": 0.05, "case": {"dt": 0.1}, "measure": "error",
         "reference": "derivation: y' = -y with y(0) = 1 gives y(1) = exp(-1)"}
DRIFT = {"name": "no bias", "kind": "special_case", "check": "the model adds no bias", "expected": 0.0,
         "tolerance": 0.5, "case": {"dt": 0.1}, "measure": "drift", "reference": "derivation: the equation has no bias"}
C_ERR = {"name": "decay error small", "oracle": "decay error", "direction": "lower", "target": 1e-5, "tolerance": 1e-7}
C_DRIFT = {"name": "no bias added", "oracle": "no bias", "direction": "lower", "target": 1e-6, "tolerance": 1e-6}
HEADLINE = {"id": "decay_headline", "kind": "mean", "estimand": "the mean of y at t = 1", "unit": "run"}
STATE: dict[str, Any] = {"exec_result": {"returncode": 0}, "result_json": {"headline_value": 987654.321}}


def _edit(find: str, replace: str, why: str = "a better integrator", file: str = "simulate.py") -> str:
    return json.dumps({"file": file, "find": find, "replace": replace, "why": why})


def _config(tmp_path: Path, rounds: int = 3) -> Config:
    return Config(
        topic="Euler on y' = -y", title="improve", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=1,
                            pilot_run=False, improve_rounds=rounds),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan="off", papers=False, review="off"),
    )


class _Model:
    """Answers the improve prompt with the given replies, in order, and keeps every prompt it was shown."""

    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts = list(replies), []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        assert prompt.startswith("# Improve the Simulation"), prompt[:80]
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "{}"

    async def aclose(self) -> None:
        return None


async def _quest(tmp_path: Path, replies: list[str], *, criteria: list[dict[str, Any]] | None = None,
                 oracles: list[dict[str, Any]] | None = None, rounds: int = 3, research: bool = False,
                 sim: str = SIM, grid: dict[str, Any] | None = None, runs: int | None = None) -> tuple[Engine, _Model]:
    """A quest that has run once: the code recorded, the protocol frozen, the first row of criteria written."""
    cfg = _config(tmp_path, rounds)
    if research:
        cfg = cfg.model_copy(update={"rigor_profile": "research"})
    engine = Engine(cfg)
    root = engine.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_bytes(sim.encode("utf-8"))
    (root / "code" / "experiment.py").write_bytes(ANALYSIS.encode("utf-8"))
    protocol = plan.normalize_protocol({"grid": grid or {"dt": [0.1, 0.05]}, "oracles": oracles or [DECAY],
                                        "criteria": criteria or [C_ERR], "metrics": [HEADLINE],
                                        **({"runs_per_setting": runs} if runs else {})})[0]
    fp.freeze(root, protocol, approved_by="human: test", source="plan.md")
    if HAS_GIT:
        assert code_project.record_change(root, "code written")
    engine._trial_mode = True
    trial_names = {c["trials"] for c in protocol["criteria"] if c.get("trials")}
    oracles_used = {c["oracle"] for c in protocol["criteria"] if c.get("oracle")}
    rows, problems = await engine._improve_measure(
        protocol, [o for o in _oracle.declared(protocol) if o["name"] in oracles_used], trial_names)
    assert not problems, problems
    cr.record(root, run="run_1", code_commit=None, results=rows)
    model = _Model(replies)
    engine._client = model
    return engine, model


def _log(engine: Engine) -> str:
    return (engine.fi_dir / "run.log").read_text(encoding="utf-8")


def _changelog(engine: Engine) -> str:
    path = engine.quest_root / "code" / "CHANGELOG.md"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


# --- the ratchet, on its own -----------------------------------------------------------------------------------------


def _row(name: str, value: float | None, *, direction: str = "lower", tolerance: float = 0.01,
         target: float | None = None, counts: bool = True) -> dict[str, Any]:
    return {"name": name, "value": value, "direction": direction, "tolerance": tolerance, "target": target,
            "counts": counts, "met": None}


def test_each_criterion_is_compared_on_its_own_by_its_own_tolerance() -> None:
    before = [_row("a", 0.5), _row("b", 0.5, tolerance=0.2), _row("c", 1.0, direction="target", target=0.0),
              _row("d", 2.0, direction="higher", tolerance=0.1)]
    after = [_row("a", 0.3), _row("b", 0.6, tolerance=0.2), _row("c", -0.5, direction="target", target=0.0),
             _row("d", 1.5, direction="higher", tolerance=0.1)]
    changes = {c["name"]: c["change"] for c in improve.compare(before, after)}
    # a: down by 0.2 > 0.01; b: up by 0.1, inside its own 0.2; c: 0.5 from the target instead of 1.0; d: lower is worse.
    assert changes == {"a": "better", "b": "same", "c": "better", "d": "worse"}
    verdict = improve.verdict(improve.compare(before, after))
    assert verdict["best"] is False and verdict["worse"] == ["d"], "one worse is enough: no total can outweigh it"


def test_a_criterion_no_longer_measured_is_worse_and_every_one_lost_is_a_broken_run() -> None:
    before = [_row("a", 0.5), _row("b", 0.5)]
    partly = improve.verdict(improve.compare(before, [_row("a", 0.1), _row("b", None)]))
    assert partly["worse"] == ["b"] and not partly["best"] and not partly["broken"]
    broken = improve.verdict(improve.compare(before, [_row("a", None), _row("b", None)]))
    assert broken["broken"] and broken["worse"] == [] and not broken["best"]


def test_a_number_the_script_measured_itself_is_never_compared() -> None:
    before = [_row("a", 0.5), _row("s", 0.5, counts=False)]
    after = [_row("a", 0.5), _row("s", 0.0, counts=False)]
    assert [c["name"] for c in improve.compare(before, after)] == ["a"]
    assert improve.verdict(improve.compare(before, after))["better"] == []


# --- the edit, before anything runs ----------------------------------------------------------------------------------


def _check(new_find: str, new_replace: str, *, file: str = "simulate.py", tried: set[str] | None = None,
           expected: list[float] | None = None, measures: set[str] | None = None,
           settings: dict[str, list[Any]] | None = None) -> tuple[str | None, str]:
    files = {"simulate.py": SIM, "experiment.py": ANALYSIS}
    edit, why = improve.parse_edit(json.loads(_edit(new_find, new_replace, file=file)))
    assert edit is not None, why
    return improve.check_edit(files, edit, ["simulate.py"], tried=tried or {improve.fingerprint(files)},
                              expected=expected or [], measures=measures if measures is not None else {"error", "y1"},
                              settings=settings if settings is not None else {"dt": [0.1]})


def test_only_changing_a_printed_number_is_refused_before_it_runs() -> None:
    text, why = _check('print("error estimate", 0.3)', 'print("error estimate", 0.0)')
    assert text is None and "changes nothing the simulation computes" in why
    text, why = _check("    print(\"error estimate\", 0.3)\n", "    # the error is tiny now\n    print('ok')\n")
    assert text is None and "changes nothing the simulation computes" in why


@pytest.mark.parametrize("file", ["experiment.py", "../needs/FROZEN_PROTOCOL.json", "code/../core/criteria.py"])
def test_an_edit_outside_the_simulation_is_refused(file: str) -> None:
    text, why = _check("x", "y", file=file)
    assert text is None and "not part of the simulation" in why


def test_an_edit_that_is_not_found_once_or_does_not_parse_or_repeats_a_version_is_refused() -> None:
    assert "not in" in _check("no such text", "x")[1]
    assert "2 times" in _check("    return ", "    return  ")[1]
    assert "no longer be valid Python" in _check(EULER, "    return y +\n")[1]
    tried = {improve.fingerprint({"simulate.py": SIM.replace(EULER, SAME), "experiment.py": ANALYSIS})}
    assert "already tried" in _check(EULER, SAME, tried=tried)[1]
    text, why = _check(EULER, RK4)
    assert text is not None and why == "" and "k4" in text


def test_writing_a_value_a_check_expects_into_the_code_is_refused() -> None:
    text, why = _check('"y1": y}', '"y1": 0.36787944117144233}', expected=[0.36787944117144233])
    assert text is None and "a value a check expects" in why
    # A round number such as 0 is not taken for one where it is not the checked number: an expected error of 0 is
    # everywhere.
    assert _check(EULER, "    return y + dt * (-y) + 0.0\n", expected=[0.0])[0] is not None


@pytest.mark.parametrize("find, replace", [
    ('"error": abs(y - math.exp(-1))', '"error": 0.0'),                  # the checked number written as 0
    ('"error": abs(y - math.exp(-1))', '"error": abs(y - math.exp(-1)) * dt ** 6'),  # shrunk, not computed better
    ('"y1": y}', '"y1": math.exp(-1)}'),                                  # the known answer, as a formula
])
def test_changing_how_the_checked_number_is_worked_out_is_refused(find: str, replace: str) -> None:
    text, why = _check(find, replace)
    assert text is None and "the number a check reads" in why


def test_recognising_the_setting_a_check_runs_on_is_refused() -> None:
    special = '    dt = cell["dt"]\n    if dt == 0.1:\n        return {"error": 0.0, "drift": 0.0, "y1": 0.0}\n'
    text, why = _check('    dt = cell["dt"]\n', special, measures=set())
    assert text is None and "a setting a check runs on" in why
    # The same comparison with a value no check runs on is a change like any other.
    assert _check('    dt = cell["dt"]\n', special.replace("0.1", "0.3"), measures=set())[0] is not None
    # A setting read straight from the cell counts too; a value of another setting, or a comparison that reads no
    # setting (a loop bound), does not.
    via_cell = special.replace("if dt == 0.1", 'if cell.get("dt") == 0.1')
    assert "a setting a check runs on" in _check('    dt = cell["dt"]\n', via_cell, measures=set())[1]
    via_alias = special.replace("    if dt == 0.1", '    h = cell["dt"]\n    if h == 0.1')
    assert "a setting a check runs on" in _check('    dt = cell["dt"]\n', via_alias, measures=set())[1]
    settings = {"dt": [0.1], "y0": [1.0], "seed": [0]}
    guard = '    dt = cell["dt"]\n    if dt == 0:\n        raise ValueError("dt")\n'
    assert _check('    dt = cell["dt"]\n', guard, measures=set(), settings=settings)[0] is not None
    loop = "    for _ in range(int(round(1 / dt))):\n"
    looped = _check(loop, "    t = 0.0\n    while t < 1.0 - 1e-12:\n        t += dt\n", measures=set(), settings=settings)
    assert looped[0] is not None, looped[1]


def test_a_criterion_measured_for_the_first_time_counts_only_when_it_is_met() -> None:
    before = [_row("a", 0.5), {**_row("b", None), "counts": False}]
    unmet = [_row("a", 0.5), {**_row("b", 0.2), "met": False}]
    met = [_row("a", 0.5), {**_row("b", 0.2), "met": True}]
    assert improve.verdict(improve.compare(before, unmet))["best"] is False
    assert improve.verdict(improve.compare(before, met))["best"] is True


def test_a_file_with_windows_line_endings_is_edited_with_the_text_the_model_saw(tmp_path: Path) -> None:
    files = {"simulate.py": SIM.replace("\n", "\r\n")}
    edit, _ = improve.parse_edit(json.loads(_edit(EULER, RK4)))
    text, why = improve.check_edit(files, edit, ["simulate.py"], tried=set(), expected=[])
    assert text is not None and "k4 = -(y + dt * k3)\r\n" in text, why
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_bytes(files["simulate.py"].encode("utf-8"))
    snap = improve.snapshot(root)
    improve.restore(root, snap)
    assert (root / "code" / "simulate.py").read_bytes() == files["simulate.py"].encode("utf-8"), "bytes kept exactly"


# --- the loop, in the engine -----------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_round_that_brings_the_error_down_is_kept_and_the_loop_stops_when_every_check_is_met(
    tmp_path: Path,
) -> None:
    engine, model = await _quest(tmp_path, [_edit(EULER, RK4)])
    out = await engine._node_improve(dict(STATE))
    root = engine.quest_root
    assert out["improve_rerun"] is True and out["improve_rounds_used"] == 1
    assert "k4" in (root / "code" / "simulate.py").read_text(encoding="utf-8")
    rows = cr.history(root)
    assert len(rows) == 2 and rows[-1]["improve"]["kept"] is True and rows[-1]["improve"]["better"] == ["decay error small"]
    assert rows[-1]["criteria"][0]["met"] is True and rows[-1]["criteria"][0]["value"] < 1e-5
    record = improve.load(root)
    assert record["best_round"] == 1 and record["stopped"] == improve.STOP_ALL_MET
    assert f"[improve] stopped: {improve.STOP_ALL_MET}. Kept the version from round 1" in _log(engine)
    assert len(model.prompts) == 1
    if HAS_GIT:
        log = _changelog(engine)
        assert "improve round 1: a better integrator | decay error small:" in log and "kept as the best version" in log
    # The study's own results never reach the loop: not the headline value, not the headline metric's name.
    prompt = model.prompts[0]
    assert "987654" not in prompt and "headline_value" not in prompt and "decay_headline" not in prompt
    assert "decay error small" in prompt and "exp(-1)" in prompt


@pytest.mark.asyncio
async def test_a_round_that_makes_one_check_worse_is_not_kept_and_only_warns_by_default(tmp_path: Path) -> None:
    worse_and_better = _edit("BIAS = 0.0\n\n\ndef step(y, dt):\n" + EULER,
                             "BIAS = 0.3\n\n\ndef step(y, dt):\n" + RK4)
    engine, model = await _quest(tmp_path, [worse_and_better, _edit(EULER, RK4)], criteria=[C_ERR, C_DRIFT],
                                 oracles=[DECAY, DRIFT])
    out = await engine._node_improve(dict(STATE))
    root = engine.quest_root
    record = improve.load(root)
    first, second = record["rounds"]
    assert first["outcome"] == "not kept" and first["worse"] == ["no bias added"] and first["better"] == ["decay error small"]
    assert second["outcome"] == "kept", "a round that made something better goes on to the next round"
    sim = (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert "BIAS = 0.0" in sim and "k4" in sim
    assert out["improve_rerun"] is True and record["stopped"] == improve.STOP_ALL_MET
    log = _log(engine)
    assert "[improve] round 1 made no bias added worse by more than the tolerance" in log
    assert "a warning only: rigor_profile: research would stop the quest here" in log
    assert not (engine.fi_dir / "pause.json").exists()
    kept = [r["improve"]["kept"] for r in cr.history(root)[1:]]
    assert kept == [False, True]
    if HAS_GIT:
        assert "improve round 1 not kept (it made no bias added worse by more than the tolerance)" in _changelog(engine)


@pytest.mark.asyncio
async def test_under_research_a_round_that_makes_a_check_worse_stops_the_quest_and_a_resume_goes_on(
    tmp_path: Path,
) -> None:
    worse_and_better = _edit("BIAS = 0.0\n\n\ndef step(y, dt):\n" + EULER,
                             "BIAS = 0.3\n\n\ndef step(y, dt):\n" + RK4)
    engine, model = await _quest(tmp_path, [worse_and_better, _edit(EULER, RK4)], criteria=[C_ERR, C_DRIFT],
                                 oracles=[DECAY, DRIFT], research=True)
    stops: list[dict[str, Any]] = []

    class Stopped(Exception):
        pass

    def pause(**kw: Any) -> None:
        stops.append(kw)
        raise Stopped

    engine._pause_for_human = pause  # type: ignore[method-assign]
    with pytest.raises(Stopped):
        await engine._node_improve(dict(STATE))
    root = engine.quest_root
    assert stops[0]["kind"] == "improve" and "made a check of correctness worse" in stops[0]["headline"]
    assert "no bias added" in stops[0]["steps"][0]
    assert (root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM, "the version kept so far is back"
    assert improve.load(root)["blocked"] is True
    # The resume: no more changes, and nothing to run again (the first version is what the last full run used).
    out = await engine._node_improve(dict(STATE))
    assert out["improve_rerun"] is False and len(model.prompts) == 1
    assert improve.load(root)["blocked"] is False
    assert "resumed after the stop for a check that got worse" in _log(engine)


@pytest.mark.asyncio
async def test_a_round_that_makes_no_check_better_stops_the_loop(tmp_path: Path) -> None:
    engine, model = await _quest(tmp_path, [_edit(EULER, SAME), _edit(EULER, RK4)])
    out = await engine._node_improve(dict(STATE))
    record = improve.load(engine.quest_root)
    assert record["stopped"] == improve.STOP_PLATEAU and len(model.prompts) == 1
    assert out["improve_rerun"] is False and record["best_round"] == 0
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM
    assert f"[improve] stopped: {improve.STOP_PLATEAU}. Kept the first version" in _log(engine)


@pytest.mark.asyncio
async def test_a_helper_module_saved_in_another_encoding_is_carried_through_byte_for_byte(tmp_path: Path) -> None:
    engine, _model = await _quest(tmp_path, [_edit(EULER, SAME)])
    helper = engine.quest_root / "code" / "constants.py"
    helper.write_bytes(b"# 25\xb0C\nK = 1\n")  # cp1252, not UTF-8
    await engine._node_improve(dict(STATE))
    assert helper.read_bytes() == b"# 25\xb0C\nK = 1\n"
    assert improve.load_snapshot(engine.quest_root, "original")["constants.py"].encode(
        "utf-8", "surrogateescape") == b"# 25\xb0C\nK = 1\n"


@pytest.mark.asyncio
async def test_the_loop_stops_when_the_rounds_are_used_up(tmp_path: Path) -> None:
    strict = {**C_ERR, "target": 1e-9, "tolerance": 1e-10}
    engine, model = await _quest(tmp_path, [_edit(EULER, HEUN), _edit(HEUN, RK3), _edit(RK3, RK4)], criteria=[strict],
                                 rounds=2)
    out = await engine._node_improve(dict(STATE))
    record = improve.load(engine.quest_root)
    assert [r["outcome"] for r in record["rounds"]] == ["kept", "kept"]
    assert record["stopped"] == improve.STOP_BUDGET and len(model.prompts) == 2 and out["improve_rounds_used"] == 2
    assert "dt ** 3 / 6" in (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8")
    # The budget is the quest's: a later pass (a redesign) has none left.
    again = await engine._node_improve({**STATE, "improve_rounds_used": 2})
    assert again == {"improve_rerun": False} and "were used earlier in this quest" in _log(engine)


@pytest.mark.asyncio
async def test_a_round_whose_run_changes_the_frozen_protocol_is_aborted_and_put_back(tmp_path: Path) -> None:
    tamper = _edit('    dt = cell["dt"]\n',
                   '    dt = cell["dt"]\n    import pathlib\n    p = pathlib.Path("needs/FROZEN_PROTOCOL.json")\n'
                   '    p.write_text(p.read_text(encoding="utf-8").replace("1e-05", "1.0"), encoding="utf-8")\n')
    engine, model = await _quest(tmp_path, [tamper, _edit(EULER, RK4)])
    root = engine.quest_root
    frozen_before = (root / "needs" / "FROZEN_PROTOCOL.json").read_bytes()
    history_before = (root / cr.HISTORY).read_bytes()
    out = await engine._node_improve(dict(STATE))
    assert (root / "needs" / "FROZEN_PROTOCOL.json").read_bytes() == frozen_before
    assert (root / cr.HISTORY).read_bytes() == history_before, "nothing the aborted round measured is recorded"
    record = improve.load(root)
    assert record["rounds"][0]["outcome"] == "aborted" and "needs/FROZEN_PROTOCOL.json" in record["rounds"][0]["reason"]
    assert record["stopped"] == improve.STOP_TAMPERED and len(model.prompts) == 1
    assert out["improve_rerun"] is False
    assert (root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM
    assert fp.load(root).get("problem") is None, "the frozen record matches its hash again"
    assert "[improve] round 1 aborted: its run changed needs/FROZEN_PROTOCOL.json" in _log(engine)
    if HAS_GIT:
        assert "improve round 1 aborted (a better integrator)" in _changelog(engine)
        assert "improve round 1 aborted: back to the first version" in _changelog(engine)


@pytest.mark.asyncio
async def test_a_round_whose_run_plants_a_git_hook_or_a_file_in_code_is_aborted_and_both_are_removed(
    tmp_path: Path,
) -> None:
    plant = _edit('    dt = cell["dt"]\n',
                  '    dt = cell["dt"]\n    import os\n    os.makedirs("code/.git/no-hooks", exist_ok=True)\n'
                  '    open("code/.git/no-hooks/pre-commit", "w").write("#!/bin/sh\\necho x > HOOK_RAN\\n")\n'
                  '    open("code/notes.txt", "w").write("x")\n')
    engine, _model = await _quest(tmp_path, [plant])
    root = engine.quest_root
    await engine._node_improve(dict(STATE))
    record = improve.load(root)
    assert record["rounds"][0]["outcome"] == "aborted"
    assert not (root / "code" / "notes.txt").exists()
    assert not (root / "code" / ".git" / "no-hooks" / "pre-commit").exists()
    assert not (root / "code" / "HOOK_RAN").exists() and not (root / "HOOK_RAN").exists(), "no hook ran at FI's commits"


@pytest.mark.skipif(__import__("os").name != "nt", reason="a junction is a Windows folder link")
@pytest.mark.asyncio
async def test_a_junction_a_round_makes_in_code_is_removed_and_what_it_points_at_is_kept(tmp_path: Path) -> None:
    link = _edit('    dt = cell["dt"]\n',
                 '    dt = cell["dt"]\n    import os, subprocess\n    if not os.path.exists("code/link"):\n'
                 '        subprocess.run(["cmd", "/c", "mklink", "/J", "code\\\\link", "data"], capture_output=True)\n')
    engine, _model = await _quest(tmp_path, [link])
    root = engine.quest_root
    (root / "data").mkdir(exist_ok=True)
    (root / "data" / "measurements.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    await engine._node_improve(dict(STATE))
    assert improve.load(root)["rounds"][0]["outcome"] == "aborted"
    assert (root / "data" / "measurements.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert not (root / "code" / "link").exists()


@pytest.mark.asyncio
async def test_a_print_only_change_does_not_count_and_the_analysis_script_is_out_of_reach(tmp_path: Path) -> None:
    replies = [_edit('print("error estimate", 0.3)', 'print("error estimate", 0.0)'),
               _edit('"headline_value": 987654.321', '"headline_value": 1.0', file="experiment.py")]
    engine, model = await _quest(tmp_path, replies, rounds=2)
    root = engine.quest_root
    out = await engine._node_improve(dict(STATE))
    record = improve.load(root)
    assert [r["outcome"] for r in record["rounds"]] == ["refused", "refused"]
    assert "changes nothing the simulation computes" in record["rounds"][0]["reason"]
    assert "not part of the simulation" in record["rounds"][1]["reason"]
    assert record["best_round"] == 0 and out["improve_rerun"] is False and record["stopped"] == improve.STOP_BUDGET
    assert len(cr.history(root)) == 1, "a refused change is not run, so nothing is measured or recorded"
    assert (root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM
    assert (root / "code" / "experiment.py").read_text(encoding="utf-8") == ANALYSIS
    if HAS_GIT:
        # Each refused round is still an entry in the code's history, with nothing changed.
        assert _changelog(engine).count("refused before it ran") == 2
        assert code_project.head(root)[1] is False
    # The second prompt names the refused change and why.
    assert "refused (it changes nothing the simulation computes" in model.prompts[1]


SIM_TRIALS = '''import random


def run_trial(cell, trial_id, seed):
    rng = random.Random(trial_id // 32)
    return {"outcome": 1.0 if rng.random() < 0.5 else 0.0}
'''


@pytest.mark.asyncio
async def test_a_criterion_about_the_trials_is_measured_from_all_of_them_and_fis_record_is_put_back(
    tmp_path: Path,
) -> None:
    """Trials that reuse one stream of random numbers in blocks of 32 do not shrink the error bar; seeding each trial
    from the seed FI gives it does. The round runs every trial of the protocol; the record of the first run's trials
    (the one the paper may still use) is exactly as it was afterwards."""
    shrink = {"name": "error bars shrink", "trials": "outcome", "direction": "target", "target": 0.5, "tolerance": 0.15}
    engine, model = await _quest(tmp_path, [_edit("random.Random(trial_id // 32)", "random.Random(seed)")],
                                 criteria=[shrink], sim=SIM_TRIALS, grid={"p": [0.5]}, runs=256)
    root = engine.quest_root
    ledger = root / "raw" / "ledger.jsonl"
    before = ledger.read_bytes()
    out = await engine._node_improve(dict(STATE))
    rows = cr.history(root)
    assert rows[0]["criteria"][0]["value"] < 0.2 and rows[0]["criteria"][0]["met"] is False
    assert abs(rows[-1]["criteria"][0]["value"] - 0.5) <= 0.15 and rows[-1]["improve"]["kept"] is True
    assert out["improve_rerun"] is True
    assert ledger.read_bytes() == before, "FI's record of the first run's trials is back as it was"
    # The kept version runs its trials again in full (a kept change to a helper module would not change the key the
    # trials already run are found by), and starts with the repair counters of a new script.
    assert not (root / ".fi" / "trials" / "run.json").exists()
    assert out["exec_reflect_iter"] == 0 and out["exec_give_up_reason"] == ""


@pytest.mark.asyncio
async def test_a_round_that_forges_the_saved_trial_record_is_aborted_and_the_real_one_is_put_back(
    tmp_path: Path,
) -> None:
    shrink = {"name": "error bars shrink", "trials": "outcome", "direction": "target", "target": 0.5, "tolerance": 0.15}
    forge = _edit("    rng = random.Random(trial_id // 32)\n",
                  "    rng = random.Random(seed)\n    open('.fi/improve/raw/0.bin', 'w').write('forged')\n")
    engine, _model = await _quest(tmp_path, [forge], criteria=[shrink], sim=SIM_TRIALS, grid={"p": [0.5]}, runs=256)
    root = engine.quest_root
    ledger = root / "raw" / "ledger.jsonl"
    before = ledger.read_bytes()
    await engine._node_improve(dict(STATE))
    record = improve.load(root)
    assert record["rounds"][0]["outcome"] == "aborted" and ".fi/improve/raw/0.bin" in record["rounds"][0]["reason"]
    assert ledger.read_bytes() == before, "the copy held in memory is put back, not the forged one on disk"
    assert (root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM_TRIALS


@pytest.mark.asyncio
async def test_a_round_never_runs_the_compiled_copy_of_an_earlier_version_of_the_same_size(tmp_path: Path) -> None:
    """Python's cache keys a compiled module on the source's size and whole-second time: two versions of one size
    written in the same second look alike. A round must run the version on disk, not the one before it."""
    import os
    import py_compile

    engine, _model = await _quest(tmp_path, [], criteria=[C_ERR, C_DRIFT], oracles=[DECAY, DRIFT])
    root = engine.quest_root
    sim = root / "code" / "simulate.py"
    earlier = SIM.replace("BIAS = 0.0", "BIAS = 0.3")
    sim.write_bytes(earlier.encode("utf-8"))
    py_compile.compile(str(sim), doraise=True)  # the earlier version's compiled copy, as a run would leave it
    stamp = sim.stat().st_mtime
    sim.write_bytes(SIM.encode("utf-8"))  # the same size
    os.utime(sim, (stamp, stamp))
    protocol = fp.protocol_of(root)
    rows, _problems = await engine._improve_measure(protocol, _oracle.declared(protocol), set())
    drift = next(r for r in rows if r["name"] == "no bias added")
    assert drift["value"] == 0.0, "the version on disk ran, not the compiled copy of the one before"


@pytest.mark.asyncio
async def test_nothing_is_changed_when_every_check_is_met_or_the_loop_is_off(tmp_path: Path) -> None:
    easy = {**C_ERR, "target": 0.05}
    engine, model = await _quest(tmp_path / "met", [_edit(EULER, RK4)], criteria=[easy])
    assert await engine._node_improve(dict(STATE)) == {"improve_rerun": False}
    assert model.prompts == [] and "every check of correctness is already met" in _log(engine)
    engine, model = await _quest(tmp_path / "off", [_edit(EULER, RK4)], rounds=0)
    assert await engine._node_improve(dict(STATE)) == {"improve_rerun": False}
    assert model.prompts == [] and "engine.improve_rounds is 0" in _log(engine)
    # A search for the best design: FI's search makes the runs, so the loop does not change the simulation under it.
    engine, model = await _quest(tmp_path / "search", [_edit(EULER, RK4)])
    search = {**STATE, "design": {"protocol": {"optimisation": {"goal": "x"}}}}
    assert await engine._node_improve(search) == {"improve_rerun": False}
    assert model.prompts == [] and "a search for the best design" in _log(engine)


@pytest.mark.asyncio
async def test_after_the_kept_version_runs_in_full_a_changed_result_is_named_in_the_changelog_only(tmp_path: Path) -> None:
    engine, _model = await _quest(tmp_path, [_edit(EULER, RK4)])
    await engine._node_improve(dict(STATE))
    after = {**STATE, "improve_rerun": True, "result_json": {"headline_value": 987654.321, "by_cell": {"dt=0.1": [0.37]}}}
    out = await engine._node_improve(after)
    assert out == {"improve_rerun": False, "improve_fell_back": False}
    assert improve.load(engine.quest_root)["headline_changed"] == ["by_cell"]
    if HAS_GIT:
        log = _changelog(engine)
        assert "the study's results changed after the kept change (by_cell)" in log and "not used to choose" in log
        assert "987654" not in log and "0.37" not in log


@pytest.mark.asyncio
async def test_a_kept_version_that_fails_its_full_run_gives_way_to_the_first_version(tmp_path: Path) -> None:
    engine, _model = await _quest(tmp_path, [_edit(EULER, RK4)])
    await engine._node_improve(dict(STATE))
    failed = {**STATE, "improve_rerun": True, "exec_result": {"returncode": 1}, "result_json": {}}
    out = await engine._node_improve(failed)
    assert out["improve_rerun"] is True and out["improve_fell_back"] is True and out["exec_reflect_iter"] == 0
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM
    assert "the first version is back and runs once more" in _log(engine)
    # If the first version fails too, the quest goes on: never a third run, even when the record on disk lost the mark
    # (the quest's state holds it too).
    record = improve.load(engine.quest_root)
    record.pop("fell_back")
    improve.save(engine.quest_root, record)
    assert (await engine._node_improve({**failed, "improve_fell_back": True}))["improve_rerun"] is False


def test_the_note_says_when_a_kept_version_gave_way_and_when_a_loop_was_cut_short() -> None:
    fell = {"stopped": improve.STOP_ALL_MET, "best_round": 0, "fell_back": "round 2",
            "rounds": [{"round": 1, "ran": True}, {"round": 2, "ran": True}]}
    text = improve.summary(fell)
    assert "The version from round 2 was kept at first, but its full run produced no result" in text
    assert "round True" not in improve.summary({**fell, "fell_back": True})
    cut = {"stopped": improve.STOP_CUT_SHORT, "best_round": 0, "rounds": [{"round": 1, "ran": True}]}
    assert "Nothing it changed was kept." in improve.summary(cut) and "The first version was kept" not in improve.summary(cut)


@pytest.mark.asyncio
async def test_a_loop_cut_short_puts_the_first_version_and_its_trial_record_back_before_it_starts_again(
    tmp_path: Path,
) -> None:
    engine, model = await _quest(tmp_path, [_edit(EULER, RK4)])
    root = engine.quest_root
    improve.save_snapshot(root, "original", improve.snapshot(root))
    ledger = root / "raw" / "ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b'{"event": "trial"}\n')
    engine._improve_save_raw()
    # The process stopped mid-round: a half-finished version in code/, a round's trials in raw/, a round's row.
    (root / "code" / "simulate.py").write_text(SIM.replace(EULER, SAME), encoding="utf-8")
    ledger.write_bytes(b"a round's trials\n")
    cr.record(root, run="run_1", code_commit=None, results=[], improve={"round": 1, "kept": False})
    improve.save(root, {"started": "then", "rounds": [{"round": 1}], "rounds_used_before": 0, "baseline_n": 1,
                        "raw_saved": True})
    out = await engine._node_improve(dict(STATE))
    assert ledger.read_bytes() == b'{"event": "trial"}\n'
    assert "an earlier improve loop was cut short" in _log(engine) and "(1 round(s) stay spent)" in _log(engine)
    # It then started again from the last full run's row (not the round's), with the round it had spent still spent.
    assert out["improve_rerun"] is True and "k4" in (root / "code" / "simulate.py").read_text(encoding="utf-8")
    assert out["improve_rounds_used"] == 2
    assert "Before this, 1 earlier pass(es)" in improve.summary(improve.load(root))


@pytest.mark.asyncio
async def test_a_stop_left_by_code_that_has_run_again_since_is_set_aside(tmp_path: Path) -> None:
    engine, model = await _quest(tmp_path, [_edit(EULER, RK4)])
    root = engine.quest_root
    improve.save(root, {"started": "then", "stopped": "x", "blocked": True, "rounds": [], "baseline_n": 99})
    out = await engine._node_improve(dict(STATE))
    assert "belongs to code that has run again since" in _log(engine)
    assert out["improve_rerun"] is True and len(model.prompts) == 1, "the loop ran for this code"


def test_the_paper_is_told_what_the_loop_did_and_that_the_results_did_not_choose_the_version(tmp_path: Path) -> None:
    cr.record(tmp_path, run="run_1", code_commit=None, results=[])  # the full run the paper is written from: n = 1
    record = {"stopped": improve.STOP_ALL_MET, "best_round": 1, "result_n": 1,
              "rounds": [{"round": 1, "ran": True, "outcome": "kept"}]}
    improve.save(tmp_path, record)
    note = improve.write_note(tmp_path)
    assert "The version from round 1 was kept" in note and "never by the study's own results" in note
    assert improve.write_note(tmp_path / "none") == ""
    # A loop over code that has run again since (a rerun from an earlier step, a redesign) is not what produced the
    # results being written up, so the paper is told nothing about it.
    cr.record(tmp_path, run="run_1", code_commit=None, results=[])
    assert improve.write_note(tmp_path) == ""


# --- through the graph -----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_quest_improves_its_simulation_runs_the_kept_version_in_full_and_writes_its_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for
    from tests.test_run_manifest import _reply

    protocol = {"grid": {"dt": [0.1, 0.05]}, "runs_per_setting": 1, "oracles": [DECAY], "criteria": [C_ERR]}
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.startswith("# Improve the Simulation"):
            prompts.append(prompt)
            return _edit(EULER, RK4)
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return _reply(SIM, ANALYSIS)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    engine = Engine(_config(tmp_path))
    artifacts = await engine.run()
    root = engine.quest_root
    assert artifacts.paper_md is not None
    assert len(prompts) == 1 and "987654" not in prompts[0]
    rows = cr.history(root)
    # The first run, the round, then the kept version's own full run (the one the paper is written from).
    assert [r.get("improve", {}).get("round") for r in rows] == [None, 1, None], rows
    assert rows[0]["criteria"][0]["met"] is False and rows[-1]["criteria"][0]["met"] is True
    assert "k4" in (root / "code" / "simulate.py").read_text(encoding="utf-8")
    log = _log(engine)
    assert "[improve] stopped: every criterion is met. Kept the version from round 1" in log
    assert log.index("[improve] stopped") < log.rindex("[execute] rc=0")
    progress = (engine.fi_dir / "progress.log").read_text(encoding="utf-8")
    assert "[improve] Changing the simulation to meet its checks of correctness (round 1 of 3)." in progress
    assert "[improve] Stopped: every criterion is met." in progress
    if HAS_GIT:
        assert "improve round 1:" in _changelog(engine)
    from core import attempt_records as ar

    assert [c["node"] for c in ar.read(engine.fi_dir, ar.MODEL_CALLS)].count("improve") == 1
