"""Regression contracts from an external re-audit's reproductions (each began as a probe asserting the defect; every one
now asserts the correct behaviour). Neutral names throughout: nothing here depends on a topic."""
from __future__ import annotations

from types import SimpleNamespace

from core import flat_output


def test_the_same_numbers_in_other_units_get_the_same_flat_decision() -> None:
    def cells(scale: float) -> list[SimpleNamespace]:
        return [SimpleNamespace(cell={"setting": i}, rows=[{"status": "ok", "values": {"size": i * scale}}])
                for i in (1, 2, 3)]
    # 1, 2 and 3 of a small unit written in the base unit, and in the small unit: one decision for both.
    assert flat_output.flat_sweep(cells(1e-12)) is None
    assert flat_output.flat_sweep(cells(1.0)) is None
    assert flat_output.flat_sweep(cells(1e12)) is None


# ---- an output that does not change is an observation, not a broken simulation ------------------------------------------

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from core import run_estimate as est
from core.engine import Engine
from core import typed_results
from tests.test_flat_output import SWEEP_ANALYSIS, _quest, _sweep_runner

CONSERVING = ('def run_cell(cell):\n'
              '    a = cell["flow"]\n'
              '    b = 10.0 - a\n'
              '    return {"total": a + b}\n')
IGNORING = 'def run_cell(cell):\n    return {"total": 10.0}\n'
DEAD_READ = 'def run_cell(cell):\n    unused = cell["flow"] * 2.0\n    return {"total": 10.0}\n'


def _run(tmp_path: Path, source: str):
    root = _quest(tmp_path, source)
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    runner = _sweep_runner(root)
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60,
                                        env={"FI_REPLICATE_SEED": "0"}))
    return root, runner, result


def test_a_conserved_total_keeps_its_result_and_is_recorded_as_observed(tmp_path: Path) -> None:
    assert not typed_results.typed_results(CONSERVING, "simulate.py")
    root, runner, result = _run(tmp_path, CONSERVING)
    assert result.returncode == 0, result.stderr[-1500:]
    assert runner.failed_script is None and runner.flat is None
    assert runner.no_difference and "cannot show that there is no effect" in runner.no_difference
    assert "does not use" not in (result.stderr or "")


@pytest.mark.parametrize("source", [IGNORING, DEAD_READ])
def test_a_program_that_lets_no_setting_reach_its_result_is_still_sent_back(tmp_path: Path, source: str) -> None:
    root, runner, result = _run(tmp_path, source)
    assert result.returncode == 1 and runner.failed_script == "simulate.py"
    assert "does not use" in result.stderr and runner.flat and runner.no_difference is None


def test_the_plan_declaring_the_quantity_invariant_is_never_a_fault(tmp_path: Path) -> None:
    root = _quest(tmp_path, IGNORING)
    (root / "code" / "experiment.py").write_text(SWEEP_ANALYSIS, encoding="utf-8")
    from core import trial_runner
    from core.execution import SharedInterpreterExecutor
    protocol = {"grid": {"flow": [1.0, 2.0, 3.0]},
                "oracles": [{"name": "mass balance", "kind": "invariant", "measure": "total", "expected": 0.0,
                             "tolerance": 1e-9, "reference": "derivation: what flows in flows out"}]}
    runner = trial_runner.TrialsRunner(
        SharedInterpreterExecutor(python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"), quest_root=root,
        protocol=protocol, deterministic=True, simulate=root / "code" / "simulate.py",
        analysis=root / "code" / "experiment.py")
    result = asyncio.run(runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=60,
                                        env={"FI_REPLICATE_SEED": "0"}))
    assert result.returncode == 0 and runner.flat is None and runner.no_difference


def test_the_run_record_says_observed_not_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, ["{}"])
    eng._record_no_difference("`total` was the same at all 3 settings tried")
    note = eng._run_checks_note()
    assert "Observed:" in note and "do not call the simulation broken" in note
    eng._record_no_difference(None)
    assert "Observed:" not in eng._run_checks_note()


@pytest.mark.parametrize("source,keys,expected", [
    (CONSERVING, ["flow"], []),
    (IGNORING, ["flow"], ["flow"]),
    (DEAD_READ, ["flow"], ["flow"]),
    ('def run_cell(cell):\n    if cell["mode"] == "a":\n        v = 1.0\n    else:\n        v = 2.0\n    return {"t": v}\n',
     ["mode"], []),
    ('def run_cell(cell):\n    if cell["mode"] == "a":\n        return {"t": 1.0}\n    return {"t": 2.0}\n', ["mode"], []),
    ('def run_cell(cell):\n    return {"t": cell.get("flow", 1.0)}\n', ["flow", "other"], ["other"]),
    ('def run_cell(cell):\n    return {"t": helper(cell)}\n', ["flow"], None),
    ('def run_cell(cell):\n    return {"t": 1.0}\n\ndef run_trial(cell, t, s):\n    return {"t": 1.0}\n', ["flow"], None),
    ("def run_cell(cell):\n    return {\"t\": eval(\"cell['flow']\")}\n", ["flow"], None),
    ("not python (", ["flow"], None),
])
def test_the_code_evidence_names_only_what_it_can_show(source: str, keys: list[str], expected: list[str] | None) -> None:
    from core import flat_output
    assert flat_output.unread_settings(source, keys) == expected


def test_a_quantity_built_from_no_setting_is_named_and_one_that_is_built_from_one_is_not() -> None:
    from core import flat_output
    source = ('def run_trial(cell, trial, seed):\n    out = {}\n    out["fixed"] = 1.0\n    out["moves"] = cell["x"] * 2\n'
              '    return out\n')
    assert flat_output.unused_outputs(source, ["fixed", "moves"]) == ["fixed"]
    assert flat_output.unused_outputs("def run_cell(cell):\n    return other(cell)\n", ["a"]) is None


def test_the_pilot_does_not_send_back_the_conserved_total() -> None:
    cells = [{"flow": f} for f in (1.0, 2.0, 3.0)]
    obs = [(dict(c), {"total": 10.0}) for c in cells for _ in range(2)]
    assert est.pilot_problems(obs, {"total": "mean"}, source=CONSERVING).problems == []
    assert [p.kind for p in est.pilot_problems(obs, {"total": "mean"}, source=IGNORING).problems] == ["flat"]


# ---- a value left out of the results does not reach a model through the run's output -----------------------------------

SENTINEL = "4242.4242"


def _with_sentinel_sim() -> str:
    from tests import test_engine_callable_simulation as cs

    return cs.SIMULATE.replace('return {"error": abs(y - math.exp(-1.0))}',
                               'return {"error": abs(y - math.exp(-1.0)), "typed_value": 4242.4242}')


def _with_sentinel_analysis() -> str:
    from tests import test_engine_callable_simulation as cs

    return cs.ANALYSIS.replace(
        "print('RESULT_JSON: ' + json.dumps({'max_error': max(errors.values())}))",
        "print('typed_value is', 4242.4242)\n"
        "print('RESULT_JSON: ' + json.dumps({'max_error': max(errors.values()), 'typed_value': 4242.4242}))")


def test_stdout_for_models_drops_the_result_line_and_any_line_naming_a_left_out_quantity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    keys = eng._left_out_quantities()
    assert keys, "the typed-in quantity is left out"
    key = sorted(keys)[0]
    raw = f"step 1 done\n{key} = {SENTINEL}\nRESULT_JSON: " + json.dumps({"kept": 1.5, key: float(SENTINEL)}) + "\ndone\n"
    shown = eng._stdout_for_models(raw)
    assert SENTINEL not in shown and "RESULT_JSON" not in shown.replace("[FI NOTE] The run's RESULT_JSON line", "")
    assert "step 1 done" in shown and "done" in shown and f"`{key}`" in shown and "must not be used" in shown
    # Output with nothing to take out is passed on as it is.
    assert eng._stdout_for_models("plain\n") == "plain\n"
    # Raw output stays in the records, off the models' path.
    eng._keep_raw_stdout(raw)
    assert SENTINEL in (eng.fi_dir / "last_run_stdout.txt").read_text(encoding="utf-8")


def test_stderr_for_models_drops_lines_naming_a_left_out_quantity_and_keeps_the_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    key = sorted(eng._left_out_quantities())[0]
    err = ("Traceback (most recent call last):\n"
           '  File "experiment.py", line 9, in <module>\n'
           f"{key} = {SENTINEL}\n"
           "ZeroDivisionError: division by zero\n")
    shown = eng._stderr_for_models(err)
    assert f"{key} = " not in shown and SENTINEL not in shown
    assert "Traceback (most recent call last):" in shown and "ZeroDivisionError: division by zero" in shown
    assert "[FI NOTE]" in shown and "1 line(s) of the run's error output" in shown and f"`{key}`" in shown
    # Error output with nothing to take out is passed on as it is.
    plain = "Traceback (most recent call last):\nValueError: x\n"
    assert eng._stderr_for_models(plain) == plain
    # FI's own message naming the quantity (what to repair) stays: it is FI's diagnosis, not a value the script printed.
    fi_line = f"FI's run gave a result that is not usable: the result `{key}` does not change when the settings change"
    shown = eng._stderr_for_models(f"{fi_line}\n{key} = {SENTINEL}\n")
    assert fi_line in shown and SENTINEL not in shown and "1 line(s) of the run's error output" in shown


def test_stdout_for_models_replaces_a_bare_left_out_value_printed_without_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    key = sorted(eng._left_out_quantities())[0]
    eng._note_typed_values({"kept": 1.5, key: float(SENTINEL)})
    shown = eng._stdout_for_models(f"Result: {SENTINEL}\nkept is 1.5\n")
    assert SENTINEL not in shown and "Result:" in shown and "kept is 1.5" in shown
    assert "(a value that is not given)" in shown


def test_the_exclusion_persists_across_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    left = eng._left_out_quantities()
    assert left
    # A new process: nothing in memory, the pass's record is what stands.
    fresh, _ = _typed_engine(tmp_path / "again", monkeypatch, ["{}"])
    (fresh.quest_root / "needs").mkdir(parents=True, exist_ok=True)
    shutil.copy(eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json", fresh.quest_root / "needs" / "TYPED_RESULTS_CHECK.json")
    fresh.__dict__.pop("_typed_result_keys", None)
    assert getattr(fresh, "_typed_result_keys", None) is None
    assert fresh._left_out_quantities() == left
    assert fresh._without_typed_results({"kept": 1.0, sorted(left)[0]: 2.0}) == {"kept": 1.0}
    assert SENTINEL not in fresh._stdout_for_models(f"{sorted(left)[0]} = {SENTINEL}\n")


def test_the_values_left_out_are_kept_so_the_writers_text_is_checked_for_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import record_claims
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    key = sorted(eng._left_out_quantities())[0]
    eng._note_typed_values({"kept": 1.5, key: float(SENTINEL)})
    record = json.loads((eng.quest_root / "needs" / "TYPED_RESULTS_CHECK.json").read_text(encoding="utf-8"))
    assert record["removed_values"] == {key: [float(SENTINEL)]}
    hits = record_claims.contradictions(f"The simulation gave {SENTINEL} at the end.", left_out_typed=record["removed_values"])
    assert hits and key in hits[0] and "typed into its code" in hits[0]
    assert record_claims.contradictions("The simulation gave 1.5 at the end.", left_out_typed=record["removed_values"]) == []


@pytest.mark.asyncio
async def test_a_left_out_value_is_in_no_model_prompt_and_not_in_the_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole graph with a fake model: the simulation types a number into its code and the analysis script prints it (in a
    RESULT_JSON line and in a plain line). Every prompt that carries the run's output (not the code, which holds its own
    constant) is free of it, and so is the paper; the raw output keeps it."""
    from core.engine import Engine
    from tests import test_engine_callable_simulation as cs

    prompts: list[str] = []
    inner = cs._fake([], _with_sentinel_sim())

    async def spying(self, messages, **kw):  # noqa: ANN001
        prompts.append("\n".join(str(m.get("content", "")) for m in messages))
        reply = await inner(self, messages, **kw)
        if "# file: experiment.py" in reply:
            reply = reply.replace(cs.ANALYSIS, _with_sentinel_analysis())
        return reply

    monkeypatch.setattr("core.engine.LLMClient.chat", spying)
    engine = Engine(cs._cfg(tmp_path))
    artifacts = await engine.run()
    raw = (engine.quest_root / ".fi" / "last_run_stdout.txt").read_text(encoding="utf-8")
    assert SENTINEL in raw, "the run printed it, and the record keeps what the run printed"
    ran = [p for p in prompts if "def run_cell" not in p and "def run_trial" not in p]
    assert ran and not any(SENTINEL in p for p in ran), [p[:200] for p in ran if SENTINEL in p][:1]
    assert any("RESULT_JSON line is not repeated here" in p or "left out of the results" in p for p in ran)
    assert artifacts.paper_md is None or SENTINEL not in Path(artifacts.paper_md).read_text(encoding="utf-8")


def test_a_note_that_says_why_a_quantity_is_out_never_repeats_its_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _asked = _typed_engine(tmp_path, monkeypatch, [json.dumps({"code": "def broken(:\n"})])
    asyncio.run(eng._results_from_computation({}))
    eng._note_typed_values({"typed_value": float(SENTINEL)})
    (eng.quest_root / "needs" / "RESULT_CHECK.json").write_text(json.dumps({
        "removed_quantities": ["other_value"], "removed_values": {"other_value": [3.14159]},
        "say": "`other_value` is 3.14159 at every one of the 8 settings although other results change with them."}),
        encoding="utf-8")
    note = eng._run_checks_note()
    assert SENTINEL not in note and "3.14159" not in note
    assert "8 settings" in note, "numbers that are not a left-out value stay"
    assert "(a value that is not given)" in note


def test_the_analysis_prompt_has_one_projection_of_the_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The audit's reproduction, with the output handed on the way the run hands it: the left-out quantity is in the raw
    output's result line, and the analysis prompt carries neither that line nor the quantity."""
    from unittest.mock import AsyncMock

    from tests.test_run_checks import _typed_engine

    eng, _ = _typed_engine(tmp_path, monkeypatch, ["{}"])
    asyncio.run(eng._results_from_computation({}))
    key = sorted(eng._left_out_quantities())[0]
    raw = {"kept_value": 73.25, key: 1.0}
    filtered = eng._without_typed_results(raw)
    assert key not in filtered
    captured: list[str] = []

    async def fake_json(prompt, **kwargs):  # noqa: ANN001
        captured.append(prompt)
        return "", {"summary": "offline", "key_findings": []}

    monkeypatch.setattr(eng, "_refresh_code_project", AsyncMock())
    monkeypatch.setattr(eng, "_ask_about_edited_project_files", lambda: None)
    monkeypatch.setattr(eng, "_chat_json", fake_json)
    stdout = "RESULT_JSON: " + json.dumps(raw)
    asyncio.run(eng._node_analyze({
        "result_json": filtered,
        "exec_result": {"returncode": 0, "duration_s": 0.1, "stdout_tail": eng._stdout_for_models(stdout)}}))
    assert len(captured) == 1
    assert f'"{key}": 1.0' not in captured[0] and "RESULT_JSON: {" not in captured[0]
    assert "73.25" in captured[0], "the accepted results are there, once"
    assert "RESULT_JSON line is not repeated here" in captured[0]


# ---- the version that runs is the version the checks were made for -----------------------------------------------------

from core import gate_receipt, plot_scope  # noqa: E402

GOOD_PLOTTED = ('import matplotlib.pyplot as plt\n'
                'def run_cell(cell):\n    return {"output": cell["x"] ** 2}\n'
                'def plot_result(res):\n'
                '    plt.plot([1, 2, 3], [4, 5, 6])\n'
                '    plt.savefig("figures/a.png")\n')
#: A reply that draws from the results AND changes what the simulation computes.
BAD_REPLY = GOOD_PLOTTED.replace('cell["x"] ** 2', "999.0").replace('plt.plot([1, 2, 3], [4, 5, 6])', 'plt.plot(res["x"], res["y"])')
#: A reply that changes only the drawing.
DRAWING_ONLY_REPLY = GOOD_PLOTTED.replace('plt.plot([1, 2, 3], [4, 5, 6])',
                                          'import json\n    res = json.load(open("r.json"))\n    plt.plot(res["x"], res["y"])')


def test_the_plot_scope_tells_drawing_from_computation() -> None:
    assert plot_scope.computation_changes(GOOD_PLOTTED, DRAWING_ONLY_REPLY) == []
    assert plot_scope.computation_changes(GOOD_PLOTTED, BAD_REPLY) == ["function run_cell was changed or removed"]
    # A script: the lists the drawing reads are plot data, so building them from the results is a drawing change; the line
    # that prints the results is not.
    script = ('import json, os\nimport matplotlib.pyplot as plt\nd = json.load(open(os.environ["FI_TRIALS"]))\n'
              'xs = [1, 2, 3]\nys = [4, 5, 6]\nfig, ax = plt.subplots()\nax.plot(xs, ys)\nfig.savefig("a.png")\n'
              'print("RESULT_JSON: " + json.dumps({"m": max(d)}))\n')
    redrawn = script.replace("xs = [1, 2, 3]\nys = [4, 5, 6]\n", "xs = list(range(len(d)))\nys = [c['v'] for c in d]\n")
    assert plot_scope.computation_changes(script, redrawn) == []
    assert plot_scope.computation_changes(script, redrawn.replace("max(d)", "5.0"))
    # Mixed function (computes and draws): the drawing is free, the computation is not.
    mixed = 'import matplotlib.pyplot as plt\ndef main():\n    e = 2.0 * 3\n    plt.plot([e])\n    plt.savefig("a.png")\n    return e\n'
    assert plot_scope.computation_changes(mixed, mixed.replace("plt.plot([e])", "plt.plot([e, e, e])")) == []
    assert plot_scope.computation_changes(mixed, mixed.replace("2.0 * 3", "9.0")) == ["function main was changed or removed"]
    # Additions are not changes; a rebinding of an existing import is.
    assert plot_scope.computation_changes(mixed, mixed + "def helper():\n    return 1\n") == []
    assert plot_scope.computation_changes("import json\nx = 1\n", "import json as j\nx = 1\n")
    assert plot_scope.computation_changes("x = 1\n", "x = (") == ["the script does not parse"]


def _figure_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str], files: dict[str, str]):
    from tests.test_run_checks import _typed_engine

    eng, asked = _typed_engine(tmp_path, monkeypatch, replies)
    code = eng.quest_root / "code"
    (code / "simulate.py").unlink(missing_ok=True)
    for name, text in files.items():
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_a_figure_repair_that_also_changes_the_simulation_is_refused_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _figure_engine(tmp_path, monkeypatch, [json.dumps({"code": BAD_REPLY})], {"simulate.py": GOOD_PLOTTED})
    path = eng.quest_root / "code" / "simulate.py"
    used, rewrote = await eng._figures_from_results({})
    assert rewrote is False and used == Engine._FIGURE_DATA_REPAIRS == len(asked)
    assert path.read_text(encoding="utf-8") == GOOD_PLOTTED, "the simulation FI checked is the one that stays"
    log = (eng.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "also changes what it computes" in log and "run_cell" in log


@pytest.mark.asyncio
async def test_a_figure_repair_that_changes_only_the_drawing_is_kept_single_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, _asked = _figure_engine(tmp_path, monkeypatch, [json.dumps({"code": DRAWING_ONLY_REPLY})], {"simulate.py": GOOD_PLOTTED})
    used, rewrote = await eng._figures_from_results({})
    assert (used, rewrote) == (1, True)
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == DRAWING_ONLY_REPLY


ANALYSIS_TYPED = ('import json, os\nimport matplotlib.pyplot as plt\n'
                  'def summarize(values):\n    return sum(values) / len(values)\n'
                  'def draw():\n    plt.plot([1, 2, 3], [4, 5, 6])\n    plt.savefig("figures/b.png")\n'
                  'draw()\n')
SIM_TWO = 'def run_cell(cell):\n    return {"output": cell["x"] ** 2}\n'


@pytest.mark.asyncio
async def test_a_figure_repair_of_the_second_script_that_changes_its_computation_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = ANALYSIS_TYPED.replace("sum(values) / len(values)", "max(values)").replace(
        "plt.plot([1, 2, 3], [4, 5, 6])", 'plt.plot(*json.load(open("r.json")))')
    drawing = ANALYSIS_TYPED.replace("plt.plot([1, 2, 3], [4, 5, 6])", 'plt.plot(*json.load(open("r.json")))')
    eng, _ = _figure_engine(tmp_path / "a", monkeypatch, [json.dumps({"code": changed})],
                            {"simulate.py": SIM_TWO, "experiment.py": ANALYSIS_TYPED})
    used, rewrote = await eng._figures_from_results({})
    assert rewrote is False and (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == ANALYSIS_TYPED
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM_TWO
    eng2, _ = _figure_engine(tmp_path / "b", monkeypatch, [json.dumps({"code": drawing})],
                             {"simulate.py": SIM_TWO, "experiment.py": ANALYSIS_TYPED})
    assert await eng2._figures_from_results({}) == (1, True)


# ---- the receipt: what changed after the checks is checked again, or the run does not start ----------------------------

PACKAGE = 'def energy(x):\n    return x ** 2\n\ndef label():\n    return "a"\n'


def _code(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code" / "pkg").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(
        "from pkg.model import energy\ndef run_cell(cell):\n    return {'e': energy(cell['x'])}\n", encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    (root / "code" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "code" / "pkg" / "model.py").write_text(PACKAGE, encoding="utf-8")
    return root


def test_the_receipt_sees_a_change_in_any_script_helper_or_package(tmp_path: Path) -> None:
    root = _code(tmp_path)
    assert gate_receipt.verdict(root) == gate_receipt.NONE
    gate_receipt.write(root)
    assert gate_receipt.verdict(root) == gate_receipt.SAME
    # FI's own files and notes are not the study's code.
    (root / "code" / "run.py").write_text("print(1)\n", encoding="utf-8")
    (root / "code" / "README.md").write_text("notes\n", encoding="utf-8")
    (root / "code" / "__pycache__").mkdir()
    (root / "code" / "__pycache__" / "x.pyc").write_bytes(b"x")
    assert gate_receipt.verdict(root) == gate_receipt.SAME
    # A comment in a helper: the same computation, the receipt is renewed.
    (root / "code" / "pkg" / "model.py").write_text(PACKAGE + "\n# a comment\n", encoding="utf-8")
    assert gate_receipt.verdict(root) == gate_receipt.DRAWING_ONLY and gate_receipt.verdict(root) == gate_receipt.SAME
    # Each way the computation can change: the simulation, the second script, the helper, a parameter file.
    for rel, text in (("simulate.py", "def run_cell(cell):\n    return {'e': 1.0}\n"),
                      ("experiment.py", "print('RESULT_JSON: {\"a\": 1}')\n"),
                      ("pkg/model.py", PACKAGE.replace("x ** 2", "x ** 3")),
                      ("params.json", '{"k": 2}')):
        gate_receipt.write(root)
        target = root / "code" / rel
        before = target.read_text(encoding="utf-8") if target.exists() else None
        target.write_text(text, encoding="utf-8")
        assert gate_receipt.verdict(root) == gate_receipt.CHANGED, rel
        if before is None:
            target.unlink()
        else:
            target.write_text(before, encoding="utf-8")


@pytest.mark.asyncio
async def test_a_computation_change_after_the_checks_makes_them_again_and_a_stop_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_run_checks import _typed_engine

    eng, _ = _typed_engine(tmp_path, monkeypatch, ["{}"])
    root = eng.quest_root
    shutil.rmtree(root / "code")
    shutil.copytree(_code(tmp_path / "src") / "code", root / "code")
    calls: list[str] = []
    verdict_stop: list[Any] = [None]

    async def gate(state, py, env, seed):  # noqa: ANN001
        calls.append("gate")
        return verdict_stop[0], None, True

    monkeypatch.setattr(eng, "_gate_pass", gate)
    gate_receipt.write(root)
    # Nothing changed: no second pass.
    assert await eng._verify_gated_code({}, "py", None, root / "code" / "simulate.py") == (None, None, False)
    assert calls == []
    # The helper's computation changed after the checks: the checks are made again, and the receipt follows.
    (root / "code" / "pkg" / "model.py").write_text(PACKAGE.replace("x ** 2", "999.0 + 0 * x"), encoding="utf-8")
    assert await eng._verify_gated_code({}, "py", None, root / "code" / "simulate.py") == (None, None, True)
    assert calls == ["gate"] and gate_receipt.verdict(root) == gate_receipt.SAME
    # When the second pass stops the quest, the run does not start and the receipt is not renewed.
    (root / "code" / "pkg" / "model.py").write_text(PACKAGE.replace("x ** 2", "0.5 + 0 * x"), encoding="utf-8")
    verdict_stop[0] = {"stopped": True}
    stop, repaired, _ = await eng._verify_gated_code({}, "py", None, root / "code" / "simulate.py")
    assert stop == {"stopped": True} and repaired is None
    assert gate_receipt.verdict(root) == gate_receipt.CHANGED


@pytest.mark.asyncio
async def test_a_rewrite_between_the_checks_and_the_run_is_checked_again_in_the_real_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole node: the figure step is made to rewrite the simulation's computation after the checks passed (the injected
    "fixed the figure but changed an equation" reply, here bypassing the refusal); the checks run a second time before
    anything is run."""
    from tests import test_engine_callable_simulation as cs

    passes: list[str] = []
    real_gate = Engine._gate_pass

    async def spying_gate(self, *args, **kwargs):  # noqa: ANN001
        passes.append((self.quest_root / "code" / "simulate.py").read_text(encoding="utf-8"))
        return await real_gate(self, *args, **kwargs)

    async def rewriting_figures(self, state):  # noqa: ANN001
        path = self.quest_root / "code" / "simulate.py"
        path.write_text(path.read_text(encoding="utf-8").replace("y = 1.0\n", "y = 1.0 + 0.0\n"), encoding="utf-8")
        return 0, True

    monkeypatch.setattr(Engine, "_gate_pass", spying_gate)
    monkeypatch.setattr(Engine, "_figures_from_results", rewriting_figures)
    monkeypatch.setattr("core.engine.LLMClient.chat", cs._fake([], cs.SIMULATE))
    engine = Engine(cs._cfg(tmp_path))
    await engine.run()
    assert len(passes) == 2, "the checks ran before the rewrite and again on the rewritten version"
    assert "y = 1.0 + 0.0" not in passes[0] and "y = 1.0 + 0.0" in passes[1]
    assert (engine.fi_dir / "run.log").read_text(encoding="utf-8").count("making the checks again") == 1


@pytest.mark.asyncio
async def test_a_run_nobody_changed_after_its_checks_makes_them_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No cost without cause: with nothing rewritten between the checks and the run, the checks are made once."""
    from tests import test_engine_callable_simulation as cs

    passes: list[int] = []
    real_gate = Engine._gate_pass

    async def spying_gate(self, *args, **kwargs):  # noqa: ANN001
        passes.append(1)
        return await real_gate(self, *args, **kwargs)

    monkeypatch.setattr(Engine, "_gate_pass", spying_gate)
    monkeypatch.setattr("core.engine.LLMClient.chat", cs._fake([], cs.SIMULATE))
    engine = Engine(cs._cfg(tmp_path))
    await engine.run()
    assert len(passes) == 1
    assert "making the checks again" not in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


def _reads(root: Path, literal: str) -> None:
    """simulate.py opens ``literal`` (a path outside code/) when it runs."""
    (root / "code" / "simulate.py").write_text(
        "from pkg.model import energy\n"
        f"PARAMS = open({literal!r}).read()\n"
        "def run_cell(cell):\n    return {'e': energy(cell['x'])}\n", encoding="utf-8")


@pytest.mark.parametrize("literal", ["../inputs/params.json", "inputs/params.json"])
def test_a_parameter_file_the_code_reads_is_part_of_the_checked_version(tmp_path: Path, literal: str) -> None:
    root = _code(tmp_path)
    _reads(root, literal)
    (root / "inputs").mkdir()
    (root / "inputs" / "params.json").write_text('{"k": 1}', encoding="utf-8")
    gate_receipt.write(root)
    assert gate_receipt.verdict(root) == gate_receipt.SAME
    (root / "inputs" / "params.json").write_text('{"k": 2}', encoding="utf-8")
    assert gate_receipt.verdict(root) == gate_receipt.CHANGED
    assert "inputs/params.json" in gate_receipt.fingerprint(root)["files"]


@pytest.mark.parametrize("literal", ["../raw/run.csv", "../.fi/notes.json", "../figures/fig.txt", "../paper/claims.json"])
def test_outputs_and_records_a_literal_names_are_not_part_of_the_checked_version(tmp_path: Path, literal: str) -> None:
    root = _code(tmp_path)
    _reads(root, literal)
    target = root / literal.replace("../", "", 1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("1,2\n", encoding="utf-8")
    gate_receipt.write(root)
    target.write_text("9,9\n", encoding="utf-8")
    assert gate_receipt.verdict(root) == gate_receipt.SAME
    assert all(not name.startswith(("raw/", ".fi/", "figures/", "paper/")) for name in gate_receipt.fingerprint(root)["files"])


def test_a_literal_naming_a_missing_file_is_ignored(tmp_path: Path) -> None:
    root = _code(tmp_path)
    _reads(root, "../inputs/missing.json")
    gate_receipt.write(root)
    now = gate_receipt.fingerprint(root)
    assert now["full"] and "inputs/missing.json" not in now["files"]
    assert gate_receipt.verdict(root) == gate_receipt.SAME


def _script_body(root: Path, body: str) -> None:
    (root / "code" / "simulate.py").write_text(
        "from pkg.model import energy\n" + body +
        "def run_cell(cell):\n    return {'e': energy(cell['x'])}\n", encoding="utf-8")


def test_a_file_the_code_writes_is_an_output_and_a_rerun_does_not_change_the_version(tmp_path: Path) -> None:
    root = _code(tmp_path)
    _script_body(root, "import pandas as pd\npd.DataFrame({'a': [1]}).to_csv('output.csv')\n")
    (root / "output.csv").write_text("a\n1\n", encoding="utf-8")
    gate_receipt.write(root)
    (root / "output.csv").write_text("a\n9\n", encoding="utf-8")  # what the next run leaves behind
    assert gate_receipt.verdict(root) == gate_receipt.SAME
    assert "output.csv" not in gate_receipt.fingerprint(root)["files"]


def test_a_file_the_code_opens_for_writing_is_an_output(tmp_path: Path) -> None:
    root = _code(tmp_path)
    _script_body(root, 'with open("data/clean.csv", "w") as handle:\n    handle.write("x,y\\n")\n')
    (root / "data").mkdir()
    (root / "data" / "clean.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    gate_receipt.write(root)
    (root / "data" / "clean.csv").write_text("x,y\n3,4\n", encoding="utf-8")
    assert gate_receipt.verdict(root) == gate_receipt.SAME


def test_a_parameter_file_in_a_binary_format_the_code_loads_is_part_of_the_checked_version(tmp_path: Path) -> None:
    root = _code(tmp_path)
    _script_body(root, 'import pickle\nMODEL = pickle.load(open("inputs/model.pkl", "rb"))\n')
    (root / "inputs").mkdir()
    (root / "inputs" / "model.pkl").write_bytes(b"\x80\x04one")
    gate_receipt.write(root)
    assert gate_receipt.verdict(root) == gate_receipt.SAME
    (root / "inputs" / "model.pkl").write_bytes(b"\x80\x04two")
    assert gate_receipt.verdict(root) == gate_receipt.CHANGED
    assert "inputs/model.pkl" in gate_receipt.fingerprint(root)["files"]


def test_a_file_the_code_both_reads_and_writes_is_an_input(tmp_path: Path) -> None:
    root = _code(tmp_path)
    _script_body(root, 'import json\nSTATE = json.load(open("state.json"))\n'
                       'json.dump(STATE, open("state.json", "w"))\n')
    (root / "state.json").write_text('{"n": 1}', encoding="utf-8")
    gate_receipt.write(root)
    (root / "state.json").write_text('{"n": 2}', encoding="utf-8")
    assert gate_receipt.verdict(root) == gate_receipt.CHANGED


def test_the_node_makes_the_receipt_check_before_the_pilot_and_the_run() -> None:
    import inspect

    body = inspect.getsource(Engine._node_execute)
    before_run = body.split("await self._figures_from_results(state)", 1)[1]
    assert before_run.index("await self._verify_gated_code(") < before_run.index("runner.execute(")
    assert "_gate_receipt.write(self.quest_root)" in body.split("await self._figures_from_results(state)", 1)[0]
