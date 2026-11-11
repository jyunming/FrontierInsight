"""Whether a fault a failing test showed changes a reported number is shown by running the study with a repair of it
(core/rerun_check.py), and the gate acts on that answer (Engine._shown_fault_outcome). Neutral made-up studies; no model is
called and no real study runs: the rerun's answer is given, or the study is a few lines of Python."""
from __future__ import annotations

import asyncio
import contextvars
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import code_project, rerun_check
from core.rerun_check import Verdict

PROTOCOL = {"metrics": [{"id": "est", "kind": "mean", "role": "result"}, {"id": "truth", "kind": "mean", "role": "reference"},
                        {"id": "cover", "kind": "proportion", "role": "result"}]}


# ---- what counts as a changed number --------------------------------------------------------------------------------------------
def test_compare_reads_a_number_as_the_same_when_it_is_only_reshaped() -> None:
    old = {"derived/derived/est/by_setting/a": 1.5}
    # the same value given as {"value": 1.5, "lo": 1.0, "hi": 2.0}: the value is the number, the ends are new numbers
    flat = dict(rerun_check._flat({"derived": {"est": {"by_setting": {"a": {"value": 1.5, "lo": 1.0, "hi": 2.0}}}}}, "derived"))
    assert flat["derived/derived/est/by_setting/a"] == 1.5
    assert rerun_check.compare(old, flat) == []


def test_compare_counts_trials_exactly_and_values_beyond_rounding() -> None:
    old = {"trials/s/ok": 100.0, "trials/s/failed": 0.0, "derived/derived/est/by_setting/s": 2.0}
    new = {"trials/s/ok": 99.0, "trials/s/failed": 1.0, "derived/derived/est/by_setting/s": 2.0 + 1e-6}
    assert sorted(c["key"] for c in rerun_check.compare(old, new)) == ["trials/s/failed", "trials/s/ok"]


def test_a_number_only_the_repaired_run_has_is_not_a_change_but_a_delivered_number_it_lacks_is() -> None:
    assert rerun_check.compare({"a": 1.0}, {"a": 1.0, "b": 5.0}) == []
    assert rerun_check.compare({"a": 1.0, "c": 2.0}, {"a": 1.0}) == [{"key": "c", "old": 2.0, "new": None}]


def test_the_other_result_files_the_paper_may_read_are_compared_too(tmp_path: Path) -> None:
    (tmp_path / "data" / "results").mkdir(parents=True)
    (tmp_path / "data" / "results" / "derived.json").write_text(json.dumps({"derived": {"est": {"by_setting": {"s": 1.0}}}}), encoding="utf-8")
    (tmp_path / "data" / "results" / "table_b.json").write_text(json.dumps({"rows": [{"a": 4.0}]}), encoding="utf-8")
    got = rerun_check.results_of(tmp_path)
    assert got["file:table_b.json/rows[0]/a"] == 4.0 and got["derived/derived/est/by_setting/s"] == 1.0


def test_a_standard_error_is_not_called_noise_when_it_moves_by_the_error_of_a_mean() -> None:
    values = [float(i % 7) for i in range(50)]
    mean = sum(values) / 50
    se = rerun_check.trial_stats(_trials(values))[0][("s", "est")][2] / math.sqrt(50)
    est = {"key": "derived/derived/est/by_setting/s", "old": mean, "new": mean + 1.5 * se}
    err = {"key": "derived/derived/est/by_setting/s/se", "old": se, "new": se + 1.5 * se}
    cov = {"key": "derived/derived/est/by_setting/s/coverage", "old": 0.95, "new": 0.96}
    end = {"key": "derived/derived/est/by_setting/s/hi", "old": mean + 2 * se, "new": mean + 2 * se + 1.5 * se}
    kept = rerun_check.counted([est, err, cov, end], PROTOCOL, _trials(values), redrawn=True)[0]
    assert [c["key"] for c in kept] == [err["key"]]  # a mean, an interval end and a coverage move with their own errors
    # a standard error moving by a few percent of its own value is inside its own sampling error
    small = {"key": err["key"], "old": se, "new": se * 1.05}
    assert rerun_check.counted([small], PROTOCOL, _trials(values), redrawn=True)[0] == []


def _trials(values: list[float], metric: str = "est", key: str = "s") -> dict[str, Any]:
    return {"cells": [{"key": key, "ok": len(values), "failed": 0, "metrics": {metric: {"values": values}}}]}


def test_a_move_inside_the_monte_carlo_uncertainty_is_not_counted_when_the_draws_changed() -> None:
    values = [float(i % 7) for i in range(50)]
    mean = sum(values) / 50
    se = rerun_check.trial_stats(_trials(values))[0][("s", "est")][2] / math.sqrt(50)
    inside = [{"key": "trials/s/est/mean", "old": mean, "new": mean + 1.5 * se}]
    beyond = [{"key": "trials/s/est/mean", "old": mean, "new": mean + 10 * se}]
    assert rerun_check.counted(inside, PROTOCOL, _trials(values), redrawn=True)[0] == []
    assert rerun_check.counted(beyond, PROTOCOL, _trials(values), redrawn=True)[0] == beyond
    # draws that did not change: any move beyond rounding is the repair's own
    assert rerun_check.counted(inside, PROTOCOL, _trials(values), redrawn=False)[0] == inside


def test_a_change_of_a_constant_metric_is_kept_not_divided_by_a_zero_uncertainty() -> None:
    values = [2.0] * 20  # every trial gave the same value: its uncertainty is 0
    assert rerun_check.uncertainty("trials/s/est/mean", 2.0, *rerun_check.trial_stats(_trials(values)), {}) == 0.0
    moved = [{"key": "trials/s/est/mean", "old": 2.0, "new": 2.001}]
    assert rerun_check.counted(moved, PROTOCOL, _trials(values), redrawn=True)[0] == moved


def test_floating_point_rounding_of_a_constant_metric_is_not_a_change() -> None:
    values = [0.0] * 20
    rounding = [{"key": "trials/s/est/mean", "old": 0.0, "new": 1e-15}]
    same = [{"key": "trials/s/est/mean", "old": 2.0, "new": 2.0}]
    assert rerun_check.counted(rounding, PROTOCOL, _trials(values), redrawn=True)[0] == []
    assert rerun_check.counted(same, PROTOCOL, _trials([2.0] * 20), redrawn=True)[0] == []
    real = [{"key": "trials/s/est/mean", "old": 0.0, "new": 1e-6}]
    assert rerun_check.counted(real, PROTOCOL, _trials(values), redrawn=True)[0] == real


def test_a_share_is_judged_by_its_binomial_error() -> None:
    trials = {"cells": [{"key": "s", "ok": 100, "failed": 0, "metrics": {}}]}
    small = [{"key": "derived/derived/cover/by_setting/s", "old": 0.95, "new": 0.96}]
    large = [{"key": "derived/derived/cover/by_setting/s", "old": 0.95, "new": 0.80}]
    assert rerun_check.counted(small, PROTOCOL, trials, redrawn=True)[0] == []
    assert rerun_check.counted(large, PROTOCOL, trials, redrawn=True)[0] == large


def test_a_reference_the_plan_declares_does_not_count() -> None:
    changes = [{"key": "derived/derived/truth/by_setting/s", "old": 1.0, "new": 2.0},
               {"key": "derived/derived/est/by_setting/s", "old": 1.0, "new": 2.0}]
    kept, notes = rerun_check.counted(changes, PROTOCOL, _trials([1.0, 2.0]), redrawn=False)
    assert [c["key"] for c in kept] == ["derived/derived/est/by_setting/s"]
    assert any("reference" in n for n in notes)


def test_draws_changed_reads_the_calls_that_draw_numbers() -> None:
    old = "import numpy as np\n\n\ndef trial(cell, ctx):\n    return ctx.rng.normal(0, 1)\n\n\ndef fit(x):\n    return x * 2\n"
    same_draws = old.replace("x * 2", "x * 3")
    other_draws = old.replace("normal(0, 1)", "normal(0, 2)")
    assert rerun_check.draws_changed({"simulate.py": old}, {"simulate.py": same_draws}) == []
    assert rerun_check.draws_changed({"simulate.py": old}, {"simulate.py": other_draws}) == ["simulate.py:trial"]


# ---- how long a rerun may take ----------------------------------------------------------------------------------------------------
def test_the_rerun_is_bounded_by_the_run_record_and_the_quests_own_limit(tmp_path: Path) -> None:
    assert rerun_check.time_allowed(tmp_path) is None  # no record, no bound: the rerun is not made
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "run_facts.json").write_text(json.dumps({"trials_run": {"wall_s": 20.0, "settings_s": 200.0}}), encoding="utf-8")
    assert rerun_check.time_allowed(tmp_path) == pytest.approx(900.0)  # 1.5 x the longer of the wall time and 3 x the settings' time
    assert rerun_check.time_allowed(tmp_path, 120) == 120.0


# ---- a repair kept in code/'s history is applied to a copy of the code ---------------------------------------------------------------
def _git(code: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=code, check=True, capture_output=True)


def _write(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))  # exactly these bytes: a line ending is part of what a patch must match


def test_the_last_refused_repair_is_a_patch_that_applies_to_code_that_has_moved_on(tmp_path: Path) -> None:
    quest = tmp_path / "quest"
    code = quest / "code"
    code.mkdir(parents=True)
    _write(code / "model.py", "def f(x):\n    return x\n\n\ndef g(x):\n    return x\n")
    _git(code, "init", "-q")
    _git(code, "add", "-A")
    _git(code, "commit", "-q", "-m", "start")
    assert code_project.keep_refused(quest, "F-1-1", {"model.py": "def f(x):\n    return x + 1\n\n\ndef g(x):\n    return x\n"}, "refused")
    # the code moves on after the try (another function is repaired): the file is no longer what the try started from
    _write(code / "model.py", "def f(x):\n    return x\n\n\ndef g(x):\n    return 2 * x\n")
    patches = code_project.refused_patches(quest, "F-1")
    assert [n for n, _p in patches] == [1]
    copy = tmp_path / "copy"
    copy.mkdir()
    _write(copy / "model.py", (code / "model.py").read_bytes().decode("utf-8"))
    assert code_project.apply_patch(copy, patches[0][1])
    got = (copy / "model.py").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert got == "def f(x):\n    return x + 1\n\n\ndef g(x):\n    return 2 * x\n"
    # a patch that does not fit changes nothing
    _write(copy / "model.py", "something else\n")
    assert not code_project.apply_patch(copy, patches[0][1])
    assert (copy / "model.py").read_bytes() == b"something else\n"
    assert code_project.refused_patches(quest, "F-2") == []


# ---- running a copy of a study --------------------------------------------------------------------------------------------------------
class _Local:
    """An executor that runs the command where it is (what the venv executor does, without a venv)."""

    async def execute(self, cmd: list[str], *, cwd: Path, timeout_s: int, env: dict[str, str] | None = None) -> Any:
        try:
            done = await asyncio.to_thread(subprocess.run, cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s, env=env)
        except subprocess.TimeoutExpired:
            return SimpleNamespace(returncode=-1, stdout="", stderr="", timed_out=True)
        return SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr, timed_out=False)


RUN_PY = '''import json, os, sys
from pathlib import Path
scale = 1.0
out = Path("output")
(out / "raw").mkdir(parents=True, exist_ok=True)
(out / "data" / "results").mkdir(parents=True, exist_ok=True)
cells = [{"key": "s", "ok": 10, "failed": 0, "metrics": {"est": {"values": [scale * i for i in range(10)]}}}]
(out / "raw" / "trials.json").write_text(json.dumps({"cells": cells}))
(out / "data" / "results" / "derived.json").write_text(json.dumps({"derived": {"est": {"by_setting": {"s": 4.5 * scale}}}}))
sys.exit(1)  # the study ends with a non-zero code after its results are written: it still counts as a run
'''


def _delivered(root: Path, scale: float) -> None:
    (root / "code").mkdir(parents=True)
    (root / "code" / "run.py").write_text(RUN_PY.replace("scale = 1.0", f"scale = {scale}"), encoding="utf-8")
    (root / "raw").mkdir()
    (root / "data" / "results").mkdir(parents=True)
    cells = [{"key": "s", "ok": 10, "failed": 0, "metrics": {"est": {"values": [scale * i for i in range(10)]}}}]
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (root / "data" / "results" / "derived.json").write_text(json.dumps({"derived": {"est": {"by_setting": {"s": 4.5 * scale}}}}),
                                                            encoding="utf-8")


def _judge(root: Path, tmp: Path, scale: float, timeout: float | None = 60) -> Verdict:
    copy = rerun_check.copy_code(root, tmp / "w")
    (copy / "run.py").write_text((copy / "run.py").read_text(encoding="utf-8").replace("scale = 1.0", f"scale = {scale}"), encoding="utf-8")
    return asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, timeout, None, source="a repair"))


def test_a_repair_that_leaves_every_reported_number_alone_is_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    got = _judge(root, tmp_path, 1.0)
    assert got.status == "could_not" and "no file" in got.why  # the copy changed no file: nothing to run
    # a repair that changes the code but no reported number
    copy = rerun_check.copy_code(root, tmp_path / "w2")
    (copy / "run.py").write_text((copy / "run.py").read_text(encoding="utf-8") + "\n# a comment\n", encoding="utf-8")
    same = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 60, None))
    assert same.status == "unchanged" and same.metrics == []


def test_a_move_of_a_millionth_counts_when_the_draws_did_not_change(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    got = _judge(root, tmp_path, 1.000001)
    assert got.status == "changed" and got.metrics == ["est"]


def _with_odd_file(tmp_path: Path, scale: float) -> Verdict:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    (root / "data" / "results" / "odd.json").write_text(json.dumps({"x": 1.0}), encoding="utf-8")
    (root / "code" / "run.py").write_text((root / "code" / "run.py").read_text(encoding="utf-8").replace(
        "sys.exit(1)", "(out / 'data' / 'results' / 'odd.json').write_text(json.dumps({'x': 1.0}))\nsys.exit(1)"), encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    text = (copy / "run.py").read_text(encoding="utf-8").replace("{'x': 1.0}", "{'x': 2.0}").replace("scale = 1.0", f"scale = {scale}")
    (copy / "run.py").write_text(text, encoding="utf-8")
    return asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 60, None))


def test_a_number_that_belongs_to_no_result_is_never_delivered_as_unchanged(tmp_path: Path) -> None:
    got = _with_odd_file(tmp_path, 1.0)
    assert got.status == "could_not" and "cannot attribute" in got.why and "file:odd.json/x" in got.keys


def test_numbers_that_belong_to_no_result_add_to_the_answer_and_never_replace_it(tmp_path: Path) -> None:
    got = _with_odd_file(tmp_path, 3.0)  # a result moved too
    assert got.status == "changed" and got.metrics == ["est"]
    assert any("cannot attribute" in n for n in got.notes)


def test_a_change_in_the_trials_that_finished_belongs_to_the_failed_trials(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "run.py").write_text((copy / "run.py").read_text(encoding="utf-8").replace('"ok": 10', '"ok": 9'), encoding="utf-8")
    got = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 60, None))
    assert got.status == "changed" and "failed trials" in got.metrics


def test_a_repair_that_moves_a_reported_number_is_changed_and_names_the_result(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    got = _judge(root, tmp_path, 3.0)
    assert got.status == "changed" and got.metrics == ["est"] and got.mode == "whole study"


def test_a_study_that_does_not_finish_in_the_time_is_could_not_and_so_is_one_with_no_bound(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    (root / "code" / "run.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "run.py").write_text("import time\ntime.sleep(30)\n# changed\n", encoding="utf-8")
    got = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 1, None))
    assert got.status == "could_not" and "did not finish" in got.why
    assert asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, None, None)).status == "could_not"


def test_only_the_analysis_is_run_again_when_only_the_analysis_changed(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    (root / "code" / "experiment.py").write_text(
        "import json, os\nfrom pathlib import Path\nt = json.load(open(os.environ['FI_TRIALS']))\n"
        "Path('data/results').mkdir(parents=True, exist_ok=True)\n"
        "Path('data/results/derived.json').write_text(json.dumps({'derived': {'est': {'by_setting': {'s': 4.5}}}}))\n", encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "experiment.py").write_text((copy / "experiment.py").read_text(encoding="utf-8").replace("4.5", "9.0"), encoding="utf-8")
    got = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 60, None))
    assert got.mode == "analysis only" and got.status == "changed" and got.metrics == ["est"]


# ---- what the gate does with the answer --------------------------------------------------------------------------------------------------
def _gate(tmp_path: Path, verdict: Verdict | None, *, headline: tuple[str, ...], test: str, quantity: str = "cover",
          function: str = "") -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from tests.test_shown_reach import _engine, _run

    engine = _engine(tmp_path)

    async def answers(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
        return {} if verdict is None else {str(i["id"]): verdict for i, _h in verdicts}

    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    return _run(engine, test, quantity=quantity, headline=headline, function=function)


from tests.test_shown_reach import DERIVE_COVER, DERIVE_EST  # noqa: E402


def test_a_fault_whose_repair_changes_no_reported_number_is_delivered_and_disclosed(tmp_path: Path) -> None:
    # the old gate stopped the paper: the fault is in the function of the main result, and the test still fails
    out, record = _gate(tmp_path, None, headline=("cover",), test=DERIVE_COVER)
    assert out and out["stuck_reason"] == "shown_fault_open"
    out, record = _gate(tmp_path / "b", Verdict("unchanged", mode="whole study"), headline=("cover",), test=DERIVE_COVER)
    assert out is None
    issue = record["issues"]["F-1"]
    assert issue["rerun_unchanged"] is True and issue["changes_result"] is False and "at_gate" not in issue and "rerun_moved_checks" not in issue


def test_numbers_that_changed_are_left_out_and_the_paper_stops_only_for_the_main_result(tmp_path: Path) -> None:
    changed = Verdict("changed", metrics=["cover"], keys=["derived/derived/cover/by_setting/s"])
    out, record = _gate(tmp_path, changed, headline=("est_a",), test=DERIVE_COVER)
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and "cover" in gate["keys"] and "run again" in gate["how"]
    out, _ = _gate(tmp_path / "b", changed, headline=("cover",), test=DERIVE_COVER)
    assert out and out["stuck_reason"] == "shown_fault_open" and "main result" in out["rationale"]


def test_a_change_in_a_result_the_main_result_is_worked_out_from_stops(tmp_path: Path) -> None:
    out, _ = _gate(tmp_path, Verdict("changed", metrics=["est_a"]), headline=("corr_ab",), test=DERIVE_EST, quantity="est_a")
    assert out and "corr_ab" in out["rationale"]


def test_a_fault_that_could_not_be_rerun_leaves_results_out_and_never_stops_the_paper_alone(tmp_path: Path) -> None:
    could_not = Verdict("could_not", why="the repaired study did not finish within 300 s")
    out, record = _gate(tmp_path, could_not, headline=("est_a",), test=DERIVE_COVER)
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and "cover" in gate["keys"] and "could not be run again" in gate["how"]


def _gate_engine(tmp_path: Path, verdict: Verdict) -> Any:
    """The engine after the gate judged one finding (about ``cover``) whose rerun gave ``verdict``."""
    from tests.test_shown_reach import DERIVE_COVER, _engine, _run

    engine = _engine(tmp_path)

    async def answers(verdicts: Any, record: Any, state: Any) -> dict[str, Any]:
        return {str(i["id"]): verdict for i, _h in verdicts}

    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    out, record = _run(engine, DERIVE_COVER, quantity="cover", headline=("est_a",))
    assert out is None
    return engine, record


def _said(engine: Any) -> tuple[str, str, str]:
    """The words of the must-fix record, the paper's limitation sentence and the run record for what the gate left out."""
    from core import disclosure, paper_facts

    kept = json.loads((engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    checks = " ".join(c.get("say", "") + c.get("note", "") for c in kept["checks"]) + " ".join(kept["removed_why"].values())
    items = " ".join(i.sentence for i in paper_facts.required_items(engine.quest_root))
    return checks, items, disclosure.run_record(engine.quest_root)


def test_a_failing_test_whose_rerun_could_not_be_made_is_not_said_to_show_a_wrong_value(tmp_path: Path) -> None:
    engine, record = _gate_engine(tmp_path, Verdict("could_not", why="the repaired study did not finish within 300 s"))
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and "cover" in gate["keys"] and "rerun" not in gate
    for text in _said(engine):
        assert "wrongly" not in text and "missing or incomplete" not in text, text
    checks, items, run = _said(engine)
    assert "did not establish whether the value is wrong or missing" in checks
    assert "did not establish whether the value is wrong or missing" in items
    assert "did not establish whether the value is wrong or missing" in run


def test_a_failing_test_whose_rerun_changed_the_number_is_said_to_change_the_reported_number(tmp_path: Path) -> None:
    engine, record = _gate_engine(tmp_path, Verdict("changed", metrics=["cover"], keys=["derived/derived/cover/by_setting/s"]))
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["outcome"] == "left_out" and gate["rerun"] == "changed"
    for text in _said(engine):
        assert "wrongly" not in text and "did not establish" not in text, text
        assert "a repair that passes" in text and "changes the reported number" in text, text


def test_a_fault_in_the_main_result_that_could_not_be_rerun_leaves_it_out_and_stops_only_for_an_empty_paper(tmp_path: Path) -> None:
    out, _ = _gate(tmp_path, Verdict("could_not", why="no run.py"), headline=("est_a",), test=DERIVE_EST, quantity="est_a")
    assert out is None  # est_b stays: the paper is not empty


def test_a_changed_standard_error_is_left_out_alone_and_does_not_stop_for_the_result_it_belongs_to(tmp_path: Path) -> None:
    # est_a is the main result and se_a its standard error: the rerun shows only se_a moved, so est_a itself is not left out
    out, record = _gate(tmp_path, Verdict("changed", metrics=["se_a"]), headline=("est_a",), test=DERIVE_COVER, quantity="se_a")
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    # the coverage FI counts from that interval is wrong with it, and is left out too; the estimate itself is not
    assert gate["outcome"] == "left_out" and gate["keys"] == ["est_a_coverage", "se_a"]


# ---- the repair is checked, the answer is placed, the time is bounded -----------------------------------------------------------------
def _engine_with_code(tmp_path: Path) -> Any:
    from tests.test_shown_reach import DERIVE_COVER, _engine, _quest

    engine = _engine(tmp_path)
    root = _quest(engine)
    rel = ".fi/finding_tests/F-1_1.py"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(DERIVE_COVER, encoding="utf-8")
    (root / ".fi" / "run_facts.json").write_text(json.dumps({"trials_run": {"wall_s": 30.0, "settings_s": 30.0}}), encoding="utf-8")
    issue = {"id": "F-1", "file": "derive.py", "function": "derive_cover", "quantity": "cover", "test": rel, "text": "wrong"}
    record = {"issues": {"F-1": dict(issue)}}

    async def run_in_env(argv: list[str], *, timeout_s: int = 180, env: dict[str, str] | None = None, **_k: Any) -> tuple[int, str, str]:
        import os

        done = subprocess.run([sys.executable, "-B", *argv], cwd=root, capture_output=True, text=True, timeout=timeout_s,
                              env={**os.environ, **(env or {})})
        return done.returncode, done.stdout, done.stderr

    engine._run_in_env = run_in_env  # type: ignore[method-assign,assignment]
    return engine, root, issue, record


NO_FIX = "import model\n\n\ndef derive_cover(table):\n    return model.helper_for_cover(1)  # reworded only\n"
FIX = "def derive_cover(table):\n    return 0.9\n"


def test_a_repair_is_run_only_when_the_findings_own_test_passes_on_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, root, issue, record = _engine_with_code(tmp_path)
    copies: list[Path] = []
    for text in (NO_FIX, FIX):
        copy = rerun_check.copy_code(root, tmp_path / f"c{len(copies)}")
        (copy / "derive.py").write_text(text, encoding="utf-8")
        copies.append(copy)
    assert asyncio.run(engine._repair_passes_own_test(issue, copies[0]))[0] is False
    assert asyncio.run(engine._repair_passes_own_test(issue, copies[1]))[0] is True
    # the whole step: a repair that leaves the test failing is never run, and never "unchanged"
    judged: list[Any] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        judged.append(a)
        return Verdict("unchanged")

    async def ask(issue: Any, rel: str, copy_dir: Path, protocol: Any, state: Any, problem: str = "") -> dict[str, str]:
        return {"derive.py": NO_FIX}

    monkeypatch.setattr(rerun_check, "judge", judge)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    got = asyncio.run(engine._rerun_one(issue, record, {}, {}))
    assert got.status == "could_not" and "own test" in got.why and judged == []

    async def ask_fixed(issue: Any, rel: str, copy_dir: Path, protocol: Any, state: Any, problem: str = "") -> dict[str, str]:
        return {"derive.py": FIX}

    engine._ask_repair_on_copy = ask_fixed  # type: ignore[method-assign,assignment]
    record["rerun_calls"] = 0
    assert asyncio.run(engine._rerun_one(issue, record, {}, {})).status == "unchanged" and len(judged) == 1


def test_nothing_is_run_when_the_time_of_the_check_is_spent_and_no_request_is_spent_in_docker(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, root, issue, record = _engine_with_code(tmp_path)
    asked: list[int] = []
    judged: list[int] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        judged.append(1)
        return Verdict("unchanged")

    async def ask(*a: Any, **k: Any) -> dict[str, str]:
        asked.append(1)
        return {"derive.py": FIX}

    monkeypatch.setattr(rerun_check, "judge", judge)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    got = asyncio.run(engine._rerun_one(issue, record, {}, {}, [5.0]))  # all the time one check may spend is gone
    assert got.status == "could_not" and "spent" in got.why and judged == []
    engine.config.execution.sandbox = "docker"
    asked.clear()
    got = asyncio.run(engine._rerun_one(issue, record, {}, {}))
    assert got.status == "could_not" and "Docker" in got.why and asked == [] and judged == []


def test_a_patch_does_not_apply_to_text_that_differs_only_in_whitespace(tmp_path: Path) -> None:
    quest = tmp_path / "quest"
    code = quest / "code"
    code.mkdir(parents=True)
    _write(code / "model.py", "def f(x):\n    return x\n")
    _git(code, "init", "-q")
    _git(code, "add", "-A")
    _git(code, "commit", "-q", "-m", "start")
    assert code_project.keep_refused(quest, "F-1-1", {"model.py": "def f(x):\n    return x + 1\n"}, "refused")
    patch = code_project.refused_patches(quest, "F-1")[0][1]
    copy = tmp_path / "copy"
    copy.mkdir()
    _write(copy / "model.py", "def f(x):\n    return  x\n")
    assert not code_project.apply_patch(copy, patch)


# ---- placing the answer ------------------------------------------------------------------------------------------------------------------
def test_numbers_that_changed_but_are_placed_on_no_result_are_not_delivered(tmp_path: Path) -> None:
    out, record = _gate(tmp_path, Verdict("changed", metrics=[], keys=["derived/derived/odd"]), headline=("est_a",), test=DERIVE_COVER)
    # judged by what the finding names, as with no rerun: nothing says which result moved, so the paper is not delivered on it
    assert out and out["stuck_reason"] == "shown_fault_open" and "cannot" in out["rationale"]


def test_numbers_that_did_not_move_do_not_answer_a_finding_about_a_figure_or_a_missing_quantity(tmp_path: Path) -> None:
    src = "import derive\n\n\ndef test_x():\n    assert derive.figure_fig1(None, None) == 1, 'wrong'\n"
    out, record = _gate(tmp_path, Verdict("unchanged", mode="whole study"), headline=("est_a",), test=src, function="figure_fig1",
                        quantity="figone")
    assert out is None
    assert "rerun_unchanged" not in record["issues"]["F-1"] and record["issues"]["F-1"].get("at_gate", {}).get("outcome") == "left_out"
    from tests.test_shown_reach import _engine

    engine = _engine(tmp_path / "m")
    issue = {"id": "F-1", "quantity": "cover"}
    missing = {"issues": {"F-1": dict(issue)},
               "read_problems": [{"id": "L1", "kind": "quantity", "status": "missing", "name": "cover", "quantities": ["cover"],
                                  "fid": "F-1", "text": "x"}]}
    assert engine._unchanged_is_shown(issue, missing, {}, [], None) is False
    assert engine._unchanged_is_shown(issue, {"issues": {"F-1": issue}}, {}, [], None) is True


def test_a_fault_that_could_not_be_rerun_stops_the_paper_when_it_leaves_the_main_result_not_computed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import shown_reach

    out, _ = _gate(tmp_path, Verdict("could_not", why="no run.py"), headline=("cover",), test=DERIVE_COVER)
    assert out is None  # only left out of the results: a rerun that could not be made does not stop the paper for that alone

    def effect(self: Any, record: Any, issue: Any, result: Any) -> Any:
        return shown_reach.Effect(not_done=["est_a"])  # the main result is not computed

    monkeypatch.setattr(shown_reach.Reach, "effect", effect)
    out, _ = _gate(tmp_path / "b", Verdict("could_not", why="no run.py"), headline=("est_a",), test=DERIVE_COVER)
    assert out and out["stuck_reason"] == "shown_fault_open" and "does not compute" in out["rationale"]


# ---- the second review ----------------------------------------------------------------------------------------------------------------
def test_a_coverage_or_rate_moves_by_its_binomial_error_when_the_draws_changed() -> None:
    trials = {"cells": [{"key": "s", "ok": 500, "failed": 0, "metrics": {}}]}
    small = {"key": "derived/derived/Ea_coverage/by_setting/s", "old": 0.95, "new": 0.96}
    leaf = {"key": "derived/derived/est/by_setting/s/coverage", "old": 0.95, "new": 0.96}
    rate = {"key": "derived/derived/fail_rate/by_setting/s", "old": 0.30, "new": 0.32}
    large = {"key": "derived/derived/Ea_coverage/by_setting/s", "old": 0.95, "new": 0.80}
    assert rerun_check.counted([small, leaf, rate], PROTOCOL, trials, redrawn=True)[0] == []
    assert rerun_check.counted([large], PROTOCOL, trials, redrawn=True)[0] == [large]
    # draws that did not change: any move counts, as before
    assert rerun_check.counted([small], PROTOCOL, trials, redrawn=False)[0] == [small]


def test_a_result_file_the_rerun_did_not_write_and_a_time_are_not_compared(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    (root / "data" / "results" / "odd.json").write_text(json.dumps({"x": 1.0, "runtime": 3.2}), encoding="utf-8")
    (root / "code" / "experiment.py").write_text(
        "import json, os\nfrom pathlib import Path\nt = json.load(open(os.environ['FI_TRIALS']))\n"
        "Path('data/results').mkdir(parents=True, exist_ok=True)\n"
        "Path('data/results/derived.json').write_text(json.dumps({'derived': {'est': {'by_setting': {'s': 4.5}}}}))\n", encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "experiment.py").write_text((copy / "experiment.py").read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    got = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 60, None))
    # odd.json is not written by an analysis-only rerun: its delivered number was not compared, so it is not "unchanged"
    assert got.mode == "analysis only" and got.status == "could_not" and "odd.json" in got.why
    assert "runtime" not in " ".join(rerun_check.results_of(root))


def test_a_changed_number_that_belongs_to_no_result_is_placed_as_well_as_the_results_that_moved(tmp_path: Path) -> None:
    from tests.test_shown_reach import DERIVE_EST

    # cover moved (not the main result), and so did a number of another file: the fault, by what its test reaches, is est_a's
    out, _ = _gate(tmp_path, Verdict("changed", metrics=["cover"], unplaced=["file:t.json/x"]), headline=("est_a",), test=DERIVE_EST,
                   quantity="est_a")
    assert out and out["stuck_reason"] == "shown_fault_open" and "main result" in out["rationale"]
    out, _ = _gate(tmp_path / "b", Verdict("changed", metrics=["cover"]), headline=("est_a",), test=DERIVE_EST, quantity="est_a")
    assert out is None  # with every moved number placed on a result, only those results are left out


def test_a_change_named_by_no_result_and_no_key_of_the_results_leaves_nothing_out_and_is_not_delivered(tmp_path: Path) -> None:
    from tests.test_shown_reach import DERIVE_EST

    out, _ = _gate(tmp_path, Verdict("changed", metrics=["failed trials"]), headline=("est_a",), test=DERIVE_EST, quantity="est_a")
    assert out and out["stuck_reason"] == "shown_fault_open"
    out, _ = _gate(tmp_path / "b", Verdict("changed", metrics=["odd_name"]), headline=("est_b",), test=DERIVE_COVER)
    # judged by the finding's own names, as with no rerun: not delivered with a "left out" key that nothing holds
    assert out and out["stuck_reason"] == "shown_fault_open" and "cannot" in out["rationale"]


# ---- a move inside the precision the plan holds a result to ---------------------------------------------------------------------
PRECISE = {**PROTOCOL, "precision": {"target_half_width": 0.1, "metric": "est"}}


def _moved(rel: float) -> list[dict[str, Any]]:
    return [{"key": "derived/derived/est/by_setting/s", "old": 22.8138685, "new": 22.8138685 * (1 + rel)}]


def _se_trials(se: float, centre: float = 22.8) -> dict[str, Any]:
    """The trials of setting ``s`` whose result ``est`` has exactly this standard error (ten values, centre +/- 3 se)."""
    return _trials([centre + (3 * se if i % 2 else -3 * se) for i in range(10)], metric="est")


def test_the_helper_trials_have_the_standard_error_asked_for() -> None:
    trials = _se_trials(0.04)
    got = rerun_check.uncertainty("derived/derived/est/by_setting/s", 22.8, *rerun_check.trial_stats(trials), {})
    assert got == pytest.approx(0.04)


def test_a_move_inside_the_results_own_standard_error_is_not_counted_and_is_noted() -> None:
    changes = [{"key": "derived/derived/est/by_setting/s", "old": 22.8138685, "new": 22.8138493}]
    kept, notes = rerun_check.counted(changes, PRECISE, _se_trials(0.04), redrawn=False)
    assert kept == [] and any("est moves by" in n and "within its own standard error" in n for n in notes)
    # the same move with random draws changed
    assert rerun_check.counted(changes, PRECISE, _se_trials(0.04), redrawn=True)[0] == []


def test_a_move_beyond_the_results_own_standard_error_is_counted() -> None:
    big = [{"key": "derived/derived/est/by_setting/s", "old": 22.8, "new": 22.95}]  # 0.15 against a standard error of 0.04
    assert rerun_check.counted(big, PRECISE, _se_trials(0.04), redrawn=False)[0] == big


def test_a_shift_is_judged_against_the_measured_error_only_not_against_a_share_of_the_target() -> None:
    # a guard, not a fail-first test: target half-width 0.1, value near 1 (a quarter of the target and 1% of the value are 0.025
    # and 0.01); the same 0.005 shift counts against a small measured error and is excused by a larger one
    near_one = [{"key": "derived/derived/est/by_setting/s", "old": 1.0, "new": 1.005}]
    assert rerun_check.counted(near_one, PRECISE, _se_trials(0.0004, 1.0), redrawn=False)[0] == near_one
    assert rerun_check.counted(near_one, PRECISE, _se_trials(0.02, 1.0), redrawn=False)[0] == []  # inside a measured error


def test_a_result_with_no_standard_error_in_the_record_is_excused_nothing_and_the_verdict_says_so() -> None:
    small = [{"key": "derived/derived/est/by_setting/s", "old": 1.0, "new": 1.005}]  # 5% of the half-width, 0.5% of the value
    kept, notes = rerun_check.counted(small, PRECISE, None, redrawn=False)
    assert kept == small and any("est has no standard error" in n for n in notes)
    assert rerun_check.counted(small, PRECISE, None, redrawn=True)[0] == small


def test_a_result_with_no_declared_precision_still_counts_a_millionth_move() -> None:
    changes = _moved(1e-6)
    assert rerun_check.counted(changes, PROTOCOL, None, redrawn=False)[0] == changes  # exact rule, as before
    assert rerun_check.compare({"derived/derived/est/by_setting/s": 22.8138685}, {"derived/derived/est/by_setting/s": 22.8138685 * (1 + 1e-6)}, 1e-9)
    # a precision on another result does not excuse this one
    other = {**PROTOCOL, "precision": {"target_half_width": 0.1, "metric": "cover"}}
    assert rerun_check.counted(changes, other, None, redrawn=False)[0] == changes


def test_the_declared_precision_does_not_excuse_a_count_or_a_coverage() -> None:
    cnt = [{"key": "trials/s/ok", "old": 10.0, "new": 9.0}]
    assert rerun_check.counted(cnt, PRECISE, None, redrawn=False)[0] == cnt
    cov = [{"key": "derived/derived/est/by_setting/s/coverage", "old": 0.95, "new": 0.9}]
    assert rerun_check.counted(cov, PRECISE, None, redrawn=False)[0] == cov


def test_a_repair_inside_the_declared_precision_is_unchanged_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    _delivered(root, 1.0)
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "run.py").write_text((copy / "run.py").read_text(encoding="utf-8").replace("scale = 1.0", "scale = 1.001"), encoding="utf-8")
    run = lambda proto: asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, proto, 60, None))  # noqa: E731
    got = run(PRECISE)  # the study's est moves by 0.0045 and the trials' mean by the same: inside the delivered run's own error (0.96)
    assert got.status == "unchanged" and any("within its own standard error" in n for n in got.notes)
    assert run(PROTOCOL).status == "changed"


K = "derived/derived/est/by_setting/s"


def _kept(changes: list[dict[str, Any]], protocol: Any = PRECISE, redrawn: bool = False, trials: Any = None) -> list[dict[str, Any]]:
    return rerun_check.counted(changes, protocol, trials, redrawn)[0]


def test_the_allowance_is_the_own_standard_error_and_stops_at_it() -> None:
    old = 100.0
    trials = _se_trials(0.03, old)
    assert _kept([{"key": K, "old": old, "new": old + 0.029}], trials=trials) == []
    over = [{"key": K, "old": old, "new": old + 0.031}]
    assert _kept(over, trials=trials) == over


def test_a_mean_shifted_by_most_of_the_target_is_counted() -> None:
    shifted = [{"key": K, "old": 22.8, "new": 22.8 + 0.09}]  # 0.9 x the half-width
    assert _kept(shifted) == shifted
    assert _kept(shifted, trials=_se_trials(0.01)) == shifted


def test_a_large_target_on_a_small_scale_value_does_not_excuse_a_large_share_of_it() -> None:
    big = {**PROTOCOL, "precision": {"target_half_width": 0.9, "metric": "est"}}
    moved = [{"key": K, "old": 0.02, "new": 0.04}]  # doubled, though far inside 0.9
    assert _kept(moved, big) == moved


def test_an_interval_that_widens_is_counted_with_the_mean_it_belongs_to() -> None:
    trials = _se_trials(0.03, 100.0)
    lo = {"key": K + "/lo", "old": 90.0, "new": 90.0 - 0.02}
    hi = {"key": K + "/hi", "old": 110.0, "new": 110.0 + 0.02}
    assert _kept([lo], trials=trials) == [] and _kept([hi], trials=trials) == []  # one end alone moves the width by 0.02: inside
    both = _kept([lo, hi], trials=trials)  # the width moves by 0.04: beyond the allowance, so both ends count
    assert both == [lo, hi]
    # a number of the setting that counts takes the allowance from the others
    big = {"key": K + "/sd", "old": 5.0, "new": 6.0}
    assert _kept([lo, big], trials=trials) == [lo, big]


def test_an_interval_stored_as_a_pair_widens_and_is_counted() -> None:
    trials = _se_trials(0.03, 100.0)
    lo = {"key": K + "/ci[0]", "old": 90.0, "new": 90.0 - 0.02}
    hi = {"key": K + "/ci[1]", "old": 110.0, "new": 110.0 + 0.02}
    assert _kept([lo], trials=trials) == [] and _kept([hi], trials=trials) == []
    assert _kept([lo, hi], trials=trials) == [lo, hi]  # the width moves by 0.04: beyond the allowance
    assert rerun_check._end_side(K + "/ci[0]") == "lo" and rerun_check._end_side(K + "/interval[1]") == "hi"
    assert rerun_check._end_side(K + "/values[0]") == ""


def test_a_precision_that_names_no_declared_result_excuses_nothing() -> None:
    changes = [{"key": K, "old": 22.8, "new": 22.8001}]
    for precision in ({"target_half_width": 0.1, "metric": "nothing"}, {"target_half_width": 0.1}, {"target_half_width": "x", "metric": "est"}):
        assert _kept(changes, {**PROTOCOL, "precision": precision}) == changes  # PROTOCOL has three metrics: none is "the only one"


def test_the_original_input_is_still_excused_when_the_own_error_covers_it() -> None:
    real = [{"key": K, "old": 22.8138685, "new": 22.8138493}]
    assert _kept(real, trials=_se_trials(0.04)) == []
    trials = _trials([float(i) for i in range(10)], metric="est")  # the delivered run's own error of "s": about 0.96
    tight = {"key": "trials/s/est/mean", "old": 4.5, "new": 4.5 + 0.02}  # beyond a quarter of 0.05, but inside its own error
    narrow = {**PROTOCOL, "precision": {"target_half_width": 0.05, "metric": "est"}}
    assert rerun_check.counted([tight], narrow, trials, redrawn=False)[0] == []


def test_a_run_whose_own_error_is_smaller_than_the_plan_allowance_is_bound_by_its_own_error() -> None:
    trials = _trials([4.5 + 0.001 * i for i in range(10)], metric="est")  # its own error: about 0.001
    move = {"key": "trials/s/est/mean", "old": 4.5045, "new": 4.5045 + 0.01}  # beyond its own error
    wide = {**PROTOCOL, "precision": {"target_half_width": 0.1, "metric": "est"}}
    assert rerun_check.counted([move], wide, trials, redrawn=False)[0] == [move]
    small = {**move, "new": 4.5045 + 0.0002}  # inside its own error too
    assert rerun_check.counted([small], wide, trials, redrawn=False)[0] == []


def test_a_rerun_that_changed_one_result_names_only_that_result_as_changed(tmp_path: Path) -> None:
    # the finding is about `est_a`; `corr_ab` is worked out from it and is left out with it, but the rerun did not show it changed
    out, record = _gate(tmp_path, Verdict("changed", metrics=["est_a"], keys=["trials/s/est_a/mean"]), headline=("est_b",),
                        test=DERIVE_EST, quantity="est_a")
    assert out is None
    gate = record["issues"]["F-1"]["at_gate"]
    assert gate["rerun"] == "changed" and {"est_a", "corr_ab"} <= set(gate["keys"])
    kept = json.loads((tmp_path / "outputs").rglob("RESULT_CHECK.json").__next__().read_text(encoding="utf-8"))
    from core import result_checks

    assert kept["removed_why"]["est_a"] == result_checks.SHOWN_FAULT_CHANGED_WHY
    assert kept["removed_why"]["corr_ab"] == result_checks.SHOWN_FAULT_WHY
    # nothing else left out with it is said changed: the record moved only the estimate's own key, not its intervals
    assert [k for k, w in kept["removed_why"].items() if w == result_checks.SHOWN_FAULT_CHANGED_WHY] == ["est_a"]
    assert "did not establish" in kept["say"] and "`corr_ab`" in kept["say"]


def test_a_key_is_said_changed_only_for_the_finding_whose_own_rerun_moved_it(tmp_path: Path) -> None:
    from core import result_checks
    from tests.test_shown_reach import _engine

    engine = _engine(tmp_path)
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "needs").mkdir(exist_ok=True)
    # two findings leave out the key `k`; only the first one's rerun moved it
    engine._leave_out_shown_faults([("qa", ["k", "m"], "how a"), ("qb", ["k"], "how b")], [], {"k": 1.0, "m": 2.0},
                                   changed={"qa": {"k"}, "qb": set()})
    kept = json.loads((engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    assert kept["removed_why"]["k"] == result_checks.SHOWN_FAULT_CHANGED_WHY and kept["removed_why"]["m"] == result_checks.SHOWN_FAULT_WHY
    notes = {c["quantity"]: c["note"] for c in kept["checks"]}
    assert "did not establish whether the value of `m`" in notes["qa"] and "changes the reported number of `k`" in notes["qa"]
    assert "did not establish" in notes["qb"] and "changes the reported number" not in notes["qb"]
# ---- a constant metric, through the whole rerun (compare, then counted) ---------------------------------------------------------------
CONST_RUN_PY = RUN_PY.replace('[scale * i for i in range(10)]', '[base] * 10').replace('4.5 * scale', 'base').replace(
    "scale = 1.0", "scale = 1.0" + chr(10) + "base = 0.0")


def _constant_judge(tmp_path: Path, new_base: float) -> Verdict:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "run.py").write_text(CONST_RUN_PY, encoding="utf-8")
    (root / "raw").mkdir()
    (root / "data" / "results").mkdir(parents=True)
    cells = [{"key": "s", "ok": 10, "failed": 0, "metrics": {"est": {"values": [0.0] * 10}}}]
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (root / "data" / "results" / "derived.json").write_text(json.dumps({"derived": {"est": {"by_setting": {"s": 0.0}}}}), encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    # the repair changes how numbers are drawn (a new seeding call), so a move must be beyond the Monte Carlo noise to count
    text = (copy / "run.py").read_text(encoding="utf-8").replace("base = 0.0", f"import random{chr(10)}random.seed(7){chr(10)}base = {new_base!r}")
    (copy / "run.py").write_text(text, encoding="utf-8")
    return asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 60, None))


def test_a_real_move_of_a_constant_metric_is_a_changed_verdict_not_an_error(tmp_path: Path) -> None:
    got = _constant_judge(tmp_path, 0.5)
    assert got.status == "changed" and got.metrics == ["est"] and got.redrawn


def test_rounding_noise_of_a_constant_metric_is_an_unchanged_verdict(tmp_path: Path) -> None:
    got = _constant_judge(tmp_path, 1e-15)
    assert got.status == "unchanged" and got.redrawn


# ---- a delivered result file the repaired run did not write is never shipped as "unchanged" ----------------------------------
def _quest_with_two_files(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "derive.py").write_text("x = 1\n", encoding="utf-8")
    res = root / "data" / "results"
    res.mkdir(parents=True)
    (res / "derived.json").write_text(json.dumps({"derived": {"est": {"by_setting": {"s": 1.0}}}}), encoding="utf-8")
    (res / "table.json").write_text(json.dumps({"est_b": 5.0}), encoding="utf-8")
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "derive.py").write_text("x = 2\n", encoding="utf-8")
    return root, copy


def _rerun_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: Path, copy: Path, *, table: bool) -> Verdict:
    out = tmp_path / "out"
    (out / "data" / "results").mkdir(parents=True)
    (out / "data" / "results" / "derived.json").write_text(
        json.dumps({"derived": {"est": {"by_setting": {"s": 1.0}}}}), encoding="utf-8")
    if table:
        (out / "data" / "results" / "table.json").write_text(json.dumps({"est_b": 5.0}), encoding="utf-8")

    async def fake_run_copy(*a: Any, **k: Any) -> tuple[Any, ...]:
        return "analysis only", out, "", 1.0, False

    monkeypatch.setattr(rerun_check, "run_copy", fake_run_copy)
    return asyncio.run(rerun_check.judge(None, "py", root, copy, {"metrics": []}, 60, None))  # disputed=None, as the engine calls


def test_an_unwritten_delivered_result_file_is_could_not_not_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: table.json left the comparison when the rerun did not write it, and the verdict was "unchanged"."""
    root, copy = _quest_with_two_files(tmp_path)
    v = _rerun_writing(tmp_path, monkeypatch, root, copy, table=False)
    assert v.status == "could_not" and "table.json" in v.why and v.unwritten == ["table.json"]


def test_a_rerun_that_compares_every_delivered_file_and_finds_no_move_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known positive."""
    root, copy = _quest_with_two_files(tmp_path)
    v = _rerun_writing(tmp_path, monkeypatch, root, copy, table=True)
    assert v.status == "unchanged" and v.unwritten == []


# ---- a delivered result file is carried or not by FI's record of the step that wrote it (core/step_writes.py) ------------------
def _quest_with_record(tmp_path: Path, wrote: dict[str, str]) -> tuple[Path, Path]:
    root, copy = _quest_with_two_files(tmp_path)
    if wrote:
        (root / ".fi").mkdir()
        (root / ".fi" / "step_writes.json").write_text(json.dumps(wrote), encoding="utf-8")
    return root, copy


def test_a_result_file_the_simulation_wrote_is_carried_by_an_analysis_only_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: table.json, unwritten by the analysis-only rerun, was carried only if no code file named it."""
    root, copy = _quest_with_record(tmp_path, {"derived.json": "analysis", "table.json": "simulation"})
    v = _rerun_writing(tmp_path, monkeypatch, root, copy, table=False)
    assert v.status == "unchanged" and v.carried == ["table.json"] and v.unwritten == []


def test_a_result_file_the_analysis_wrote_stays_uncompared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known positive: the step that was rerun wrote table.json and the rerun did not: still could_not."""
    root, copy = _quest_with_record(tmp_path, {"derived.json": "analysis", "table.json": "analysis"})
    v = _rerun_writing(tmp_path, monkeypatch, root, copy, table=False)
    assert v.status == "could_not" and v.unwritten == ["table.json"] and v.carried == []


def test_a_result_file_no_step_is_recorded_for_is_not_compared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No record at all, or a record without the file: its writer is unknown, so it is never carried (the old code read the
    file name in the code and carried it when no changed file named it)."""
    for wrote in ({}, {"derived.json": "analysis"}):
        root, copy = _quest_with_record(tmp_path / str(len(wrote)), wrote)
        v = _rerun_writing(tmp_path / str(len(wrote)), monkeypatch, root, copy, table=False)
        assert v.status == "could_not" and v.unwritten == ["table.json"] and v.carried == []


def test_a_name_in_the_code_decides_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The old code carried table.json because only report.py named it; the record says the analysis wrote it."""
    root, copy = _quest_with_record(tmp_path, {"derived.json": "analysis", "table.json": "analysis"})
    (root / "code" / "report.py").write_text('open("table.json", "w")\n', encoding="utf-8")
    v = _rerun_writing(tmp_path, monkeypatch, root, copy, table=False)
    assert v.status == "could_not" and v.unwritten == ["table.json"]


def test_a_whole_study_rerun_carries_nothing_it_reran(tmp_path: Path) -> None:
    root, _copy = _quest_with_record(tmp_path, {"derived.json": "analysis", "table.json": "simulation"})
    assert rerun_check.affected_files(root, {"table.json"}, "whole study") == {"table.json"}
    assert rerun_check.affected_files(root, {"table.json"}, "analysis only") == set()


def test_the_record_says_the_last_step_that_wrote_a_file_and_skips_overlapping_runs(tmp_path: Path) -> None:
    from core import step_writes

    root = tmp_path / "quest"
    res = root / "data" / "results"
    res.mkdir(parents=True)
    with step_writes.watching(root) as w:  # simulation writes a.json and b.json; the analysis rewrites b.json and writes c.json
        (res / "a.json").write_text("1", encoding="utf-8")
        (res / "b.json").write_text("1", encoding="utf-8")
        w.mark(step_writes.SIMULATION)
        (res / "b.json").write_text("22", encoding="utf-8")
        (res / "c.json").write_text("1", encoding="utf-8")
        w.mark(step_writes.ANALYSIS)
    assert step_writes.read(root) == {"a.json": "simulation", "b.json": "analysis", "c.json": "analysis"}
    other = tmp_path / "quest2"
    (other / "data" / "results").mkdir(parents=True)
    with step_writes.watching(other) as first:
        with step_writes.watching(other) as second:  # two runs at once: nobody can tell who wrote what
            (other / "data" / "results" / "x.json").write_text("1", encoding="utf-8")
            first.mark(step_writes.ANALYSIS)
            second.mark(step_writes.ANALYSIS)
    assert step_writes.read(other) == {}
    with pytest.raises(RuntimeError):  # a run that stops before its analysis records nothing
        with step_writes.watching(other) as w:
            (other / "data" / "results" / "y.json").write_text("1", encoding="utf-8")
            raise RuntimeError
    assert step_writes.read(other) == {}


def test_a_changed_rerun_adds_the_numbers_of_an_unwritten_affected_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: a changed rerun with an unwritten file added none of that file's numbers to what the finding reaches."""
    root, copy = _quest_with_record(tmp_path, {"derived.json": "analysis", "table.json": "analysis"})
    out = tmp_path / "out"
    (out / "data" / "results").mkdir(parents=True)
    (out / "data" / "results" / "derived.json").write_text(
        json.dumps({"derived": {"est": {"by_setting": {"s": 9.0}}}}), encoding="utf-8")

    async def fake_run_copy(*a: Any, **k: Any) -> tuple[Any, ...]:
        return "analysis only", out, "", 1.0, False

    monkeypatch.setattr(rerun_check, "run_copy", fake_run_copy)
    v = asyncio.run(rerun_check.judge(None, "py", root, copy, {"metrics": []}, 60, None))
    assert v.status == "changed" and v.unwritten == ["table.json"]
    assert any(k.startswith("file:table.json") for k in v.keys) and any(k.startswith("file:table.json") for k in v.unplaced)


# ---- nested steps are one run; an unattributable period forgets old writers ----------------------------------------------------


def _results(root: Path) -> Path:
    res = root / "data" / "results"
    res.mkdir(parents=True, exist_ok=True)
    return res


def test_steps_started_inside_a_watched_run_are_the_same_run_and_still_record(tmp_path: Path) -> None:
    from core import outside_run, step_writes

    root = tmp_path / "quest"
    res = _results(root)
    with step_writes.watching(root) as w:
        inner = outside_run.begin_run(root)  # what OutsideExecutor / execute_keeping_only_finished do for each setting
        start = outside_run.working_states(root)
        (res / "a.json").write_text("1", encoding="utf-8")
        assert outside_run.run_overlapped(root, inner) is False
        recorded = outside_run.record_finished(root, start, outside_run.run_overlapped(root, inner))
        outside_run.end_run(root, inner)
        w.mark(step_writes.SIMULATION)
        w.mark(step_writes.ANALYSIS)
    assert "data/results/a.json" in recorded  # the finished-files record works inside the watched run
    assert step_writes.read(root) == {"a.json": "simulation"}  # and so does the step record


def test_settings_at_the_same_time_inside_one_run_still_overlap_each_other(tmp_path: Path) -> None:
    from core import outside_run, step_writes

    root = tmp_path / "quest"
    _results(root)
    with step_writes.watching(root) as w:
        a = outside_run.begin_run(root)
        b = outside_run.begin_run(root)
        assert outside_run.run_overlapped(root, a) and outside_run.run_overlapped(root, b)
        outside_run.end_run(root, b)
        outside_run.end_run(root, a)
        w.mark(step_writes.ANALYSIS)


def test_a_run_beside_a_watched_run_overlaps_it_and_its_old_writers_are_forgotten(tmp_path: Path) -> None:
    from core import outside_run, step_writes

    root = tmp_path / "quest"
    res = _results(root)
    (root / ".fi").mkdir()
    (root / ".fi" / "step_writes.json").write_text(json.dumps({"table.json": "simulation", "keep.json": "simulation"}), encoding="utf-8")
    (res / "table.json").write_text("1", encoding="utf-8")
    (res / "keep.json").write_text("1", encoding="utf-8")
    with step_writes.watching(root) as w:
        beside = contextvars.Context().run(outside_run.begin_run, root)  # another run in the quest, outside the watched one's scope
        (res / "table.json").write_text("22", encoding="utf-8")  # rewritten by the analysis, but it cannot be told
        w.mark(step_writes.SIMULATION)
        w.mark(step_writes.ANALYSIS)
        outside_run.end_run(root, beside)
    # the record keeps nothing for the file that period may have touched, so an analysis-only rerun does not carry it
    assert step_writes.read(root) == {"keep.json": "simulation"}
    assert rerun_check.affected_files(root, {"table.json"}, "analysis only") == {"table.json"}
    assert rerun_check.affected_files(root, {"keep.json"}, "analysis only") == set()


def test_a_run_that_stops_early_forgets_the_writers_of_the_files_it_changed(tmp_path: Path) -> None:
    from core import step_writes

    root = tmp_path / "quest"
    res = _results(root)
    (root / ".fi").mkdir()
    (root / ".fi" / "step_writes.json").write_text(json.dumps({"table.json": "simulation"}), encoding="utf-8")
    (res / "table.json").write_text("1", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with step_writes.watching(root) as w:
            (res / "table.json").write_text("22", encoding="utf-8")
            w.mark(step_writes.SIMULATION)
            raise RuntimeError
    assert step_writes.read(root) == {}


# ---- a rerun that takes long says where it is in run.log ---------------------------------------------------------------------------
def _slow_quest(tmp_path: Path, qid: str) -> tuple[Path, Path]:
    """A quest folder whose study has three settings that take about a second each, and a copy of its code with one file changed."""
    root = tmp_path / qid
    code = root / "code"
    code.mkdir(parents=True)
    (code / "run.py").write_text(code_project.RUN_SOURCE, encoding="utf-8")
    (code / "study.json").write_text(json.dumps({"entry": "run_cell", "grid": {"a": [1, 2, 3]}, "runs_per_setting": 1}), encoding="utf-8")
    (code / "simulate.py").write_text("import time\n\n\ndef run_cell(cell):\n    time.sleep(1.0)\n    return {'est': float(cell['a'])}\n",
                                      encoding="utf-8")
    (code / "experiment.py").write_text("import json, pathlib\np = pathlib.Path('data/results')\np.mkdir(parents=True, exist_ok=True)\n"
                                        "(p / 'derived.json').write_text(json.dumps({'derived': {'est': {'by_setting': {'x': 1.0}}}}))\n",
                                        encoding="utf-8")
    (root / "raw").mkdir()
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": []}), encoding="utf-8")
    (root / "data" / "results").mkdir(parents=True)
    (root / "data" / "results" / "derived.json").write_text(json.dumps({"derived": {"est": {"by_setting": {"x": 1.0}}}}), encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "simulate.py").write_text((copy / "simulate.py").read_text(encoding="utf-8") + "\n# repaired\n", encoding="utf-8")
    return root, copy


def test_a_long_rerun_says_in_run_log_which_finding_it_is_and_how_far_it_has_got(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import _close_quest_logger, _quest_logger

    qid = "q_progress"
    root, copy = _slow_quest(tmp_path, qid)
    monkeypatch.setattr(rerun_check, "PROGRESS_EVERY_S", 1.2)
    _quest_logger(qid, root / ".fi")
    try:
        asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, PROTOCOL, 120, None, source="a repair", finding="F-9"))
    finally:
        _close_quest_logger(qid)
    text = (root / ".fi" / "run.log").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if "rerun of finding F-9" in ln]
    assert lines, text
    assert any("of 3 settings done" in ln for ln in lines), lines
    print("\n".join(lines))
