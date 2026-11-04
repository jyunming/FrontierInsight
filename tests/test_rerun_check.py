"""Whether a fault a failing test showed changes a reported number is shown by running the study with a repair of it
(core/rerun_check.py), and the gate acts on that answer (Engine._shown_fault_outcome). Neutral made-up studies; no model is
called and no real study runs: the rerun's answer is given, or the study is a few lines of Python."""
from __future__ import annotations

import asyncio
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


def test_a_standard_error_or_a_coverage_is_not_called_noise_when_it_moves_by_the_error_of_a_mean() -> None:
    values = [float(i % 7) for i in range(50)]
    mean = sum(values) / 50
    se = rerun_check.trial_stats(_trials(values))[0][("s", "est")][2] / math.sqrt(50)
    est = {"key": "derived/derived/est/by_setting/s", "old": mean, "new": mean + 1.5 * se}
    err = {"key": "derived/derived/est/by_setting/s/se", "old": se, "new": se + 1.5 * se}
    cov = {"key": "derived/derived/est/by_setting/s/coverage", "old": 0.95, "new": 0.96}
    end = {"key": "derived/derived/est/by_setting/s/hi", "old": mean + 2 * se, "new": mean + 2 * se + 1.5 * se}
    kept = rerun_check.counted([est, err, cov, end], PROTOCOL, _trials(values), redrawn=True)[0]
    assert [c["key"] for c in kept] == [err["key"], cov["key"]]  # a mean and an interval end move with the mean's error
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
    assert issue["rerun_unchanged"] is True and issue["changes_result"] is False and "at_gate" not in issue


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
