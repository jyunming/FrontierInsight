"""Once a rerun of the whole study, with a repair of one finding, did not finish in all the time a rerun is allowed, FI does not
run the whole study again for another finding in the same check: a repair of another line does not make the study shorter.
That finding's answer is "could not rerun" and says why; it is judged as a finding no rerun answered. Neutral made-up study
(tests/test_rerun_check.py); no model is called and nothing long is run (the rerun itself is replaced)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core import rerun_check
from core.rerun_check import Verdict


def _two_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: Verdict) -> tuple[Any, dict[str, Any], list[Any], list[float]]:
    from tests.test_rerun_check import _engine_with_code

    engine, root, issue, record = _engine_with_code(tmp_path)
    engine.config.execution.timeout_s = 900  # a rerun is allowed 1.5 x 3 x 30 s = 135 s
    issue["file"] = record["issues"]["F-1"]["file"] = "simulate.py"
    (root / "code" / "simulate.py").write_text("def run_cell(cell):\n    return {'q': 1.0}\n", encoding="utf-8")
    other = {**issue, "id": "F-2"}
    record["issues"]["F-2"] = dict(other)
    judged: list[float] = []

    async def judge(*a: Any, **k: Any) -> Verdict:
        judged.append(a[5])
        return answer

    async def ask(issue: Any, *a: Any, **k: Any) -> dict[str, str]:  # each finding gets its own repaired code
        return {"simulate.py": f"def run_cell(cell):\n    return {{'q': 1.0}}  # repaired for {issue['id']}\n"}

    async def passes(issue: Any, copy_dir: Path) -> tuple[bool, str]:
        return True, ""

    async def no_refusal(*a: Any, **k: Any) -> str:
        return ""

    monkeypatch.setattr(rerun_check, "judge", judge)
    engine._ask_repair_on_copy = ask  # type: ignore[method-assign,assignment]
    engine._repair_passes_own_test = passes  # type: ignore[method-assign,assignment]
    engine._repair_small_run_refusal = no_refusal  # type: ignore[method-assign,assignment]
    record["rerun_calls"] = 0
    return engine, record, [issue, other], judged


def test_after_a_whole_study_rerun_ran_out_of_all_its_time_another_finding_is_not_run_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # on the old code: the second finding's rerun was started too, and ran out of its time again (two 8 h runs in one study)
    engine, record, (first, second), judged = _two_findings(
        tmp_path, monkeypatch, Verdict("could_not", "the repaired study did not finish within 135 s", mode="whole study",
                                       timed_out=True))
    del engine._rerun_shown  # the fixture's "no reruns": the real one is what is judged here
    got = asyncio.run(engine._rerun_shown([(first, "its test still fails on it"), (second, "its test still fails on it")],
                                          record, {}))
    assert got["F-1"].status == "could_not" and got["F-1"].timed_out
    assert len(judged) == 1  # the study ran once
    two = got["F-2"]
    assert two.status == "could_not" and "did not finish within 135 s" in two.why and "F-1" in two.why
    assert "does not run it again for another finding" in two.why
    assert "does not run it again for another finding" in record["issues"]["F-2"]["rerun"]["why"]  # kept for the paper


def test_a_rerun_cut_short_by_the_time_left_in_the_check_does_not_stop_the_next_one(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # known positive: the time out was the check's, not the study's, so a later rerun may still fit
    engine, record, (first, second), judged = _two_findings(
        tmp_path, monkeypatch, Verdict("could_not", "did not finish", mode="whole study", timed_out=True))
    timed_out: dict[str, Any] = {}
    asyncio.run(engine._rerun_one(first, record, {}, {}, [70.0], False, timed_out))
    assert not timed_out
    asyncio.run(engine._rerun_one(second, record, {}, {}, [400.0], False, timed_out))
    assert len(judged) == 2


def test_a_rerun_that_finished_does_not_stop_the_next_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, record, (first, second), judged = _two_findings(tmp_path, monkeypatch, Verdict("unchanged", mode="whole study"))
    budget, timed_out = [270.0], {}
    asyncio.run(engine._rerun_one(first, record, {}, {}, budget, False, timed_out))
    asyncio.run(engine._rerun_one(second, record, {}, {}, budget, False, timed_out))
    assert len(judged) == 2 and not timed_out


def test_a_kept_answer_that_ran_out_of_all_its_time_also_keeps_the_other_findings_from_running_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a check resumed after the long rerun: its answer is read from the record (not run again), and it counts the same way
    engine, record, (first, second), judged = _two_findings(
        tmp_path, monkeypatch, Verdict("could_not", "the repaired study did not finish within 135 s", mode="whole study",
                                       timed_out=True))
    asyncio.run(engine._rerun_one(first, record, {}, {}, [400.0], False, {}))  # the earlier check: the study ran out of time
    assert len(judged) == 1 and record.get("rerun_could_not")
    timed_out: dict[str, Any] = {}  # the next check
    again = asyncio.run(engine._rerun_one(first, record, {}, {}, [400.0], False, timed_out))
    assert again.timed_out and len(judged) == 1  # the kept answer, not a new run
    assert timed_out.get("finding") == "F-1"
    two = asyncio.run(engine._rerun_one(second, record, {}, {}, [400.0], False, timed_out))
    assert len(judged) == 1 and two.status == "could_not" and "does not run it again for another finding" in two.why


def test_a_kept_answer_for_a_crash_does_not_keep_the_other_findings_from_running(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # known positive: a repair that crashed says nothing about how long the study takes
    engine, record, (first, second), judged = _two_findings(
        tmp_path, monkeypatch, Verdict("could_not", "the repaired simulation does not run", mode="whole study", crashed=True))
    timed_out: dict[str, Any] = {}
    asyncio.run(engine._rerun_one(first, record, {}, {}, [400.0], False, timed_out))
    asyncio.run(engine._rerun_one(second, record, {}, {}, [400.0], False, timed_out))
    assert not timed_out and len(judged) >= 2
