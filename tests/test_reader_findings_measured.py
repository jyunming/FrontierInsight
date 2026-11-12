"""A second model's word that a quantity is computed differently from the plan keeps a result out of the paper only after a
test FI asked for shows the difference and a rerun of the study with a repair that passes it moves a number
(core/engine.py::_measure_reader_findings). Neutral made-up study; the model's test and the rerun are replaced by the answer they
would give (what the gate does with each answer is judged)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from core import code_review
from core import paper_facts as pf
from core import result_checks, rerun_check
from core.rerun_check import Verdict
from tests.test_shown_reach import RESULT, _engine, _quest

ISSUE = {"id": "F-9", "file": "derive.py", "function": "", "quantity": "cover", "kind": "quantity", "changes_result": True,
         "demonstrated": None, "outcome": "open", "text": "cover is computed another way", "severity": "high"}
NOTE = "the plan's dependent variable `cover` is computed differently from the plan in the code (x)"
CHECK = {"kind": "code_reader", "note": NOTE, "quantity": "cover", "keys": ["cover"]}
SAY = ("A second model compared the code with the plan and found that `cover` is computed differently from the plan (x); the "
       "repairs did not remove it. Do NOT report `cover`.")


def _setup(tmp_path: Path, monkeypatch: Any, *, shown: bool, verdict: Verdict | None, calls: list[Any],
           extra_checks: list[dict[str, Any]] | None = None) -> tuple[Any, dict[str, Any]]:
    engine = _engine(tmp_path)
    root = _quest(engine)
    rc = {"removed_quantities": ["cover"], "removed_figures": [], "removed_values": {"cover": [0.94]},
          "checks": [dict(CHECK), *(extra_checks or [])], "note": NOTE, "say": SAY}
    (root / "needs" / "RESULT_CHECK.json").write_text(json.dumps(rc), encoding="utf-8")
    monkeypatch.setattr(code_review, "differing_quantities", lambda *a, **k: [
        {"name": "cover", "fid": "F-9", "kind": "dependent variable", "status": "different", "note": "x"}])

    async def demonstrate(issue: Any, record: Any, state: Any) -> bool:
        calls.append("demonstrate")
        if shown:
            issue.update(demonstrated=True, test=".fi/finding_tests/F-9_1.py")
        return shown

    async def answers(verdicts: Any, record: Any, state: Any, strict: bool = False) -> dict[str, Any]:
        calls.append(("rerun", strict, [i.get("disputed") for i, _h in verdicts]))
        return {} if verdict is None else {str(i["id"]): verdict for i, _h in verdicts}

    engine._demonstrate_reader_finding = demonstrate  # type: ignore[method-assign,assignment]
    engine._rerun_shown = answers  # type: ignore[method-assign,assignment]
    return engine, {"issues": {"F-9": dict(ISSUE)}}


def _measure(engine: Any, record: dict[str, Any], headline: list[str], *, reach: bool = True) -> Any:
    from core import frozen_protocol, shown_reach

    root = engine.quest_root
    got = shown_reach.Reach(root, frozen_protocol.protocol_of(root), headline, []) if reach else None
    state = {"design": None, "result_json": RESULT, "exec_result": {"returncode": 0}}
    return asyncio.run(engine._measure_reader_findings(record, state, got, headline))


def _rc(root: Path) -> dict[str, Any]:
    return json.loads((root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))


def test_a_shown_difference_whose_passing_repair_moves_no_number_gives_the_result_back_and_says_the_method_differs(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("unchanged", mode="analysis only"), calls=calls)
    assert _measure(engine, record, ["est_a"]) is None
    assert calls == ["demonstrate", ("rerun", True, [["cover"]])]  # the repair is held to the test; the reader's quantity is the disputed one
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == [] and rc["checks"] == [] and "Do NOT report" not in rc["say"]
    assert rc["kept_differences"][0]["measured"]["status"] == "unchanged" and "FI observed this" in rc["say"] and "do not say that the plan's definition was run" in rc["say"]
    items = [i for i in pf.required_items(engine.quest_root) if i.kind == pf.METHOD_DIFFERS]
    assert items and "failed on the code that produced the results" in items[0].sentence
    assert "plan's definition gave" not in rc["say"] and "run again with the plan" not in items[0].sentence
    assert "cover" not in result_checks.left_out_by_record(engine.quest_root)


def test_a_difference_no_test_shows_is_not_a_reason_to_keep_a_result_out(tmp_path, monkeypatch) -> None:
    """Outcome changed against the first version of the fix: no test could be written, so no rerun is made and the claim stands
    unshown; the result stays and the paper says so (a word alone never withholds)."""
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=False, verdict=Verdict("unchanged"), calls=calls)
    assert _measure(engine, record, ["cover"]) is None
    assert calls == ["demonstrate"]  # nothing is rerun and nothing is stopped
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == [] and rc["kept_differences"][0]["measured"]["status"] == "not_shown"
    assert "none did" in rc["say"]
    items = [i for i in pf.required_items(engine.quest_root) if i.kind == pf.METHOD_DIFFERS]
    assert items and "no test FI wrote could show it" in items[0].sentence


def test_a_shown_difference_that_moves_a_number_takes_out_what_moved_and_what_is_worked_out_from_it(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("changed", metrics=["cover", "corr_ab"]), calls=calls)
    assert _measure(engine, record, ["est_a"]) is None  # est_a, the main result, did not move
    rc = _rc(engine.quest_root)
    assert "cover" in rc["removed_quantities"] and "corr_ab" in rc["removed_quantities"] and "kept_differences" not in rc
    assert "est_a" not in rc["removed_quantities"]


def test_a_changed_main_result_stops_with_the_records_and_without_them(tmp_path, monkeypatch) -> None:
    for number, use_reach in enumerate((True, False)):
        calls: list[Any] = []
        engine, record = _setup(tmp_path / str(number), monkeypatch, shown=True, verdict=Verdict("changed", metrics=["est_a"]), calls=calls)
        out = _measure(engine, record, ["est_a"], reach=use_reach)
        assert out and out["stuck_reason"] == "shown_fault_open" and "est_a" in out["rationale"]
        assert "cover" in _rc(engine.quest_root)["removed_quantities"]


def test_a_difference_that_could_not_be_rerun_stays_out_and_the_record_says_nothing_measured_it(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("could_not", why="the study runs in Docker"), calls=calls)
    assert _measure(engine, record, ["cover"]) is None  # never a stop on a reading alone
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and rc["checks"][0]["measured"]["status"] == "could_not"


def test_a_key_another_check_still_takes_out_is_not_said_to_be_kept(tmp_path, monkeypatch) -> None:
    """The answers block, the Limitations and the writer's text must agree: the other check's withholding wins."""
    other = {"kind": "shown_fault", "note": "a test shows cover is wrong", "quantity": "cover_other", "keys": ["cover"]}
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("unchanged"), calls=calls, extra_checks=[other])
    _measure(engine, record, ["est_a"])
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == ["cover"] and not rc.get("kept_differences")
    assert "Report it" not in rc["say"] and "stays in the results" not in rc["say"]
    assert not [i for i in pf.required_items(engine.quest_root) if i.kind == pf.METHOD_DIFFERS]
    assert "cover" in result_checks.left_out_by_record(engine.quest_root)


# ---- the repair's own file marker: only prose right before an opening fence ----------------------------------------------------
def test_a_file_marker_in_prose_right_before_the_fence_names_the_file() -> None:
    for reply in ("# file: derive.py\n```python\nx = 1\n```\n", "Here it is.\n\n# file: derive.py\n\n```python\nx = 1\n```\n",
                  "```python\n# file: derive.py\nx = 1\n```\n"):
        assert code_review.parse_files(reply, ["derive.py"]) == {"derive.py": "x = 1\n"}
    assert code_review.parse_files("```python\nx = 1\n```\n", ["derive.py"]) == {}


def test_a_marker_like_line_inside_a_fence_is_code_and_never_moved() -> None:
    reply = "```python\n# file: a.py\nx = 1\n# file: b.py\n```\n"
    assert code_review.parse_files(reply, ["a.py", "b.py"]) == {"a.py": "x = 1\n# file: b.py\n"}
    reply = '```python\n# file: a.py\ns = """\n# file: b.py\n```x\n"""\n```\n'
    assert code_review.parse_files(reply, ["a.py", "b.py"]) == {}  # a fence inside a string ends the block, as before: never rewritten
    reply = '```python\n# file: a.py\ns = """\n# file: b.py\n"""\n```\n'
    assert code_review.parse_files(reply, ["a.py", "b.py"]) == {"a.py": 's = """\n# file: b.py\n"""\n'}


# ---- the rerun of a reader's claim -------------------------------------------------------------------------------------------------
def test_a_precision_on_a_result_worked_out_afterwards_is_the_plans_precision() -> None:
    plan = {"metrics": [{"id": "est", "kind": "mean"}], "derived": [{"id": "limit"}],
            "precision": {"target_half_width": 1e-4, "metric": "limit"}}
    assert rerun_check.declared_precision(plan) == ("limit", 1e-4)
    base = {"derived/derived/limit": 22.81384918}
    changes = rerun_check.compare(base, {"derived/derived/limit": 22.81384926}, 1e-9)
    # a result worked out afterwards has no standard error in the trial record: nothing excuses its move, and the verdict says so
    kept, notes = rerun_check.counted(changes, plan, None, False)
    assert changes and kept == changes and any("limit has no standard error" in x for x in notes)
    moved = rerun_check.compare(base, {"derived/derived/limit": 22.9}, 1e-9)
    assert len(rerun_check.counted(moved, plan, None, False)[0]) == 1  # a real move still counts


def _judge_quest(tmp_path: Path, write_table: bool) -> tuple[Path, Path]:
    from tests.test_rerun_check import _delivered

    root = tmp_path / "quest"
    _delivered(root, 1.0)
    (root / "data" / "results" / "table.json").write_text(json.dumps({"cover": 0.5}), encoding="utf-8")
    line = "(out / 'data' / 'results' / 'table.json').write_text(json.dumps({'cover': 0.5}))\n"
    (root / "code" / "run.py").write_text((root / "code" / "run.py").read_text(encoding="utf-8").replace("sys.exit(1)", line + "sys.exit(1)"),
                                          encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    text = (copy / "run.py").read_text(encoding="utf-8")
    (copy / "run.py").write_text((text if write_table else text.replace(line, "")) + "\n# a repair\n", encoding="utf-8")
    return root, copy


def _run_judge(root: Path, copy: Path, **kw: Any) -> Verdict:
    from tests.test_rerun_check import PROTOCOL as P, _Local

    return asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, P, 60, None, **kw))


def test_a_result_file_the_repair_stopped_writing_is_unmeasured_for_a_reader_repair(tmp_path) -> None:
    """Before: the file's numbers were dropped from the comparison and the other, unchanged numbers made it `unchanged`."""
    root, copy = _judge_quest(tmp_path, write_table=False)
    shown = _run_judge(root, copy)  # a shown fault (no disputed quantity) is not "unchanged" either: the file was not compared
    assert shown.status == "could_not" and "table.json" in shown.why
    got = _run_judge(root, copy, disputed=["cover"])
    assert got.status == "could_not" and "did not compare every delivered number of cover" in got.why
    root, copy = _judge_quest(tmp_path / "b", write_table=True)
    assert _run_judge(root, copy, disputed=["cover"]).status == "unchanged"  # compared, and the same
    assert _run_judge(root, copy, disputed=["nothing"]).status == "could_not"  # a quantity nothing holds was not compared


def test_a_repair_that_loses_a_delivered_number_is_not_a_measurement_for_a_reader_repair(tmp_path) -> None:
    from tests.test_rerun_check import PROTOCOL as P, _Local, _delivered

    root = tmp_path / "quest"
    _delivered(root, 1.0)
    copy = rerun_check.copy_code(root, tmp_path / "w")
    text = (copy / "run.py").read_text(encoding="utf-8").replace('{"est": {"by_setting": {"s": 4.5 * scale}}}', "{}")
    (copy / "run.py").write_text(text, encoding="utf-8")
    assert asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, P, 60, None)).status == "changed"
    new = asyncio.run(rerun_check.judge(_Local(), sys.executable, root, copy, P, 60, None, disputed=["est"]))
    assert new.status == "could_not" and "gave no number" in new.why and new.crashed


def test_one_setting_in_a_file_the_repair_did_not_write_leaves_the_answer_open(tmp_path) -> None:
    """Setting `s` of the disputed quantity is in derived.json and unchanged; setting `b` was in table.json, which the repair stopped writing."""
    from tests.test_rerun_check import _delivered

    root = tmp_path / "quest"
    _delivered(root, 1.0)
    (root / "data" / "results" / "table.json").write_text(json.dumps({"est": {"b": 1.0}}), encoding="utf-8")
    line = "(out / 'data' / 'results' / 'table.json').write_text(json.dumps({'est': {'b': 1.0}}))\n"
    (root / "code" / "run.py").write_text((root / "code" / "run.py").read_text(encoding="utf-8").replace("sys.exit(1)", line + "sys.exit(1)"),
                                          encoding="utf-8")
    copy = rerun_check.copy_code(root, tmp_path / "w")
    (copy / "run.py").write_text((copy / "run.py").read_text(encoding="utf-8").replace(line, "") + "\n# a repair\n", encoding="utf-8")
    got = _run_judge(root, copy, disputed=["est"])
    assert got.status == "could_not" and "file:table.json/est/b" in got.why


def test_a_kept_answer_is_reused_only_when_it_was_made_the_same_way(tmp_path) -> None:
    from core import code_review as cr
    from core.engine import Engine

    engine = _engine(tmp_path)
    root = _quest(engine)
    ran: list[bool] = []

    async def one(issue: Any, record: Any, state: Any, done: Any, budget: Any = None, strict: bool = False) -> Verdict:
        ran.append(strict)
        return Verdict("unchanged")

    engine._rerun_one = one  # type: ignore[method-assign,assignment]
    record = {"issues": {"F-9": dict(ISSUE)}}
    plain = {**ISSUE, "id": "F-9"}
    strict = {**plain, "disputed": ["cover"]}
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    assert ran == [False]
    asyncio.run(Engine._rerun_shown(engine, [(strict, "")], record, {}, strict=True))
    assert ran == [False, True]  # an ordinary answer is never reused for a reader's claim
    asyncio.run(Engine._rerun_shown(engine, [(strict, "")], record, {}, strict=True))
    assert ran == [False, True]  # the same way, the same disputed quantity: reused
    asyncio.run(Engine._rerun_shown(engine, [({**plain, "disputed": ["other"]}, "")], record, {}, strict=True))
    assert ran == [False, True, True]  # another disputed set is asked again
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    assert ran == [False, True, True]  # a strict answer also stands for an ordinary one
    assert cr.version(root / "code")


def test_a_kept_answer_follows_every_file_of_the_study_and_only_a_measured_one_is_kept(tmp_path) -> None:
    from core.engine import Engine

    engine = _engine(tmp_path)
    root = _quest(engine)
    ran: list[str] = []
    status = ["unchanged"]

    async def one(issue: Any, record: Any, state: Any, done: Any, budget: Any = None, strict: bool = False) -> Verdict:
        ran.append(status[0])
        return Verdict(status[0])

    engine._rerun_one = one  # type: ignore[method-assign,assignment]
    record = {"issues": {"F-9": dict(ISSUE)}}
    plain = {**ISSUE, "id": "F-9"}
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    assert ran == ["unchanged"], "the same files: reused"
    (root / "inputs").mkdir(exist_ok=True)
    (root / "inputs" / "table.csv").write_text("a\n1\n", encoding="utf-8")  # the person's file: the old key never saw it
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    assert ran == ["unchanged", "unchanged"], "a changed inputs/ file asks again"
    (root / "inputs" / "table.csv").write_text("a\n2\n", encoding="utf-8")
    (root / "data" / "results").mkdir(parents=True, exist_ok=True)
    (root / "data" / "results" / "derived.json").write_text("{}", encoding="utf-8")  # what the study delivered: the comparison's base
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    assert ran == ["unchanged"] * 3, "a changed delivered record asks again"
    (root / "data" / "results" / "derived.json").write_text('{"a": 1}', encoding="utf-8")
    status[0] = "could_not"
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    asyncio.run(Engine._rerun_shown(engine, [(plain, "")], record, {}))
    assert ran == ["unchanged"] * 3 + ["could_not", "could_not"], "a could-not answer is never reused"


def test_without_records_every_changed_number_is_left_out_and_the_headline_still_stops(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("changed", metrics=["cover", "corr_ab"]), calls=calls)
    assert _measure(engine, record, ["est_a"], reach=False) is None
    removed = _rc(engine.quest_root)["removed_quantities"]
    assert "cover" in removed and "corr_ab" in removed and "est_a" not in removed


def test_a_changed_number_no_result_owns_is_also_placed_by_what_the_finding_reaches(tmp_path, monkeypatch) -> None:
    from core import shown_reach

    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("changed", metrics=["cover"], unplaced=["file:x/y"]), calls=calls)
    monkeypatch.setattr(shown_reach.Reach, "effect", lambda self, rec, issue, result=None: shown_reach.Effect(left_out=["corr_ab"]))
    assert _measure(engine, record, ["est_a"]) is None
    assert "corr_ab" in _rc(engine.quest_root)["removed_quantities"]
    calls2: list[Any] = []
    engine, record = _setup(tmp_path / "b", monkeypatch, shown=True, verdict=Verdict("changed", metrics=["cover"]), calls=calls2)
    assert _measure(engine, record, ["est_a"]) is None  # nothing unplaced: only what the changed number reaches
    assert "corr_ab" not in _rc(engine.quest_root)["removed_quantities"]


def test_a_test_request_that_could_not_be_made_is_undecided_never_not_shown(tmp_path, monkeypatch) -> None:
    from core import finding_repair

    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=False, verdict=None, calls=calls)

    async def not_asked(issue: Any, record: Any, state: Any) -> bool:
        calls.append("demonstrate")
        raise finding_repair.NotAsked("the request could not be made (RuntimeError: no answer)")

    engine._demonstrate_reader_finding = not_asked  # type: ignore[method-assign,assignment]
    assert _measure(engine, record, ["est_a"]) is None
    rc = _rc(engine.quest_root)
    assert calls == ["demonstrate"]
    assert rc["removed_quantities"] == ["cover"] and rc["checks"][0]["measured"]["status"] == "could_not"
    assert rc["checks"][0]["measured"]["why"].startswith("not asked")


def test_after_a_test_request_not_made_the_next_pass_decides_as_usual(tmp_path, monkeypatch) -> None:
    from core import finding_repair

    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=False, verdict=None, calls=calls)
    first = engine._demonstrate_reader_finding

    async def not_asked(issue: Any, record: Any, state: Any) -> bool:
        raise finding_repair.NotAsked("the request could not be made (RuntimeError: no answer)")

    engine._demonstrate_reader_finding = not_asked  # type: ignore[method-assign,assignment]
    assert _measure(engine, record, ["est_a"]) is None
    engine._demonstrate_reader_finding = first  # type: ignore[method-assign,assignment]
    assert _measure(engine, record, ["est_a"]) is None  # answered now: no test shows it
    rc = _rc(engine.quest_root)
    assert rc["removed_quantities"] == [] and rc["kept_differences"][0]["measured"]["status"] == "not_shown"


def test_a_rerun_whose_repair_request_was_not_made_is_asked_again_on_the_same_code(tmp_path, monkeypatch) -> None:
    calls: list[Any] = []
    engine, record = _setup(tmp_path, monkeypatch, shown=True, verdict=Verdict("could_not", why="not asked: x", mode="not_asked"),
                            calls=calls)
    assert _measure(engine, record, ["est_a"]) is None
    rc = _rc(engine.quest_root)
    assert rc["checks"][0]["measured"]["status"] == "could_not" and "version" not in rc["checks"][0]["measured"]
    assert _measure(engine, record, ["est_a"]) is None
    assert [c for c in calls if c != "demonstrate"] == [("rerun", True, [["cover"]])] * 2, "the same code is asked again"
