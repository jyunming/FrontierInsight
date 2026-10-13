"""Replay of three real quests whose code reader's findings were repaired badly (tests/fixtures/replay/code_findings_*/):
the same issue came back reworded under a cap of twelve, repairs fixed about a third of what they were sent and re-broke
code, a finding the reader got wrong was sent to repair anyway, and a contradiction inside the plan was sent to the code. The
code and the findings are the real last versions; the models are scripted from files in the fixtures (their names live there,
not here). FI now keeps every finding by a stable id, asks for a failing test of each before repairing it, repairs one
function at a time, keeps a change only when nothing that passed before fails, and puts a contradiction of the plan to the
plan's model."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from core import code_findings as cf
from core import code_review as cr
from core import finding_repair as fr
from tests.test_code_review import DESIGN, PROTOCOL, STATE, WITH_PLAN, _engine

pytestmark = pytest.mark.usefixtures("this_python")
FIXTURES = Path(__file__).parent / "fixtures" / "replay"
SMALL = FIXTURES / "code_findings_small"  # a package of four small functions
CONFLICT = FIXTURES / "code_findings_conflict"  # a plan that defines one quantity two ways
WIDE = FIXTURES / "code_findings_wide"  # 35 findings in six readings, a list that rotates


def _info(fixture: Path) -> dict[str, Any]:
    return json.loads((fixture / "scripted" / "info.json").read_text(encoding="utf-8"))


def _script(fixture: Path, name: str) -> str:
    return (fixture / "scripted" / f"{name}.txt").read_text(encoding="utf-8")


INFO = _info(SMALL)
FIRST, SECOND = INFO["first"], INFO["second"]
PACKAGE = INFO["package"]
MODEL_PY = Path(PACKAGE) / INFO["module"]
FIRST_TEST, FIRST_GOOD, FIRST_REVERTED = _script(SMALL, "first_test"), _script(SMALL, "first_good"), _script(SMALL, "first_reverted")
SECOND_TEST, SECOND_GOOD = _script(SMALL, "second_test"), _script(SMALL, "second_good")


def _record(fixture: Path) -> dict[str, Any]:
    return json.loads((fixture / "code_review.json").read_text(encoding="utf-8"))


def _install(engine: Any, fixture: Path) -> Path:
    """The real last version of a quest's code in the engine's quest folder."""
    code = engine.quest_root / "code"
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(fixture / "code", code, ignore=shutil.ignore_patterns("__pycache__"))
    return code


def _finding(problem: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """A real logic finding of the fixture, as the reader's reply gives it."""
    return {"where": problem["where"], "what": problem["text"], "why": problem.get("note", ""), "fix": problem.get("fix", ""),
            "severity": problem.get("severity", "medium"), "quantity": problem.get("name", ""),
            "changes_result": True, **extra}


def _logic(fixture: Path, pid: str) -> dict[str, Any]:
    return next(p for p in _record(fixture)["problems"] if p["id"] == pid)


def _block(text: str) -> str:
    return f"```python\n{text}```"


def _in(function: str) -> str:
    return f"the function `{function}`"


class Models:
    """The reader, the code model and the other model in one fake client. ``findings(n, code)`` gives the findings of reading
    n; ``verdicts(reqs)`` the requirement verdicts (default: all implemented); ``test(prompt)`` the test file written for a
    request; ``repair(who, prompt)`` the reply to a repair request (``who`` is the model asked: ``code`` or ``other``)."""

    def __init__(self, findings: Callable[[int, str], list[dict[str, Any]]], test: Callable[[str], str],
                 repair: Callable[[str, str], str], verdicts: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None) -> None:
        self.findings, self.test, self.repair, self.verdicts = findings, test, repair, verdicts
        self.readings: list[str] = []
        self.tests: list[str] = []
        self.repairs: list[tuple[str, str]] = []
        self.seen: list[tuple[str, int]] = []  # (kind, the count on disk when the request was made)
        self.root: Path | None = None

    def _on_disk(self) -> int:
        record = json.loads((self.root / ".fi" / "code_review.json").read_text(encoding="utf-8")) if self.root else {}
        return int(record.get("finding_calls") or 0)

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        node = str(kw.get("node") or "")
        if prompt.lstrip().startswith(fr.TEST_HEAD):
            self.tests.append(prompt)
            self.seen.append(("test", self._on_disk()))
            return _block(self.test(prompt))
        if prompt.lstrip().startswith(fr.REPAIR_HEAD):
            who = "other" if node.endswith(".repair") else "code"
            self.repairs.append((who, prompt))
            self.seen.append(("repair", self._on_disk()))
            return self.repair(who, prompt)
        assert prompt.lstrip().startswith("# Code Against the Plan"), prompt[:80]
        self.readings.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        code = prompt.split("# The code", 1)[1]
        verdicts = self.verdicts(reqs) if self.verdicts else [
            {"id": r["id"], "status": "implemented", "file": "simulate.py", "line": 1} for r in reqs]
        return json.dumps({"requirements": verdicts, "findings": self.findings(len(self.readings), code),
                           "checked": ["a point of the logic"], "summary": "s"})

    async def aclose(self) -> None:
        return None


def _engine_with(tmp_path: Path, fixture: Path, models: Models) -> Any:
    engine = _engine(tmp_path)
    _install(engine, fixture)
    engine._client = models
    models.root = engine.quest_root
    return engine


def _small_models(**changes: Any) -> Models:
    first = _finding(_logic(SMALL, FIRST["finding"]), severity="high")

    def findings(n: int, code: str) -> list[dict[str, Any]]:
        return [] if FIRST["fixed_marker"] in code else [first]

    models = Models(findings, lambda prompt: FIRST_TEST, lambda who, prompt: _block(FIRST_GOOD))
    for key, value in changes.items():
        setattr(models, key, value)
    return models


@pytest.mark.asyncio
async def test_a_real_finding_is_shown_by_a_failing_test_then_repaired_in_its_function_only(tmp_path: Path) -> None:
    models = _small_models()
    engine = _engine_with(tmp_path, SMALL, models)
    model_py = engine.quest_root / "code" / MODEL_PY
    before = model_py.read_text(encoding="utf-8")
    assert FIRST["broken_marker"] in before
    assert await engine._code_review_gate(WITH_PLAN) is None
    after = model_py.read_text(encoding="utf-8")
    assert FIRST["fixed_marker"] in after and FIRST["broken_marker"] not in after
    # Only the function was replaced: the functions after it are byte for byte what they were.
    assert before.split(f"def {SECOND['function']}", 1)[1] == after.split(f"def {SECOND['function']}", 1)[1]
    assert len(models.tests) == 1 and len(models.repairs) == 1
    repair = models.repairs[0][1]
    assert _in(FIRST["function"]) in repair and f"def {SECOND['function']}" in repair.split("offers")[1], \
        "the function is shown, the rest of the file only by its signatures"
    assert SECOND["line_marker"] not in repair
    record = cr.load(engine.quest_root)
    (issue,) = [i for i in record["issues"].values() if i["function"] == FIRST["function"]]
    assert issue["demonstrated"] is True and issue["outcome"] == "fixed" and issue["fixed_by"] == "planner-model"
    assert issue["repairs"][0]["outcome"] == "accepted" and issue["test"].startswith(".fi/finding_tests/")
    assert record["accepted_tests"] == [issue["test"]] and (engine.quest_root / issue["test"]).is_file()
    assert [r["open"] for r in record["readings"]] == [1, 0] and record["converged"] is True


@pytest.mark.asyncio
async def test_a_finding_whose_test_passes_on_the_code_is_not_demonstrated_and_not_repaired(tmp_path: Path) -> None:
    """A real finding that was false: the code does have every value the reader said it lacked."""
    info = _info(CONFLICT)
    false_one = next(p for p in _record(CONFLICT)["problems"] if p["id"] == info["false_finding"])
    assert false_one["status"] == "different" and info["false_name"] in false_one["text"]

    def verdicts(reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in reqs:
            hit = r["kind"] == "sweep"
            out.append({"id": r["id"], "status": "different" if hit else "implemented", "file": "experiment.py",
                        "line": info["line"], "note": "the settings are not all varied" if hit else "", "changes_result": True})
        return out

    models = Models(lambda n, code: [], lambda prompt: _script(CONFLICT, "grid_test"), lambda who, prompt: "", verdicts=verdicts)
    engine = _engine_with(tmp_path, CONFLICT, models)
    before = {p.name: p.read_text(encoding="utf-8") for p in (engine.quest_root / "code").glob("*.py")}
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert models.repairs == [], "the test passes on the current code: nothing is repaired"
    assert {p.name: p.read_text(encoding="utf-8") for p in (engine.quest_root / "code").glob("*.py")} == before
    record = cr.load(engine.quest_root)
    (issue,) = record["issues"].values()
    assert issue["demonstrated"] is False and issue["outcome"] == "not demonstrated" and "passes on the current code" in issue["reason"]
    assert issue["function"] == info["function"] and record["send_backs"] == 1
    assert record["final"] is True and cr.not_computed(engine.quest_root, engine.quest_root / "code")  # still recorded
    assert "passes on the current code" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_reply_that_also_redefines_an_earlier_fixed_function_does_not_undo_that_fix(tmp_path: Path) -> None:
    first = _finding(_logic(SMALL, FIRST["finding"]), severity="high")
    line = next(n for n, text in enumerate((SMALL / "code" / MODEL_PY).read_text(encoding="utf-8").splitlines(), 1)
                if SECOND["line_marker"] in text)
    second = {"where": f"{MODEL_PY.as_posix()}:{line}", "what": SECOND["what"], "why": "w", "fix": "f", "severity": "medium",
              "quantity": SECOND["quantity"], "changes_result": True}

    def findings(n: int, code: str) -> list[dict[str, Any]]:
        out = [] if FIRST["fixed_marker"] in code else [first]
        return out + ([] if SECOND["fixed_marker"] in code else [second])

    def test(prompt: str) -> str:
        return SECOND_TEST if SECOND["what"] in prompt.split("# The finding", 1)[1] else FIRST_TEST

    def repair(who: str, prompt: str) -> str:
        if _in(FIRST["function"]) in prompt:
            return _block(FIRST_GOOD)
        # The second finding: the first try also puts the first function back as it was, the second does not.
        broke = sum(1 for _w, p in models.repairs if _in(SECOND["function"]) in p) == 1
        return _block((FIRST_REVERTED + "\n\n" if broke else "") + SECOND_GOOD)

    models = Models(findings, test, repair)
    engine = _engine_with(tmp_path, SMALL, models)
    model_py = engine.quest_root / "code" / MODEL_PY
    assert await engine._code_review_gate(WITH_PLAN) is None
    record = cr.load(engine.quest_root)
    first_issue = next(i for i in record["issues"].values() if i["function"] == FIRST["function"])
    other = next(i for i in record["issues"].values() if i["function"] == SECOND["function"])
    assert first_issue["outcome"] == "fixed" and other["outcome"] == "fixed"
    assert [r["outcome"] for r in other["repairs"]] == ["accepted"]
    assert "not used" in (engine.fi_dir / "run.log").read_text(encoding="utf-8"), "the out-of-scope definition is logged"
    text = model_py.read_text(encoding="utf-8")
    assert FIRST["fixed_marker"] in text and FIRST["broken_marker"] not in text, "the earlier fix was not undone"
    assert SECOND["fixed_marker"] in text
    assert len(record["accepted_tests"]) == 2


@pytest.mark.asyncio
async def test_a_function_that_still_fails_is_rewritten_once_by_the_other_model_and_the_record_says_which(tmp_path: Path) -> None:
    def repair(who: str, prompt: str) -> str:
        return _block(FIRST_GOOD) if who == "other" else _block(FIRST_REVERTED)

    models = _small_models(repair=repair)
    engine = _engine_with(tmp_path, SMALL, models)
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert [w for w, _p in models.repairs] == ["code"] * fr.REPAIR_CALLS + ["other"]
    record = cr.load(engine.quest_root)
    (issue,) = [i for i in record["issues"].values() if i["function"] == FIRST["function"]]
    assert issue["outcome"] == "fixed" and issue["fixed_by"] == "reader-model" and issue["fixed_by_role"] == "other"
    assert [r["by"] for r in issue["repairs"]] == ["code"] * 3 + ["other"]
    assert issue["calls"] == {"test": 1, "repair": fr.REPAIR_CALLS, "other": 1}
    assert cf.summary(record)["fixed_by"] == {"reader-model": 1}


@pytest.mark.asyncio
async def test_without_another_model_configured_nothing_is_rewritten_by_one(tmp_path: Path) -> None:
    models = _small_models(repair=lambda who, prompt: "")
    engine = _engine(tmp_path, other=False)
    _install(engine, SMALL)
    engine._client = models
    models.root = engine.quest_root
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert models.readings == [], "the reader is the model that wrote the code: nothing is read, nothing is repaired"


@pytest.mark.asyncio
async def test_every_request_is_counted_on_disk_before_it_is_made_and_the_counts_are_bounded(tmp_path: Path) -> None:
    models = _small_models(repair=lambda who, prompt: "")
    engine = _engine_with(tmp_path, SMALL, models)
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert models.seen == [("test", 1)] + [("repair", n) for n in range(2, 2 + fr.REPAIR_CALLS + fr.OTHER_CALLS)], \
        "when each request is made, it is already counted in .fi/code_review.json"
    record = cr.load(engine.quest_root)
    assert record["finding_calls"] == 1 + fr.REPAIR_CALLS + fr.OTHER_CALLS <= cr.MAX_FINDING_CALLS
    (issue,) = [i for i in record["issues"].values() if i["function"] == FIRST["function"]]
    assert issue["outcome"] == "not repaired" and issue["demonstrated"] is True
    # A resume does not ask again: the version is read, the issue is final, nothing is spent.
    again = type(engine)(engine.config, resume_quest_id=engine.quest_id)
    again._client = models
    asked = len(models.tests) + len(models.repairs) + len(models.readings)
    assert await again._code_review_gate(WITH_PLAN) is None
    assert len(models.tests) + len(models.repairs) + len(models.readings) == asked


@pytest.mark.asyncio
async def test_a_test_that_fails_for_another_reason_shows_nothing_and_is_asked_for_again_within_its_bound(tmp_path: Path) -> None:
    answers = iter(["import nothing_here_exists\n\n\ndef test_x():\n    assert True\n",
                    "def test_y():\n    raise ValueError('the test itself is broken')\n", FIRST_TEST])
    models = _small_models(test=lambda prompt: next(answers))
    engine = _engine_with(tmp_path, SMALL, models)
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert len(models.tests) == fr.TEST_CALLS == 3
    assert "the test could not show anything" in models.tests[1].split("# What went wrong with the last try")[1]
    assert "ModuleNotFoundError" in models.tests[1] and "ValueError" in models.tests[2]
    (issue,) = [i for i in cr.load(engine.quest_root)["issues"].values() if i["function"] == FIRST["function"]]
    assert issue["demonstrated"] is True and issue["outcome"] == "fixed" and issue["calls"]["test"] == 3
    # Never a pass: a test that never fails by an assertion leaves the finding not demonstrated.
    models2 = _small_models(test=lambda prompt: "def test_z():\n    raise ValueError('broken')\n", repair=lambda w, p: "")
    engine2 = _engine_with(tmp_path / "b", SMALL, models2)
    await engine2._code_review_gate(WITH_PLAN)
    (issue2,) = [i for i in cr.load(engine2.quest_root)["issues"].values() if i["function"] == FIRST["function"]]
    assert issue2["outcome"] == "not demonstrated" and models2.repairs == []


def _readable_plan(engine: Any) -> None:
    """A plan the engine can read, whose one metric is defined as a mean absolute error."""
    from core import plan as _plan

    proto = {**PROTOCOL, "precision": {"metric": "settling_time", "target_half_width": 0.05}, "runs_per_setting": 3,
             "metrics": [{"id": "settling_time", "estimand": "mean absolute error of the settling time", "kind": "mean",
                          "unit": "one run of the spring"}]}
    text = _plan.render(engine.config.topic, {}, {**DESIGN, "protocol": proto})
    _plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    _plan.record_version(engine.quest_root, text, by="model")


@pytest.mark.asyncio
async def test_a_contradiction_in_the_plan_is_put_to_the_plan_and_never_to_the_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real finding: the metric is defined one way in one part of the plan and another way in another."""
    info = _info(CONFLICT)
    conflict = _finding(_logic(CONFLICT, info["conflict_finding"]), plan_conflict=True, why=info["conflict_why"])
    models = Models(lambda n, code: [conflict], lambda prompt: "def test_x():\n    assert False\n", lambda who, prompt: "")
    engine = _engine_with(tmp_path, CONFLICT, models)
    _readable_plan(engine)
    asked: list[dict[str, Any]] = []

    async def fake_ask(design: Any, sha: str, **kw: Any) -> tuple[Any, str]:
        asked.append(kw)
        return design, sha

    monkeypatch.setattr(engine, "_ask_plan_to_complete", fake_ask)
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert models.tests == [] and models.repairs == [], "nothing is sent to the code for it"
    assert len(asked) == 1 and asked[0]["file"] == "code_plan_conflict_asked.json"
    assert info["conflict_why"] in asked[0]["request"](None, "") and "contradicts itself" in asked[0]["request"](None, "")
    record = cr.load(engine.quest_root)
    (issue,) = record["issues"].values()
    assert issue["outcome"] == "plan problem" and issue["plan_conflict"] is True and issue["plan_asked"] is True
    assert issue["plan_changed"] is False and record["plan_conflicts"][0]["id"] == issue["id"]
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "the PLAN contradicts itself" in note and "not a fault of the code" in note
    # What the plan's model may rewrite is bounded: only definitions, never a check or a tolerance.
    before = (engine.quest_root / "plan.md").read_text(encoding="utf-8")
    from core import plan as _plan

    revised = _plan.edit_design_block(before, lambda d: {**d, "protocol": {
        **d["protocol"], "metrics": [{**m, "estimand": "the spread of the estimate"} for m in d["protocol"]["metrics"]]}})
    kept = asked[0]["keep"](before, _plan.raw_design_block(revised))  # the model's answer is JSON shaped like the design block
    assert "the spread of the estimate" in kept
    loosened = _plan.edit_design_block(before, lambda d: {**d, "protocol": {
        **d["protocol"], "oracles": [{**o, "tolerance": 1.0} for o in d["protocol"]["oracles"]]}})
    with pytest.raises(ValueError):
        asked[0]["keep"](before, _plan.raw_design_block(loosened))


@pytest.mark.asyncio
async def test_once_the_protocol_is_frozen_a_contradiction_of_the_plan_is_only_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import frozen_protocol as _frozen

    info = _info(CONFLICT)
    conflict = _finding(_logic(CONFLICT, info["conflict_finding"]), plan_conflict=True)
    models = Models(lambda n, code: [conflict], lambda prompt: "", lambda who, prompt: "")
    engine = _engine_with(tmp_path, CONFLICT, models)
    _readable_plan(engine)
    monkeypatch.setattr(_frozen, "load", lambda root: {"protocol": PROTOCOL})
    called: list[Any] = []

    async def fake_ask(*a: Any, **kw: Any) -> tuple[Any, str]:
        called.append(1)
        return a[0], a[1]

    monkeypatch.setattr(engine, "_ask_plan_to_complete", fake_ask)
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert called == [] and models.tests == []
    (issue,) = cr.load(engine.quest_root)["issues"].values()
    assert issue["outcome"] == "plan problem" and issue["plan_asked"] is False


def test_duplicates_in_the_real_readings_merge_into_stable_ids() -> None:
    expected = {WIDE: (35, 30), CONFLICT: (13, 9), SMALL: (10, 10)}
    for fixture, (before, after) in expected.items():
        problems = _record(fixture)["problems"]
        cf.index_problems(problems, fixture / "code")
        assert len(problems) == before and len({p["fid"] for p in problems}) == after, fixture.name
    problems = _record(CONFLICT)["problems"]
    cf.index_problems(problems, CONFLICT / "code")
    by_id: dict[str, list[str]] = {}
    for p in problems:
        by_id.setdefault(p["fid"], []).append(p["id"])
    # One quantity told as a dependent variable, as a metric, and once in a finding of logic is one issue.
    merged = sorted(sorted(v) for v in by_id.values() if len(v) > 1)
    assert merged == sorted(sorted(group) for group in _info(CONFLICT)["merged"])
    wide = _record(WIDE)["problems"]
    cf.index_problems(wide, WIDE / "code")
    same = {p["id"]: p["fid"] for p in wide}
    assert same["R19"] == same["L10"] and same["R22"] == same["L11"] and same["R17"] == same["L6"] and same["R18"] == same["L7"]


def test_a_reworded_repeat_keeps_its_id_even_when_the_code_has_moved(tmp_path: Path) -> None:
    problems = _record(WIDE)["problems"]
    cf.index_problems(problems, WIDE / "code")
    first = {p["id"]: p["fid"] for p in problems}
    moved = tmp_path / "code"
    shutil.copytree(WIDE / "code", moved, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("simulate.py", "experiment.py"):
        path = moved / name
        path.write_text("# a note added at the top\n# and another\n\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    again = _record(WIDE)["problems"]
    for p in again:
        if p["kind"] == "logic":
            p["text"] = "reworded: " + p["text"][::-1]  # the reader's words; a requirement's text is the plan's
        if p.get("line"):
            p["line"] += 3
        if p.get("where"):
            name, _, line = p["where"].partition(":")
            p["where"] = f"{name}:{int(line) + 3}" if line.isdigit() else p["where"]
    cf.index_problems(again, moved)
    assert {p["id"]: p["fid"] for p in again} == first


def test_the_reader_s_list_is_no_longer_cut_at_twelve() -> None:
    logic = [p for p in _record(WIDE)["problems"] if p["kind"] == "logic"]
    assert len(logic) == 12, "the real list was exactly twelve in every reading"
    reply = {"findings": [{"where": p["where"], "what": p["text"]} for p in logic] +
                         [{"where": f"experiment.py:{i}", "what": f"another {i}"} for i in range(1, 9)]}
    assert len(cr.parse_logic(reply, [])) == 20


@pytest.mark.asyncio
async def test_what_was_found_is_kept_in_the_record_with_its_history_across_the_readings_of_a_quest(tmp_path: Path) -> None:
    """The reader flags a repaired issue again in other words: it keeps its id, its first reading and both statuses, and is
    reopened rather than counted as a new finding."""
    first = _finding(_logic(SMALL, FIRST["finding"]), severity="high")
    reworded = {**first, "what": INFO["reworded_what"], "where": f"{INFO['module']}:19"}
    models = Models(lambda n, code: [first if n == 1 else reworded], lambda prompt: FIRST_TEST, lambda who, prompt: _block(FIRST_GOOD))
    engine = _engine_with(tmp_path, SMALL, models)
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert len(record["issues"]) == 1, "the reworded finding is the same issue"
    (issue,) = record["issues"].values()
    assert [r["n"] for r in issue["readings"]] == [1, 2] and issue["first_seen"] == 1
    assert len(issue["wordings"]) == 2 and issue["rounds"] == 1 and issue["outcome"] == "reopened"
    assert [r["outcome"] for r in issue["repairs"]] == ["accepted"]
    assert record["converged"] is False, "a round that fixed a demonstrated issue the reader still flags is not progress"
    assert DESIGN and STATE  # (kept imported: the neutral fixtures of the code-review tests)
