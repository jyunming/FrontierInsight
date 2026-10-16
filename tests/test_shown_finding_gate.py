"""The evidence gate stops a quest whose code holds a fault that changes a reported number, that a failing test showed, and
whose test still fails on the code that produced the results. A reader's word alone, a test that now passes, a test that
cannot be run, and a quest without a simulation never stop it. Real tiny test files and the real test runner; no model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from core import code_findings as cf
from core import code_review
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

pytestmark = pytest.mark.usefixtures("this_python")

TYPED_IN = "def dose():\n    return 60.0 if True else 40.0\n"
COMPUTED = "def dose():\n    return 59.0\n"
FAILING = ("import experiment\n\n\ndef test_dose_is_not_typed_in():\n"
           "    assert experiment.dose() != 60.0, 'dose is the typed-in 60.0'\n")
PASSING = "import experiment\n\n\ndef test_dose_is_not_typed_in():\n    assert experiment.dose() >= 0\n"
STATE: dict[str, Any] = {"exec_result": {"returncode": 0}, "result_json": {"dose": 60.0}}

#: What the code model answers when FI asks it once more for a test (``Engine._chat`` is replaced; no model is called).
REPLIES: list[str] = []
ASKED: list[str] = []


@pytest.fixture(autouse=True)
def _no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    REPLIES.clear()
    ASKED.clear()

    async def chat(self, prompt, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        ASKED.append(str(prompt))
        return REPLIES.pop(0) if REPLIES else "no test"

    monkeypatch.setattr(Engine, "_chat", chat)


class _Sink(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(topic="a dose to size", title="dose", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "experiment.py").write_text(TYPED_IN, encoding="utf-8")
    return engine


def _issue(fid: str = "F-1", **over: Any) -> dict[str, Any]:
    base = {"id": fid, "changes_result": True, "demonstrated": True, "failure_kind": cf.ASSERTION, "outcome": cf.GAVE_UP,
            "plan_conflict": False,
            "readings": [{"n": 1, "status": "open"}], "severity": "high", "first_seen": 1, "file": "experiment.py",
            "function": "main", "text": "dose_to_size is set to a typed-in value", "test": f".fi/finding_tests/{fid}_1.py"}
    return {**base, **over}


def _record(engine: Engine, *issues: dict[str, Any], test: str | None = FAILING) -> None:
    if test is not None:
        folder = engine.quest_root / ".fi" / "finding_tests"
        folder.mkdir(parents=True, exist_ok=True)
        for i in issues:
            (engine.quest_root / i["test"]).write_text(test, encoding="utf-8")
    code_review.save(engine.quest_root, {"issues": {i["id"]: i for i in issues}})


@pytest.mark.asyncio
async def test_a_shown_open_fault_whose_test_still_fails_stops_the_quest_with_no_paper(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict["stuck"] is True and verdict["stuck_reason"] == "shown_fault_open"
    assert verdict["verdict"] == "insufficient" and verdict["gaps"] == [verdict["rationale"]]
    assert "shown by a failing test" in verdict["rationale"] and "experiment.py main" in verdict["rationale"]
    assert "typed-in value" in verdict["rationale"]
    stuck = await engine._node_stuck_no_findings({**STATE, "evidence_assessment": verdict})
    assert "failing test" in stuck["stuck"]["problem"]
    assert (engine.quest_root / "needs" / "STUCK.json").is_file()
    assert not (engine.quest_root / "paper.md").exists()


@pytest.mark.asyncio
async def test_the_same_files_give_the_same_answer_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    assert await engine._shown_fault_verdict(STATE) == await engine._shown_fault_verdict(STATE)


@pytest.mark.asyncio
async def test_a_test_that_now_passes_on_the_final_code_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    (engine.quest_root / "code" / "experiment.py").write_text(COMPUTED, encoding="utf-8")
    assert await engine._shown_fault_verdict(STATE) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("over", [
    {"demonstrated": False}, {"demonstrated": None}, {"changes_result": False}, {"plan_conflict": True},
])
async def test_an_issue_that_is_not_shown_open_and_result_changing_is_never_run(
        tmp_path: Path, over: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(**over))
    ran: list[Any] = []

    async def spy(argv, **kw):  # noqa: ANN001
        ran.append(argv)
        return 0, "", ""

    monkeypatch.setattr(Engine, "_run_in_env", spy)
    assert await engine._shown_fault_verdict(STATE) is None
    assert ran == [], "no test is run for it"


@pytest.mark.asyncio
@pytest.mark.parametrize("test", [
    # the shape of a live run: the test breaks on the rewritten code (an error, not an assertion), the fault is still there
    "import experiment\n\n\ndef test_x():\n    assert experiment.dose.co_names\n",
    # the function the test calls was renamed
    "import experiment\n\n\ndef test_x():\n    assert experiment.compute_dose() != 60.0\n",
    "import module_that_is_not_there\n\n\ndef test_x():\n    assert False\n",
])
async def test_a_test_that_no_longer_runs_on_the_final_code_shows_nothing_and_stops_nothing(tmp_path: Path, test: str) -> None:
    # only a failing assertion about values shows a fault; an error in the test shows nothing (r14b stopped on a figure
    # test that raised AttributeError once experiment.py was rewritten)
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=test)
    assert await engine._shown_fault_verdict(STATE) is None
    error = code_review.load(engine.quest_root)["issues"]["F-1"]["gate_error"]
    assert error["place"] == "test" and error["at"].startswith("F-1_1.py:")


WRAPPED = ("import experiment\n\n\ndef test_x():\n    try:\n        experiment.dose(1)\n    except Exception as exc:\n"
           "        assert False, f'dose raised: {exc}'\n")
CRASHES = "def dose():\n    return {}['missing']\n"


@pytest.mark.asyncio
async def test_an_exception_turned_into_an_assertion_shows_nothing_and_a_crash_in_the_code_is_placed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=WRAPPED)
    assert await engine._shown_fault_verdict(STATE) is None
    error = code_review.load(engine.quest_root)["issues"]["F-1"]["gate_error"]
    assert error["kind"] == "wrapped" and error["place"] == "test"
    (engine.quest_root / "code" / "experiment.py").write_text(CRASHES, encoding="utf-8")
    _record(engine, _issue(), test="import experiment\n\n\ndef test_x():\n    assert experiment.dose() != 60.0\n")
    # a crash inside the program goes to the crash repair (tests/test_gate_crash_repair.py), never a shown fault
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict.get("repair_crash") is True and not verdict.get("stuck")
    assert code_review.load(engine.quest_root)["issues"]["F-1"]["crash"]["at"] == "code/experiment.py:2"


def _says_unconfirmed(engine: Engine) -> None:
    from core import paper_facts

    [item] = [i for i in paper_facts.required_items(engine.quest_root) if i.kind == paper_facts.CODE_FINDING]
    assert item.id == "finding-f-1" and "not confirmed that the fault is fixed" in item.sentence, item.sentence


@pytest.mark.asyncio
async def test_a_missing_test_file_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=None)
    assert await engine._shown_fault_verdict(STATE) is None
    # ... but nothing shows the fault fixed: the paper must say so
    assert code_review.load(engine.quest_root)["issues"]["F-1"]["gate_error"]["kind"] == "missing"
    _says_unconfirmed(engine)


@pytest.mark.asyncio
async def test_a_test_runner_that_gives_no_answer_stops_nothing_and_says_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())

    async def no_answer(self, argv, **kw):  # noqa: ANN001
        return 1, "", "the environment is broken"

    monkeypatch.setattr(Engine, "_run_in_env", no_answer)
    sink = _Sink()
    engine._log.addHandler(sink)
    try:
        assert await engine._shown_fault_verdict(STATE) is None
    finally:
        engine._log.removeHandler(sink)
    assert any("gave no answer" in m for m in sink.messages)


@pytest.mark.asyncio
async def test_one_passing_test_does_not_hide_another_that_still_fails(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue("F-1", first_seen=1), _issue("F-2", first_seen=2))
    (engine.quest_root / ".fi" / "finding_tests" / "F-1_1.py").write_text(PASSING, encoding="utf-8")
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and "still fails" in verdict["rationale"]


@pytest.mark.asyncio
async def test_a_quest_without_a_simulation_or_without_a_run_is_not_judged(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())
    assert await engine._shown_fault_verdict({**STATE, "no_simulation_resolved": True}) is None
    assert await engine._shown_fault_verdict({"result_json": {}}) is None
    assert await engine._shown_fault_verdict({**STATE, "exec_result": {}}) is None


@pytest.mark.asyncio
async def test_only_the_most_urgent_few_are_run_and_a_later_failing_one_still_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    many = [_issue(f"F-{n}", first_seen=n) for n in range(1, 13)]
    _record(engine, *many)
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict["stuck_reason"] == "shown_fault_open"
    assert len(cf.shown_open({"issues": {i["id"]: i for i in many}})[: cf.MAX_SHOWN_RERUNS]) == cf.MAX_SHOWN_RERUNS


def test_shown_open_ranks_result_changing_high_severity_first() -> None:
    a = _issue("F-a", severity="low", first_seen=1)
    b = _issue("F-b", severity="high", first_seen=2)
    assert [i["id"] for i in cf.shown_open({"issues": {"F-a": a, "F-b": b}})] == ["F-b", "F-a"]


NOT_FLAGGED = [{"outcome": cf.FIXED}, {"outcome": cf.GONE, "readings": [{"n": 1, "status": "open"}, {"n": 2, "status": "not flagged"}]},
               {"readings": [{"n": 1, "status": "open"}, {"n": 2, "status": "not flagged"}]},
               {"outcome": cf.REOPENED, "demonstrated": None, "fixed_by": "a-model"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("over", NOT_FLAGGED)
async def test_a_repaired_or_no_longer_flagged_fault_is_not_run_again(
        tmp_path: Path, over: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # a finding test turns any exception into a failed assertion, so after a repair changes an interface its old test
    # "fails" on correct code: running these again stopped both trustworthy papers of the suite
    engine = _engine(tmp_path)
    _record(engine, _issue(**over))
    ran: list[Any] = []

    async def spy(self, argv, **kw):  # noqa: ANN001
        ran.append(argv)
        return 0, "", ""

    monkeypatch.setattr(Engine, "_run_in_env", spy)
    assert await engine._shown_fault_verdict(STATE) is None
    assert ran == []


HANG = "import time\n\n\ndef test_x():\n    time.sleep(60)\n"


@pytest.mark.asyncio
async def test_a_test_that_hangs_shows_nothing_and_does_not_hide_another_that_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "SHOWN_TEST_TIMEOUT_S", 5)
    engine = _engine(tmp_path)
    _record(engine, _issue("F-1", first_seen=1), _issue("F-2", first_seen=2))
    (engine.quest_root / ".fi" / "finding_tests" / "F-1_1.py").write_text(HANG, encoding="utf-8")
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and "still fails" in verdict["rationale"]
    (engine.quest_root / ".fi" / "finding_tests" / "F-2_1.py").write_text(PASSING, encoding="utf-8")
    assert await engine._shown_fault_verdict(STATE) is None


@pytest.mark.asyncio
async def test_a_sole_fault_whose_test_hangs_stops_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cf, "SHOWN_TEST_TIMEOUT_S", 5)
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=HANG)
    assert await engine._shown_fault_verdict(STATE) is None
    assert code_review.load(engine.quest_root)["issues"]["F-1"]["gate_error"]["kind"] == "no_answer"
    _says_unconfirmed(engine)


@pytest.mark.asyncio
async def test_an_error_in_the_second_run_does_not_hide_a_failing_first_run(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    fail_then_error = ("from pathlib import Path\n\n\ndef test_x():\n    p = Path('flag2.txt')\n"
                       "    if p.exists():\n        raise RuntimeError('second run')\n"
                       "    p.write_text('1')\n    assert False, 'first run'\n")
    _record(engine, _issue(), test=fail_then_error)
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and "still fails" in verdict["rationale"]


@pytest.mark.asyncio
async def test_a_test_that_fails_once_and_then_passes_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    flaky = ("from pathlib import Path\n\n\ndef test_x():\n    p = Path('flag.txt')\n"
             "    if not p.exists():\n        p.write_text('1')\n        assert False, 'first run'\n")
    _record(engine, _issue(), test=flaky)
    assert await engine._shown_fault_verdict(STATE) is None


@pytest.mark.asyncio
async def test_an_error_inside_the_check_with_a_flagged_fault_stops_the_quest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue())

    async def boom(self, rel):  # noqa: ANN001
        raise OSError("disk")

    monkeypatch.setattr(Engine, "_rerun_finding_test", boom)
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and "could not run its test again" in verdict["rationale"]


TWO_RESULTS: dict[str, Any] = {"exec_result": {"returncode": 0}, "result_json": {"dose": 60.0, "contrast": 0.42}}


@pytest.mark.asyncio
@pytest.mark.parametrize("test", [FAILING])
async def test_a_fault_in_another_quantity_is_left_out_and_the_paper_says_so(tmp_path: Path, test: str) -> None:
    """A fault outside the study's main quantity, whose test still fails, takes that quantity out of the results the paper
    may use; nothing stops. The record of it is the paper's run record and the quality report."""
    import json

    from core import disclosure, quality_report

    engine = _engine(tmp_path)
    _record(engine, _issue(quantity="dose"), test=test)
    assert await engine._shown_fault_verdict(TWO_RESULTS) is None
    checked = json.loads((engine.quest_root / "needs" / "RESULT_CHECK.json").read_text(encoding="utf-8"))
    assert checked["removed_quantities"] == ["dose"] and checked["removed_values"] == {"dose": [60.0]}
    assert "a test FI ran" in checked["removed_why"]["dose"]
    assert "computes `dose` wrongly" in disclosure.run_record(engine.quest_root)
    issues = quality_report._issue_rows(code_review.load(engine.quest_root))
    assert issues[0]["at_gate"]["outcome"] == "left_out"


@pytest.mark.asyncio
async def test_the_same_fault_in_the_main_quantity_still_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(quantity="dose", headline=True))
    verdict = await engine._shown_fault_verdict(TWO_RESULTS)
    assert verdict and verdict["stuck_reason"] == "shown_fault_open"
    assert not (engine.quest_root / "needs" / "RESULT_CHECK.json").exists()


@pytest.mark.asyncio
async def test_a_fault_whose_removal_leaves_no_result_still_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(quantity="dose"))
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and "leaves no result" in verdict["rationale"]


@pytest.mark.asyncio
async def test_a_fault_in_a_function_the_run_never_calls_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    (engine.quest_root / "code" / "experiment.py").write_text(
        TYPED_IN + "\n\ndef plot_unused():\n    return 0\n\nprint(dose())\n", encoding="utf-8")
    _record(engine, _issue(function="plot_unused"))
    assert await engine._shown_fault_verdict(STATE) is None
    assert code_review.load(engine.quest_root)["issues"]["F-1"]["never_run"] is True
    _record(engine, _issue(function="dose"))  # the same fault in the function the run calls still stops
    verdict = await engine._shown_fault_verdict(STATE)
    assert verdict and verdict["stuck_reason"] == "shown_fault_open"


def test_a_fault_may_be_about_the_main_quantity_by_its_name_or_its_keys() -> None:
    result = {"total_yield": 3.2, "yield_ci": [3.0, 3.4], "runtime": 1.0}
    issue = {"id": "F-1", "quantity": "yield"}
    assert code_review.about_headline({}, issue, result, ["total_yield"])  # one name holds the other
    assert code_review.about_headline({}, {"id": "F-2", "quantity": "totalyield"}, result, ["total_yield"])
    assert not code_review.about_headline({}, {"id": "F-3", "quantity": "runtime"}, result, ["total_yield"])
    assert code_review.about_headline({}, {"id": "F-4", "headline": True}, result, [])
    # a short headline name (AUC) by its own name and by the keys it holds
    assert code_review.about_headline({}, {"id": "F-5", "quantity": "AUC"}, {"auc": 0.85, "n": 3}, ["auc"])
    assert code_review.about_headline({}, {"id": "F-6", "quantity": "auc"}, {"auc": 0.85}, ["AUC"])
    assert not code_review.about_headline({}, {"id": "F-7", "quantity": "n"}, {"auc": 0.85, "n": 3}, ["auc"])


@pytest.mark.asyncio
async def test_a_fault_whose_values_other_keys_also_hold_still_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(quantity="dose"))
    verdict = await engine._shown_fault_verdict(
        {"exec_result": {"returncode": 0}, "result_json": {"dose": 60.0, "dose_raw": 60.0, "contrast": 0.42}})
    assert verdict and "under other names too" in verdict["rationale"]


@pytest.mark.asyncio
async def test_a_left_out_quantity_is_not_said_once_it_is_no_longer_flagged(tmp_path: Path) -> None:
    from core import disclosure

    engine = _engine(tmp_path)
    _record(engine, _issue(quantity="dose"))
    assert await engine._shown_fault_verdict(TWO_RESULTS) is None
    assert "computes `dose` wrongly" in disclosure.run_record(engine.quest_root)
    record = code_review.load(engine.quest_root)
    record["issues"]["F-1"]["outcome"] = cf.FIXED  # a later repair fixed it
    code_review.save(engine.quest_root, record)
    assert "dose" not in disclosure.run_record(engine.quest_root)
    assert await engine._shown_fault_verdict(TWO_RESULTS) is None  # the gate again: nothing shown, the old mark is gone
    assert "at_gate" not in code_review.load(engine.quest_root)["issues"]["F-1"]


# ---- a test that errors in its own code is asked for once more, as an assertion about values ------------------------------

CO_NAMES = "import experiment\n\n\ndef test_x():\n    assert experiment.dose.co_names\n"  # r7's shape: an error in the test
AS_VALUES = ("```python\nimport experiment\n\n\ndef test_dose_is_computed():\n"
             "    assert experiment.dose() != 60.0, 'dose is the typed-in 60.0'\n```")


@pytest.mark.asyncio
async def test_a_test_that_errors_in_itself_is_asked_for_once_more_and_a_failing_assertion_stops(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=CO_NAMES)
    REPLIES.append(AS_VALUES)
    verdict = await engine._shown_fault_verdict(STATE)
    assert len(ASKED) == 1 and "has no attribute 'co_names'" in ASKED[0] and "FAIL BY AN ASSERTION" in ASKED[0]
    assert verdict and verdict["stuck_reason"] == "shown_fault_open"
    issue = code_review.load(engine.quest_root)["issues"]["F-1"]
    assert issue["test"] == ".fi/finding_tests/F-1_2.py" and issue["failure_kind"] == cf.ASSERTION
    assert issue["asked_again"]["outcome"] == "fail"
    # the same files give the same answer, and the model is not asked again
    assert (await engine._shown_fault_verdict(STATE))["stuck_reason"] == "shown_fault_open" and len(ASKED) == 1


@pytest.mark.asyncio
async def test_a_test_asked_for_once_more_that_errors_again_is_recorded_and_stops_nothing(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=CO_NAMES)
    REPLIES.append("```python\nimport experiment\n\n\ndef test_x():\n    assert experiment.missing()\n```")
    assert await engine._shown_fault_verdict(STATE) is None
    issue = code_review.load(engine.quest_root)["issues"]["F-1"]
    assert issue["asked_again"]["outcome"] == "error" and issue["gate_error"]["place"] == "test"
    assert await engine._shown_fault_verdict(STATE) is None and len(ASKED) == 1  # asked once, ever


@pytest.mark.asyncio
async def test_a_test_asked_for_once_more_that_passes_clears_it(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=CO_NAMES)
    (engine.quest_root / "code" / "experiment.py").write_text(COMPUTED, encoding="utf-8")
    REPLIES.append(AS_VALUES)
    assert await engine._shown_fault_verdict(STATE) is None
    assert code_review.load(engine.quest_root)["issues"]["F-1"]["asked_again"]["outcome"] == "pass"


@pytest.mark.asyncio
async def test_a_model_that_cannot_be_asked_stops_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _record(engine, _issue(), test=CO_NAMES)

    async def down(self, prompt, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        raise RuntimeError("the provider is down")

    monkeypatch.setattr(Engine, "_chat", down)
    assert await engine._shown_fault_verdict(STATE) is None
    assert code_review.load(engine.quest_root)["issues"]["F-1"]["gate_error"]["place"] == "test"
