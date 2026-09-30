"""code/CHANGELOG.md: each entry names its kind of change (added / changed / fixed / tidied), decided by the step that made
it, and says whether the results changed: the checks of correctness before and after, and whether the study's results
changed (by name only). A line not known yet says "not measured yet" and is filled in by the run that follows."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import changelog, code_project, criteria as cr, frozen_protocol as fp, plan
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine

HAS_GIT = shutil.which("git") is not None
needs_git = pytest.mark.skipif(not HAS_GIT, reason="git not installed")

CASE_ORACLE = {"name": "rk4 error", "kind": "special_case", "check": "error at t=1 of y'=-y against exp(-1)",
               "expected": 0.0, "tolerance": 1e-5, "case": {"dt": 0.1}, "measure": "error",
               "reference": "derivation: y' = -y, y(0) = 1 gives y(1) = exp(-1) = 0.3679"}
CRITERION = {"name": "rk4 error small", "oracle": "rk4 error", "direction": "lower", "target": 1e-5, "tolerance": 1e-7}
PROTOCOL = {"grid": {"dt": [0.1, 0.05]}, "oracles": [CASE_ORACLE], "criteria": [CRITERION]}


def _row(n: int, value: float | None, digest: dict[str, str] | None, *, sha: str = "p1",
         improve: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"n": n, "protocol_sha256": sha,
                           "criteria": [{"name": "rk4 error small", "value": value, "met": value is not None
                                         and value < 1e-5, "counts": True}]}
    if digest:
        row["result_digest"] = digest
    if improve:
        row["improve"] = improve
    return row


# --- the words, on their own ------------------------------------------------------------------------------------------


def test_the_four_kinds_and_nothing_else(caplog: pytest.LogCaptureFixture) -> None:
    assert changelog.CATEGORIES == ("added", "changed", "fixed", "tidied")
    for word in changelog.CATEGORIES:
        assert changelog.category(word.upper()) == word
    log = logging.getLogger("test_changelog")
    with caplog.at_level(logging.WARNING, logger="test_changelog"):
        assert changelog.category("refactored", log=log) == "changed"
    assert "not a kind of change" in caplog.text


def test_only_files_fis_runs_never_read_make_a_tidy_up() -> None:
    assert changelog.documentation_only(["README.md", "requirements.txt", "run.py", "study.json", "METHODS.md",
                                         "tests/test_oracles.py"])
    for code in ("simulate.py", "experiment.py", "fi_search.py", "mypkg/model.py"):
        assert not changelog.documentation_only(["README.md", code])
    assert not changelog.documentation_only([])


def test_a_first_run_has_nothing_to_compare_with() -> None:
    line = changelog.run_line([], _row(1, 3.3e-7, {"peak": "a"}))
    assert line == ("Did the results change: not compared: no earlier run's results are on record. Checks of "
                    "correctness now: rk4 error small: 3.3e-07 (met).")


def test_same_checks_and_same_results_is_no() -> None:
    history = [_row(1, 3.3e-7, {"peak": "a", "t": "b"}), _row(2, 3.3e-7, {"peak": "a", "t": "b"})]
    line = changelog.run_line(history, history[-1])
    assert line == ("Did the results change: no. The study's results: unchanged. Checks of correctness: "
                    "rk4 error small: 3.3e-07 (same, met).")


def test_a_check_that_moved_says_so_apart_from_the_studys_results() -> None:
    history = [_row(1, 3.3e-7, {"peak": "a"}), _row(2, 2e-4, {"peak": "a"})]
    line = changelog.run_line(history, history[-1])
    assert line == ("Did the results change: the study's results did not; a check of correctness did. Checks of "
                    "correctness: rk4 error small: from 3.3e-07 to 0.0002 (not met).")


def test_a_check_measured_on_one_side_only_is_shown_not_counted_as_moved() -> None:
    history = [_row(1, None, {"peak": "a"}), _row(2, 3.3e-7, {"peak": "a"})]
    line = changelog.run_line(history, history[-1])
    assert line.startswith("Did the results change: no. The study's results: unchanged.")
    assert "rk4 error small: from not measured to 3.3e-07 (met)" in line


def test_a_changed_result_is_named_never_its_value() -> None:
    history = [_row(1, 3.3e-7, {"peak": "a", "t": "b"}), _row(2, 3.3e-7, {"peak": "c", "t": "b", "new": "d"})]
    line = changelog.run_line(history, history[-1])
    assert line.startswith("Did the results change: yes: the study's results changed (new, peak). Checks of correctness:")


def test_the_comparison_skips_improve_rounds_and_runs_without_results() -> None:
    # A round's row and a later full run's row both differ from the first run; neither is what the last run is
    # compared with (a round never runs the study in full; the third run produced no results).
    history = [_row(1, 1e-3, {"peak": "a"}), _row(2, 1e-9, {"peak": "z"}, improve={"round": 1}), _row(3, 5e-4, None),
               _row(4, 1e-3, {"peak": "a"})]
    line = changelog.run_line(history, history[-1])
    assert line.startswith("Did the results change: no. The study's results: unchanged.")
    assert "0.001 (same, not met)" in line


def test_an_amended_plan_does_not_compare_its_checks() -> None:
    history = [_row(1, 1e-3, {"peak": "a"}), _row(2, 1e-9, {"peak": "a"}, sha="p2")]
    line = changelog.run_line(history, history[-1])
    assert "The plan was amended between the two runs, so its checks of correctness are not compared" in line
    assert line.startswith("Did the results change: no. The study's results: unchanged.")


def test_a_round_not_kept_is_not_measured_and_a_put_back_says_what_measured_it() -> None:
    before = [{"name": "err", "value": 1e-3, "met": False, "counts": True}]
    after = [{"name": "err", "value": 2e-3, "met": False, "counts": True}]
    line = changelog.round_not_kept(before, after)
    assert line.startswith("Did the results change: not measured: this version was not kept")
    assert "err: from 0.001 to 0.002 (not met)" in line
    assert "which a full run already measured" in changelog.put_back("the first version")
    later = changelog.put_back("round 1's version", ran_in_full=False)
    assert "full run already measured" not in later and "the full run after the improve loop says" in later


def test_several_changes_measured_by_one_run_say_so() -> None:
    assert changelog.together("X.", 0) == "X."
    assert changelog.together("X.", 1).endswith("together with 1 other change that was waiting for a run.)")
    assert changelog.together("X.", 2).endswith("together with 2 other changes that were waiting for a run.)")


# --- the entries in code/ ---------------------------------------------------------------------------------------------


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    return root


def _entries(root: Path) -> list[str]:
    text = (root / "code" / "CHANGELOG.md").read_text(encoding="utf-8")
    return ["## " + part for part in text.split("\n## ")[1:]]


@needs_git
def test_an_entry_names_its_kind_and_waits_for_its_run(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert code_project.record_change(root, "code written", category="added")
    (entry,) = _entries(root)
    assert " - Added: code written\n" in entry and "Files: experiment.py" in entry
    assert changelog.PENDING in entry
    text = (root / "code" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith(changelog.HEADER)


@needs_git
def test_a_change_to_the_project_files_alone_is_tidied_and_its_results_are_known(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    code_project.record_change(root, "code written", category="added")
    (root / "code" / "README.md").write_text("# Study\n", encoding="utf-8")
    (root / "code" / "requirements.txt").write_text("numpy==2.0.0\n", encoding="utf-8")
    assert code_project.record_change(root, "changes since the code was last recorded", category="changed")
    last = _entries(root)[-1]
    assert " - Tidied: changes since the code was last recorded" in last
    assert ("Did the results change: no: only README.md, requirements.txt changed, which FI's own runs of the study do "
            "not read.") in last
    # A change to the code itself is never tidied.
    (root / "code" / "experiment.py").write_text("print('v2')\n", encoding="utf-8")
    (root / "code" / "README.md").write_text("# Study v2\n", encoding="utf-8")
    assert code_project.record_change(root, "a person's edit", category="changed")
    assert " - Changed: a person's edit" in _entries(root)[-1] and changelog.PENDING in _entries(root)[-1]


@needs_git
def test_filling_in_commits_only_the_changelog(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    code_project.record_change(root, "code written", category="added")
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {\"a\": 1}')\n", encoding="utf-8")
    code_project.record_change(root, "the code as it ran", category="fixed")
    (root / "code" / "experiment.py").write_text("print('an edit not recorded yet')\n", encoding="utf-8")
    assert code_project.fill_pending(root, changelog.results("no. The study's results: unchanged.")) == 2
    text = (root / "code" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.PENDING not in text
    assert text.count("no. The study's results: unchanged. (One run measured this change together with 1 other") == 2
    shown = code_project._git(root / "code", "show", "--name-only", "--format=", "HEAD").stdout.split()
    assert shown == ["CHANGELOG.md"], "the fill-in commits CHANGELOG.md alone"
    status = code_project._git(root / "code", "status", "--porcelain").stdout
    assert "experiment.py" in status and "CHANGELOG.md" not in status  # the edit waits for its own entry
    assert code_project.fill_pending(root, changelog.results("x")) == 0  # nothing is waiting any more


@needs_git
def test_a_fill_in_that_cannot_be_committed_leaves_code_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    root = _quest(tmp_path)
    code_project.record_change(root, "code written", category="added")
    before = (root / "code" / "CHANGELOG.md").read_bytes()
    real = code_project._git

    def failing(code_dir: Path, *args: str) -> Any:
        if args and args[0] == "commit":
            raise subprocess.TimeoutExpired(["git", "commit"], 60)
        return real(code_dir, *args)

    monkeypatch.setattr(code_project, "_git", failing)
    assert code_project.fill_pending(root, changelog.results("no.")) == 0
    monkeypatch.setattr(code_project, "_git", real)
    assert (root / "code" / "CHANGELOG.md").read_bytes() == before
    assert code_project.head(root)[1] is False, "no uncommitted CHANGELOG.md to read as a change to the code"
    # And a CHANGELOG.md changed on its own is committed without an entry about itself.
    (root / "code" / "CHANGELOG.md").write_bytes(before + b"A note of my own.\n")
    assert code_project.record_change(root, "the code as it ran", category="changed") is True  # committed
    assert len(_entries(root)) == 1 and code_project.head(root)[1] is False


@needs_git
def test_putting_back_an_earlier_version_closes_what_was_waiting(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    code_project.record_change(root, "code written", category="added")
    code_project.fill_pending(root, changelog.results("not compared."))
    original = (root / "code" / "experiment.py").read_text(encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('v2')\n", encoding="utf-8")
    code_project.record_change(root, "improve round 1: a smaller step | kept", category="changed")
    (root / "code" / "experiment.py").write_text(original, encoding="utf-8")
    code_project.record_change(root, "improve: back to the first version", category="changed",
                               results=changelog.put_back("the first version"), undo_pending="undone before a run")
    text = (root / "code" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.PENDING not in text
    assert "Did the results change: not measured: undone before a run" in text
    assert "Did the results change: no: this puts back the first version, which a full run already measured." in text


# --- the engine: each kind from the step that made the change ---------------------------------------------------------


def _implement_engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=2, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        pauses=PausesConfig(review="off"), output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    return eng


class _Model:
    """No real model: every question gets the same small script."""

    async def chat(self, messages, **kw):  # noqa: ANN001
        return "```python\nimport json\nprint('RESULT_JSON: ' + json.dumps({'peak': 1.0}))\n```\nDEPS: none"


def _last_heading(root: Path) -> str:
    return _entries(root)[-1].split("\n", 1)[0]


@needs_git
@pytest.mark.parametrize("case, kind, words", [
    ("first", "Added", "code written"),
    ("refine", "Added", "added what a refine asked for: the runtime for n=64"),
    ("review", "Fixed", "what the review found wrong in"),
    ("redesign", "Changed", "revised after review (round 1)"),
])
def test_implement_names_the_kind_of_change(tmp_path: Path, case: str, kind: str, words: str) -> None:
    from tests.test_reexecute_route import XG3_REVIEW, _state

    eng = _implement_engine(tmp_path)
    eng._client = _Model()  # type: ignore[assignment]
    state: dict[str, Any] = {"topic": "t", "design": {}}
    if case in ("refine", "review", "redesign"):
        (tmp_path / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
        assert code_project.record_change(tmp_path, "code written", category="added")
    if case == "refine":
        state = {**state, "iteration": 1, "refine_extend": ["the runtime for n=64"], "refine_scope": "data"}
    elif case == "review":
        state = {**_state(XG3_REVIEW), "design": {}}
    elif case == "redesign":
        state = {**state, "iteration": 1}
    asyncio.run(eng._node_implement(state))  # type: ignore[arg-type]
    heading = _last_heading(tmp_path)
    assert f" - {kind}: {words}" in heading, heading
    assert changelog.PENDING in _entries(tmp_path)[-1]


def _criteria_engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="RK4 on y' = -y", title="criteria", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, execute_replicates=1, pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(plan="off", papers=False),
    ))
    root = eng.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "code" / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    fp.freeze(root, plan.normalize_protocol(PROTOCOL)[0], approved_by="human: test", source="plan.md")
    return eng


def _judged(root: Path, value: float) -> None:
    (root / "needs" / "ORACLE_CHECK.json").write_text(json.dumps({"status": "ok", "attempts": [{"judged": [
        {"name": "rk4 error", "value": value, "expected": 0.0, "measured_by": "engine", "passed_by_engine": True}]}]}),
        encoding="utf-8")


def _edit(root: Path, text: str) -> None:
    (root / "code" / "experiment.py").write_text(text, encoding="utf-8")


@needs_git
def test_each_run_fills_in_whether_the_results_changed(tmp_path: Path) -> None:
    eng = _criteria_engine(tmp_path)
    root = eng.quest_root
    assert code_project.record_change(root, "code written", category="added")
    assert changelog.PENDING in _entries(root)[-1]

    # A run that crashed (no results): nothing is filled in, nothing is guessed.
    _judged(root, 3.3e-7)
    asyncio.run(eng._record_criteria({}, attempt="a1", result=None))
    assert changelog.PENDING in _entries(root)[-1]

    # FI repaired the script and it ran with results: the repair is "fixed"; both waiting entries are filled.
    _edit(root, "print('RESULT_JSON: {\"peak\": 1}')\n")
    asyncio.run(eng._record_criteria({"exec_reflect_iter": 1}, attempt="a2", result={"peak": 0.987654, "t": 2.0},
                                     repaired=True))
    entries = _entries(root)
    assert " - Fixed: the code as it ran (changed since it was last recorded)" in entries[-1]
    assert all(changelog.PENDING not in e for e in entries)
    assert ("Did the results change: not compared: no earlier run's results are on record. Checks of correctness "
            "now: rk4 error small: 3.3e-07 (met). (One run measured this change together with 1 other") in entries[-1]
    assert code_project.head(root)[1] is False

    # A person's edit (no repair) that leaves the study's results unchanged, while the check moved.
    _edit(root, "print('RESULT_JSON: {\"peak\": 2}')\n")
    _judged(root, 2e-7)
    asyncio.run(eng._record_criteria({}, attempt="a3", result={"peak": 0.987654, "t": 2.0}))
    last = _entries(root)[-1]
    assert " - Changed: the code as it ran" in last
    assert ("Did the results change: the study's results did not; a check of correctness did. Checks of correctness: "
            "rk4 error small: "
            "from 3.3e-07 to 2e-07 (met).") in last

    # A change whose results changed: named, never the value.
    _edit(root, "print('RESULT_JSON: {\"peak\": 3}')\n")
    asyncio.run(eng._record_criteria({}, attempt="a4", result={"peak": 0.5, "t": 2.0}))
    last = _entries(root)[-1]
    assert ("Did the results change: yes: the study's results changed (peak). Checks of correctness: rk4 error small: "
            "2e-07 (same, met).") in last
    text = (root / "code" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "0.987654" not in text and "0.5" not in text

    # The same code again: no entry, and the row still records the results' fingerprint for the next comparison.
    before = len(_entries(root))
    asyncio.run(eng._record_criteria({}, attempt="a5", result={"peak": 0.5, "t": 2.0}))
    assert len(_entries(root)) == before
    rows = cr.history(root)
    assert "result_digest" not in rows[0] and set(rows[-1]["result_digest"]) == {"peak", "t"}
