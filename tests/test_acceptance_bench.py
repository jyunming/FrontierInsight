"""The acceptance bench (dev/evaluation/acceptance/runner.py) runs each case's probe and scores what it printed."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from dev.evaluation.acceptance import runner


def _probe(folder: Path, name: str, body: str) -> None:
    (folder / name).write_text(textwrap.dedent(body), encoding="utf-8")


def _cases(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "cases.yaml"
    import yaml

    path.write_text(yaml.safe_dump({"root": str(tmp_path), "probes": str(tmp_path), "cases": rows}), encoding="utf-8")
    return path


def test_each_outcome_is_counted_by_what_was_expected(tmp_path: Path) -> None:
    (tmp_path / "tree" / "core").mkdir(parents=True)
    (tmp_path / "tree" / "core" / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    _probe(tmp_path, "says.py", """
        import json, sys
        print("some log line")
        print("ACCEPT_JSON: " + json.dumps({"got": sys.argv[3], "detail": "from " + sys.argv[2]}))
    """)
    _probe(tmp_path, "silent.py", "print('nothing to read')\n")
    _probe(tmp_path, "crash.py", "raise SystemExit('the replay could not load the record')\n")
    cases = _cases(tmp_path, [
        {"id": "go-ok", "expect": "go_on", "probe": "says.py", "args": ["go_on"], "quest": "q1"},
        {"id": "false-stop", "expect": "go_on", "probe": "says.py", "args": ["stop"]},
        {"id": "missed-stop", "expect": "stop", "probe": "says.py", "args": ["go_on"]},
        {"id": "caught", "expect": "catch", "probe": "says.py", "args": ["catch"]},
        {"id": "missed", "expect": "catch", "probe": "says.py", "args": ["miss"]},
        {"id": "flag", "expect": "clean", "probe": "says.py", "args": ["catch"]},
        {"id": "kept", "expect": "keep", "probe": "says.py", "args": ["refuse"]},
        {"id": "silent", "expect": "go_on", "probe": "silent.py"},
        {"id": "crash", "expect": "stop", "probe": "crash.py"},
        {"id": "none", "expect": "go_on", "note": "no probe yet"},
    ])
    report = runner.score(tmp_path / "tree", cases, tmp_path / "out", jobs=3)
    t = report["totals"]
    assert (t["cases"], t["measured"], t["passed"]) == (10, 7, 2)
    assert (t["false_stop"], t["missed_stop"], t["missed_fault"], t["false_flag"], t["other_fail"]) == (1, 1, 1, 1, 1)
    assert (t["faults"], t["faults_caught"], t["not_measured"], t["no_probe"]) == (2, 1, 2, 1)
    assert report["core_lines"] == 2
    assert report["check_count"] is None  # this tree has no core/checks.py
    by_id = {r["id"]: r for r in report["results"]}
    assert by_id["go-ok"]["detail"].endswith("q1")  # the quest path is given under the root
    assert "could not load the record" in by_id["crash"]["detail"]
    assert (tmp_path / "out" / "score.md").read_text(encoding="utf-8").count("| ") > 10


def test_a_case_that_passed_before_and_fails_now_is_a_regression(tmp_path: Path) -> None:
    (tmp_path / "tree" / "core").mkdir(parents=True)
    _probe(tmp_path, "says.py", """
        import json, sys
        print("ACCEPT_JSON: " + json.dumps({"got": sys.argv[3]}))
    """)
    old = {"head": "abc", "core_lines": 9, "results": [{"id": "a", "passed": True}, {"id": "b", "passed": False}]}
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    cases = _cases(tmp_path, [{"id": "a", "expect": "go_on", "probe": "says.py", "args": ["stop"]},
                              {"id": "b", "expect": "go_on", "probe": "says.py", "args": ["stop"]},
                              {"id": "c", "expect": "go_on", "probe": "says.py", "args": ["stop"]}])
    report = runner.score(tmp_path / "tree", cases, tmp_path / "out", against=tmp_path / "old.json")
    assert report["against"]["regressions"] == ["a"]


def test_the_checks_of_a_tree_are_counted_from_its_own_registry(tmp_path: Path) -> None:
    core = tmp_path / "tree" / "core"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text("", encoding="utf-8")
    assert runner.check_count(tmp_path / "tree") is None
    (core / "checks.py").write_text("CHECKS = (1, 2, 3)\n", encoding="utf-8")
    assert runner.check_count(tmp_path / "tree") == 3
    report = {"tree": "t", "head": "h", "when": "now", "core_lines": 5, "check_count": 3,
              "totals": {k: 0 for k in ("cases", "measured", "passed", "false_stop", "missed_stop", "faults",
                                        "faults_caught", "missed_fault", "false_flag", "other_fail", "not_measured",
                                        "no_probe")},
              "results": [], "against": {"head": "g", "regressions": [], "core_lines_before": 9, "check_count_before": 7}}
    text = runner.markdown(report)
    assert "checks 3" in text and "checks 7 -> 3" in text


def test_only_the_last_outcome_line_counts_and_a_bad_one_is_not_measured() -> None:
    assert runner.outcome('ACCEPT_JSON: {"got": "stop"}\nACCEPT_JSON: {"got": "go_on"}\n') == {"got": "go_on"}
    assert runner.outcome("ACCEPT_JSON: not json\n") is None
    assert runner.outcome('ACCEPT_JSON: {"detail": "no outcome"}\n') is None


def test_a_cases_file_with_two_cases_of_one_id_is_refused(tmp_path: Path) -> None:
    cases = _cases(tmp_path, [{"id": "a", "expect": "go_on"}, {"id": "a", "expect": "stop"}])
    try:
        runner.load(cases)
    except ValueError as e:
        assert "two cases share" in str(e)
    else:
        raise AssertionError("a duplicate id was accepted")
