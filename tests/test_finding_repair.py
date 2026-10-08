"""Test-first repair of one finding at a time (core/finding_repair.py), without the engine: the splice of a reply's functions
into a file, the harness that runs a finding's test, and the loop itself with a scripted model and an in-memory record.
Neutral code (a mass on a damped spring)."""
from __future__ import annotations

import asyncio
import re
import string
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from core import code_findings as cf
from core import finding_repair as fr

SOURCE = '''import math


# E1
def quality(m, w, c):
    return m * w / c


def damping(m, k, c):
    return c / (2.0 * math.sqrt(m * k))


def helper(x):
    return x
'''


def _block(text: str) -> str:
    return f"```python\n{text}```"


def test_a_reply_replaces_the_functions_it_defines_and_nothing_else() -> None:
    new, names, why = fr.splice(SOURCE, _block("def quality(m, w, c):\n    return m * w / (c + 0.0)\n"))
    assert why == "" and names == ["quality"]
    assert "(c + 0.0)" in new and "# E1\ndef quality" in new, "the label above the function stays"
    assert new.split("def damping")[1] == SOURCE.split("def damping")[1], "the other functions are as they were"
    assert new.startswith("import math")


def test_a_function_keeps_its_parameters_and_a_reply_may_change_only_a_few_functions() -> None:
    new, _names, why = fr.splice(SOURCE, _block("def quality(m, c, w):\n    return 1\n"))
    assert new is None and "must keep its parameters (m, w, c)" in why
    many = "".join(f"def {n}(m, w, c):\n    return 1\n\n" for n in ("quality",)) + (
        "def damping(m, k, c):\n    return 1\n\ndef helper(x):\n    return 2\n\ndef quality2(a):\n    return 0\n")
    new, names, why = fr.splice(SOURCE, _block(many + "def damping2(x):\n    return 1\n"))
    assert new is not None and names == ["quality", "damping", "helper", "quality2", "damping2"][:len(names)]
    too_many = SOURCE + "def extra(a):\n    return a\n"
    reply = "".join(f"def {n}(*a):\n    return 1\n\n" for n in ())  # placeholder: built below from the file itself
    funcs = ["quality(m, w, c)", "damping(m, k, c)", "helper(x)", "extra(a)"]
    reply = "".join(f"def {f}:\n    return 1\n\n" for f in funcs)
    got, _n, why = fr.splice(too_many, _block(reply))
    assert got is None and "replaces 4 functions" in why


def test_new_helpers_and_imports_come_with_the_reply_and_a_reply_with_no_function_is_no_reply() -> None:
    reply = _block("import numpy as np\n\n\ndef quality(m, w, c):\n    return np.divide(m * w, c)\n\n\ndef _safe(x):\n    return x\n")
    new, names, why = fr.splice(SOURCE, reply)
    assert why == "" and names == ["quality", "_safe"] and "import numpy as np" in new and "def _safe" in new
    assert fr.splice(SOURCE, "no code here")[0] is None and fr.splice(SOURCE, _block("x = 1\n"))[0] is None
    assert fr.splice(SOURCE, _block("def quality(m, w, c:\n"))[0] is None
    assert fr.splice("def broken(:\n", _block("def a():\n    pass\n"))[0] is None


def test_the_scope_of_a_repair_is_a_top_level_function_the_file_has() -> None:
    assert fr.scope_function({"function": "quality"}, SOURCE) == "quality"
    for name in ("", "<module>", "Klass.method", "absent"):
        assert fr.scope_function({"function": name}, SOURCE) == ""
    assert "def damping(m, k, c)" in fr.signatures(SOURCE, "quality") and "def quality" not in fr.signatures(SOURCE, "quality")
    assert fr.function_source(SOURCE, "helper").startswith("def helper")


def test_a_test_must_define_a_test_function_and_parse() -> None:
    assert fr.reply_test(_block("def test_x():\n    assert 1\n")).startswith("def test_x")
    assert fr.reply_test("def test_y():\n    assert 1\n").startswith("def test_y")
    assert fr.reply_test(_block("x = 1\n")) == "" and fr.reply_test(_block("def test_x(:\n")) == "" and fr.reply_test("") == ""


def _run(tmp_path: Path, files: dict[str, str]) -> dict[str, fr.TestRun]:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    (code / "spring.py").write_text("def quality(m, w, c):\n    return m * w / c\n", encoding="utf-8")
    paths = []
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        paths.append(str(tmp_path / name))
    done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source(str(code), paths)], capture_output=True, text=True,
                          cwd=tmp_path, timeout=60)
    return {Path(k).name: v for k, v in fr.parse_runs(done.stdout, paths).items()}


def test_the_harness_tells_a_failure_by_assertion_from_a_test_that_could_not_run(tmp_path: Path) -> None:
    runs = _run(tmp_path, {
        "passes.py": "import spring\n\ndef test_a():\n    assert spring.quality(2, 3, 2) == 3\n",
        "fails.py": "import spring\n\ndef test_a():\n    assert spring.quality(2, 3, 2) == 4, 'wrong by design'\n",
        "raises.py": "import spring\n\ndef test_a():\n    spring.quality(1, 1, 0)\n",
        "no_import.py": "import not_a_module_here\n\ndef test_a():\n    pass\n",
        "no_test.py": "x = 1\n",
        "mixed.py": "def test_a():\n    assert False, 'one'\n\ndef test_b():\n    raise KeyError('two')\n"})
    assert runs["passes.py"].status == "pass"
    assert runs["fails.py"].status == "fail" and "wrong by design" in runs["fails.py"].detail
    assert runs["raises.py"].status == "error" and "ZeroDivisionError" in runs["raises.py"].detail, "a crash is not a demonstration"
    assert runs["no_import.py"].status == "error" and "ModuleNotFoundError" in runs["no_import.py"].detail
    assert runs["no_test.py"].status == "error"
    assert runs["mixed.py"].status == "error", "a test that errors shows nothing, whatever else it asserts"
    assert fr.parse_runs("nothing printed", ["a.py"])["a.py"].status == "error"


def test_what_passed_before_must_still_pass_and_a_static_check_may_not_find_more() -> None:
    before = {"equation:E1": "pass", "finding:a": "pass", "finding:b": "fail"}
    assert fr.regressions(before, {"equation:E1": "pass", "finding:a": "fail", "finding:b": "pass"}) == ["finding:a"]
    assert fr.regressions(before, {"finding:a": "pass"}) == ["equation:E1"], "a check that was not run now is not a pass"
    assert fr.regressions(before, {**before, "finding:b": "error"}) == []
    assert fr.worse_static({"grid_keys": 1, "typed_constants": 0}, {"grid_keys": 1, "typed_constants": 1}) == ["typed_constants"]
    assert fr.worse_static({"grid_keys": 2}, {"grid_keys": 1}) == []


# --- the loop, with a scripted model --------------------------------------------------------------------------------------


class Run:
    """Everything the loop needs, in memory: a code folder, a record, scripted replies, and the counts on disk at each request."""

    def __init__(self, tmp_path: Path, issues: list[dict[str, Any]], replies: dict[str, list[str]], *, other: str = "") -> None:
        self.root = tmp_path
        self.code = tmp_path / "code"
        self.code.mkdir(parents=True)
        (self.code / "spring.py").write_text(SOURCE, encoding="utf-8")
        self.record: dict[str, Any] = {"issues": {}, "readings": [{"n": 1}]}
        for i in issues:
            full = {"id": i["id"], "kind": "quantity", "file": "spring.py", "function": "quality", "text": "quality is wrong",
                    "why": "w", "fix": "f", "severity": "high", "changes_result": True, "readings": [{"n": 1, "status": "open"}],
                    "demonstrated": None, "outcome": "open", "repairs": [], "tests": 0, "calls": {"test": 0, "repair": 0, "other": 0},
                    "rounds": 0, "wordings": [], **i}
            self.record["issues"][full["id"]] = full
        self.replies = {k: list(v) for k, v in replies.items()}
        self.asked: list[tuple[str, str, int]] = []
        self.prompts: list[str] = []
        self.static = {"grid_keys": 0}
        self.snapshot_now: dict[str, str] = {"equation:E1": "pass"}
        self.saves = 0
        self.keeps = ""
        self.other = other
        prompts = {"code_finding_test": string.Template("TEST $finding\n$code\n$problem"),
                   "code_review_repair": string.Template("REPAIR $finding\n$code\n$test\n$failure\n$how\n$problem")}
        self.env = fr.Env(
            root=tmp_path, code=self.code, record=self.record, save=self._save, ask=self._ask, run_tests=self._run_tests,
            snapshot=self._snapshot, static=lambda: dict(self.static), keeps=lambda old, new, rel: self.keeps, prompts=prompts,
            topic="a spring", equations="[]", scrub=lambda text: text, log=_Log(), package="", other=other, code_model="writer")

    def _save(self) -> None:
        self.saves += 1

    async def _ask(self, prompt: str, who: str, kind: str) -> str:
        self.prompts.append(prompt)
        self.asked.append((who, kind, int(self.record.get("finding_calls") or 0)))
        queue = self.replies.get(f"{who}:{kind}") or []
        return queue.pop(0) if queue else ""

    async def _run_tests(self, paths: list[str]) -> dict[str, fr.TestRun]:
        out = {}
        for p in paths:
            done = subprocess.run([sys.executable, "-B", "-c", fr.harness_source(str(self.code), [str(self.root / p)])],
                                  capture_output=True, text=True, cwd=self.root, timeout=60)
            out[p] = fr.parse_runs(done.stdout, [str(self.root / p)])[str(self.root / p)]
        return out

    async def _snapshot(self) -> dict[str, str]:
        out = dict(self.snapshot_now)
        accepted = list(self.record.get("accepted_tests") or [])
        if accepted:
            out.update({f"finding:{p}": r.status for p, r in (await self._run_tests(accepted)).items()})
        return out

    def go(self) -> list[str]:
        return asyncio.run(fr.repair_issues(self.env))


class _Log:
    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


FAILING = "import spring\n\ndef test_q():\n    assert spring.quality(2, 3, 2) == 3.5, 'the factor is wrong'\n"
GOOD = "def quality(m, w, c):\n    return m * w / c + 0.5\n"
BAD = "def quality(m, w, c):\n    return 7\n"


def test_a_failing_test_then_a_repair_that_passes_it_is_kept(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(GOOD)]})
    assert run.go() == ["spring.py"]
    issue = run.record["issues"]["F-1"]
    assert issue["demonstrated"] is True and issue["outcome"] == "fixed" and issue["fixed_by"] == "writer"
    assert "m * w / c + 0.5" in (run.code / "spring.py").read_text(encoding="utf-8")
    assert run.record["accepted_tests"] == [issue["test"]] and issue["repairs"] == [
        {"try": 1, "by": "code", "model": "writer", "outcome": "accepted"}]
    assert [(w, k) for w, k, _n in run.asked] == [("code", "test"), ("code", "repair")]


def test_a_test_that_passes_on_the_current_code_leaves_the_finding_alone(tmp_path: Path) -> None:
    passing = "import spring\n\ndef test_q():\n    assert spring.quality(2, 3, 2) == 3\n"
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(passing)]})
    assert run.go() == []
    issue = run.record["issues"]["F-1"]
    assert issue["demonstrated"] is False and issue["outcome"] == "not demonstrated"
    assert [k for _w, k, _n in run.asked] == ["test"], "no repair is asked for"
    assert (run.code / "spring.py").read_text(encoding="utf-8") == SOURCE


def test_a_test_that_fails_once_and_passes_the_next_time_is_not_trusted(tmp_path: Path) -> None:
    flaky = ("import pathlib\nNOTE = pathlib.Path(__file__).with_suffix('.seen')\n\ndef test_q():\n"
             "    if not NOTE.exists():\n        NOTE.write_text('x')\n        assert False, 'first run only'\n")
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(flaky)] * 3})
    assert run.go() == []
    assert run.record["issues"]["F-1"]["outcome"] == "not demonstrated"
    assert [k for _w, k, _n in run.asked] == ["test"] * fr.TEST_CALLS


def test_a_repair_that_does_not_pass_its_own_test_is_undone(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(BAD), _block(GOOD)]})
    assert run.go() == ["spring.py"]
    issue = run.record["issues"]["F-1"]
    assert [r["outcome"] for r in issue["repairs"]] == ["rejected", "accepted"]
    assert "the finding's own test still fails" in issue["repairs"][0]["why"]
    assert "return 7" not in (run.code / "spring.py").read_text(encoding="utf-8")


def test_a_repair_that_breaks_a_check_that_passed_is_undone_and_the_old_text_kept(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(GOOD)] * 3})
    seen: list[dict[str, str]] = []

    async def breaking() -> dict[str, str]:
        seen.append(dict(run.snapshot_now))
        return {"equation:E1": "pass"} if len(seen) == 1 else {"equation:E1": "fail"}

    run.env.snapshot = breaking
    assert run.go() == []
    issue = run.record["issues"]["F-1"]
    assert issue["outcome"] == "not repaired" and all(r["outcome"] == "rejected" for r in issue["repairs"])
    assert "it broke checks that passed before: equation:E1" in issue["repairs"][0]["why"]
    assert (run.code / "spring.py").read_text(encoding="utf-8") == SOURCE, "never a version that broke something"
    assert "accepted_tests" not in run.record or run.record["accepted_tests"] == []


def test_a_repair_that_adds_a_fault_to_a_static_check_or_loses_what_the_file_had_is_undone(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(GOOD)] * 3})
    counts = iter([{"grid_keys": 0}, {"grid_keys": 1}] + [{"grid_keys": 1}] * 10)
    run.env.static = lambda: next(counts)
    assert run.go() == []
    assert "it added faults to the static checks: grid_keys" in run.record["issues"]["F-1"]["repairs"][0]["why"]
    assert (run.code / "spring.py").read_text(encoding="utf-8") == SOURCE
    run2 = Run(tmp_path / "b", [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(GOOD)] * 3})
    run2.keeps = "it dropped `helper`, which the file had"
    assert run2.go() == []
    assert "it dropped `helper`" in run2.record["issues"]["F-1"]["repairs"][0]["why"]


def test_the_other_model_gets_one_try_after_the_code_model_s_and_its_name_is_recorded(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(BAD)] * 3,
                                          "other:repair": [_block(GOOD)]}, other="the-other")
    assert run.go() == ["spring.py"]
    issue = run.record["issues"]["F-1"]
    assert [(r["by"], r["outcome"]) for r in issue["repairs"]] == [("code", "rejected")] * 3 + [("other", "accepted")]
    assert issue["fixed_by"] == "the-other" and issue["fixed_by_role"] == "other"
    assert issue["calls"] == {"test": 1, "repair": 3, "other": 1}


def test_with_no_other_model_the_tries_end_at_the_code_model_s(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(BAD)] * 5})
    assert run.go() == []
    issue = run.record["issues"]["F-1"]
    assert issue["outcome"] == "not repaired" and len(issue["repairs"]) == fr.REPAIR_CALLS
    assert not any(w == "other" for w, _k, _n in run.asked)


def test_every_request_is_counted_before_it_is_made_and_the_quest_total_is_a_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import code_review as cr

    run = Run(tmp_path, [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(BAD)] * 5})
    run.go()
    assert [n for _w, _k, n in run.asked] == list(range(1, len(run.asked) + 1)), "the count includes the request being made"
    monkeypatch.setattr(cr, "MAX_FINDING_CALLS", 2)
    run2 = Run(tmp_path / "b", [{"id": "F-1"}], {"code:test": [_block(FAILING)], "code:repair": [_block(BAD)] * 5})
    run2.go()
    assert len(run2.asked) == 2 and run2.record["finding_calls"] == 2


def test_issues_are_taken_in_rank_order_and_one_that_changes_nothing_waits_for_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = [{"id": "F-minor", "changes_result": False, "severity": "high", "function": "helper"},
             {"id": "F-low", "changes_result": True, "severity": "low", "function": "damping"},
             {"id": "F-top", "changes_result": True, "severity": "high", "function": "quality"}]
    tests = {"quality": FAILING.replace("3.5", "99"),
             "damping": "import spring\n\ndef test_d():\n    assert spring.damping(1, 1, 1) == 99\n",
             "helper": "import spring\n\ndef test_h():\n    assert spring.helper(1) == 99\n"}
    order: list[str] = []

    class Ordered(Run):
        async def _ask(self, prompt: str, who: str, kind: str) -> str:
            self.asked.append((who, kind, int(self.record.get("finding_calls") or 0)))
            name = re.search(r"the function `(\w+)`", prompt).group(1)  # type: ignore[union-attr]
            if kind == "test":
                order.append(name)
                return _block(tests[name])
            return ""

    run = Ordered(tmp_path, items, {})
    run.go()
    assert order == ["quality", "damping", "helper"], "what changes a result first, the one that changes nothing last"
    # The ones that change nothing are repaired only while the quest has used less than half its requests.
    monkeypatch.setattr(fr, "MINOR_LIMIT", 0)
    order.clear()
    run2 = Ordered(tmp_path / "b", items, {})
    run2.go()
    assert order == ["quality", "damping"]
    assert "recorded, not repaired" in run2.record["issues"]["F-minor"]["note"]
    assert run2.record["issues"]["F-minor"]["outcome"] == "open"


def test_a_contradiction_of_the_plan_is_never_an_issue_for_the_code(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1", "plan_conflict": True}], {"code:test": [_block(FAILING)]})
    assert run.go() == [] and run.asked == []


def test_an_issue_with_no_file_in_the_code_is_not_demonstrated(tmp_path: Path) -> None:
    run = Run(tmp_path, [{"id": "F-1", "file": "gone.py"}], {})
    assert run.go() == [] and run.asked == []
    assert run.record["issues"]["F-1"]["outcome"] == "not demonstrated"


def test_a_finding_that_names_no_function_is_repaired_in_its_file_as_a_whole(tmp_path: Path) -> None:
    whole = "# file: spring.py\n" + SOURCE.replace("return m * w / c", "return m * w / c + 0.5")
    run = Run(tmp_path, [{"id": "F-1", "function": ""}], {"code:test": [_block(FAILING)], "code:repair": [_block(whole)]})
    assert run.go() == ["spring.py"]
    assert run.record["issues"]["F-1"]["outcome"] == "fixed" and "(the function" not in run.prompts[1]
    assert "m * w / c + 0.5" in (run.code / "spring.py").read_text(encoding="utf-8")


def test_the_accepted_state_is_the_baseline_for_the_next_issue(tmp_path: Path) -> None:
    items = [{"id": "F-a", "function": "quality"}, {"id": "F-b", "function": "damping", "severity": "medium"}]
    second = "import spring\n\ndef test_d():\n    assert spring.damping(1, 1, 1) == 99\n"
    damp = "def damping(m, k, c):\n    return 99\n"
    run = Run(tmp_path, items, {"code:repair": [_block(GOOD), _block(damp)]})
    replies = iter([_block(FAILING), _block(second)])

    async def ask(prompt: str, who: str, kind: str) -> str:
        run.asked.append((who, kind, 0))
        return next(replies) if kind == "test" else (run.replies["code:repair"].pop(0))

    run.env.ask = ask
    assert run.go() == ["spring.py"]
    assert run.record["accepted_tests"] == [i["test"] for i in run.record["issues"].values()]
    assert cf.FIXED == "fixed" and all(i["outcome"] == "fixed" for i in run.record["issues"].values())


def test_a_reply_may_replace_only_the_function_the_finding_is_about_and_add_helpers() -> None:
    skipped: list[str] = []
    reply = _block("def quality(m, w, c):\n    return m * w / (c + 1)\n\n\ndef damping(m, k, c):\n    return 0\n\n\n"
                   "def _extra(x):\n    return x\n")
    new, names, why = fr.splice(SOURCE, reply, {"quality"}, skipped)
    assert why == "" and names == ["quality", "_extra"] and skipped == ["damping"]
    assert "(c + 1)" in new and "def _extra" in new
    assert "return c / (2.0 * math.sqrt(m * k))" in new, "the unrelated function is as it was"
    assert "return 0" not in new
    # A reply that does not define the function the finding is about changes nothing.
    none, _n, why = fr.splice(SOURCE, _block("def damping(m, k, c):\n    return 0\n"), {"quality"}, [])
    assert none is None and "does not define the function the finding is about" in why


METHODS = '''class Spring:
    def stiffness(self, k):
        return k

    def damping(self, c):
        return c

    def total(self, k, c):
        def inner(x):
            return x
        return inner(k) + c
'''


def test_the_other_functions_of_a_file_are_compared_one_by_one_including_methods_and_nested_ones() -> None:
    assert fr.other_functions_changed(METHODS, METHODS, "Spring.stiffness") == []
    edited = METHODS.replace("return k\n", "return k * 2\n", 1)
    assert fr.other_functions_changed(METHODS, edited, "Spring.stiffness") == []
    assert fr.other_functions_changed(METHODS, edited, "Spring.damping") == ["Spring.stiffness"]
    nested = METHODS.replace("            return x\n", "            return x + 1\n")
    assert fr.other_functions_changed(METHODS, nested, "Spring.total.inner") == []
    assert fr.other_functions_changed(METHODS, nested, "Spring.total") == ["Spring.total.inner"]
    gone = METHODS.replace("    def damping(self, c):\n        return c\n\n", "")
    assert fr.other_functions_changed(METHODS, gone, "Spring.stiffness") == ["Spring.damping"]


def test_a_repair_of_a_method_that_alters_another_function_is_rejected_and_one_that_does_not_is_kept(tmp_path: Path) -> None:
    test = "import spring\n\ndef test_s():\n    assert spring.Spring().stiffness(1) == 2, 'doubled'\n"
    run = Run(tmp_path, [{"id": "F-1", "function": "Spring.stiffness"}], {})
    (run.code / "spring.py").write_text(METHODS, encoding="utf-8")
    both = "# file: spring.py\n" + METHODS.replace("return k\n", "return k * 2\n", 1).replace("return c\n", "return c + 1\n", 1)
    only = "# file: spring.py\n" + METHODS.replace("return k\n", "return k * 2\n", 1)
    run.replies = {"code:test": [_block(test)], "code:repair": [_block(both), _block(only)]}
    assert run.go() == ["spring.py"]
    issue = run.record["issues"]["F-1"]
    assert [r["outcome"] for r in issue["repairs"]] == ["rejected", "accepted"]
    assert "which the finding is not about (only `Spring.stiffness` may change)" in issue["repairs"][0]["why"]
    text = (run.code / "spring.py").read_text(encoding="utf-8")
    assert "return k * 2" in text and "return c + 1" not in text
