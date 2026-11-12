"""A finding that a result the plan declares is not computed can be repaired (core/finding_repair.py).

Two parts of one cause, shown on a neutral made-up study (a table of derived results, as FI's analysis frame keeps):

* the repair request now carries the names the plan declares (the same block the test writer gets), so the model does not
  invent a name that the study refuses when it runs;
* a finding of kind ``quantity`` (a result is missing) may list the result in a table of named entries the file has
  (``DERIVED["name"] = ...``): there is no entry of it anywhere yet, so the rule that allows only the entries of the
  finding's own quantity found no table to put it in. Every other statement is held to the old rules."""
from __future__ import annotations

import string
from pathlib import Path

from core import finding_repair as fr
from tests.test_finding_repair import Run, _block

AGENTS = Path(__file__).parent.parent / "agents"

TABLE = '''ROLES = {"mean_x": "result", "mean_y": "result"}
DERIVED = {"mean_x": {"by": "derive_mean_x"}, "mean_y": {"by": "derive_mean_y"}}
LIMIT = 3


def derive_mean_x(rows):
    return sum(r["x"] for r in rows) / len(rows)


def derive_mean_y(rows):
    return sum(r["y"] for r in rows) / len(rows)
'''

NEW_FN = '''

def derive_spread_width(rows):
    ys = [r["y"] for r in rows]
    return max(ys) - min(ys)
'''

ISSUE = {"id": "F-1", "kind": "quantity", "file": "spring.py", "function": "", "quantity": "spread_width",
         "text": "the spread width is not computed", "wordings": ["the dependent variable spread_width is computed"]}

TEST = ("import spring\n\ndef test_listed():\n"
        "    assert 'spread_width' in spring.DERIVED, 'the spread width is not listed among the derived results'\n")


def _run(tmp_path: Path, replies: list[str], issue: dict | None = None) -> Run:
    run = Run(tmp_path, [issue or ISSUE], {"code:test": [_block(TEST)], "code:repair": replies})
    (run.code / "spring.py").write_text(TABLE, encoding="utf-8")
    return run


def test_a_missing_result_may_be_listed_in_a_table_the_file_has(tmp_path: Path) -> None:
    good = TABLE + NEW_FN + 'DERIVED["spread_width"] = {"by": "derive_spread_width"}\n'
    run = _run(tmp_path, ["# file: spring.py\n" + good])
    run.replies["code:repair"] = [_block("# file: spring.py\n" + good)]
    assert run.go() == ["spring.py"]
    repairs = run.record["issues"]["F-1"]["repairs"]
    assert [r["outcome"] for r in repairs] == ["accepted"], repairs
    text = (run.code / "spring.py").read_text(encoding="utf-8")
    assert 'DERIVED["spread_width"]' in text and "def derive_spread_width" in text


def test_a_repair_that_also_changes_another_result_keeps_only_the_missing_result(tmp_path: Path) -> None:
    """What a repair changes beyond the finding is put back (the scope rule keeps the part the finding is about), so the
    missing result is listed and nothing else of the file is what the reply made it."""
    listed = 'DERIVED["spread_width"] = {"by": "derive_spread_width"}\n'
    whole = TABLE + NEW_FN + listed
    replies = {
        "another entry": whole.replace('"mean_y": {"by": "derive_mean_y"}', '"mean_y": {"by": "derive_spread_width"}'),
        "another value": whole.replace("LIMIT = 3", "LIMIT = 9"),
        "another key": whole + 'DERIVED["mean_z"] = {"by": "derive_mean_y"}\n',
    }
    for name, reply in replies.items():
        root = tmp_path / name.replace(" ", "_")
        root.mkdir()
        run = _run(root, [_block("# file: spring.py\n" + reply)])
        assert run.go() == ["spring.py"], name
        text = (run.code / "spring.py").read_text(encoding="utf-8")
        assert 'DERIVED["spread_width"]' in text, name
        assert '"mean_y": {"by": "derive_mean_y"}' in text and "LIMIT = 3" in text and "mean_z" not in text, name


def test_the_new_entry_rule_is_only_for_a_finding_of_a_missing_result() -> None:
    old = TABLE
    new = TABLE + 'DERIVED["spread_width"] = {"by": "x"}\n'
    assert fr.unspoken_frame_changes(old, new, ISSUE) == []
    assert fr.unspoken_frame_changes(old, new, {**ISSUE, "kind": "logic"}), "a logic finding does not list a new result"
    assert fr.unspoken_frame_changes(old, new, {**ISSUE, "quantity": "other_thing"}), "only the finding's own quantity"
    assert fr.unspoken_frame_changes(old, TABLE + 'LIMIT2 = 1\n', ISSUE), "a new constant is not an entry"
    assert fr.unspoken_frame_changes(old, TABLE + 'LIMIT["spread_width"] = 1\n', ISSUE), "LIMIT is not a table of entries"
    # a table with a computed key or a spread is not a plain table of named entries
    spread = 'EXTRA = {**ROLES}\n'
    assert fr.unspoken_frame_changes(old + spread, old + spread + 'EXTRA["spread_width"] = 1\n', ISSUE)
    # the entries the table had are not touched by the new one
    assert fr.unspoken_frame_changes(old, old.replace('"mean_x": "result"', '"mean_x": "reference"') +
                                     'ROLES["spread_width"] = "result"\n', ISSUE)


def test_the_repair_request_carries_the_names_the_plan_declares(tmp_path: Path) -> None:
    template = string.Template((AGENTS / "code_review_repair.md").read_text(encoding="utf-8"))
    assert "$declared" in template.template
    run = _run(tmp_path, [_block("# file: spring.py\n" + TABLE)])
    run.env.prompts["code_review_repair"] = template
    run.go()
    asked = [p for p in run.prompts if fr.REPAIR_HEAD in p]
    assert asked and "# The names the plan declares" in asked[0]


# --- a repair that adds a missing result may not change the other results of FI's small run -------------------------------
# The fact is FI's own: the values the small run gave for every result before the repair are compared with the values it gives
# after it (``Env.small_values``); a result that is new after the repair is not compared. Here the "small run" is the file's own
# results on fixed rows, as FI's run of the study would give them.
ROWS = [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 6.0}, {"x": 5.0, "y": 4.0}]

FUNCS = '''ROLES = {"mean_x": "result", "mean_y": "result"}


def results(rows):
    out = {}
    out["mean_x"] = sum(r["x"] for r in rows) / len(rows)
    out["mean_y"] = sum(r["y"] for r in rows) / len(rows)
    return out
'''
FUNCS_TEST = ("import spring\n\ndef test_listed():\n"
              "    assert 'spread_width' in spring.results([{'x': 1.0, 'y': 2.0}]), 'the spread width is not computed'\n")


def _small_values_of(code: Path):
    async def values() -> dict:
        ns: dict = {}
        exec(compile((code / "spring.py").read_text(encoding="utf-8"), "spring.py", "exec"), ns)
        if "results" in ns:
            got = ns["results"](ROWS)
        else:
            got = {k: ns[v["by"]](ROWS) for k, v in ns["DERIVED"].items()}
        return {"x=1": {k: [v] for k, v in got.items()}}
    return values


def _run_with_small(tmp_path: Path, reply: str, *, issue: dict | None = None, source: str = TABLE, test: str = TEST) -> Run:
    run = Run(tmp_path, [issue or ISSUE], {"code:test": [_block(test)], "code:repair": [_block("# file: spring.py\n" + reply)]})
    (run.code / "spring.py").write_text(source, encoding="utf-8")
    run.env.small_values = _small_values_of(run.code)
    return run


LISTED = 'DERIVED["spread_width"] = {"by": "derive_spread_width"}\n'
PLANTED = (TABLE.replace("def derive_mean_y(rows):\n    return sum(r[\"y\"] for r in rows) / len(rows)",
                         "def derive_mean_y(rows):\n    return 0.0") + NEW_FN + LISTED)


def test_a_repair_that_also_changes_another_result_in_the_small_run_is_refused(tmp_path: Path) -> None:
    run = _run_with_small(tmp_path, PLANTED)
    run.go()
    repairs = run.record["issues"]["F-1"]["repairs"]
    assert repairs and all(r["outcome"] != "accepted" for r in repairs), repairs
    reasons = " ".join(str(r.get("reason") or r.get("why") or r) for r in repairs)
    assert "it changed `mean_y` in FI's small run" in reasons and "`spread_width` is missing" in reasons, reasons
    assert (run.code / "spring.py").read_text(encoding="utf-8") == TABLE, "the code is left as it was"


def test_a_repair_that_only_adds_the_missing_result_is_accepted_with_the_small_run(tmp_path: Path) -> None:
    run = _run_with_small(tmp_path, TABLE + NEW_FN + LISTED)
    assert run.go() == ["spring.py"]
    assert [r["outcome"] for r in run.record["issues"]["F-1"]["repairs"]] == ["accepted"]


def test_wiring_the_new_result_into_the_function_that_returns_the_results_is_accepted(tmp_path: Path) -> None:
    wired = FUNCS.replace("    return out\n", '    ys = [r["y"] for r in rows]\n    out["spread_width"] = max(ys) - min(ys)\n    return out\n')
    run = _run_with_small(tmp_path, wired, source=FUNCS, test=FUNCS_TEST)
    assert run.go() == ["spring.py"]
    assert [r["outcome"] for r in run.record["issues"]["F-1"]["repairs"]] == ["accepted"]


def test_a_finding_of_another_kind_is_not_held_to_the_rule(tmp_path: Path) -> None:
    run = _run_with_small(tmp_path, PLANTED, issue={**ISSUE, "kind": "logic"})
    run.go()
    reasons = " ".join(str(r) for r in run.record["issues"]["F-1"]["repairs"])
    assert "in FI's small run" not in reasons, reasons


def test_results_of_two_small_runs_are_compared_by_name_and_value() -> None:
    from core import small_run as sr

    before = {"s1": {"a": [1.0, 2.0], "b": [3.0]}, "s2": {"a": [5.0]}}
    assert sr.changed_results(before, {"s1": {"a": [1.0, 2.0], "b": [3.0], "new": [9.0]}, "s2": {"a": [5.0]}}) == []
    assert sr.changed_results(before, {"s1": {"a": [1.0, 2.0], "b": [3.5]}, "s2": {"a": [5.0]}}) == ["b"]
    assert sr.changed_results(before, {"s1": {"a": [1.0, 2.0]}, "s2": {"a": [6.0]}}) == ["a", "b"], "a result taken out changed"
    assert sr.changed_results(before, {"s1": {"a": [1.0, float("nan")], "b": [3.0]}, "s2": {"a": [5.0]}}) == ["a"]
    assert sr.changed_results({"s1": {"a": [float("nan")]}}, {"s1": {"a": [float("nan")]}}) == []
    assert sr.changed_results({}, {"s1": {"a": [1.0]}}) == []


def test_values_of_a_record_are_read_by_setting_and_name() -> None:
    from core import small_run as sr

    record = {"settings": [{"key": "x=1", "values": {"a": [1.0, 2.0]}}, {"key": "x=2", "values": {}}, {"key": "x=3"}]}
    assert sr.values_of(record) == {"x=1": {"a": [1.0, 2.0]}, "x=2": {}, "x=3": {}}
    assert sr.values_of({}) == {}
