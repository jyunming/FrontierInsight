"""A model's JSON reply with prose around it that holds braces of its own is read only where no doubt is left."""

from __future__ import annotations

import logging

from core.engine import _parse_json_lenient
from core.script_reply import resolve

PROSE_THEN_FENCE = (
    "The failing case is `{\"limit\": 5}` and the loop never stops for it.\n"
    "Also a set such as {1, 2} is not meant.\n\n"
    "```json\n"
    "{\"functions\": \"def step(x):\\n    return x + 1\\n\", \"patch_summary\": \"fix step\"}\n"
    "```\n"
)


def test_object_in_fence_after_prose_with_braces_is_read() -> None:
    got = _parse_json_lenient(PROSE_THEN_FENCE)
    assert got is not None
    assert set(got) == {"functions", "patch_summary"}
    assert resolve("def step(x):\n    return x\n", got.get("code"), got.get("functions")).kind == "functions"


def test_stray_closing_brace_after_a_whole_object() -> None:
    assert _parse_json_lenient('{"a": 1, "b": {"c": 2}}}') == {"a": 1, "b": {"c": 2}}


def test_reply_read_as_before_is_unchanged() -> None:
    assert _parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_lenient('Here: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}


def test_path_logs_which_object_was_taken(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="frontier_insight.engine"):
        _parse_json_lenient(PROSE_THEN_FENCE)
    assert any("whole fenced block" in r.getMessage() for r in caplog.records)


def test_no_object_gives_none_and_warns(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="frontier_insight.engine"):
        assert _parse_json_lenient("Sorry, a {broken brace} and no answer.", node="n") is None
    assert any("JSON parse failed" in r.getMessage() for r in caplog.records)


def test_empty_quoted_braces_are_not_an_answer() -> None:
    assert _parse_json_lenient("The case {} and {not json} only.") is None


def test_answer_first_then_a_quoted_object_is_not_guessed() -> None:
    text = 'The answer: {"code": "print(1)", "deps": []} and the old case {"limit": 5} stays.'
    assert _parse_json_lenient(text) is None


def test_two_unfenced_objects_give_none() -> None:
    assert _parse_json_lenient('First {"limit": 5} then {"code": "x"} done.') is None


def test_a_reply_cut_short_inside_its_object_gives_none() -> None:
    # the inner object parses on its own; it is not the reply's answer
    text = 'Then {"functions": "def f(): pass", "meta": {"a": 1}, "note": "cut'
    assert _parse_json_lenient(text) is None
    text = 'Case {1, 2} here.\r\n\r\n{"code": "print(1)", "deps": {"numpy": "1"}, "patch_summary": "cu'
    assert _parse_json_lenient(text) is None


def test_an_odd_number_of_fences_gives_none() -> None:
    text = 'Case {"limit": 5} quoted.\n```json\n{"a": 1}\n```\nMore:\n```python\nprint(1)\n'
    assert _parse_json_lenient(text) is None


def test_a_fenced_list_after_prose_with_braces_gives_none() -> None:
    text = 'The case is {1, 2} here.\n```json\n[{"a": 1}]\n```\n'
    assert _parse_json_lenient(text) is None


def test_want_with_no_match_gives_none_and_warns(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="frontier_insight.engine"):
        assert _parse_json_lenient(PROSE_THEN_FENCE, node="n", want=("missing_key",)) is None
    assert any("JSON parse failed" in r.getMessage() for r in caplog.records)
    assert not any("braces around its object" in r.getMessage() for r in caplog.records)


def test_an_answer_that_does_not_parse_is_not_replaced_by_a_quoted_object() -> None:
    head = 'The failing case is {"threshold": 0.01} here.\n\n```json\n'
    raw = head + '{"functions": "def f():' + chr(10) + '    return 1' + chr(10) + '"}\n```\n'
    assert _parse_json_lenient(raw) is None
    escaped = head + '{"functions": "def f():' + chr(92) + 'n    return 1' + chr(92) + 'n"}\n```\n'
    assert list(_parse_json_lenient(escaped)) == ["functions"]
