"""A model's JSON reply with prose around it that holds braces of its own is still read."""

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


def test_last_top_level_object_is_taken_when_none_is_fenced() -> None:
    text = 'First try {"limit": 5} did not work. The answer: {"code": "print(1)", "deps": []} done.'
    assert _parse_json_lenient(text) == {"code": "print(1)", "deps": []}


def test_one_fenced_object_wins_over_a_later_quoted_one() -> None:
    text = ('```json\n{"functions": "def f():\\n    pass\\n"}\n```\n'
            'As a note, the case {"limit": 5} stays as it was.')
    assert list(_parse_json_lenient(text)) == ["functions"]


def test_stray_closing_brace_after_a_whole_object() -> None:
    assert _parse_json_lenient('{"a": 1, "b": {"c": 2}}}') == {"a": 1, "b": {"c": 2}}


def test_reply_read_as_before_is_unchanged() -> None:
    assert _parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json_lenient('Here: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}


def test_no_object_gives_none_and_warns(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="frontier_insight.engine"):
        assert _parse_json_lenient("Sorry, a {broken brace} and no answer.", node="n") is None
    assert any("JSON parse failed" in r.getMessage() for r in caplog.records)


def test_empty_quoted_braces_are_not_an_answer() -> None:
    assert _parse_json_lenient("The case {} and {not json} only.") is None


def test_path_logs_which_object_was_taken(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="frontier_insight.engine"):
        _parse_json_lenient(PROSE_THEN_FENCE)
    assert any("fenced block" in r.getMessage() for r in caplog.records)
