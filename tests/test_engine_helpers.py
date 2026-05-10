"""Direct unit tests for the helper functions in `core.engine`.

These avoid the full LangGraph DAG + venv + subprocess machinery exercised
by `test_engine_smoke.py` so they run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import (
    Engine,
    _extract_result_json,
    _new_quest_id,
    _parse_json_lenient,
    _slugify,
    _strip_outer_fence,
)


# ---- _parse_json_lenient -------------------------------------------------


def test_parse_json_lenient_strict_object() -> None:
    assert _parse_json_lenient('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_lenient_fenced_with_lang() -> None:
    text = '```json\n{"a": 1}\n```'
    assert _parse_json_lenient(text) == {"a": 1}


def test_parse_json_lenient_fenced_no_lang() -> None:
    text = '```\n{"a": 1}\n```'
    assert _parse_json_lenient(text) == {"a": 1}


def test_parse_json_lenient_prose_wrapped() -> None:
    text = 'Here is your answer:\n\n{"verdict": "accept", "score": 4}\n\nThanks!'
    assert _parse_json_lenient(text) == {"verdict": "accept", "score": 4}


def test_parse_json_lenient_with_leading_garbage_and_trailing_commentary() -> None:
    text = (
        'Sure! I think this is good.\n'
        '{"hypothesis": "y=x", "deps": ["matplotlib"]}\n'
        'Hope this helps.'
    )
    assert _parse_json_lenient(text) == {"hypothesis": "y=x", "deps": ["matplotlib"]}


def test_parse_json_lenient_garbage_input_returns_none() -> None:
    assert _parse_json_lenient("this is not JSON at all") is None


def test_parse_json_lenient_empty_input_returns_none() -> None:
    assert _parse_json_lenient("") is None


def test_parse_json_lenient_array_returns_none() -> None:
    # Top-level non-dict JSON must be rejected — callers expect a dict.
    assert _parse_json_lenient("[1, 2, 3]") is None


def test_parse_json_lenient_unbalanced_braces_returns_none() -> None:
    # No closing brace at all — the rfind fallback finds nothing valid.
    assert _parse_json_lenient('{"a": 1') is None


def test_parse_json_lenient_nested_object() -> None:
    text = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
    assert _parse_json_lenient(text) == {"outer": {"inner": [1, 2, {"deep": True}]}}


# ---- _extract_result_json ------------------------------------------------


def test_extract_result_json_single_match() -> None:
    stdout = 'epoch 1 done\nRESULT_JSON: {"score": 0.42}\n'
    assert _extract_result_json(stdout) == {"score": 0.42}


def test_extract_result_json_multiple_matches_last_wins() -> None:
    stdout = (
        'preview RESULT_JSON: {"score": 0.1}\n'
        'middle log line\n'
        'RESULT_JSON: {"score": 0.99, "final": true}\n'
    )
    assert _extract_result_json(stdout) == {"score": 0.99, "final": True}


def test_extract_result_json_no_match_returns_none() -> None:
    assert _extract_result_json("plain stdout with no marker\n") is None


def test_extract_result_json_empty_returns_none() -> None:
    assert _extract_result_json("") is None


def test_extract_result_json_malformed_json_returns_none() -> None:
    # Marker present but the JSON is broken — must not raise.
    stdout = 'RESULT_JSON: {"score": NOT_VALID}\n'
    assert _extract_result_json(stdout) is None


def test_extract_result_json_ignores_marker_without_braces() -> None:
    # No braces on the line at all.
    assert _extract_result_json("RESULT_JSON: oops\n") is None


# ---- _strip_outer_fence --------------------------------------------------


def test_strip_outer_fence_removes_lang_fence() -> None:
    assert _strip_outer_fence("```python\nprint('x')\n```") == "print('x')"


def test_strip_outer_fence_removes_bare_fence() -> None:
    assert _strip_outer_fence("```\nhello\n```") == "hello"


def test_strip_outer_fence_passes_through_unfenced() -> None:
    assert _strip_outer_fence("hello\nworld") == "hello\nworld"


def test_strip_outer_fence_passes_through_partial_fence() -> None:
    # Opening fence only — not a full match, return unchanged.
    assert _strip_outer_fence("```\nhello") == "```\nhello"


def test_strip_outer_fence_with_surrounding_whitespace() -> None:
    # Leading/trailing whitespace is stripped before matching, so this works.
    assert _strip_outer_fence("\n```json\n{\"a\":1}\n```\n") == '{"a":1}'


# ---- _slugify ------------------------------------------------------------


def test_slugify_basic() -> None:
    assert _slugify("Hello World") == "hello-world"


def test_slugify_collapses_runs_of_punct() -> None:
    assert _slugify("foo!!!  bar___baz") == "foo-bar-baz"


def test_slugify_empty_returns_untitled() -> None:
    assert _slugify("") == "untitled"


def test_slugify_all_non_alnum_returns_untitled() -> None:
    assert _slugify("!!!---???") == "untitled"


def test_slugify_unicode_falls_back_to_untitled() -> None:
    # The regex keeps only ASCII a-z0-9, so pure non-ASCII becomes empty.
    assert _slugify("日本語") == "untitled"


def test_slugify_unicode_mixed_keeps_ascii_run() -> None:
    assert _slugify("hello 日本 world") == "hello-world"


def test_slugify_strips_leading_trailing_dashes() -> None:
    assert _slugify("---abc---") == "abc"


def test_slugify_lowercases_and_keeps_digits() -> None:
    assert _slugify("Run42 v2") == "run42-v2"


# ---- _new_quest_id -------------------------------------------------------


def test_new_quest_id_unique_across_rapid_calls() -> None:
    ids = {_new_quest_id("same seed") for _ in range(64)}
    # uuid hex suffix guarantees uniqueness even at sub-second granularity.
    assert len(ids) == 64


def test_new_quest_id_includes_slug() -> None:
    qid = _new_quest_id("Photonic Crystal Optimization")
    assert "photonic-crystal-optimization" in qid


def test_new_quest_id_falls_back_for_empty_seed() -> None:
    qid = _new_quest_id("")
    assert "untitled" in qid


def test_new_quest_id_truncates_long_slug() -> None:
    long_seed = "a" * 200
    qid = _new_quest_id(long_seed)
    # The slug portion is capped at 32 chars; the whole id is "<ts>-<slug>-<6hex>".
    parts = qid.split("-")
    # First chunk is the unix timestamp (digits), last is the 6-hex tail.
    assert parts[0].isdigit()
    assert len(parts[-1]) == 6


# ---- _route_after_review -------------------------------------------------


def _route_config(
    tmp_path: Path, *, review_loop: bool, max_iterations: int
) -> Config:
    return Config(
        topic="route audit",
        title="route-audit",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=max_iterations, review_loop=review_loop),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def test_route_after_review_loop_disabled_always_done(tmp_path: Path) -> None:
    engine = Engine(_route_config(tmp_path, review_loop=False, max_iterations=5))
    state = {"review": {"verdict": "revise"}, "iteration": 0}
    assert engine._route_after_review(state) == "done"


def test_route_after_review_accept_returns_done(tmp_path: Path) -> None:
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    state = {"review": {"verdict": "accept"}, "iteration": 0}
    assert engine._route_after_review(state) == "done"


def test_route_after_review_revise_under_cap_returns_revise(tmp_path: Path) -> None:
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    state = {"review": {"verdict": "revise"}, "iteration": 1}
    assert engine._route_after_review(state) == "revise"


def test_route_after_review_revise_at_cap_returns_done(tmp_path: Path) -> None:
    # iteration just incremented to max_iterations -> no further revisions.
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    state = {"review": {"verdict": "revise"}, "iteration": 2}
    assert engine._route_after_review(state) == "done"


def test_route_after_review_missing_review_field_returns_done(tmp_path: Path) -> None:
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    # No "review" in state at all — should not raise; default verdict is "accept".
    assert engine._route_after_review({"iteration": 0}) == "done"


def test_route_after_review_missing_iteration_treated_as_zero(tmp_path: Path) -> None:
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=1))
    state = {"review": {"verdict": "revise"}}
    # iteration defaults to 0, 0 < 1 => revise.
    assert engine._route_after_review(state) == "revise"


# ---- graph topology ------------------------------------------------------


def test_conditional_edge_targets_match_route_return_values(tmp_path: Path) -> None:
    """The conditional dict {"revise": "design", "done": END} must cover both
    possible return values of `_route_after_review` and nothing else.
    """
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=3))

    seen: set[str] = set()
    for verdict in ("accept", "revise", "reject", "weird-other"):
        for it in (0, 1, 2, 3, 99):
            seen.add(engine._route_after_review({"review": {"verdict": verdict}, "iteration": it}))

    assert seen == {"revise", "done"}


def test_build_graph_review_has_conditional_edges_to_design_and_end(tmp_path: Path) -> None:
    """Verify topology: linear ideate->...->review and conditional review->design|END."""
    from langgraph.graph import END, START

    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    g = engine._build_graph()

    expected_nodes = {
        "ideate", "literature", "design", "implement",
        "execute", "analyze", "write", "review",
    }
    assert expected_nodes.issubset(set(g.nodes))

    # Linear edges that must be present.
    plain_edges = set(g.edges)
    assert (START, "ideate") in plain_edges
    assert ("ideate", "literature") in plain_edges
    assert ("literature", "design") in plain_edges
    assert ("design", "implement") in plain_edges
    assert ("implement", "execute") in plain_edges
    assert ("execute", "analyze") in plain_edges
    assert ("analyze", "write") in plain_edges
    assert ("write", "review") in plain_edges

    # Conditional edge mapping: route output -> next node.
    branches = g.branches.get("review") or {}
    assert branches, "expected at least one conditional branch on `review`"
    branch = next(iter(branches.values()))
    # path_map is the {"revise": "design", "done": END} dict.
    assert branch.ends == {"revise": "design", "done": END}
