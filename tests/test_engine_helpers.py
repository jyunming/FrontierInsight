"""Direct unit tests for the helper functions in `core.engine`.

These avoid the full LangGraph DAG + venv + subprocess machinery exercised
by `test_engine_smoke.py` so they run in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
    _parse_implement_response,
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


# ---- _parse_json_lenient logging contract --------------------------------


def test_parse_json_lenient_logs_warning_on_parse_failure(caplog) -> None:
    """When JSON parsing definitively fails, a single WARNING line
    must fire with the raw LLM text (truncated). Without this,
    callers using the ``parsed or {"foo": "(parse failed)"}`` idiom
    silently inject dummy values into QuestState and a developer
    debugging a prompt change has no way to see what the model
    actually returned. Audit regression guard."""
    import logging

    bad_text = "I'm sorry, I can't generate JSON. Here is some prose instead."
    with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
        result = _parse_json_lenient(bad_text)

    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.message for r in warnings]
    msg = warnings[0].getMessage()
    assert "JSON parse failed" in msg
    # Raw text snippet must appear so the dev can see what broke.
    assert "I'm sorry" in msg
    assert "prose instead" in msg


def test_parse_json_lenient_logs_truncated_raw_text(caplog) -> None:
    """Very long bad outputs must be truncated in the log to keep
    run.log readable; the truncation marker must be present so the
    reader knows there's more."""
    import logging

    huge = "garbage " * 1000  # ~8 KB, well over default 500-char cap
    with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
        _parse_json_lenient(huge)

    msg = caplog.records[-1].getMessage()
    # Should show "chars truncated" marker, not the full 8 KB.
    assert "chars truncated" in msg
    assert len(msg) < 2000, f"log msg should not contain full 8KB payload"


def test_parse_json_lenient_includes_node_tag_when_passed(caplog) -> None:
    """Callers may pass ``node="design"`` (or any node name) and the
    log line includes ``node=design`` for quick grep. Empty node
    (the default) must NOT inject a dangling ``node=`` substring."""
    import logging

    with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
        _parse_json_lenient("nope", node="design")
    msg = caplog.records[-1].getMessage()
    assert "node=design" in msg

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
        _parse_json_lenient("nope")  # no node= kwarg
    msg = caplog.records[-1].getMessage()
    assert "node=" not in msg


def test_parse_json_lenient_empty_input_does_not_log(caplog) -> None:
    """Empty input is not a parse failure — there was nothing to parse.
    Don't spam WARNING for every node that happens to get an empty
    response (which is itself a different problem, surfaced elsewhere)."""
    import logging

    with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
        result = _parse_json_lenient("")
    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [], [r.message for r in warnings]


def test_parse_json_lenient_array_does_not_log(caplog) -> None:
    """When the LLM returns valid JSON but the wrong shape (a list,
    not an object), the JSON itself parsed fine — the contract is
    upstream of this function. Don't WARN."""
    import logging

    with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
        result = _parse_json_lenient("[1, 2, 3]")
    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [], [r.message for r in warnings]


def test_parse_json_lenient_whitespace_only_does_not_log(caplog) -> None:
    """Whitespace-only and empty-fence-only inputs collapse to empty
    after ``_strip_outer_fence(...).strip()``. Both must be treated as
    silent no-ops, NOT as parse failures — otherwise a model that
    happens to emit trailing whitespace would log WARNING noise on
    every node. Regression for PR #54 bot comment."""
    import logging

    # Cases that genuinely collapse to "" after _strip_outer_fence + strip:
    #   - pure whitespace at top level
    #   - well-formed empty fenced block (requires content-newline pair
    #     inside, so "```\n\n```" matches the regex with empty content)
    #   - same but with a language tag
    # NOT tested here: malformed fences like "```\n```" (single \n between
    # open and close) don't match the regex, fall through to the no-braces
    # path, and DO log a WARNING — which is correct behavior (the LLM
    # emitted something syntactically weird and the user should know).
    cases = [
        "   ",                   # pure whitespace
        "\n\n\t",                # whitespace newlines/tabs
        "```\n\n```",            # empty fenced block (regex matches, content="")
        "```python\n\n```",      # empty fenced block with language
    ]
    for text in cases:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="frontier_insight.engine"):
            assert _parse_json_lenient(text) is None, text
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [], (
            f"{text!r} produced a spurious WARNING: "
            f"{[r.message for r in warnings]!r}"
        )


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
        "clarify", "ideate", "literature", "design", "implement",
        "execute", "analyze", "write", "review",
    }
    assert expected_nodes.issubset(set(g.nodes))

    # Linear edges that must be present. Phase I inserted `clarify`
    # between START and `ideate`. The no-simulation mode turned
    # ``design → implement`` into a conditional edge (implement vs
    # wait_for_data) — see the branches check below.
    plain_edges = set(g.edges)
    assert (START, "clarify") in plain_edges
    assert ("clarify", "ideate") in plain_edges
    assert ("ideate", "literature") in plain_edges
    assert ("literature", "design") in plain_edges
    assert ("implement", "execute") in plain_edges
    # Phase K replaced `execute → analyze` with `execute → execute_reflect`
    # plus a conditional `execute_reflect → execute | analyze` edge.
    assert ("execute", "execute_reflect") in plain_edges
    # no-simulation chain: wait_for_data → data_load → analyze. Both
    # new nodes feed analyze on the no-sim path.
    assert ("wait_for_data", "data_load") in plain_edges
    assert ("data_load", "analyze") in plain_edges
    # Phase L replaced `analyze → write` with `analyze → cross_check` plus
    # a conditional `cross_check → write | design` edge.
    assert ("analyze", "cross_check") in plain_edges
    assert ("write", "review") in plain_edges

    # Conditional branches: review-revise, execute-reflect-retry,
    # cross-check-redesign, AND the new design → implement | wait_for_data.
    assert "review" in g.branches
    assert "execute_reflect" in g.branches
    assert "cross_check" in g.branches
    assert "design" in g.branches, (
        "design must have a conditional edge for no-simulation routing"
    )

    review_branch = next(iter(g.branches["review"].values()))
    assert review_branch.ends == {"revise": "design", "done": END}
    reflect_branch = next(iter(g.branches["execute_reflect"].values()))
    assert reflect_branch.ends == {"retry": "execute", "proceed": "analyze"}
    cross_branch = next(iter(g.branches["cross_check"].values()))
    assert cross_branch.ends == {"write": "write", "redesign": "design"}
    design_branch = next(iter(g.branches["design"].values()))
    assert design_branch.ends == {
        "implement": "implement", "wait_for_data": "wait_for_data",
    }


# ---- _parse_implement_response ------------------------------------------


def test_parse_implement_response_fenced_block_plus_deps_line() -> None:
    """Happy path for the new fenced-block format (the whole reason this
    parser exists — kills the JSON-escape overhead that was hanging
    implement on long streams)."""
    text = (
        "```python\n"
        "import numpy as np\n"
        "print('RESULT_JSON: {}')\n"
        "```\n"
        "DEPS: numpy, matplotlib\n"
    )
    code, deps = _parse_implement_response(text)
    assert "import numpy as np" in code
    assert "print('RESULT_JSON: {}')" in code
    assert deps == ["numpy", "matplotlib"]


def test_parse_implement_response_fenced_no_language_tag() -> None:
    """Some models emit ``` instead of ```python. Still parse."""
    text = "```\nprint('ok')\n```\nDEPS: numpy"
    code, deps = _parse_implement_response(text)
    assert code == "print('ok')"
    assert deps == ["numpy"]


def test_parse_implement_response_deps_with_brackets_and_quotes() -> None:
    """Tolerate ``DEPS: ['numpy', 'scipy']`` even though we asked for
    bare names — models often regress to list syntax."""
    text = "```python\npass\n```\nDEPS: ['numpy', 'scipy']"
    code, deps = _parse_implement_response(text)
    assert code == "pass"
    assert deps == ["numpy", "scipy"]


def test_parse_implement_response_no_deps_line_returns_empty_list() -> None:
    """Missing DEPS line is fine — design_deps union covers most cases.
    Empty list means 'I declare nothing', not 'parser broke'."""
    text = "```python\nimport sys\nprint('ok')\n```"
    code, deps = _parse_implement_response(text)
    assert code == "import sys\nprint('ok')"
    assert deps == []


def test_parse_implement_response_falls_back_to_legacy_json() -> None:
    """Models occasionally regress to the old ``{"code": "...", "deps": [...]}``
    shape; don't punish them — fall through to _parse_json_lenient."""
    text = '{"code": "print(42)", "deps": ["numpy"]}'
    code, deps = _parse_implement_response(text)
    assert code == "print(42)"
    assert deps == ["numpy"]


def test_parse_implement_response_empty_and_garbage_return_empty() -> None:
    assert _parse_implement_response("") == ("", [])
    assert _parse_implement_response("no code here, sorry") == ("", [])


def test_parse_implement_response_picks_first_fenced_block() -> None:
    """``_PY_FENCE_RE`` uses ``.search()`` with a non-greedy ``(.*?)``,
    so it always returns the FIRST fenced block — that is the contract.
    The prompt explicitly asks for exactly one fence; if a model emits
    a tiny prose-example fence before the real one, the contract is
    violated and the small fence wins (caller can log/retry)."""
    text = (
        "```python\nimport numpy\nprint(1)\n```\n"
        "DEPS: numpy\n"
    )
    code, deps = _parse_implement_response(text)
    assert "import numpy" in code
    assert deps == ["numpy"]


def test_parse_implement_response_pep508_extras_are_preserved() -> None:
    """Regression for PR #27 review: a naive ``raw.strip("[](){}")`` would
    chew the trailing ``]`` off ``pandas[performance]``, producing the
    broken spec ``pandas[performance`` that pip can't install. Only one
    matched OUTER pair should be peeled."""
    text = (
        "```python\nimport pandas\n```\n"
        "DEPS: pandas[performance], numpy\n"
    )
    code, deps = _parse_implement_response(text)
    assert "pandas[performance]" in deps
    assert "numpy" in deps


def test_parse_implement_response_pep508_extras_inside_brackets() -> None:
    """Same as above but with the outer list-syntax brackets too:
    ``DEPS: [pandas[performance], scipy]``. Strip exactly one outer
    matched pair; preserve the inner ``[performance]`` extras."""
    text = (
        "```python\npass\n```\n"
        "DEPS: [pandas[performance], scipy]\n"
    )
    code, deps = _parse_implement_response(text)
    assert "pandas[performance]" in deps
    assert "scipy" in deps


def test_parse_implement_response_ignores_deps_assignment_inside_fence() -> None:
    """Regression for PR #27 review: a Python statement like
    ``deps = ["numpy"]`` inside the fenced experiment code must NOT be
    misread as the metadata DEPS line. The parser only searches the
    post-fence tail for DEPS."""
    text = (
        "```python\n"
        "deps = ['fake_inside_fence']\n"
        "import scipy\n"
        "```\n"
        "DEPS: scipy\n"
    )
    code, deps = _parse_implement_response(text)
    assert "import scipy" in code
    assert deps == ["scipy"]
    assert "fake_inside_fence" not in deps


def test_parse_implement_response_legacy_deps_as_string() -> None:
    """Regression for PR #27 review: legacy JSON shape with deps as a
    string (``"deps": "numpy"``) — coerce to a single-element list, NOT
    a per-character list."""
    text = '{"code": "print(1)", "deps": "numpy"}'
    code, deps = _parse_implement_response(text)
    assert code == "print(1)"
    assert deps == ["numpy"]


def test_parse_implement_response_legacy_deps_as_comma_string() -> None:
    """And a comma-separated single string splits correctly."""
    text = '{"code": "print(1)", "deps": "numpy, scipy"}'
    code, deps = _parse_implement_response(text)
    assert sorted(deps) == ["numpy", "scipy"]


# ---- _format_lit / _format_lit_from_state: literature context window ----


def test_format_lit_uses_2000_char_window() -> None:
    """Regression for the paper-depth fix: the per-paper excerpt fed to
    write.md was 600 chars — too thin to discuss prior work by content.
    Bumped to 2000 chars. Pin both helpers (in-memory + state-loaded)."""
    from core.engine import (
        _LIT_EXCERPT_CHARS, _format_lit, _format_lit_from_state,
    )

    assert _LIT_EXCERPT_CHARS == 2000

    long_body = "A" * 5000
    from core.knowledge import RetrievedDoc
    docs = [RetrievedDoc(
        content=long_body, metadata={"title": "huge paper"},
    )]
    rendered = _format_lit(docs)
    # We render `[i] title\n<2000 chars of content>`; the content slice
    # is exactly _LIT_EXCERPT_CHARS, not the full body.
    assert "A" * _LIT_EXCERPT_CHARS in rendered
    assert "A" * (_LIT_EXCERPT_CHARS + 1) not in rendered

    # Same invariant on the from-state path used during checkpoint resume.
    state = {"literature": [{
        "content": long_body, "metadata": {"title": "huge paper"},
    }]}
    rendered_state = _format_lit_from_state(state)  # type: ignore[arg-type]
    assert "A" * _LIT_EXCERPT_CHARS in rendered_state
    assert "A" * (_LIT_EXCERPT_CHARS + 1) not in rendered_state


# ---- Prompt-template shape: review + analyze receive $clarify_block ----


def test_review_prompt_advertises_rigor_and_depth_axes() -> None:
    """Pin the contract: the review prompt asks the model to return
    rigor_score and depth_score keys. If these are removed the revise
    loop loses its depth signal (the whole point of this change)."""
    from pathlib import Path
    review_md = (Path(__file__).resolve().parent.parent
                 / "agents" / "review.md").read_text(encoding="utf-8")
    assert "rigor_score" in review_md
    assert "depth_score" in review_md
    # Must also accept the clarify block so it can read study_depth.
    assert "$clarify_block" in review_md


def test_analyze_prompt_accepts_clarify_block() -> None:
    """Wired up so analyze respects the study_depth setting end-to-end."""
    from pathlib import Path
    analyze_md = (Path(__file__).resolve().parent.parent
                  / "agents" / "analyze.md").read_text(encoding="utf-8")
    assert "$clarify_block" in analyze_md


def test_write_prompt_authors_title_not_slug() -> None:
    """write.md must instruct the model to AUTHOR a Title-Case title
    instead of just echoing `# $title` (the YAML slug)."""
    from pathlib import Path
    write_md = (Path(__file__).resolve().parent.parent
                / "agents" / "write.md").read_text(encoding="utf-8")
    # Old contract:  "Begin with `# $title`"     (model echoed the slug)
    # New contract:  "MUST be a proper Title-Case academic title"
    assert "Title-Case" in write_md or "Title Case" in write_md
    assert "Do NOT use the raw slug" in write_md


# ---- _quest_logger lifecycle ---------------------------------------------


def test_quest_logger_releases_file_handler_on_close(tmp_path: Path) -> None:
    """After ``_close_quest_logger``, the per-quest run.log FileHandler
    must be closed AND removed from the logger's handler list. On
    Windows, this is what releases the file lock so a subsequent
    ``shutil.rmtree`` of the quest directory can succeed.

    Regression for the Windows test-cleanup cascade: prior to the
    Engine.run finally-block fix, the FileHandler stayed open for the
    process lifetime and broke .pytest_tmp cleanup across multiple
    test sessions in this codebase."""
    import logging
    from core.engine import _quest_logger, _close_quest_logger

    fi_dir = tmp_path / ".fi"
    logger = _quest_logger("test-quest-lifecycle-aabbcc", fi_dir)

    # FileHandler was added.
    file_handlers = [
        h for h in logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    handler = file_handlers[0]
    # ``logging.FileHandler`` opens the file eagerly at construction
    # (delay=False by default), so handler.stream is already an open
    # file object here. Log a line for good measure so the test also
    # exercises the write path.
    logger.info("smoke")
    assert handler.stream is not None
    assert not handler.stream.closed

    # Close it.
    _close_quest_logger("test-quest-lifecycle-aabbcc")

    # Handler is gone from the logger.
    assert not any(
        isinstance(h, logging.FileHandler) for h in logger.handlers
    ), f"FileHandler still attached: {logger.handlers!r}"
    # The handler's stream is closed.
    assert handler.stream is None or handler.stream.closed


def test_quest_logger_can_be_reopened_after_dir_recreate(tmp_path: Path) -> None:
    """The hard case: a test deletes the quest dir and recreates it
    with the SAME quest_id. The second ``_quest_logger`` call must
    NOT inherit the now-stale FileHandler from the first call —
    otherwise log lines go to a deleted path and the new run.log
    is never written.

    This is the precise failure mode that broke Windows test cleanup
    multiple times before the lifecycle fix landed."""
    import logging
    import shutil
    from core.engine import _quest_logger, _close_quest_logger

    qid = "test-recreate-aabbcc"
    fi_dir_1 = tmp_path / "run1" / ".fi"
    _quest_logger(qid, fi_dir_1)
    logging.getLogger(f"frontier_insight.{qid}").info("first run")
    _close_quest_logger(qid)
    # Whole tree is removable now.
    shutil.rmtree(tmp_path / "run1")
    assert not (tmp_path / "run1").exists()

    # Recreate the SAME quest_id with a different fi_dir.
    fi_dir_2 = tmp_path / "run2" / ".fi"
    logger2 = _quest_logger(qid, fi_dir_2)
    logging.getLogger(f"frontier_insight.{qid}").info("second run")

    # The handler MUST point at the new path, not the deleted one.
    file_handlers = [
        h for h in logger2.handlers if isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename).resolve() == (
        fi_dir_2 / "run.log"
    ).resolve(), (
        f"second _quest_logger call returned a stale handler pointing at "
        f"{file_handlers[0].baseFilename!r} — should have pointed at "
        f"{(fi_dir_2 / 'run.log')!r}"
    )

    # And the second run.log actually got written.
    assert (fi_dir_2 / "run.log").exists()
    assert "second run" in (fi_dir_2 / "run.log").read_text(encoding="utf-8")

    _close_quest_logger(qid)


def test_close_quest_logger_idempotent() -> None:
    """``_close_quest_logger`` must be safe to call multiple times —
    the Engine.run finally-block calls it, and the artifact-collection
    try/finally calls it again; a second call must not error."""
    from core.engine import _close_quest_logger

    qid = "test-idempotent-aabbcc"
    _close_quest_logger(qid)
    _close_quest_logger(qid)   # no logger exists yet → no-op
    _close_quest_logger(qid)


def test_close_quest_logger_logger_with_no_handlers_is_safe() -> None:
    """If a logger was never instantiated (no _quest_logger call),
    closing it must still be a no-op rather than raising."""
    import logging
    from core.engine import _close_quest_logger

    qid = "test-never-instantiated-aabbcc"
    # Forcibly clear any handlers (Python's logging module may have
    # left an empty Logger object from a prior test run).
    logger = logging.getLogger(f"frontier_insight.{qid}")
    assert logger.handlers == []
    _close_quest_logger(qid)
    assert logger.handlers == []


# ---- no-simulation mode helpers ------------------------------------------


def test_list_user_data_files_skips_readme_and_dotfiles(tmp_path: Path) -> None:
    """The data_load node walks <quest_root>/data/ on every resume.
    The engine-authored README.md and any dot-prefixed files must be
    excluded so they don't pollute the prompt corpus."""
    from core.engine import _list_user_data_files

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "README.md").write_text("auto-written by FI\n")
    (data_dir / "survey.csv").write_text("a,b,c\n1,2,3\n")
    (data_dir / "notes.md").write_text("# field notes\n")
    (data_dir / ".DS_Store").write_bytes(b"\x00\x01")
    (data_dir / ".hidden.txt").write_text("ignored\n")

    files = _list_user_data_files(data_dir)
    names = [p.name for p in files]
    assert "survey.csv" in names
    assert "notes.md" in names
    assert "README.md" not in names, (
        "README.md is the FI-authored instruction file; must not be "
        "treated as user data"
    )
    assert ".DS_Store" not in names
    assert ".hidden.txt" not in names


def test_list_user_data_files_returns_empty_when_dir_missing(
    tmp_path: Path,
) -> None:
    from core.engine import _list_user_data_files
    assert _list_user_data_files(tmp_path / "does-not-exist") == []


def test_list_user_data_files_is_deterministic(tmp_path: Path) -> None:
    """Stable sort by path so the data_load prompt is reproducible
    across resumes — the LLM sees the same file IDs each time."""
    from core.engine import _list_user_data_files

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ["z.csv", "a.csv", "m.csv"]:
        (data_dir / name).write_text("x\n")

    first = [p.name for p in _list_user_data_files(data_dir)]
    second = [p.name for p in _list_user_data_files(data_dir)]
    assert first == second == ["a.csv", "m.csv", "z.csv"]


def test_render_data_readme_includes_topic_and_resume_command(
    tmp_path: Path,
) -> None:
    """The README dropped into <quest_root>/data/ must tell the user
    (a) the topic so they remember what they were collecting data for,
    (b) the hypothesis from the design node,
    (c) the resume command with the right quest_id pasted in."""
    from core.engine import _render_data_readme

    state = {
        "topic": "Belgium vs Taiwan culture comparison",
        "design": {
            "hypothesis": "Trust in institutions correlates with X.",
            "method": "Survey 100 respondents per country.",
            "variables": {"trust_score": "Likert 1-7"},
        },
    }
    md = _render_data_readme(state, "1778751621-test-aabbcc")

    assert "Belgium vs Taiwan culture comparison" in md
    assert "Trust in institutions" in md
    assert "fi --resume 1778751621-test-aabbcc" in md
    # Format hints the user actually needs.
    assert ".csv" in md
    assert ".md" in md
    assert ".pdf" in md
    # The variables block should make it through.
    assert "trust_score" in md


def test_render_data_readme_handles_missing_design(tmp_path: Path) -> None:
    """If the design node failed or its output is sparse, the README
    should still be generated — just with fallback text for the
    missing fields. No KeyError."""
    from core.engine import _render_data_readme

    state = {"topic": "Some qualitative question"}  # no design key
    md = _render_data_readme(state, "1700000000-foo-aabbcc")
    assert "Some qualitative question" in md
    # The hypothesis fallback should mention the design node may have failed.
    assert "design node may have failed" in md


def test_engine_config_no_simulation_default_false() -> None:
    """Backward compat: existing YAMLs without an engine.no_simulation
    field must continue to work. The default must be False so quests
    keep simulating unless the user opts in."""
    from core.config import EngineConfig
    assert EngineConfig().no_simulation is False
    assert EngineConfig(no_simulation=True).no_simulation is True


def test_resolve_no_simulation_from_clarify_yaml_wins(tmp_path: Path) -> None:
    """When ``engine.no_simulation: true`` is set in YAML, the resolved
    flag is True regardless of clarify_answers — even if clarify says
    'theoretical'. The YAML is the explicit user override."""
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(no_simulation=True, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    assert engine._resolve_no_simulation_from_clarify(
        {"empirical_vs_theoretical": "theoretical"},
    ) is True


def test_resolve_no_simulation_from_clarify_empirical_answer_triggers(
    tmp_path: Path,
) -> None:
    """When YAML doesn't set the flag but the clarify answer is
    'empirical', the resolved flag becomes True. Auto-detect path."""
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(no_simulation=False, clarify_mode="auto"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    assert engine._resolve_no_simulation_from_clarify(
        {"empirical_vs_theoretical": "empirical"},
    ) is True
    # And other answers don't trigger it.
    assert engine._resolve_no_simulation_from_clarify(
        {"empirical_vs_theoretical": "theoretical"},
    ) is False
    assert engine._resolve_no_simulation_from_clarify(
        {"empirical_vs_theoretical": "mixed"},
    ) is False
    assert engine._resolve_no_simulation_from_clarify({}) is False


def test_route_after_design_no_sim_flag_routes_to_wait_for_data(
    tmp_path: Path,
) -> None:
    """The routing function — the heart of the no-sim graph edge.
    When state carries ``no_simulation_resolved=True``, design must
    route to ``wait_for_data``. Otherwise to ``implement``."""
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    assert engine._route_after_design({"no_simulation_resolved": True}) == "wait_for_data"
    assert engine._route_after_design({"no_simulation_resolved": False}) == "implement"
    assert engine._route_after_design({}) == "implement"


def test_data_load_prompt_is_in_loaded_prompts() -> None:
    """``_load_prompts`` must include ``data_load.md`` so
    ``_node_data_load`` can ``self._prompts["data_load"].substitute(...)``
    without a KeyError. Regression for missed prompt registration."""
    from core.engine import _load_prompts
    prompts = _load_prompts()
    assert "data_load" in prompts
    # And the template must have the variables the node passes in.
    src = prompts["data_load"].template
    # ``string.Template`` accepts both ``$var`` and ``${var}`` syntax;
    # the prompt uses the ``${var}`` form (less ambiguous near JSON
    # braces). Check for either spelling.
    for var in ("topic", "design_block", "file_manifest", "content_blocks"):
        assert (f"${var}" in src) or ("${" + var + "}" in src), (
            f"data_load.md missing template var ${var}"
        )


@pytest.mark.asyncio
async def test_engine_run_closes_logger_on_exception_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer ``try/finally`` in ``Engine.run`` must close the
    per-quest run.log FileHandler even when the graph invoke (or any
    inner code) raises. The earlier version of this fix only closed
    on the success path — exceptions leaked the file lock and broke
    Windows test cleanup.

    Regression for PR #56 bot comment + the docstring claim of
    'every exit path'."""
    import logging
    import shutil
    from core.engine import Engine

    cfg = Config(
        topic="exception path test",
        title="exception-path-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)

    # Force ``executor.setup`` to raise; this fires BEFORE the
    # inner try/finally for LLM cleanup, so without the outer
    # try/finally fix, the logger would leak.
    async def boom(*a, **kw):
        raise RuntimeError("executor setup intentionally failed")
    monkeypatch.setattr(engine.executor, "setup", boom)

    with pytest.raises(RuntimeError, match="executor setup intentionally failed"):
        await engine.run()

    # Logger handlers must be gone. If they aren't, the FileHandler
    # is still holding the file lock — confirm by deleting the dir.
    quest_logger = logging.getLogger(f"frontier_insight.{engine.quest_id}")
    file_handlers = [
        h for h in quest_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    assert file_handlers == [], (
        f"FileHandler leaked on exception path: {file_handlers!r}"
    )
    # And actually rmtree the quest dir to verify the file lock is
    # released. On Windows this would fail with PermissionError if the
    # leak regressed.
    shutil.rmtree(engine.quest_root)
    assert not engine.quest_root.exists()
