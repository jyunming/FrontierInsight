"""Direct unit tests for the helper functions in `core.engine`.

These avoid the full LangGraph DAG + venv + subprocess machinery exercised
by `test_engine_smoke.py` so they run in milliseconds.
"""

from __future__ import annotations

import re
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
    every node."""
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


def test_slugify_pure_non_latin_uses_hash_fallback() -> None:
    """Quest-id slug must be ASCII (the digest / critique / --resume
    regex enforces it). Pre-fix: pure non-ASCII collapsed to the
    constant ``"untitled"`` so every CJK quest shared the same id
    prefix. Post-fix: deterministic 8-hex hash so each distinct
    non-ASCII topic gets a distinct quest_id."""
    out = _slugify("日本語")
    assert out != "untitled"
    assert out.startswith("i18n-")
    assert len(out) == len("i18n-") + 8
    # Same input → same hash (determinism for /resume-by-topic).
    assert _slugify("日本語") == out
    # Different inputs → different hashes.
    assert _slugify("近視的遺傳影響") != out
    # ASCII characters never inside the i18n-... output.
    assert re.fullmatch(r"i18n-[0-9a-f]{8}", out)


def test_slugify_mixed_script_keeps_ascii_run() -> None:
    """Mixed CJK + ASCII keeps the ASCII portion verbatim — that's the
    most-readable quest_id available without transliterating CJK."""
    assert _slugify("hello 日本 world") == "hello-world"
    assert _slugify("Genetic 遺傳 impact") == "genetic-impact"


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
    tmp_path: Path, *, review_loop: bool, max_iterations: int,
    evidence_gate: bool = True,
) -> Config:
    return Config(
        topic="route audit",
        title="route-audit",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=max_iterations, review_loop=review_loop,
            evidence_gate=evidence_gate,
            # These tests exercise the legacy non-gated revise/done
            # routing — gate "off" so the human-feedback short-circuit
            # doesn't intercept.
            human_feedback_gate="off",
        ),
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


# ---- evidence_gate -------------------------------------------------------


def test_route_after_evidence_gate_reads_node_decision(tmp_path: Path) -> None:
    """The router just echoes the route the node already computed, and
    fails open to ``write`` when the gate was a passthrough."""
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    assert engine._route_after_evidence_gate(
        {"evidence_assessment": {"route": "broaden_lit"}}) == "broaden_lit"
    assert engine._route_after_evidence_gate(
        {"evidence_assessment": {"route": "write"}}) == "write"
    assert engine._route_after_evidence_gate({}) == "write"


async def test_evidence_gate_disabled_is_passthrough(tmp_path: Path) -> None:
    """engine.evidence_gate=False ⇒ no LLM call, empty patch, route=write."""
    engine = Engine(_route_config(
        tmp_path, review_loop=True, max_iterations=2, evidence_gate=False))
    patch = await engine._node_evidence_gate(
        {"topic": "t", "literature": [], "analysis": {}})
    assert patch == {}


async def test_evidence_gate_fails_open_on_unparseable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-JSON / invalid-verdict response must NOT block the quest —
    the gate defaults to a 'sufficient' verdict that routes to write."""
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))

    async def fake_chat(prompt, node=None):  # noqa: ANN001
        return "the evidence looks fine to me"  # not JSON

    monkeypatch.setattr(engine, "_chat", fake_chat)
    patch = await engine._node_evidence_gate(
        {"topic": "t", "literature": [{"content": "x"}], "analysis": {}})
    assert patch["evidence_assessment"]["verdict"] == "sufficient"
    assert patch["evidence_assessment"]["route"] == "write"


async def test_evidence_gate_broaden_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 'broaden' verdict re-enters literature at most
    evidence_gate_max_broaden (default 1) times, then proceeds to write
    even though the verdict is still 'broaden' — it can't spin."""
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))

    async def fake_chat(prompt, node=None):  # noqa: ANN001
        return '{"verdict": "broaden", "rationale": "thin", "gaps": ["more data"]}'

    monkeypatch.setattr(engine, "_chat", fake_chat)
    # Budget available → broaden, bump the counter.
    p1 = await engine._node_evidence_gate({"topic": "t", "evidence_broaden_count": 0})
    assert p1["evidence_assessment"]["route"] == "broaden_lit"
    assert p1["evidence_broaden_count"] == 1
    # Budget exhausted → write despite the same verdict; counter not bumped.
    p2 = await engine._node_evidence_gate({"topic": "t", "evidence_broaden_count": 1})
    assert p2["evidence_assessment"]["route"] == "write"
    assert "evidence_broaden_count" not in p2


async def test_evidence_gate_shows_source_titles_to_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is asked to judge whether sources are on-topic, so it must
    SEE them — not just a count. The source title + snippet must reach the
    prompt (an off-topic 'Banana Bread' source for a 'SpaceX revenue' topic
    can only be caught if the agent can read it)."""
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))
    captured: dict[str, str] = {}

    async def fake_chat(prompt, node=None):  # noqa: ANN001
        captured["prompt"] = prompt
        return '{"verdict": "insufficient", "gaps": ["on-topic sources"]}'

    monkeypatch.setattr(engine, "_chat", fake_chat)
    state = {
        "topic": "SpaceX revenue trend 2020-2024",
        "literature": [
            {"content": "Banana bread recipe with walnuts and cinnamon.",
             "metadata": {"title": "Best Banana Bread"}},
        ],
    }
    await engine._node_evidence_gate(state)
    assert "Best Banana Bread" in captured["prompt"]
    assert "Banana bread recipe" in captured["prompt"]


async def test_evidence_gate_fails_open_on_malformed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signal-gathering is inside the fail-open guard with per-item type
    checks: non-dict literature items, a non-dict analysis, and non-dict /
    wrong-shaped cross_check entries (resumed or malformed checkpoint) must
    NOT raise — the gate degrades to write and still tallies the good rows."""
    engine = Engine(_route_config(tmp_path, review_loop=True, max_iterations=2))

    async def fake_chat(prompt, node=None):  # noqa: ANN001
        return '{"verdict": "sufficient", "gaps": []}'

    monkeypatch.setattr(engine, "_chat", fake_chat)
    bad_state = {
        "topic": "t",
        "literature": ["not-a-dict", {"content": "real source"}, 42],
        "analysis": "not-a-dict",          # wrong shape
        "cross_check": [
            "nope",                          # non-dict entry
            {"hits": "not-a-list"},          # hits wrong shape
            {"hits": [{"stance": "supporting"}, "x"]},  # one good supporting
        ],
    }
    patch = await engine._node_evidence_gate(bad_state)
    assert patch["evidence_assessment"]["route"] == "write"
    # Only the one literature dict with non-empty content counts.
    assert patch["evidence_assessment"]["n_sources"] == 1
    assert patch["evidence_assessment"]["n_supporting"] == 1
    # The typed protocol is always recorded (built defensively before the
    # fail-open guard), even on malformed state.
    assert "research_protocol" in patch
    assert patch["research_protocol"]["topic_type"]


def test_execute_reflect_retries_on_empty_result_json(tmp_path: Path) -> None:
    """rc=0 but NO RESULT_JSON marker is stored as {} — that must trigger
    repair (retry), not be mistaken for a parsed result and proceed."""
    engine = Engine(_route_config(tmp_path, review_loop=False, max_iterations=1))
    # Empty result + budget remaining → retry.
    assert engine._route_after_execute_reflect(
        {"exec_result": {"returncode": 0}, "result_json": {},
         "exec_reflect_iter": 0}) == "retry"
    # A real (non-empty, non-degenerate) result → proceed.
    assert engine._route_after_execute_reflect(
        {"exec_result": {"returncode": 0}, "result_json": {"metric": 0.5},
         "exec_reflect_iter": 0}) == "proceed"
    # Empty result but retry budget exhausted → proceed (can't spin).
    assert engine._route_after_execute_reflect(
        {"exec_result": {"returncode": 0}, "result_json": {},
         "exec_reflect_iter": 99}) == "proceed"


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

    # Linear edges that must be present. `clarify` runs between
    # START and `ideate`. The no-simulation mode turned
    # ``design → implement`` into a conditional edge (implement vs
    # wait_for_data) — see the branches check below.
    plain_edges = set(g.edges)
    assert (START, "clarify") in plain_edges
    assert ("clarify", "ideate") in plain_edges
    assert ("ideate", "literature") in plain_edges
    assert ("literature", "design") in plain_edges
    assert ("implement", "execute") in plain_edges
    # `execute → execute_reflect` replaces the old `execute → analyze`
    # edge, plus a conditional `execute_reflect → execute | analyze`.
    assert ("execute", "execute_reflect") in plain_edges
    # no-simulation chain: auto_collect_data → wait_for_data →
    # data_load → web_plots → web_figures → analyze. auto_collect_data is
    # the agent-side retrieval (Axon + web) that runs BEFORE the user-data
    # pause; web_plots derives figures from the collected sources and
    # web_figures embeds license-clean illustrative figures before analyze
    # consumes them.
    assert ("auto_collect_data", "wait_for_data") in plain_edges
    assert ("wait_for_data", "data_load") in plain_edges
    assert ("data_load", "web_plots") in plain_edges
    assert ("web_plots", "web_figures") in plain_edges
    assert ("web_figures", "analyze") in plain_edges
    # `analyze → cross_check` replaces the old `analyze → write` edge,
    # plus a conditional `cross_check → write | design` edge.
    assert ("analyze", "cross_check") in plain_edges
    # `write → claim_check → review` (claim_check grounds each paper claim to
    # evidence; it's a passthrough when engine.claim_grounding is off).
    assert ("write", "claim_check") in plain_edges
    assert ("claim_check", "review") in plain_edges

    # Conditional branches: review-revise, execute-reflect-retry,
    # cross-check-redesign, AND the new design → implement | wait_for_data.
    assert "review" in g.branches
    assert "execute_reflect" in g.branches
    assert "cross_check" in g.branches
    assert "design" in g.branches, (
        "design must have a conditional edge for no-simulation routing"
    )

    review_branch = next(iter(g.branches["review"].values()))
    assert review_branch.ends == {
        "revise": "design",
        "done": END,
        # The human-feedback gate adds a third terminal: when
        # ``engine.human_feedback_gate: "after_review"`` the router
        # picks ``human_feedback`` instead of revise/done so the user
        # can accept / reject / refine.
        "human_feedback": "human_feedback",
    }
    # human_feedback node itself routes back to design (refine) or END
    # (accept/reject). Same ``revise``/``done`` labels the auto path uses.
    assert "human_feedback" in g.branches
    hf_branch = next(iter(g.branches["human_feedback"].values()))
    assert hf_branch.ends == {"revise": "design", "done": END}
    reflect_branch = next(iter(g.branches["execute_reflect"].values()))
    assert reflect_branch.ends == {"retry": "execute", "proceed": "analyze"}
    cross_branch = next(iter(g.branches["cross_check"].values()))
    # Three terminal labels: ``broaden_lit`` re-enters the literature
    # node (iterative literature loop), ``redesign`` re-enters design
    # for ``re_experiment``, and the happy-path "write" label now lands
    # on the evidence_gate node (which makes the real write-vs-broaden call).
    assert cross_branch.ends == {
        "write": "evidence_gate",
        "redesign": "design",
        "broaden_lit": "literature",
    }
    # evidence_gate routes to write (sufficient / insufficient-but-write)
    # or back to literature for one bounded broaden pass.
    assert "evidence_gate" in g.nodes
    assert "evidence_gate" in g.branches
    ev_branch = next(iter(g.branches["evidence_gate"].values()))
    assert ev_branch.ends == {"write": "write", "broaden_lit": "literature"}
    design_branch = next(iter(g.branches["design"].values()))
    # Two-stage implement: the simulate-path routing key stays
    # ``implement`` (for resume contract compatibility — the 609990
    # checkpoint pins ``next=("implement",)``) but the target is the
    # outline node, which then chains to the body node also named
    # ``implement``.
    assert design_branch.ends == {
        "implement": "implement_outline",
        "auto_collect_data": "auto_collect_data",
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
    """A naive ``raw.strip("[](){}")`` would chew the trailing ``]``
    off ``pandas[performance]``, producing the broken spec
    ``pandas[performance`` that pip can't install. Only one matched
    OUTER pair should be peeled."""
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
    """A Python statement like ``deps = ["numpy"]`` inside the fenced
    experiment code must NOT be misread as the metadata DEPS line.
    The parser only searches the post-fence tail for DEPS."""
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
    """Legacy JSON shape with deps as a string
    (``"deps": "numpy"``) — coerce to a single-element list, NOT
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
    # Include a year so the entry passes _is_citable (title alone is
    # not enough — title-only entries get filtered upstream so the
    # writer LLM never sees "[i] item-i" stubs that would tempt it to
    # invent an author).
    cite_meta = {"title": "huge paper", "year": 2024}
    docs = [RetrievedDoc(content=long_body, metadata=cite_meta)]
    rendered = _format_lit(docs)
    # We render `[i] title\n<2000 chars of content>`; the content slice
    # is exactly _LIT_EXCERPT_CHARS, not the full body.
    assert "A" * _LIT_EXCERPT_CHARS in rendered
    assert "A" * (_LIT_EXCERPT_CHARS + 1) not in rendered

    # Same invariant on the from-state path used during checkpoint resume.
    state = {"literature": [{"content": long_body, "metadata": cite_meta}]}
    rendered_state = _format_lit_from_state(state)  # type: ignore[arg-type]
    assert "A" * _LIT_EXCERPT_CHARS in rendered_state
    assert "A" * (_LIT_EXCERPT_CHARS + 1) not in rendered_state


# ---- _is_citable: drop weak Axon entries so the writer can't fabricate ----


def test_is_citable_drops_title_only_entries() -> None:
    """An entry with only a title (no year/author/venue/doi/url) is
    not enough for a real citation — the writer LLM previously invented
    "Prior work (2026)" and bogus URLs to fill the missing slots. Drop
    such entries upstream so the writer never sees them."""
    from core.engine import _is_citable
    assert not _is_citable({"title": "huge paper"})
    assert not _is_citable({"title": "", "year": 2024})  # no real title
    assert not _is_citable({})


def test_is_citable_keeps_entries_with_one_id_field() -> None:
    """Title + ANY one of (authors / year / venue / doi / url / arxiv_id)
    is enough — the citation will at least be traceable."""
    from core.engine import _is_citable
    for field, value in [
        ("authors", ["Mack, C."]),
        ("year", 2024),
        ("published", "2024-04-01"),
        ("venue", "J. Micro/Nanolith."),
        ("publisher", "SPIE"),
        ("doi", "10.1117/12.1234567"),
        ("arxiv_id", "2403.12345"),
        ("url", "https://example.org/p.pdf"),
    ]:
        meta = {"title": "Stochastic LER", field: value}
        assert _is_citable(meta), f"entry with {field} should be citable: {meta}"


def test_format_lit_drops_unusable_entries_entirely() -> None:
    """When the Axon pull is all weak metadata, _format_lit should
    emit the empty-knowledge-base sentinel rather than a numbered
    list of stub entries (which the writer would treat as real and
    fabricate authors/URLs for)."""
    from core.engine import _format_lit
    from core.knowledge import RetrievedDoc
    weak = [
        RetrievedDoc(content="...", metadata={"title": "x"}),     # no id
        RetrievedDoc(content="...", metadata={}),                  # nothing
        RetrievedDoc(content="...", metadata={"title": ""}),       # blank title
    ]
    assert _format_lit(weak) == "(no prior work surfaced from the knowledge base)"


def test_is_audience_appropriate_external_drops_fi_internal_kinds() -> None:
    """External-facing papers must not cite cross-quest memory
    artifacts (kind=fi_critique / fi_digest / fi_portfolio /
    fi_proposal / fi_summary / fi_source_catalog) — an outside reader
    can't look them up."""
    from core.engine import _is_audience_appropriate
    for kind in [
        "fi_critique", "fi_digest", "fi_portfolio", "fi_proposal",
        "fi_summary", "fi_summary_input", "fi_source_catalog",
        "fi_paper_spine",
    ]:
        assert not _is_audience_appropriate({"kind": kind}, "external"), (
            f"audience=external must drop kind={kind!r}"
        )


def test_is_audience_appropriate_external_keeps_real_papers() -> None:
    """External literature (arxiv / openalex / crossref / etc.) and
    fi_local_paper (the user's own real papers) are kept under
    audience=external."""
    from core.engine import _is_audience_appropriate
    for kind in ["arxiv", "openalex", "crossref", "fi_local_paper", "", None]:
        meta = {"kind": kind} if kind is not None else {}
        assert _is_audience_appropriate(meta, "external"), (
            f"audience=external must keep kind={kind!r}"
        )


def test_is_audience_appropriate_internal_keeps_everything() -> None:
    """Internal-facing papers can cite anything in Axon."""
    from core.engine import _is_audience_appropriate
    for kind in [
        "fi_critique", "fi_proposal", "arxiv", "fi_local_paper",
        "fi_source_catalog",
    ]:
        assert _is_audience_appropriate({"kind": kind}, "internal"), (
            f"audience=internal must keep kind={kind!r}"
        )


def test_format_lit_drops_internal_kinds_when_external() -> None:
    """End-to-end: an Axon pull mixing real papers and FI-internal
    cross-quest memory should produce a References section with only
    the real papers when audience=external."""
    from core.engine import _format_lit
    from core.knowledge import RetrievedDoc
    docs = [
        # An external citation: kept.
        RetrievedDoc(
            content="External abstract.",
            metadata={
                "title": "Stochastic LER in EUV", "year": 2024,
                "kind": "arxiv", "arxiv_id": "2404.12345",
            },
        ),
        # An FI proposal: dropped for external audience.
        RetrievedDoc(
            content="A prior FI proposal.",
            metadata={
                "title": "Why simple intensity features fail",
                "year": 2026, "kind": "fi_proposal",
            },
        ),
    ]
    external = _format_lit(docs, audience="external")
    assert "Stochastic LER in EUV" in external
    assert "Why simple intensity features fail" not in external, (
        "fi_proposal must NOT appear in an external-audience paper"
    )

    internal = _format_lit(docs, audience="internal")
    assert "Stochastic LER in EUV" in internal
    assert "Why simple intensity features fail" in internal, (
        "audience=internal keeps the cross-quest entry"
    )


def test_format_lit_renumbers_after_dropping() -> None:
    """When some entries get filtered, the kept entries should be
    renumbered [1], [2], ... — no gaps. Otherwise the writer sees
    "[1] huge paper" then "[3] another paper" and may think [2]
    exists but was hidden, prompting a hallucinated citation."""
    from core.engine import _format_lit
    from core.knowledge import RetrievedDoc
    docs = [
        RetrievedDoc(content="ignore me", metadata={"title": "stub"}),  # dropped
        RetrievedDoc(content="A" * 50, metadata={"title": "kept-1", "year": 2024}),
        RetrievedDoc(content="ignore me", metadata={}),                 # dropped
        RetrievedDoc(content="B" * 50, metadata={"title": "kept-2", "doi": "10.x/y"}),
    ]
    rendered = _format_lit(docs)
    assert "[1] kept-1" in rendered or "[1]" in rendered and "kept-1" in rendered
    assert "[2] kept-2" in rendered or "[2]" in rendered and "kept-2" in rendered
    assert "[3]" not in rendered, "should not leave a gap after drops"


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


def test_design_prompt_supports_no_simulation_mode() -> None:
    """In no_simulation mode the design node must NOT plan an executable
    Python experiment (the no-sim path never runs it, and the writer then
    narrates the engine's own 'the simulation was not run' mechanics).
    design.md exposes the override slot; the engine fills it with a
    non-empty directive that zeroes deps/figures."""
    from pathlib import Path
    from core.engine import _NO_SIM_DESIGN_DIRECTIVE
    design_md = (Path(__file__).resolve().parent.parent
                 / "agents" / "design.md").read_text(encoding="utf-8")
    assert "$study_mode_directive" in design_md
    assert "NO-SIMULATION" in _NO_SIM_DESIGN_DIRECTIVE
    # It must override the executable-experiment default, not append to it.
    assert "Ignore the" in _NO_SIM_DESIGN_DIRECTIVE
    assert '"dependencies": []' in _NO_SIM_DESIGN_DIRECTIVE
    assert '"figures_planned": []' in _NO_SIM_DESIGN_DIRECTIVE


def test_write_prompt_forbids_narrating_the_engine() -> None:
    """A paper must never describe the machinery that produced it. write.md
    carries a hard rule banning pipeline/process/failure meta-commentary
    ('the automated research pipeline failed to run the simulation'), plus
    a slot for the no-simulation framing note."""
    from pathlib import Path
    from core.engine import _NO_SIM_WRITE_NOTE
    write_md = (Path(__file__).resolve().parent.parent
                / "agents" / "write.md").read_text(encoding="utf-8")
    assert "$study_mode_note" in write_md
    assert "Never narrate the engine" in write_md
    # The specific phrases that leaked into the real paper must be named.
    assert "automated research pipeline" in write_md
    assert "was not executed" in write_md
    assert _NO_SIM_WRITE_NOTE.strip()


def test_all_artifact_prompts_forbid_engine_narration() -> None:
    """The no-engine-narration rule must hold for EVERY published artifact,
    not just the paper — a poster/slide/talk gets its own LLM call from the
    paper, so each can re-introduce 'the automated pipeline …' unless its
    own prompt forbids it. Pin that all four artifact authors carry a
    'do not narrate the pipeline/engine' directive."""
    import re
    from pathlib import Path
    agents = Path(__file__).resolve().parent.parent / "agents"
    # Match the underlying directive, not one exact phrasing: a
    # do-not/never + narrate/describe/mention/reference, close to a
    # pipeline/engine/system/tool/automation noun. This survives a
    # legitimate reword ("never mention the system", "do not describe the
    # engine") while still catching an accidental deletion of the rule.
    directive = re.compile(
        r"(?:do not|don't|never)[^.\n]{0,40}"
        r"(?:narrat|describ|mention|referenc)[^.\n]{0,40}"
        r"(?:pipeline|engine|system|tooling|tool|automation|machinery)",
        re.IGNORECASE,
    )
    for prompt in ("write.md", "poster.md", "slides.md", "speech.md"):
        text = (agents / prompt).read_text(encoding="utf-8")
        assert directive.search(text), (
            f"agents/{prompt} is missing a no-engine/pipeline-narration "
            f"rule — an audience artifact must not describe the tool that "
            f"produced it"
        )


def test_no_sim_directive_renders_into_design_prompt() -> None:
    """End-to-end at the template layer: the directive lands in the design
    prompt when no_simulation is on and vanishes when off — the exact
    branch the design node takes on state['no_simulation_resolved']."""
    import string
    from pathlib import Path
    from core.engine import _NO_SIM_DESIGN_DIRECTIVE
    tmpl = string.Template(
        (Path(__file__).resolve().parent.parent
         / "agents" / "design.md").read_text(encoding="utf-8")
    )
    kw = dict(topic="t", chosen_idea="{}", literature_block="lit",
              review_feedback="r", timeout_s="60", clarify_block="c")
    on = tmpl.substitute(study_mode_directive=_NO_SIM_DESIGN_DIRECTIVE, **kw)
    off = tmpl.substitute(study_mode_directive="", **kw)
    assert "NO-SIMULATION" in on
    assert "NO-SIMULATION" not in off


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


def test_resolve_no_simulation_from_clarify_empirical_answer_triggers_legacy(
    tmp_path: Path,
) -> None:
    """Legacy fallback path: when no ``simulatability`` slot is
    present (older clarify schema / quests resumed from old
    checkpoints), an ``empirical_vs_theoretical: empirical`` still
    triggers no_simulation. New quests should populate the
    ``simulatability`` slot directly instead."""
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


def test_resolve_no_simulation_from_clarify_simulatability_no_triggers(
    tmp_path: Path,
) -> None:
    """The NEW preferred decision path: when clarify produces a
    ``simulatability`` slot with default ``"no"``, the engine routes
    to the no-simulation flow."""
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
    assert engine._resolve_no_simulation_from_clarify({
        "simulatability": {
            "default": "no",
            "reason": "Belgium/Taiwan cultural attitudes require survey data.",
        }
    }) is True


def test_resolve_no_simulation_from_clarify_simulatability_yes_does_not_trigger(
    tmp_path: Path,
) -> None:
    """``simulatability: yes`` keeps the simulation path. Same for
    ``uncertain`` (the review prompt may add scrutiny later, but the
    routing stays simulate-first)."""
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
    assert engine._resolve_no_simulation_from_clarify({
        "simulatability": {"default": "yes", "reason": "ODE simulation."},
    }) is False
    assert engine._resolve_no_simulation_from_clarify({
        "simulatability": {"default": "uncertain", "reason": "marginal."},
    }) is False


def test_resolve_no_simulation_simulatability_str_shape(tmp_path: Path) -> None:
    """The auto-mode clarify reducer (``{k: v["default"]}`` at the
    Engine._clarify_node ``mode=="auto"`` branch) collapses every slot
    to its bare default value — including ``simulatability``. The
    resolver must accept that string-shaped answer the same way it
    accepts the unit-test dict shape. Without this, an auto-mode
    quest with ``simulatability=yes`` falls through to the legacy
    ``empirical_vs_theoretical=empirical`` rule and incorrectly
    routes to NO_SIMULATION.

    Pins both str values that should route to SIMULATE plus the
    str value that should route to NO_SIMULATION."""
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
    # Bare-string simulatability — what auto-mode actually produces.
    assert engine._resolve_no_simulation_from_clarify(
        {"simulatability": "no"},
    ) is True
    assert engine._resolve_no_simulation_from_clarify(
        {"simulatability": "yes"},
    ) is False
    assert engine._resolve_no_simulation_from_clarify(
        {"simulatability": "uncertain"},
    ) is False
    # Bare-string simulatability with whitespace + uppercase — same
    # normalization the dict path applies.
    assert engine._resolve_no_simulation_from_clarify(
        {"simulatability": "  NO  "},
    ) is True
    # And bare-string simulatability=yes still overrides the legacy
    # empirical fallback when both are present.
    assert engine._resolve_no_simulation_from_clarify({
        "simulatability": "yes",
        "empirical_vs_theoretical": "empirical",
    }) is False


def test_resolve_no_simulation_simulatability_beats_empirical_when_both_set(
    tmp_path: Path,
) -> None:
    """When BOTH slots are present, the new ``simulatability`` slot
    wins. The legacy ``empirical_vs_theoretical`` fallback only fires
    when ``simulatability`` is missing.

    Real-world case: a topic the LLM judged 'empirical' methodology
    BUT also 'yes' on simulatability — e.g. an empirical-style study
    of an algorithm's behavior that's still pure Python. Should NOT
    route to no_simulation."""
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
    assert engine._resolve_no_simulation_from_clarify({
        "empirical_vs_theoretical": "empirical",
        "simulatability": {"default": "yes", "reason": "Pure Python."},
    }) is False, "simulatability=yes must override empirical_vs_theoretical=empirical"


def test_resolve_no_simulation_yaml_still_wins_over_simulatability(
    tmp_path: Path,
) -> None:
    """YAML's ``engine.no_simulation: true`` is the explicit user
    override and beats any LLM judgement, including a
    ``simulatability: yes`` from clarify."""
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(no_simulation=True, clarify_mode="auto"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    assert engine._resolve_no_simulation_from_clarify({
        "simulatability": {"default": "yes", "reason": "looks simulatable"},
    }) is True


def test_resolve_no_simulation_logs_resolution_with_source_and_reason(
    tmp_path: Path,
) -> None:
    """Every resolution path logs an INFO line naming the source
    (yaml / clarify_simulatability / clarify_empirical_legacy /
    default) and the reason (when available). This is the
    transparency contract — a user reading run.log can see WHY
    the engine routed the way it did."""
    import logging as _logging

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

    captured: list[_logging.LogRecord] = []
    sink = _logging.Handler()
    sink.emit = captured.append  # type: ignore[assignment]
    engine._log.addHandler(sink)
    try:
        engine._resolve_no_simulation_from_clarify({
            "simulatability": {
                "default": "no",
                "reason": "Cultural data needs surveys.",
            }
        })
    finally:
        engine._log.removeHandler(sink)

    info_lines = [r.getMessage() for r in captured if r.levelno == _logging.INFO]
    assert any("simulatability resolved: NO_SIMULATION" in m for m in info_lines)
    assert any("source=clarify_simulatability" in m for m in info_lines)
    assert any("Cultural data needs surveys" in m for m in info_lines)


def test_resolve_no_simulation_warns_on_unknown_simulatability_value(
    tmp_path: Path,
) -> None:
    """When the LLM returns a simulatability.default outside the
    documented {yes, no, uncertain} set (e.g. ``"maybe"``, a typo, or
    free-form prose), the engine falls through to the legacy
    empirical_vs_theoretical check. That fallthrough must be VISIBLE —
    a WARNING with the offending value lands in run.log so the user
    can spot the LLM drift without diffing clarify answers against
    the engine source. Empty / missing decision strings DO NOT warn
    (they're the legitimate "slot was omitted" case)."""
    import logging as _logging

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

    captured: list[_logging.LogRecord] = []
    sink = _logging.Handler()
    sink.emit = captured.append  # type: ignore[assignment]
    engine._log.addHandler(sink)
    try:
        # Unrecognized value — must WARN and fall through to legacy.
        result = engine._resolve_no_simulation_from_clarify({
            "simulatability": {"default": "maybe", "reason": "unclear"}
        })
    finally:
        engine._log.removeHandler(sink)
    assert result is False, "unknown value should NOT trigger no-simulation"
    warnings = [r.getMessage() for r in captured if r.levelno == _logging.WARNING]
    assert warnings, "expected a WARNING about the unrecognized value"
    assert any("'maybe'" in m or "maybe" in m for m in warnings), (
        "WARNING must include the offending value so the user can grep for it"
    )
    assert any("yes, no, uncertain" in m for m in warnings), (
        "WARNING must name the documented allowed set"
    )

    # Empty / missing default — silent fallthrough (no WARNING).
    captured.clear()
    engine._log.addHandler(sink)
    try:
        engine._resolve_no_simulation_from_clarify({
            "simulatability": {"default": "", "reason": ""}
        })
    finally:
        engine._log.removeHandler(sink)
    warnings = [r.getMessage() for r in captured if r.levelno == _logging.WARNING]
    assert not warnings, (
        f"empty/missing simulatability.default must NOT warn — the slot "
        f"being omitted is a legitimate legacy-prompt case. Got: {warnings}"
    )


def test_clarify_prompt_has_simulatability_slot() -> None:
    """``agents/clarify.md`` must request the new ``simulatability``
    slot from the LLM. Regression guard so a future prompt rewrite
    doesn't silently lose the routing signal."""
    clarify_md = (Path(__file__).resolve().parent.parent
                  / "agents" / "clarify.md").read_text(encoding="utf-8")
    assert "\"simulatability\"" in clarify_md, (
        "agents/clarify.md must declare a 'simulatability' slot — "
        "it's the routing signal for no_simulation mode."
    )
    # The prompt must spell out the three allowed answers in the
    # slot's declared `default` line. Substring matches against the
    # whole file would trivially pass on prose like "yes, the topic..."
    # — pin to the exact declaration so a future rewrite that drops
    # any of the three options fails the guard.
    expected_default = '"yes" or "no" or "uncertain"'
    assert expected_default in clarify_md, (
        f"agents/clarify.md must declare the simulatability default as "
        f"{expected_default!r} so the LLM emits one of those tokens "
        f"(the engine matches case-insensitively but expects one of "
        f"the three). A looser substring guard would miss a future "
        f"prompt rewrite that drops one of the options."
    )


def test_route_after_design_no_sim_flag_routes_to_auto_collect_data(
    tmp_path: Path,
) -> None:
    """The routing function — the heart of the no-sim graph edge.
    When state carries ``no_simulation_resolved=True``, design must
    route to ``auto_collect_data`` — the auto-collect node runs
    BEFORE wait_for_data. Otherwise to ``implement``."""
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
    assert engine._route_after_design({"no_simulation_resolved": True}) == "auto_collect_data"
    assert engine._route_after_design({"no_simulation_resolved": False}) == "implement"
    assert engine._route_after_design({}) == "implement"


# ---- paper.pdf pre-flight check ------------------------------------------


def test_output_config_require_pdf_default_false() -> None:
    """Back-compat: existing YAMLs without ``output.require_pdf``
    must keep working. Default must be False so quests retain the
    graceful-skip-with-diagnostic behavior from #55."""
    from core.config import OutputConfig
    assert OutputConfig().require_pdf is False
    assert OutputConfig(require_pdf=True).require_pdf is True


def _make_engine_for_preflight(
    tmp_path: Path, *, kinds: list[str], require_pdf: bool = False,
) -> "Engine":
    cfg = Config(
        topic="preflight test",
        title="preflight",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(
            kinds=kinds, output_dir=tmp_path / "outputs",
            require_pdf=require_pdf,
        ),
    )
    return Engine(cfg)


def test_preflight_pdf_pass_when_everything_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: pandoc + pdflatex both reachable → no warning,
    no error."""
    from generation import paper as paper_mod

    monkeypatch.setattr(
        paper_mod.shutil, "which",
        lambda name: "/fake/pandoc" if name == "pandoc"
        else "/fake/pdflatex" if name == "pdflatex" else None,
    )
    # The engine module also calls shutil.which directly.
    from core import engine as engine_mod
    monkeypatch.setattr(
        engine_mod.shutil, "which",
        lambda name: "/fake/pandoc" if name == "pandoc"
        else "/fake/pdflatex" if name == "pdflatex" else None,
    )

    engine = _make_engine_for_preflight(tmp_path, kinds=["paper_md", "paper_pdf"])
    # No raise, no warning needed.
    engine._preflight_paper_pdf()


def test_preflight_pdf_skipped_when_pdf_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the user didn't ask for paper_pdf, the pre-flight does
    nothing — even when pandoc is missing."""
    from generation import paper as paper_mod
    from core import engine as engine_mod
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _n: None)
    monkeypatch.setattr(engine_mod.shutil, "which", lambda _n: None)

    engine = _make_engine_for_preflight(tmp_path, kinds=["paper_md"])
    engine._preflight_paper_pdf()  # no raise, no warning — early-exit


def test_preflight_pdf_warns_when_pandoc_missing_and_not_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``require_pdf: false`` is the default — a missing prereq must
    log a WARNING (with the install recipe) but NOT raise. The quest
    continues; paper.md still lands; paper_pdf_skipped.md will be
    written at the end."""
    import logging as _logging
    from generation import paper as paper_mod
    from core import engine as engine_mod
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _n: None)
    monkeypatch.setattr(engine_mod.shutil, "which", lambda _n: None)

    engine = _make_engine_for_preflight(
        tmp_path, kinds=["paper_md", "paper_pdf"], require_pdf=False,
    )
    # Per-quest logger has propagate=False (per _quest_logger), so
    # pytest's caplog at the root logger never receives records from
    # it. Attach a plain ``logging.Handler`` whose ``emit`` is swapped
    # for a list ``append`` — the simplest way to capture records
    # emitted by THIS logger without going through pytest's
    # propagation-based capture machinery. (Not
    # ``logging.handlers.MemoryHandler``, which buffers + flushes to a
    # target handler and is overkill for this assertion.)
    captured_records: list[_logging.LogRecord] = []
    sink = _logging.Handler()
    sink.setLevel(_logging.WARNING)
    sink.emit = captured_records.append  # type: ignore[assignment]
    engine._log.addHandler(sink)
    try:
        engine._preflight_paper_pdf()
    finally:
        engine._log.removeHandler(sink)

    warnings = [r for r in captured_records if r.levelno == _logging.WARNING]
    assert warnings, "expected a WARNING about missing prereqs"
    msg = warnings[-1].getMessage()
    assert "paper_pdf requested" in msg
    assert "pandoc" in msg
    assert "--install-tectonic" in msg, "fix recipe must mention --install-tectonic"
    assert "require_pdf=True" in msg, "warning must point at the strict-mode flag"


def test_preflight_pdf_raises_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``require_pdf: true`` upgrades the warning to a hard failure
    that aborts the quest BEFORE any LLM call happens. Saves the user
    ~15 min of LLM cost on a quest doomed to skip the PDF."""
    from generation import paper as paper_mod
    from core import engine as engine_mod
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _n: None)
    monkeypatch.setattr(engine_mod.shutil, "which", lambda _n: None)

    engine = _make_engine_for_preflight(
        tmp_path, kinds=["paper_md", "paper_pdf"], require_pdf=True,
    )
    with pytest.raises(RuntimeError) as ei:
        engine._preflight_paper_pdf()
    msg = str(ei.value)
    assert "paper_pdf requested" in msg
    assert "Aborting before LLM calls" in msg
    assert "winget" in msg or "brew" in msg or "package manager" in msg


def test_preflight_pdf_raises_on_missing_latex_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pandoc IS present but no LaTeX engine is reachable, the
    pre-flight still triggers in strict mode."""
    from generation import paper as paper_mod
    from core import engine as engine_mod

    def fake_which(name):
        # pandoc found, pdflatex/tectonic not.
        return "/fake/pandoc" if name == "pandoc" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)
    monkeypatch.setattr(engine_mod.shutil, "which", fake_which)
    # paper_mod._find_pdf_engine ALSO probes REPO_ROOT/tools/tectonic.*
    # as a fallback. On a dev box where `--install-tectonic` has run,
    # that file exists and the test thinks the engine IS reachable.
    # Point REPO_ROOT at a clean tmp dir so the probe misses.
    monkeypatch.setattr(paper_mod, "REPO_ROOT", tmp_path)

    engine = _make_engine_for_preflight(
        tmp_path, kinds=["paper_md", "paper_pdf"], require_pdf=True,
    )
    with pytest.raises(RuntimeError, match="LaTeX engine"):
        engine._preflight_paper_pdf()


@pytest.mark.asyncio
async def test_engine_run_invokes_preflight_before_executor_setup_and_llm_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``Engine.run`` MUST call ``_preflight_paper_pdf``
    before ``executor.setup`` and before ``resolve_endpoint_async``.

    The contract the pre-flight exists to enforce is "abort BEFORE
    spending LLM money on a quest that can't produce its requested
    PDF." If a future refactor moved the preflight call below
    ``executor.setup`` (which does venv creation, a few-second op)
    or below ``resolve_endpoint_async`` (which talks to the provider
    socket and starts metering), strict mode would silently turn
    into "abort AFTER setup costs" — exactly the failure mode the
    pre-flight is supposed to prevent.

    This test mocks the three downstream surfaces and proves they
    aren't reached when require_pdf=True and prereqs are missing.
    """
    from unittest.mock import AsyncMock, MagicMock
    from core import engine as engine_mod
    from generation import paper as paper_mod

    # Both pandoc and pdflatex absent — preflight should raise.
    monkeypatch.setattr(engine_mod.shutil, "which", lambda _n: None)
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _n: None)

    engine = _make_engine_for_preflight(
        tmp_path, kinds=["paper_md", "paper_pdf"], require_pdf=True,
    )

    # Replace ALL three "after preflight" surfaces with stubs that
    # record whether they ran. The assertion below is "no stub was
    # called" — i.e. preflight aborted before any of them.
    executor_setup = AsyncMock()
    engine.executor.setup = executor_setup  # type: ignore[method-assign]

    resolve_called = MagicMock()
    async def fake_resolve(*args, **kwargs):
        resolve_called(*args, **kwargs)
        return MagicMock(base_url="http://x", model="m")
    monkeypatch.setattr(engine_mod, "resolve_endpoint_async", fake_resolve)

    llm_client_called = MagicMock()
    monkeypatch.setattr(engine_mod, "LLMClient", llm_client_called)

    with pytest.raises(RuntimeError) as ei:
        await engine.run()

    # The error must come from the preflight (not from a downstream
    # mock raising), so check the marker string.
    assert "Aborting before LLM calls" in str(ei.value)
    assert executor_setup.await_count == 0, (
        "executor.setup must not run when preflight aborts the quest"
    )
    assert resolve_called.call_count == 0, (
        "resolve_endpoint_async must not be called when preflight "
        "aborts the quest"
    )
    assert llm_client_called.call_count == 0, (
        "LLMClient must not be constructed when preflight aborts the "
        "quest (constructing it implies an endpoint was already resolved)"
    )


# ---- _node_auto_collect_data --------------------------------------------
#
# Tests for the agent-side data collection node that runs BEFORE
# wait_for_data in no-simulation mode. Mocks ``Knowledge.asearch``
# directly — no real Axon corpus required.


def _make_no_sim_engine(
    tmp_path: Path,
    *,
    auto_collect_data: bool = True,
    knowledge_enabled: bool = True,
    auto_collect_top_k: int = 5,
) -> "Engine":
    # ALWAYS construct Engine with KnowledgeConfig(enabled=False) so
    # ``Knowledge.__init__`` skips its slow embedding/retriever bring-
    # up (15+ s on this machine). Then overwrite ``engine.knowledge``
    # with a MagicMock whose ``enabled`` matches what the test
    # actually wants and whose ``asearch`` callers replace with the
    # per-test return value / side-effect.
    from unittest.mock import MagicMock, AsyncMock
    cfg = Config(
        topic="cross-cultural collectivism in Belgium vs Taiwan",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            auto_collect_data=auto_collect_data,
            auto_collect_top_k=auto_collect_top_k,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),  # cheap init
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    engine = Engine(cfg)
    mock_knowledge = MagicMock()
    mock_knowledge.enabled = knowledge_enabled
    # Default asearch: returns no docs. Per-test code overrides.
    mock_knowledge.asearch = AsyncMock(return_value=[])
    engine.knowledge = mock_knowledge  # type: ignore[assignment]
    return engine


def test_auto_collect_top_k_default_is_5() -> None:
    """Default top_k is 5 — the prompt-budget rationale documented on
    the field. Regression guard so a future bump doesn't silently
    blow out the data_load LLM call's content budget."""
    from core.config import EngineConfig
    assert EngineConfig().auto_collect_top_k == 5
    assert EngineConfig().auto_collect_data is True, (
        "auto_collect_data must default ON — the user explicitly asked "
        "for agent-side data collection, not user-only"
    )


@pytest.mark.asyncio
async def test_node_auto_collect_data_passthrough_when_flag_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``engine.auto_collect_data: false`` makes the node a logged
    passthrough — no Axon call, no files written. The downstream
    pause behavior is untouched."""
    from unittest.mock import AsyncMock

    engine = _make_no_sim_engine(tmp_path, auto_collect_data=False)
    spy = AsyncMock(return_value=[])
    engine.knowledge.asearch = spy  # type: ignore[method-assign]

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 0}
    spy.assert_not_called()  # critical — no LLM/RAG cost on opt-out
    auto_dir = engine.quest_root / "data" / "auto_collected"
    assert not auto_dir.exists(), (
        "passthrough must not create the auto_collected dir — that "
        "would leak intent into the user's filesystem unnecessarily"
    )


@pytest.mark.asyncio
async def test_node_auto_collect_data_passthrough_when_knowledge_disabled(
    tmp_path: Path,
) -> None:
    """``knowledge.enabled: false`` (no Axon to query) → passthrough.
    Don't crash, don't try to call asearch on a disabled retriever."""
    from unittest.mock import AsyncMock

    engine = _make_no_sim_engine(tmp_path, knowledge_enabled=False)
    spy = AsyncMock(return_value=[])
    engine.knowledge.asearch = spy  # type: ignore[method-assign]

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 0}
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_node_auto_collect_data_writes_files_on_axon_hits(
    tmp_path: Path,
) -> None:
    """Happy path: Axon returns N docs → the node writes N files into
    ``<quest_root>/data/auto_collected/`` with rank-prefixed names and
    YAML front matter carrying provenance. wait_for_data will then
    see those files via its rglob walk and proceed without pausing."""
    from unittest.mock import AsyncMock
    from core.knowledge import RetrievedDoc

    docs = [
        RetrievedDoc(
            content="Hofstede 2010 cultural dimensions: Belgium has IDV=75.",
            metadata={"source": "hofstede_belgium.pdf", "kind": "fi_local_paper"},
        ),
        RetrievedDoc(
            content="Taiwan IDV=17 — strongly collectivist per Hofstede.",
            metadata={"source": "hofstede_taiwan.pdf", "title": "Cultural Dims TW"},
        ),
        RetrievedDoc(
            content="World Values Survey wave 7 — public-trust measures.",
            metadata={"url": "https://wvs.example/wave7", "year": 2022},
        ),
    ]
    engine = _make_no_sim_engine(tmp_path, auto_collect_top_k=3)
    engine.knowledge.asearch = AsyncMock(return_value=docs)  # type: ignore[method-assign]

    result = await engine._node_auto_collect_data({
        "topic": "Belgium vs Taiwan culture",
        "design": {"hypothesis": "IDV gap predicts trust dynamics"},
    })

    assert result == {"auto_collected_count": 3}
    auto_dir = engine.quest_root / "data" / "auto_collected"
    files = sorted(auto_dir.glob("*.md"))
    assert len(files) == 3, f"expected 3 files, got {[f.name for f in files]}"
    # Rank-prefixed, zero-padded so a 10+ doc retrieval sorts naturally.
    assert files[0].name.startswith("001_")
    assert files[1].name.startswith("002_")
    assert files[2].name.startswith("003_")
    # Front matter present + the actual content.
    body0 = files[0].read_text(encoding="utf-8")
    assert body0.startswith("---\n")
    assert "auto_collected: true" in body0
    assert "rank: 1" in body0
    assert "Hofstede 2010" in body0
    # Metadata-rich doc[1] gets its title + source rendered.
    body1 = files[1].read_text(encoding="utf-8")
    assert "title:" in body1
    assert "Cultural Dims TW" in body1
    # URL-based doc[2] gets the URL rendered, not a None source.
    body2 = files[2].read_text(encoding="utf-8")
    assert "url:" in body2
    assert "wave7" in body2


@pytest.mark.asyncio
async def test_node_auto_collect_data_front_matter_is_yaml_parseable(
    tmp_path: Path,
) -> None:
    """The YAML front matter must be safely parseable by any standard
    YAML loader — so downstream tools (or a future data_load that
    parses provenance) can round-trip metadata values containing
    quotes, backslashes, colons, and unicode without mangling.

    Front matter must NOT use Python ``repr`` (which is not YAML-safe
    — a value like ``"O'Brien"`` would parse back as ``"O\\'Brien"``
    from a YAML reader)."""
    from unittest.mock import AsyncMock
    from core.knowledge import RetrievedDoc
    import yaml as _yaml

    docs = [
        RetrievedDoc(
            content="hit body",
            metadata={
                # Punctuation that breaks Python repr-vs-YAML round-trip:
                "source": "O'Brien_2021.pdf",
                "title": "Trust: A YAML-hostile string (with colons)",
                "url": "https://example.com/path?q=value&r=2",
                "kind": "fi_local_paper",
            },
        ),
    ]
    engine = _make_no_sim_engine(tmp_path, auto_collect_top_k=1)
    engine.knowledge.asearch = AsyncMock(return_value=docs)  # type: ignore[method-assign]

    await engine._node_auto_collect_data({"topic": "x", "design": {}})

    files = sorted(
        (engine.quest_root / "data" / "auto_collected").glob("*.md")
    )
    assert len(files) == 1
    body = files[0].read_text(encoding="utf-8")
    # Extract the front matter block.
    assert body.startswith("---\n")
    _, fm_block, _ = body.split("---\n", 2)
    parsed = _yaml.safe_load(fm_block)
    assert isinstance(parsed, dict), f"front matter is not a YAML mapping: {fm_block!r}"
    # Values round-trip cleanly.
    assert parsed["source"] == "O'Brien_2021.pdf"
    assert parsed["title"] == "Trust: A YAML-hostile string (with colons)"
    assert parsed["url"] == "https://example.com/path?q=value&r=2"
    assert parsed["auto_collected"] is True
    assert parsed["rank"] == 1


@pytest.mark.asyncio
async def test_node_auto_collect_data_cleans_up_dir_when_all_writes_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case: when EVERY write raises OSError
    (e.g. permissions or full disk after the mkdir succeeded), the
    node must NOT leave an empty ``auto_collected/`` directory behind
    — that would mislead the user into thinking auto-collection
    produced output. Verify cleanup happens."""
    from unittest.mock import AsyncMock
    from pathlib import Path as _Path
    from core.knowledge import RetrievedDoc

    docs = [
        RetrievedDoc(content=f"body{i}", metadata={"source": f"x{i}.pdf"})
        for i in range(3)
    ]
    engine = _make_no_sim_engine(tmp_path, auto_collect_top_k=3)
    engine.knowledge.asearch = AsyncMock(return_value=docs)  # type: ignore[method-assign]

    # Force every write_text to fail. mkdir is still allowed (so we
    # can prove the cleanup path runs).
    original_write_text = _Path.write_text
    def fake_write_text(self, *_a, **_kw):  # type: ignore[no-untyped-def]
        if "auto_collected" in str(self):
            raise OSError("simulated disk full")
        return original_write_text(self, *_a, **_kw)
    monkeypatch.setattr(_Path, "write_text", fake_write_text)

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 0}
    auto_dir = engine.quest_root / "data" / "auto_collected"
    assert not auto_dir.exists(), (
        f"empty auto_collected/ must be cleaned up when all writes "
        f"failed; got dir={auto_dir} children="
        f"{list(auto_dir.iterdir()) if auto_dir.exists() else 'n/a'}"
    )


# ---- _run_dataset_adapters ----------------------------------------------


@pytest.mark.asyncio
async def test_dataset_adapters_passthrough_when_list_empty(
    tmp_path: Path,
) -> None:
    """Default ``EngineConfig.dataset_adapters: []`` — no adapter
    runs, no subdirs created. Axon-only auto-collect path is
    preserved."""
    from unittest.mock import AsyncMock
    from core.knowledge import RetrievedDoc

    engine = _make_no_sim_engine(tmp_path)
    engine.knowledge.asearch = AsyncMock(return_value=[  # type: ignore[method-assign]
        RetrievedDoc(content="axon hit", metadata={"source": "p.pdf"}),
    ])

    await engine._node_auto_collect_data({"topic": "x", "design": {}})

    auto_dir = engine.quest_root / "data" / "auto_collected"
    # Only top-level Axon files; no per-adapter subdirs.
    children = sorted(p.name for p in auto_dir.iterdir())
    assert children == ["001_p.md"], children


@pytest.mark.asyncio
async def test_dataset_adapters_unknown_name_logs_warning_and_skips(
    tmp_path: Path,
) -> None:
    """An unrecognized adapter name (typo in YAML) must NOT crash;
    log a WARNING listing the known names and skip. The rest of the
    flow proceeds normally."""
    import logging as _logging
    from unittest.mock import AsyncMock
    from core.knowledge import RetrievedDoc

    engine = _make_no_sim_engine(tmp_path)
    engine.config.engine.dataset_adapters = ["typo_adapter"]  # type: ignore[misc]
    engine.knowledge.asearch = AsyncMock(return_value=[  # type: ignore[method-assign]
        RetrievedDoc(content="axon", metadata={"source": "x.pdf"}),
    ])

    captured: list[_logging.LogRecord] = []
    sink = _logging.Handler()
    sink.emit = captured.append  # type: ignore[assignment]
    engine._log.addHandler(sink)
    try:
        result = await engine._node_auto_collect_data({"topic": "x", "design": {}})
    finally:
        engine._log.removeHandler(sink)

    # Axon write still happened; adapter contributed nothing.
    assert result == {"auto_collected_count": 1}
    warnings = [r.getMessage() for r in captured if r.levelno == _logging.WARNING]
    assert any("unknown dataset adapter" in m and "typo_adapter" in m for m in warnings), (
        f"expected unknown-adapter WARNING; got: {warnings}"
    )


@pytest.mark.asyncio
async def test_dataset_adapters_invokes_registered_adapter_and_writes_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: a registered adapter is named in
    ``engine.dataset_adapters`` → it's instantiated, ``search()`` is
    awaited with the same query the Axon path used, results are
    written to ``data/auto_collected/<adapter_name>/``, and the
    final auto_collected_count includes both Axon + adapter writes."""
    from unittest.mock import AsyncMock
    from core.datasets import ADAPTER_REGISTRY
    from core.datasets.base import DatasetAdapter, DatasetRow
    from core.knowledge import RetrievedDoc

    class FakeAdapter(DatasetAdapter):
        name = "fake"
        last_query: str | None = None

        async def search(self, query: str, *, top_k: int) -> list[DatasetRow]:
            FakeAdapter.last_query = query
            return [
                DatasetRow(
                    content="# Mocked dataset row\n\n| col | val |\n|-----|-----|\n| x | 1 |",
                    metadata={"source": "fake", "indicator_id": "ABC", "title": "Mock Row"},
                ),
                DatasetRow(
                    content="another row",
                    metadata={"source": "fake", "indicator_id": "XYZ"},
                ),
            ]

    monkeypatch.setitem(ADAPTER_REGISTRY, "fake", FakeAdapter)

    engine = _make_no_sim_engine(tmp_path)
    engine.config.engine.dataset_adapters = ["fake"]  # type: ignore[misc]
    engine.knowledge.asearch = AsyncMock(return_value=[  # type: ignore[method-assign]
        RetrievedDoc(content="axon hit", metadata={"source": "p.pdf"}),
    ])

    result = await engine._node_auto_collect_data({
        "topic": "Belgium vs Taiwan trust",
        "design": {"hypothesis": "IDV gap predicts trust"},
    })

    # 1 Axon + 2 adapter rows = 3 total.
    assert result == {"auto_collected_count": 3}
    # Adapter wrote into a per-adapter subdir.
    sub_dir = engine.quest_root / "data" / "auto_collected" / "fake"
    files = sorted(sub_dir.glob("*.md"))
    assert len(files) == 2
    assert files[0].name == "001_abc.md"  # slugged from indicator_id
    assert files[1].name == "002_xyz.md"
    # Adapter saw a query built from topic + hypothesis.
    assert FakeAdapter.last_query is not None
    assert "Belgium" in FakeAdapter.last_query
    assert "IDV gap" in FakeAdapter.last_query
    # The rendered file has the adapter name folded into the YAML
    # front matter so a downstream tool can attribute the row.
    body = files[0].read_text(encoding="utf-8")
    assert "adapter: fake" in body


@pytest.mark.asyncio
async def test_dataset_adapters_exception_is_caught_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter that raises during ``.search()`` MUST NOT abort the
    no-simulation flow. Log a WARNING, skip the adapter's results,
    continue."""
    from unittest.mock import AsyncMock
    from core.datasets import ADAPTER_REGISTRY
    from core.datasets.base import DatasetAdapter
    from core.knowledge import RetrievedDoc

    class FlakyAdapter(DatasetAdapter):
        name = "flaky"
        async def search(self, query: str, *, top_k: int):
            raise RuntimeError("simulated API outage")

    monkeypatch.setitem(ADAPTER_REGISTRY, "flaky", FlakyAdapter)

    engine = _make_no_sim_engine(tmp_path)
    engine.config.engine.dataset_adapters = ["flaky"]  # type: ignore[misc]
    engine.knowledge.asearch = AsyncMock(return_value=[  # type: ignore[method-assign]
        RetrievedDoc(content="axon", metadata={"source": "p.pdf"}),
    ])

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    # Axon write present (1); adapter contributed nothing.
    assert result == {"auto_collected_count": 1}
    # The subdir for the flaky adapter must NOT exist (no half-baked
    # artifact).
    assert not (engine.quest_root / "data" / "auto_collected" / "flaky").exists()


@pytest.mark.asyncio
async def test_dataset_adapters_run_even_when_axon_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``knowledge.enabled=False`` must NOT skip the dataset adapter
    step. A user who opts in to ``dataset_adapters: [worldbank]`` with
    no Axon configured still expects the adapter to fire — the Axon
    short-circuit must not return early before adapters can run."""
    from core.datasets import ADAPTER_REGISTRY
    from core.datasets.base import DatasetAdapter, DatasetRow

    class AlwaysReturnsAdapter(DatasetAdapter):
        name = "always"
        async def search(self, query: str, *, top_k: int):
            return [DatasetRow(
                content="adapter still fired",
                metadata={"source": "always", "title": "Adapter ran"},
            )]

    monkeypatch.setitem(ADAPTER_REGISTRY, "always", AlwaysReturnsAdapter)

    engine = _make_no_sim_engine(tmp_path, knowledge_enabled=False)
    engine.config.engine.dataset_adapters = ["always"]  # type: ignore[misc]
    # Don't even need to mock asearch — Axon path short-circuits.

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 1}, (
        "dataset adapter must run regardless of Axon's state"
    )
    assert (engine.quest_root / "data" / "auto_collected" / "always").exists()


@pytest.mark.asyncio
async def test_dataset_adapters_run_even_when_axon_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same regression as above for the Axon-raises path."""
    from unittest.mock import AsyncMock
    from core.datasets import ADAPTER_REGISTRY
    from core.datasets.base import DatasetAdapter, DatasetRow

    class AlwaysReturnsAdapter(DatasetAdapter):
        name = "always"
        async def search(self, query: str, *, top_k: int):
            return [DatasetRow(content="row", metadata={"source": "always"})]

    monkeypatch.setitem(ADAPTER_REGISTRY, "always", AlwaysReturnsAdapter)

    engine = _make_no_sim_engine(tmp_path)
    engine.config.engine.dataset_adapters = ["always"]  # type: ignore[misc]
    engine.knowledge.asearch = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("axon down"),
    )

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    # 1 adapter row, 0 from Axon.
    assert result == {"auto_collected_count": 1}


@pytest.mark.asyncio
async def test_dataset_adapters_run_when_axon_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Axon legitimately returned nothing; dataset adapters should
    still run."""
    from unittest.mock import AsyncMock
    from core.datasets import ADAPTER_REGISTRY
    from core.datasets.base import DatasetAdapter, DatasetRow

    class AlwaysReturnsAdapter(DatasetAdapter):
        name = "always"
        async def search(self, query: str, *, top_k: int):
            return [DatasetRow(content="row", metadata={"source": "always"})]

    monkeypatch.setitem(ADAPTER_REGISTRY, "always", AlwaysReturnsAdapter)

    engine = _make_no_sim_engine(tmp_path)
    engine.config.engine.dataset_adapters = ["always"]  # type: ignore[misc]
    engine.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 1}


@pytest.mark.asyncio
async def test_render_auto_collected_md_coerces_non_scalar_metadata(
    tmp_path: Path,
) -> None:
    """``_render_auto_collected_md`` must coerce list / dict values
    to YAML scalars (strings) so the front
    matter stays flat. Without this, an adapter passing
    ``metadata={"tags": ["a", "b"]}`` would emit nested YAML
    that changes the file head shape downstream consumers expect."""
    from core.engine import _render_auto_collected_md
    import yaml as _yaml

    body = _render_auto_collected_md(
        idx=1,
        meta={
            "source": "test",
            "tags": ["culture", "trust"],  # list → coerce to str
            "extra": {"nested": "dict"},   # dict → coerce to str
            "count": 42,                    # int → preserve
            "ratio": 3.14,                  # float → preserve
            "verified": True,               # bool → preserve
        },
        content="body",
    )
    _, fm, _ = body.split("---\n", 2)
    parsed = _yaml.safe_load(fm)
    # Scalars preserved.
    assert parsed["count"] == 42
    assert parsed["ratio"] == 3.14
    assert parsed["verified"] is True
    # Non-scalars rendered as strings (flat front matter shape).
    assert isinstance(parsed["tags"], str)
    assert "culture" in parsed["tags"]
    assert isinstance(parsed["extra"], str)
    assert "nested" in parsed["extra"]


def test_dataset_adapter_top_k_rejects_zero_and_negative() -> None:
    """Pydantic ``ge=1`` validation on ``dataset_adapter_top_k``.
    Same rationale as ``auto_collect_top_k``: top_k=0 is useless."""
    from core.config import EngineConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EngineConfig(dataset_adapter_top_k=0)
    with pytest.raises(ValidationError):
        EngineConfig(dataset_adapter_top_k=-3)
    assert EngineConfig(dataset_adapter_top_k=1).dataset_adapter_top_k == 1


def test_engine_config_dataset_adapters_default_empty() -> None:
    """Default ``dataset_adapters: []`` keeps auto-collect Axon-only.
    Opt-in only — no surprise external API calls when a user upgrades
    without touching YAML."""
    from core.config import EngineConfig
    assert EngineConfig().dataset_adapters == []
    assert EngineConfig().dataset_adapter_top_k == 3


def test_auto_collect_top_k_rejects_zero_and_negative() -> None:
    """``Field(default=5, ge=1)`` on ``auto_collect_top_k``: passing
    top_k=0 to Axon would mean "request zero hits" which is useless;
    a typo / negative value should fail at YAML parse time, not
    silently pass through."""
    from core.config import EngineConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EngineConfig(auto_collect_top_k=0)
    with pytest.raises(ValidationError):
        EngineConfig(auto_collect_top_k=-1)
    # Positive values work — boundary check at 1.
    assert EngineConfig(auto_collect_top_k=1).auto_collect_top_k == 1


@pytest.mark.asyncio
async def test_node_auto_collect_data_uses_topic_and_hypothesis_in_query(
    tmp_path: Path,
) -> None:
    """The Axon query is built from topic + design.hypothesis — that's
    a sharper retrieval signal than topic alone once design has run.
    Regression guard so a refactor doesn't accidentally drop the
    hypothesis."""
    from unittest.mock import AsyncMock

    engine = _make_no_sim_engine(tmp_path)
    spy = AsyncMock(return_value=[])
    engine.knowledge.asearch = spy  # type: ignore[method-assign]

    await engine._node_auto_collect_data({
        "topic": "Belgium vs Taiwan culture",
        "design": {"hypothesis": "IDV gap predicts trust dynamics"},
    })

    spy.assert_awaited_once()
    args, kwargs = spy.call_args
    query = args[0]
    assert "Belgium" in query
    assert "IDV gap" in query, "hypothesis must be part of the Axon query"
    assert kwargs.get("top_k") == 5


@pytest.mark.asyncio
async def test_node_auto_collect_data_falls_through_on_axon_exception(
    tmp_path: Path,
) -> None:
    """An exception from Axon must NOT crash the no-sim flow. Log a
    WARNING and return a zero-count state so wait_for_data takes
    over and pauses for user-supplied data (the safety net)."""
    from unittest.mock import AsyncMock

    engine = _make_no_sim_engine(tmp_path)
    engine.knowledge.asearch = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("axon down")
    )

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 0}
    auto_dir = engine.quest_root / "data" / "auto_collected"
    # An exception during retrieval shouldn't leave a stub dir behind.
    assert not auto_dir.exists() or not any(auto_dir.iterdir())


@pytest.mark.asyncio
async def test_node_auto_collect_data_zero_hits_falls_through(
    tmp_path: Path,
) -> None:
    """Axon returned but found nothing — log INFO, return zero, let
    wait_for_data pause for the user. Don't create the empty
    auto_collected dir (would mislead the user into thinking the agent
    found something)."""
    from unittest.mock import AsyncMock

    engine = _make_no_sim_engine(tmp_path)
    engine.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]

    result = await engine._node_auto_collect_data({"topic": "x", "design": {}})

    assert result == {"auto_collected_count": 0}
    auto_dir = engine.quest_root / "data" / "auto_collected"
    assert not auto_dir.exists(), (
        "0-hit path must not create an empty auto_collected dir — the "
        "user would mistake it for a partial success"
    )


@pytest.mark.asyncio
async def test_node_auto_collect_data_files_pass_wait_for_data_filter(
    tmp_path: Path,
) -> None:
    """Integration check across γ → β boundary: files written by
    auto_collect_data MUST be picked up by ``_list_user_data_files``
    (used by ``wait_for_data`` to decide whether to pause). Without
    this, auto-collect would happily write files and wait_for_data
    would still pause because of a path-filtering mismatch."""
    from unittest.mock import AsyncMock
    from core.knowledge import RetrievedDoc
    from core.engine import _list_user_data_files

    engine = _make_no_sim_engine(tmp_path)
    engine.knowledge.asearch = AsyncMock(return_value=[  # type: ignore[method-assign]
        RetrievedDoc(content="hit", metadata={"source": "p.pdf"}),
    ])

    await engine._node_auto_collect_data({"topic": "x", "design": {}})

    listed = _list_user_data_files(engine.quest_root / "data")
    assert len(listed) == 1
    assert listed[0].parent.name == "auto_collected"
    assert listed[0].suffix == ".md"


@pytest.mark.asyncio
async def test_node_data_load_filters_readme_from_walked_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_walk_folder`` from core.summarizer doesn't know about the
    FI-authored README. Without filtering, the prompt would include
    the README's contents (topic + hypothesis + resume command)
    AS IF IT WERE user-supplied evidence, and the LLM might cite it
    as a primary source in key_findings.

    ``_node_data_load`` must drop the top-level README before
    rendering manifest / content blocks so the LLM only sees the
    user's actual data."""
    from unittest.mock import AsyncMock

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

    # Build the same data dir layout the engine would have produced:
    # README.md from FI + the user's actual data files.
    data_dir = engine.quest_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "README.md").write_text(
        "# Drop your data here\n\n"
        "## The hypothesis\n\n"
        "> Trust correlates with X (this should NOT end up as a 'finding')\n",
        encoding="utf-8",
    )
    (data_dir / "survey.csv").write_text("country,trust\nBE,0.6\nTW,0.7\n", encoding="utf-8")
    (data_dir / "notes.md").write_text(
        "# Field notes\n\nObserved that public-trust institutions...\n",
        encoding="utf-8",
    )

    captured_prompt: dict[str, str] = {}

    async def fake_chat(prompt, *, node="", **kw):
        captured_prompt["text"] = prompt
        return '{"summary": "ok", "key_findings": [], "measurements": {}}'

    monkeypatch.setattr(engine, "_chat", fake_chat)

    state = {"topic": "test", "design": {"hypothesis": "h"}}
    out = await engine._node_data_load(state)

    # README content must NOT appear in the prompt the LLM saw.
    prompt_text = captured_prompt["text"]
    assert "Drop your data here" not in prompt_text, (
        "FI-authored README leaked into the data_load prompt"
    )
    assert "this should NOT end up" not in prompt_text

    # The actual user data DOES appear.
    assert "survey.csv" in prompt_text
    assert "notes.md" in prompt_text
    assert "0.6" in prompt_text or "trust" in prompt_text

    # data_files returned should contain ONLY the user's files (2),
    # not 3 (which would mean the README slipped through).
    assert len(out["data_files"]) == 2


@pytest.mark.asyncio
async def test_node_data_load_handles_readme_only_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the only file in <quest_root>/data/ is the FI-authored
    README (user resumed too early without dropping data), the node
    must return an empty result_json rather than synthesize findings
    from the README's own instructions."""
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
    data_dir = engine.quest_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "README.md").write_text("# Drop your data here\n", encoding="utf-8")

    # The LLM must not be called in this case.
    called = {"n": 0}

    async def fake_chat(prompt, *, node="", **kw):  # pragma: no cover
        called["n"] += 1
        return "{}"

    monkeypatch.setattr(engine, "_chat", fake_chat)
    out = await engine._node_data_load({"topic": "t", "design": {}})
    assert out == {"result_json": {}, "data_files": []}
    assert called["n"] == 0, "LLM was called despite empty user data"


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

    Pins the docstring claim of 'every exit path'."""
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


# ---- _run_ideate_tournament ---------------------------------------------


def _make_tournament_engine(tmp_path: Path) -> "Engine":
    cfg = Config(
        topic="some research topic",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(ideate_tournament=True, ideate_reflect=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    return Engine(cfg)


def test_engine_config_ideate_tournament_default_false() -> None:
    """Tournament must be OPT-IN — it costs 2 extra LLM calls vs.
    the single-shot critique. Default-off keeps the cost floor at 7-18
    calls/quest per docs/USAGE.md."""
    from core.config import EngineConfig
    assert EngineConfig().ideate_tournament is False


def test_ideate_tournament_prompt_loads() -> None:
    """The ``ideate_tournament`` prompt is added to ``_load_prompts``.
    Regression guard so a future prompt-list edit doesn't silently
    drop the tournament prompt and leave the feature unable to run."""
    from core.engine import _load_prompts
    prompts = _load_prompts()
    assert "ideate_tournament" in prompts
    t = prompts["ideate_tournament"]
    # The prompt must accept the four variables the engine substitutes.
    rendered = t.substitute(topic="x", clarify_block="y", idea_a="a", idea_b="b")
    assert "winner" in rendered.lower()


@pytest.mark.asyncio
async def test_run_ideate_tournament_picks_majority_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 ideas → 3 pairwise matches. If idea 0 wins both its matches,
    it has 2 wins; the other two have 1 win and 0 wins. Tournament
    must pick idea 0."""
    engine = _make_tournament_engine(tmp_path)
    ideas = [
        {"title": "Alpha", "summary": "first"},
        {"title": "Beta", "summary": "second"},
        {"title": "Gamma", "summary": "third"},
    ]
    # Match outcomes (a_idx, b_idx) → winner letter
    outcomes = {
        (0, 1): "A",  # Alpha beats Beta
        (0, 2): "A",  # Alpha beats Gamma
        (1, 2): "B",  # Gamma beats Beta
    }
    call_count = {"n": 0}

    async def fake_chat(self, prompt, *, node, **kw):
        call_count["n"] += 1
        # Extract a/b indices from the prompt text (each prompt has the
        # full idea JSON inline, so we match on title to know which pair).
        is_alpha_in_a = '"Alpha"' in prompt.split('# Idea B')[0]
        is_beta_in_a = '"Beta"' in prompt.split('# Idea B')[0]
        if is_alpha_in_a and '"Beta"' in prompt.split('# Idea B')[1]:
            w = outcomes[(0, 1)]
        elif is_alpha_in_a and '"Gamma"' in prompt.split('# Idea B')[1]:
            w = outcomes[(0, 2)]
        elif is_beta_in_a and '"Gamma"' in prompt.split('# Idea B')[1]:
            w = outcomes[(1, 2)]
        else:
            raise AssertionError(f"unexpected pair in prompt: {prompt[:200]}")
        return f'{{"winner": "{w}", "reason": "tractability", "margin": "decisive"}}'

    # monkeypatch.setattr is auto-scoped to this test — pytest's
    # teardown restores Engine._chat even if an assertion raises mid-test.
    # Patching the class attribute directly with try/finally is fragile
    # to a setup-time exception leaking the replacement into later tests.
    from core.engine import Engine as _Engine
    monkeypatch.setattr(_Engine, "_chat", fake_chat)

    winner, record = await engine._run_ideate_tournament(
        {"topic": "x"}, ideas, initial_chosen=ideas[1],
    )

    assert winner["title"] == "Alpha"
    assert record["winner_idx"] == 0
    # Alpha wins both its matches (vs Beta, vs Gamma) → idx 0: 2 wins.
    # Beta loses both its matches (vs Alpha, vs Gamma) → idx 1: 0 wins.
    # Gamma loses to Alpha, beats Beta → idx 2: 1 win.
    assert record["wins"] == [2, 0, 1]
    assert record["outcome"] == "swapped"  # initial_chosen was Beta
    assert call_count["n"] == 3  # C(3, 2)


@pytest.mark.asyncio
async def test_run_ideate_tournament_falls_back_when_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every match returns malformed JSON (winner != "A" or "B"),
    no idea accumulates wins. Tournament must fall back to
    ``initial_chosen`` rather than crashing or picking arbitrarily."""
    engine = _make_tournament_engine(tmp_path)
    ideas = [{"title": "A", "summary": "x"}, {"title": "B", "summary": "y"}]
    initial = ideas[1]

    async def fake_chat(self, prompt, *, node, **kw):
        return '{"winner": "TIE", "reason": "?"}'  # invalid winner

    from core.engine import Engine as _Engine
    monkeypatch.setattr(_Engine, "_chat", fake_chat)
    winner, record = await engine._run_ideate_tournament(
        {"topic": "x"}, ideas, initial_chosen=initial,
    )

    assert winner["title"] == "B"  # initial_chosen preserved
    assert record["outcome"] == "inconclusive_fallback"
    assert record["winner_idx"] is None


@pytest.mark.asyncio
async def test_run_ideate_tournament_dispatches_matches_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 pairwise matches with 200ms simulated latency each. Parallel
    dispatch → wall-clock ~200ms; serial would be 600ms. Threshold
    set well below the serial floor so the test stays unambiguous
    even under CI scheduling jitter / GC pauses (a tighter
    130 ms-with-50 ms-sleeps threshold would flake on constrained
    runners)."""
    import asyncio as _asyncio
    import time as _time

    engine = _make_tournament_engine(tmp_path)
    ideas = [{"title": "A"}, {"title": "B"}, {"title": "C"}]

    async def slow_chat(self, prompt, *, node, **kw):
        await _asyncio.sleep(0.2)
        return '{"winner": "A", "reason": "x", "margin": "decisive"}'

    from core.engine import Engine as _Engine
    monkeypatch.setattr(_Engine, "_chat", slow_chat)

    start = _time.monotonic()
    await engine._run_ideate_tournament(
        {"topic": "x"}, ideas, initial_chosen=ideas[0],
    )
    elapsed = _time.monotonic() - start

    assert elapsed < 0.5, (
        f"3 matches must dispatch concurrently; wall-clock {elapsed:.3f}s "
        f"suggests serial execution (serial floor is ~0.6s for 3 × 200ms; "
        f"parallel ceiling is ~0.2s + overhead)."
    )


# ---- non-scientific paper formats + write-persona ----------------------


def test_paper_format_literal_includes_non_scientific_values() -> None:
    """PaperFormat covers essay/report/policy_brief/whitepaper as
    well as the scientific venues. Regression guard so a future
    Literal edit can't silently drop a format."""
    from typing import get_args
    from core.config import PaperFormat
    values = set(get_args(PaperFormat))
    # Scientific (must remain — back-compat).
    for v in ("generic", "neurips", "iclr", "ieee_access", "nature_mi"):
        assert v in values, f"scientific format {v!r} was dropped"
    # Non-scientific formats.
    for v in ("essay", "report", "policy_brief", "whitepaper"):
        assert v in values, f"non-scientific format {v!r} missing"


def test_paper_format_subsets_partition_the_literal() -> None:
    """``SCIENTIFIC_PAPER_FORMATS`` ∪ ``NON_SCIENTIFIC_PAPER_FORMATS``
    must equal the full ``PaperFormat`` set with empty intersection.
    Without this guard, adding a new venue to the Literal could leave
    it unclassified by either subset and ``_resolve_write_persona``
    would silently fall through to the default voice — invisible
    regression."""
    from typing import get_args
    from core.config import (
        NON_SCIENTIFIC_PAPER_FORMATS,
        PaperFormat,
        SCIENTIFIC_PAPER_FORMATS,
    )
    all_values = set(get_args(PaperFormat))
    assert SCIENTIFIC_PAPER_FORMATS | NON_SCIENTIFIC_PAPER_FORMATS == all_values
    assert SCIENTIFIC_PAPER_FORMATS & NON_SCIENTIFIC_PAPER_FORMATS == set()


def test_paper_format_templates_exist_with_body_placeholder() -> None:
    """Each non-scientific format must have a usable LaTeX template
    that pandoc accepts — stub templates (one-line `%` comment files)
    would fail at compile time. Each template needs a `$body$`
    placeholder + a `\\begin{document}` block at minimum so pandoc
    treats them as templates rather than invalid documents."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    for fmt in ("essay", "report", "policy_brief", "whitepaper"):
        path = repo_root / "templates" / "paper" / fmt / "template.tex"
        assert path.exists(), f"missing template.tex for {fmt}"
        body = path.read_text(encoding="utf-8")
        assert "$body$" in body, (
            f"templates/paper/{fmt}/template.tex must contain "
            f"pandoc's $body$ placeholder so the paper.md body lands "
            f"in the rendered PDF"
        )
        assert "\\begin{document}" in body, (
            f"templates/paper/{fmt}/template.tex must be a complete "
            f"LaTeX document (avoids the stub-template trap)"
        )


def _make_write_persona_engine(
    tmp_path: Path, *, paper_format: str = "generic"
) -> "Engine":
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(
            output_dir=tmp_path / "outputs", paper_format=paper_format,
        ),
    )
    return Engine(cfg)


def test_resolve_write_persona_scientific_returns_empty(tmp_path: Path) -> None:
    """The five scientific venues (generic + 4 named) keep
    ``write.md``'s built-in functional framing — the persona block
    is empty so the prompt's default voice carries it."""
    for fmt in ("generic", "neurips", "iclr", "ieee_access", "nature_mi"):
        engine = _make_write_persona_engine(tmp_path, paper_format=fmt)
        assert engine._resolve_write_persona({"clarify_answers": {}}) == "", (
            f"scientific format {fmt!r} must return empty persona block"
        )


def test_resolve_write_persona_non_scientific_returns_named_voice(
    tmp_path: Path,
) -> None:
    """Each non-scientific format swaps the default voice for a
    format-appropriate persona (essayist / consulting analyst /
    policy analyst / industry analyst)."""
    expectations = {
        "essay": "essayist",
        "report": "consulting analyst",
        "policy_brief": "policy analyst",
        "whitepaper": "industry analyst",
    }
    for fmt, marker in expectations.items():
        engine = _make_write_persona_engine(tmp_path, paper_format=fmt)
        persona = engine._resolve_write_persona({"clarify_answers": {}})
        assert persona, f"{fmt!r} must produce a non-empty persona block"
        assert marker in persona.lower(), (
            f"{fmt!r} persona missing the expected marker {marker!r}: "
            f"{persona[:200]!r}"
        )


def test_resolve_write_persona_clarify_answers_win_over_yaml(
    tmp_path: Path,
) -> None:
    """The clarify agent's pick (or user override in interactive
    mode) wins over the YAML default. So a quest configured with
    ``paper_format: generic`` whose clarify answer landed on
    ``paper_venue: essay`` produces the essayist persona, not the
    scientific default."""
    engine = _make_write_persona_engine(tmp_path, paper_format="generic")
    persona = engine._resolve_write_persona({
        "clarify_answers": {"paper_venue": "essay"}
    })
    assert "essayist" in persona.lower()

    # Reverse: clarify=generic + YAML=essay → clarify wins.
    engine2 = _make_write_persona_engine(tmp_path, paper_format="essay")
    persona2 = engine2._resolve_write_persona({
        "clarify_answers": {"paper_venue": "generic"}
    })
    assert persona2 == "", (
        "clarify answer 'generic' must override YAML 'essay' → empty persona"
    )


def test_resolve_write_persona_unknown_format_falls_back_to_empty(
    tmp_path: Path,
) -> None:
    """A clarify answer outside the Literal (e.g. a typo or future
    format not yet wired) must fall back to the default voice, not
    crash. The Literal validation happens upstream at YAML load
    time — this is defense-in-depth for the clarify path which
    accepts free-form strings."""
    engine = _make_write_persona_engine(tmp_path)
    persona = engine._resolve_write_persona({
        "clarify_answers": {"paper_venue": "made_up_format"}
    })
    assert persona == ""


@pytest.mark.asyncio
async def test_run_ideate_tournament_skipped_when_fewer_than_two_ideas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C(N, 2) is 0 for N=1, so the tournament has nothing to do.
    ``_node_ideate``'s guard ``len(ideas) >= 2`` keeps the helper
    from being called in this case — but if it ever IS called with
    fewer ideas, the helper should also handle it gracefully (no
    LLM calls, return initial_chosen)."""
    engine = _make_tournament_engine(tmp_path)
    ideas = [{"title": "Solo"}]

    chat_calls = 0
    async def fake_chat(self, prompt, *, node, **kw):
        nonlocal chat_calls
        chat_calls += 1
        return '{"winner": "A"}'

    from core.engine import Engine as _Engine
    monkeypatch.setattr(_Engine, "_chat", fake_chat)
    winner, record = await engine._run_ideate_tournament(
        {"topic": "x"}, ideas, initial_chosen=ideas[0],
    )

    # No matches to play; outcome falls through to inconclusive_fallback.
    assert chat_calls == 0
    assert winner["title"] == "Solo"
    assert record["outcome"] == "inconclusive_fallback"
