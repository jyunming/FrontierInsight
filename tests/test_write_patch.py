"""A revise for flagged passages edits those passages and leaves the paper as it was.

Until now a reviewed draft went back to the writer with the whole review and
one instruction, to write the whole paper again. Measured on the stored quests,
every such rewrite added background sentences with citations their sources do
not back (28 of 38 graded papers ended at the iteration cap still flagged
``unsupported_claim``), one swapped two citation numbers, and one dropped all
three figures. When every must-fix hit names one passage, the writer now gets the
earlier draft and only those passages and returns ``find``/``replace`` edits; the
engine applies an edit only where its ``find`` occurs exactly once and leaves
every other character alone.

The chat client is a fake that records each call and reports token usage from
the lengths it saw, so the cost log shows what each call cost. No model runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core import paper_patch
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import (
    Engine, _aggregate_cost_rows, _finalize_paper_sources, _hit_name, _paper_basis, build_references,
    cited_references,
)

TOPIC = "How fast do stochastic epidemics die out?"


def _paper(title: str, doi: str) -> dict[str, Any]:
    return {"content": f"{title}. An abstract.", "metadata": {
        "source": "crossref", "title": title, "doi": doi,
        "authors": ["A. Author"], "year": 2020, "venue": "J. Tests"}}


# Labelled 1..4 in this order. The draft below cites them in that order, so the
# write node's renumbering leaves every number as the writer gave it.
LIT = [_paper("Alpha", "10.1/a"), _paper("Beta", "10.1/b"), _paper("Gamma", "10.1/c"), _paper("Delta", "10.1/d")]

DESIGN = {"hypothesis": "DESIGN-SENTINEL", "figures_planned": ["figures/final_size.png"]}
ANALYSIS = {"key_findings": ["The mean final size over 300 runs was 0.583."]}
FIGURES = ["figures/final_size.png"]

CLAIM = "Extinction is certain below R0 = 1.2 [3]."
CAPTION = "**Figure 1.** Final size against R0 for the deterministic model and the mean of the simulations."
NEW_CLAIM = "Extinction is likely at low R0 [3]."

# What the writer returns for a first draft (the engine lists the sources).
DRAFT = (
    "# Extinction Times in Stochastic Epidemics\n\n"
    "## Abstract\nWe measure how often an epidemic started by one case dies out. The extinction "
    "probability was 0.34 at R0 = 1.5.\n\n"
    "**Keywords:** SIR, extinction, Gillespie, branching process\n\n"
    "## Introduction\nDeterministic models predict a final size for every R0 [1]. Stochastic models add "
    "early extinction [2]. " + CLAIM + " Branching processes give the extinction probability 1/R0 [2, 4].\n\n"
    "## Results\n"
    f"![{CAPTION}](figures/final_size.png)\n\n"
    "Figure 1 shows the final size rising with R0. The mean over 300 runs was 0.583.\n\n"
    "## Discussion\nThe gap between the two is the early extinction [2]. A larger population would shrink "
    "the gap [1].\n"
)


class FakeClient:
    """Answers each chat call from ``replies`` in order, remembers it, and
    reports usage from the sizes it saw (a quarter of a token per character)."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []  # (node, prompt)
        self.last_usage: dict[str, int] | None = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[-1]["content"]
        self.calls.append((kw.get("node", ""), prompt))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        self.last_usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(reply) // 4}
        return reply

    @property
    def nodes(self) -> list[str]:
        return [node for node, _ in self.calls]


def _engine(
    tmp_path: Path, replies: list[Any], **provider: Any,
) -> tuple[Engine, FakeClient, list[tuple[int, str]]]:
    eng = Engine(Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai", **provider),
        engine=EngineConfig(max_iterations=3, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    client = FakeClient(replies)
    eng._client = client  # type: ignore[assignment]
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    logged: list[tuple[int, str]] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logged.append((record.levelno, record.getMessage()))

    eng._log.addHandler(_Capture())
    return eng, client, logged


def _study(**extra: Any) -> dict[str, Any]:
    return {
        "topic": TOPIC, "title": "t", "iteration": 0, "literature": LIT, "design": DESIGN,
        "analysis": ANALYSIS, "figures": FIGURES, **extra,
    }


def _first_draft(eng: Engine) -> tuple[dict[str, Any], str]:
    out = asyncio.run(eng._node_write(_study()))  # type: ignore[arg-type]
    return out, Path(out["paper_md"]).read_text(encoding="utf-8")


def _revise_state(out: dict[str, Any], hits: list[str], **extra: Any) -> dict[str, Any]:
    """The state the graph hands the write node after a review of the draft the
    last write returned: its literature, its paper and its basis."""
    review = {"verdict": "revise", "must_flag_hits": hits, **extra.pop("review", {})}
    grounding = extra.pop("claim_grounding", {
        "claims": [{"claim": "Extinction is certain below R0 = 1.2.", "basis": "unsupported",
                    "evidence": "Neither the results nor [3] say so."}],
        "unsupported": ["Extinction is certain below R0 = 1.2."],
    })
    return _study(
        literature=out["literature"], paper_md=out["paper_md"], paper_basis=out["paper_basis"],
        iteration=1, review=review, claim_grounding=grounding, **extra,
    )


def _revise(eng: Engine, out: dict[str, Any], hits: list[str], **extra: Any) -> str:
    result = asyncio.run(eng._node_write(_revise_state(out, hits, **extra)))  # type: ignore[arg-type]
    return Path(result["paper_md"]).read_text(encoding="utf-8")


def _edits(*pairs: tuple[str, str]) -> str:
    return json.dumps([{"find": f, "replace": r} for f, r in pairs])


def _cost_rows(root: Path) -> list[dict[str, Any]]:
    path = root / ".fi" / "cost.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _warnings(logged: list[tuple[int, str]]) -> list[str]:
    return [msg for level, msg in logged if level >= logging.WARNING]


# --- one flagged sentence: only that sentence changes ---------------------------------

def test_a_revise_for_one_flagged_sentence_changes_only_that_sentence(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, _edits((CLAIM, NEW_CLAIM))])
    out1, before = _first_draft(eng)
    assert CLAIM in before and client.nodes == ["write"]

    out2 = asyncio.run(eng._node_write(_revise_state(out1, ["unsupported_claim"])))  # type: ignore[arg-type]
    after = Path(out2["paper_md"]).read_text(encoding="utf-8")

    assert client.nodes == ["write", "write.patch"], "no whole-paper call for this round"
    # Everything but the one sentence is byte-identical, the source lists and figures included.
    assert after == before.replace(CLAIM, NEW_CLAIM)
    assert after.replace(NEW_CLAIM, CLAIM) == before
    assert f"![{CAPTION}](figures/final_size.png)" in after
    assert [r["n"] for r in cited_references(out2["literature"], after)] == [1, 2, 3, 4]
    assert [r["title"] for r in build_references(out2["literature"])] == ["Alpha", "Beta", "Gamma", "Delta"]
    assert out2["paper_basis"] == out1["paper_basis"], "the study is the same, so the next round may edit again"
    assert not _warnings(logged)
    assert any("applied 1 edit(s)" in m and "as it was" in m for _l, m in logged)


def test_the_writer_gets_the_draft_and_the_passages_and_not_the_whole_study(tmp_path: Path) -> None:
    eng, client, _ = _engine(tmp_path, [DRAFT, _edits((CLAIM, NEW_CLAIM))])
    out1, before = _first_draft(eng)
    _revise(eng, out1, ["unsupported_claim"])
    prompt = client.calls[1][1]
    assert before in prompt, "the whole earlier draft, exactly as it is"
    assert "Extinction is certain below R0 = 1.2." in prompt, "the claim the check flagged"
    assert json.dumps(CLAIM) in prompt, "and the draft's own words for it, escaped as `find` needs"
    assert "Neither the results nor [3] say so." in prompt, "with the check's reason"
    assert "[3] A. Author (2020). Gamma" in prompt, "the prior-work block the labels come from"
    assert "DESIGN-SENTINEL" not in prompt and "Write the whole paper again" not in prompt
    assert "DESIGN-SENTINEL" in client.calls[0][1], "the first draft's prompt is the writer's, as before"


def test_two_edits_are_applied_together(tmp_path: Path) -> None:
    new_caption = "**Figure 1.** Final size against R0: the deterministic curve and the mean of 300 simulations."
    finding = ('figure_caption: the caption of figures/final_size.png names "extinction", but the figure '
               'does not show it on its axis "final size"')
    eng, client, logged = _engine(tmp_path, [DRAFT, _edits((CAPTION, new_caption), (CLAIM, NEW_CLAIM))])
    out1, before = _first_draft(eng)
    after = _revise(
        eng, out1, ["unsupported_claim", "figure_caption"], review={"figure_caption_warnings": [finding]},
    )
    assert client.nodes == ["write", "write.patch"]
    assert after == before.replace(CLAIM, NEW_CLAIM).replace(CAPTION, new_caption)
    assert "](figures/final_size.png)" in after
    assert json.dumps(CAPTION) in client.calls[1][1], "the caption is handed over as the caption, not the link"
    assert not _warnings(logged)


def test_a_flagged_sentence_taken_out_renumbers_the_sources_consistently(tmp_path: Path) -> None:
    """Taking out the only sentence that cites [3] leaves Gamma uncited, so Delta
    becomes [3]: the text and the References agree, and nothing else moves."""
    eng, client, _ = _engine(tmp_path, [DRAFT, _edits((CLAIM + " ", ""))])
    out1, before = _first_draft(eng)
    result = asyncio.run(eng._node_write(_revise_state(out1, ["unsupported_claim"])))  # type: ignore[arg-type]
    after = Path(result["paper_md"]).read_text(encoding="utf-8")
    assert client.nodes == ["write", "write.patch"]
    assert "Extinction is certain" not in after
    assert "extinction probability 1/R0 [2, 3]." in after, "Delta was [4] and is now [3]"
    assert [r["title"] for r in cited_references(result["literature"], after)] == ["Alpha", "Beta", "Delta"]
    assert [ln for ln in after.split("## References\n", 1)[1].splitlines() if ln.strip()] == [
        "1. A. Author (2020). Alpha. J. Tests. DOI: 10.1/a",
        "2. A. Author (2020). Beta. J. Tests. DOI: 10.1/b",
        "3. A. Author (2020). Delta. J. Tests. DOI: 10.1/d",
    ]
    # Outside the sentence and the one renumbered bracket, the paper is as it was.
    body_before = before.split("## References")[0].replace(CLAIM + " ", "")
    assert after.split("## References")[0].replace("[2, 3]", "[2, 4]") == body_before


def test_a_persona_prefixed_hit_is_a_hit_like_any_other(tmp_path: Path) -> None:
    eng, client, _ = _engine(tmp_path, [DRAFT, _edits((CLAIM, NEW_CLAIM))])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, ["[methodologist] unsupported_claim"])
    assert client.nodes == ["write", "write.patch"]
    assert _hit_name("[methodologist] unsupported_claim") == "unsupported_claim"


# --- what the model gives back cannot be used: the whole paper is written again --------

def _fallback(tmp_path: Path, reply: Any) -> tuple[FakeClient, list[tuple[int, str]], str]:
    rewrite = DRAFT.replace(CLAIM, NEW_CLAIM)
    eng, client, logged = _engine(tmp_path, [DRAFT, reply, rewrite])
    out1, _before = _first_draft(eng)
    return client, logged, _revise(eng, out1, ["unsupported_claim"])


def test_an_edit_whose_find_is_not_in_the_draft_falls_back_to_a_whole_rewrite(tmp_path: Path) -> None:
    client, logged, after = _fallback(tmp_path, _edits(("Extinction is certain everywhere.", "x")))
    assert client.nodes == ["write", "write.patch", "write"]
    assert NEW_CLAIM in after, "the whole-paper call's paper is what is delivered"
    assert any(
        "the edits could not be used (edit 1: the text to find is not in the draft" in m
        and "writing the whole paper again" in m for m in _warnings(logged)
    )


def test_an_edit_whose_find_occurs_twice_falls_back(tmp_path: Path) -> None:
    assert DRAFT.count("early extinction") == 2
    client, logged, _after = _fallback(tmp_path, _edits(("early extinction", "early die-out")))
    assert client.nodes == ["write", "write.patch", "write"]
    assert any("is in the draft 2 times" in m for m in _warnings(logged))


@pytest.mark.parametrize(("reply", "why"), [
    ("Here are the edits I would make: replace the certain claim.", "the reply is not JSON"),
    ("# Extinction Times\n\n## Abstract\nA whole new paper.\n", "the reply is a whole paper, not edits"),
    ("[]", "the reply holds no edits"),
    ('[{"replace": "x"}]', "edit 1 is not a find/replace pair"),
    ('{"answer": 3}', "the reply is JSON but not a list of edits"),
    ("", "the reply is empty"),
    (_edits((CLAIM, CLAIM)), "the edits change nothing"),
    (_edits((CLAIM, "x"), (CLAIM, "y")), "two edits overlap"),
    (_edits((CAPTION, "**Figure 1.** Changed."), ("](figures/final_size.png)", "](figures/other.png)")),
     "an edit changed a figure link or a heading"),
    (_edits(("## Discussion", "## Conclusions")), "an edit changed a figure link or a heading"),
    # The engine writes the source lists from the citations in the text: an edit
    # to a reference entry would be undone by the next step, so it is refused.
    (_edits(("A. Author (2020). Alpha. J. Tests. DOI: 10.1/a", "A. Author (2021). Alpha.")),
     "the text to find is not in the draft"),
])
def test_a_reply_that_is_not_usable_edits_falls_back_and_says_why(tmp_path: Path, reply: str, why: str) -> None:
    client, logged, after = _fallback(tmp_path, reply)
    assert client.nodes == ["write", "write.patch", "write"]
    assert any(why in m for m in _warnings(logged)), _warnings(logged)
    assert after.startswith("# Extinction Times") and NEW_CLAIM in after


def test_edits_that_replace_most_of_the_paper_are_a_rewrite_in_disguise(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, "placeholder", DRAFT])
    out1, before = _first_draft(eng)
    body = before[:before.index("## References") - 2]
    client.replies[0] = _edits((body, "A short paper."))
    _revise(eng, out1, ["unsupported_claim"])
    assert client.nodes == ["write", "write.patch", "write"]
    assert any("the edits replace" in m for m in _warnings(logged))


def test_a_failed_edit_call_falls_back_to_the_whole_paper(tmp_path: Path) -> None:
    client, logged, _after = _fallback(tmp_path, RuntimeError("prompt too long"))
    assert client.nodes == ["write", "write.patch", "write"]
    assert any("the call for edits failed (RuntimeError: prompt too long)" in m for m in _warnings(logged))


# --- rounds that are not answered with edits ------------------------------------------

@pytest.mark.parametrize("hits", [
    ["unsupported_claim", "over_page_limit: the rendered PDF is 5 pages; the limit is 4. Cut about 200 words"],
    ["figure_missing: the paper leaves out figures/x.png, which the design planned"],
    ["citations_unchecked: the claim check could not run on this draft"],
    ["unsupported_claim", "single_point_eval"],
])
def test_a_hit_about_the_whole_paper_writes_the_whole_paper(tmp_path: Path, hits: list[str]) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, hits)
    assert client.nodes == ["write", "write"], "today's path: no call for edits"
    assert any("writing the whole paper again:" in m for lvl, m in logged if lvl == logging.INFO)
    assert "Write the whole paper again" in client.calls[1][1]


def test_a_flag_that_names_no_passage_writes_the_whole_paper(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, ["unsupported_claim"], claim_grounding={"claims": [], "unsupported": []})
    assert client.nodes == ["write", "write"]
    assert any("lists none" in m for _l, m in logged)


def test_a_study_that_was_run_again_writes_the_whole_paper(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, ["unsupported_claim"], analysis={"key_findings": ["A new result."]})
    assert client.nodes == ["write", "write"]
    assert any("has been run again since" in m for _l, m in logged)


def test_a_draft_written_before_the_basis_was_recorded_is_written_again(tmp_path: Path) -> None:
    eng, client, _logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    state = _revise_state(out1, ["unsupported_claim"])
    del state["paper_basis"]
    asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    assert client.nodes == ["write", "write"]


def test_no_draft_on_disk_writes_the_whole_paper(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    state = _revise_state(out1, ["unsupported_claim"])
    Path(state["paper_md"]).unlink()
    asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    assert client.nodes == ["write", "write"]
    assert any("no earlier draft on disk" in m for _l, m in logged)


def test_a_draft_longer_than_a_model_reads_writes_the_whole_paper(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, before = _first_draft(eng)
    Path(out1["paper_md"]).write_text(before + ("Filler sentence. " * 9000), encoding="utf-8")
    _revise(eng, out1, ["unsupported_claim"])
    assert client.nodes == ["write", "write"]
    assert any("characters a model reads" in m for _l, m in logged)


def test_a_first_draft_is_written_as_before(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT])
    out = asyncio.run(eng._node_write(_study()))  # type: ignore[arg-type]
    assert client.nodes == ["write"]
    assert client.calls[0][1].rstrip().endswith("## Review of the previous draft\n(none — first draft)")
    assert not [m for _l, m in logged if "whole paper again" in m or "editing the earlier draft" in m]
    assert set(out) == {"paper_md", "literature", "paper_basis"}


def test_a_revise_with_no_must_fix_hit_is_written_as_before(tmp_path: Path) -> None:
    eng, client, logged = _engine(tmp_path, [DRAFT, DRAFT])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, [], claim_grounding={})
    assert client.nodes == ["write", "write"]
    assert not [m for _l, m in logged if "whole paper again" in m]


# --- what the edit call costs ---------------------------------------------------------

def test_the_edit_call_is_a_smaller_call_than_the_whole_paper_call(tmp_path: Path) -> None:
    """The same flagged claim, answered once with edits and once by writing the
    paper again: the cost log has the edit call under its own node, and it is the
    smaller call, in its reply above all."""
    edit_dir, whole_dir = tmp_path / "edit", tmp_path / "whole"
    eng, _client, _ = _engine(edit_dir, [DRAFT, _edits((CLAIM, NEW_CLAIM))])
    out1, _before = _first_draft(eng)
    _revise(eng, out1, ["unsupported_claim"])

    eng2, _client2, _ = _engine(whole_dir, [DRAFT, DRAFT.replace(CLAIM, NEW_CLAIM)])
    out1b, _ = _first_draft(eng2)
    _revise(eng2, out1b, ["unsupported_claim", "over_page_limit: the rendered PDF is 5 pages"])

    patch_row = next(r for r in _cost_rows(edit_dir) if r["node"] == "write.patch")
    rewrite_row = [r for r in _cost_rows(whole_dir) if r["node"] == "write"][1]
    # A one-kilobyte paper: the reply is a few dozen tokens against a paper's worth.
    assert patch_row["usage"]["completion_tokens"] * 3 < rewrite_row["usage"]["completion_tokens"]
    assert patch_row["usage"]["prompt_tokens"] < rewrite_row["usage"]["prompt_tokens"]
    by_node = _aggregate_cost_rows(_cost_rows(edit_dir))["by_node"]
    assert by_node["write.patch"]["requests"] == 1 and by_node["write"]["requests"] == 1
    assert by_node["write.patch"]["total_tokens"] < by_node["write"]["total_tokens"]


def test_the_edit_call_gets_the_writers_time_budget() -> None:
    """A long prompt on a slow server needs the time a whole-paper write gets, or
    the call for edits times out and every round falls back."""
    from unittest.mock import AsyncMock, MagicMock

    from core.provider import LLMClient, ResolvedEndpoint, node_budget

    assert node_budget({"write": 600.0}, "write.patch", 120.0) == 600.0
    assert node_budget({"write": 600.0, "write.patch": 90.0}, "write.patch", 120.0) == 90.0
    assert node_budget({"write": 600.0}, "write", 120.0) == 600.0
    assert node_budget({"write": 600.0}, "clarify", 120.0) == 120.0
    assert node_budget({"write": 600.0}, "rewrite.patch", 120.0) == 120.0, "only the part before the dot counts"

    response = MagicMock()
    response.json = MagicMock(return_value={"choices": [{"message": {"content": "[]"}}]})
    response.raise_for_status = MagicMock()
    http = MagicMock()
    http.post = AsyncMock(return_value=response)
    client = LLMClient(
        ResolvedEndpoint(base_url="https://x/v1", model="m", api_key="k"),
        http=http, timeout_s=120.0, node_http_timeout_s={"write": 600.0},
    )
    asyncio.run(client.chat([{"role": "user", "content": "hi"}], node="write.patch"))
    assert http.post.call_args.kwargs["timeout"] == 600.0


def test_the_edit_call_uses_the_writers_model_setting(tmp_path: Path) -> None:
    eng, _client, _ = _engine(tmp_path, [], node_models={"write": "writer-model"})
    assert eng._model_for_node("write.patch") == "writer-model"
    eng2, _client2, _ = _engine(tmp_path / "other", [], node_models={"write.patch": "editor-model", "write": "w"})
    assert eng2._model_for_node("write.patch") == "editor-model"


# --- through the graph ---------------------------------------------------------------

def test_through_the_graph_the_second_write_edits_the_first(tmp_path: Path) -> None:
    """The basis and the draft survive the checkpoint, so the write node the
    review sends the quest back to edits the draft the first one wrote."""
    eng, client, _ = _engine(tmp_path, [DRAFT, _edits((CLAIM, NEW_CLAIM))])
    reviews = iter([
        {"review": {"verdict": "revise", "must_flag_hits": ["unsupported_claim"]}, "iteration": 1},
        {"review": {"verdict": "accept", "must_flag_hits": []}},
    ])
    grounding = {
        "claims": [{"claim": "Extinction is certain below R0 = 1.2.", "basis": "unsupported", "evidence": "x"}],
        "unsupported": ["Extinction is certain below R0 = 1.2."],
    }

    async def claim_check(state: dict[str, Any]) -> dict[str, Any]:
        return {"claim_grounding": grounding}

    async def review(state: dict[str, Any]) -> dict[str, Any]:
        return next(reviews)

    async def never(state: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("the experiment is not run again")

    for name in ("design", "implement_outline", "implement", "execute", "analyze"):
        setattr(eng, f"_node_{name}", never)
    eng._node_claim_check = claim_check  # type: ignore[method-assign]
    eng._node_review = review  # type: ignore[method-assign]
    eng._route_after_evidence_gate = lambda state: "write"  # type: ignore[method-assign]
    graph = eng._build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "patch"}}
    graph.update_state(config, _study(), as_node="evidence_gate")
    asyncio.run(graph.ainvoke(None, config))

    assert client.nodes == ["write", "write.patch"]
    final = graph.get_state(config).values
    first = _finalize_paper_sources(DRAFT, LIT, "external")[0]
    assert Path(final["paper_md"]).read_text(encoding="utf-8") == first.replace(CLAIM, NEW_CLAIM)
    assert final["paper_basis"] == _paper_basis(_study())  # type: ignore[arg-type]


# --- the basis -----------------------------------------------------------------------

def test_the_basis_changes_with_the_study_and_not_with_the_review() -> None:
    base = _study()
    assert _paper_basis(base) == _paper_basis(_study())  # type: ignore[arg-type]
    assert _paper_basis(base) != _paper_basis(_study(analysis={"key_findings": []}))  # type: ignore[arg-type]
    assert _paper_basis(base) != _paper_basis(_study(feedback_history=[{"iteration": 0, "text": "x"}]))  # type: ignore[arg-type]
    assert _paper_basis(base) == _paper_basis(_study(review={"verdict": "revise"}, iteration=2))  # type: ignore[arg-type]


# === core.paper_patch, without an engine ================================================

PAPER = (
    "# Title\n\n"
    "## Abstract\nFirst sentence of the abstract. The rate was 0.34 at \\(R_0 = 1.5\\).\n\n"
    "## Methods\nWe run the model. It has a rate \\(\\beta\\) and a limit \\(\\times\\) of two.\n\n"
    "| a | b |\n|---|---|\n| 0.34 | 0.58 |\n\n"
    "- First item is here. Second sentence of the item.\n"
    "- Another item.\n\n"
    "![**Figure 1.** A curve.](figures/curve.png)\n\n"
    "## References\n1. A. Author (2020). Alpha.\n"
)
END = PAPER.index("## References")


def test_units_are_exact_substrings_inside_the_region() -> None:
    units = paper_patch.sentence_units(PAPER[:END])
    texts = [PAPER[s:e] for s, e, _b in units]
    assert "First sentence of the abstract." in texts
    assert "The rate was 0.34 at \\(R_0 = 1.5\\)." in texts
    assert "| 0.34 | 0.58 |" in texts, "a table row is one unit"
    assert "First item is here." in texts and "Second sentence of the item." in texts, "a list item is split"
    assert "![**Figure 1.** A curve.](figures/curve.png)" in texts
    assert all(e <= END and s < e and PAPER[s:e] == PAPER[s:e].strip() for s, e, _b in units)


def test_a_flagged_passage_is_found_as_it_stands_or_as_the_sentence_that_covers_it() -> None:
    units = paper_patch.sentence_units(PAPER[:END])
    assert paper_patch.locate_passage("First sentence of the abstract.", PAPER, END, units) == (
        "First sentence of the abstract.", "exact")
    # A paraphrase (no LaTeX, no `is`): not a substring, but the one sentence that covers it.
    assert paper_patch.locate_passage("The rate was 0.34 at R0 = 1.5", PAPER, END, units) == (
        "The rate was 0.34 at \\(R_0 = 1.5\\).", "sentence")
    assert paper_patch.locate_passage(
        "Something the draft never says about quantum gravity.", PAPER, END, units) == ("", "")
    assert paper_patch.locate_passage("too short", PAPER, END, units) == ("", "")
    got, _how = paper_patch.locate_passage("The 0.34 rate was measured when R0 was 1.5 in the abstract.", PAPER, END, units)
    assert got == "" or got in PAPER, "whatever is returned is the draft's own text"


def test_two_sentences_that_fit_equally_are_not_guessed_between() -> None:
    sentence = "The outbreak probability rises with the reproduction number in large populations."
    twice = f"# T\n\n{sentence}\n\n{sentence}\n"
    units = paper_patch.sentence_units(twice)
    assert paper_patch.locate_passage(
        "Outbreak probability rises with reproduction number in large populations", twice, len(twice), units,
    ) == ("", "")


def test_the_source_lists_are_outside_the_region_edits_reach() -> None:
    edits = [paper_patch.Edit("Alpha", "Beta")]
    with pytest.raises(paper_patch.PatchError, match="is not in the draft"):
        paper_patch.apply_edits(PAPER, END, edits)
    assert "Beta" in paper_patch.apply_edits(PAPER, len(PAPER), edits).text, "in the region the word is unique"


def test_edits_apply_together_and_in_any_order() -> None:
    a = paper_patch.Edit("First sentence of the abstract.", "Opening.")
    b = paper_patch.Edit("Another item.", "")
    one = paper_patch.apply_edits(PAPER, END, [a, b])
    two = paper_patch.apply_edits(PAPER, END, [b, a])
    assert one.text == two.text == PAPER.replace("First sentence of the abstract.", "Opening.").replace(
        "Another item.", "")
    assert one.applied == 2 and one.removed_chars == len(a.find) + len(b.find)


def test_a_noop_edit_is_counted_and_not_applied() -> None:
    done = paper_patch.apply_edits(PAPER, END, [
        paper_patch.Edit("We run the model.", "We run the model."), paper_patch.Edit("Another item.", "One item."),
    ])
    assert done.applied == 1 and done.ignored == 1


def test_the_reply_may_be_fenced_wrapped_in_an_object_or_escape_a_backslash_badly() -> None:
    body = '[{"find": "Another item.", "replace": "One item."}]'
    assert paper_patch.parse_edits(body) == [paper_patch.Edit("Another item.", "One item.")]
    assert paper_patch.parse_edits(f"```json\n{body}\n```") == paper_patch.parse_edits(body)
    assert paper_patch.parse_edits('{"edits": ' + body + "}") == paper_patch.parse_edits(body)
    assert paper_patch.parse_edits("Here you go:\n" + body + "\nDone.") == paper_patch.parse_edits(body)
    assert paper_patch.parse_edits('[{"find": "a", "replace": null}]') == [paper_patch.Edit("a", "")]
    # LaTeX written with one backslash is not valid JSON; the backslash the model meant is kept.
    bad = r'[{"find": "\(R_0 = 1.5\)", "replace": "\(R_0 = 3\)"}]'
    done = paper_patch.apply_edits(PAPER, END, paper_patch.parse_edits(bad))
    assert "at \\(R_0 = 3\\)." in done.text and done.text.count("\\(R_0 = 1.5\\)") == 0


def test_latex_commands_that_json_reads_as_control_characters_are_put_back() -> None:
    # ``\beta`` and ``\times`` written with one backslash are valid JSON escapes
    # (a backspace, a tab) that the draft does not hold, so they are read as LaTeX again.
    reply = r'[{"find": "\(\beta\) and a limit \(\times\) of two", "replace": "\(\beta\) and no limit"}]'
    edits = paper_patch.parse_edits(reply)
    assert "\x08" in edits[0].find and "\t" in edits[0].find
    done = paper_patch.apply_edits(PAPER, END, edits)
    assert "a rate \\(\\beta\\) and no limit." in done.text and "\x08" not in done.text and "\t" not in done.text


def test_an_edit_may_not_touch_the_figures_or_the_headings() -> None:
    with pytest.raises(paper_patch.PatchError, match="figure link or a heading"):
        paper_patch.apply_edits(PAPER, END, [paper_patch.Edit("figures/curve.png", "figures/other.png")])
    with pytest.raises(paper_patch.PatchError, match="figure link or a heading"):
        paper_patch.apply_edits(PAPER, END, [paper_patch.Edit("## Methods", "## Approach")])
    # The caption's words are the caption's to change.
    done = paper_patch.apply_edits(PAPER, END, [paper_patch.Edit("A curve.", "The curve rises.")])
    assert "![**Figure 1.** The curve rises.](figures/curve.png)" in done.text


def test_a_caption_finding_is_located_by_its_figure_file() -> None:
    got, reason = paper_patch.plan_passages(
        PAPER, END, hits=[("figure_caption", "figure_caption")], claims=[],
        captions=['figure_caption: the caption of figures/curve.png names "x", but the figure does not show it'],
    )
    assert reason == "" and [(p.kind, p.located, p.how) for p in got] == [
        ("caption", "**Figure 1.** A curve.", "caption")]
    got, _reason = paper_patch.plan_passages(
        PAPER, END, hits=[("figure_caption", "figure_caption")], claims=[],
        captions=["figure_caption: the caption of figures/nothing.png names x"],
    )
    assert got[0].located == "" and "not found by the engine" in paper_patch.format_passages(got)


def test_a_number_is_located_through_the_context_the_check_quotes() -> None:
    hit = ("unsourced_number: the paper prints 0.34, and nothing in this run accounts for it: it matches no value "
           "— “The rate was 0.34 at R0 = 1.5”")
    got, reason = paper_patch.plan_passages(PAPER, END, hits=[("unsourced_number", hit)], claims=[], captions=[])
    assert reason == "" and got[0].kind == "number"
    assert got[0].flagged == "The rate was 0.34 at R0 = 1.5"
    assert got[0].located == "The rate was 0.34 at \\(R_0 = 1.5\\)." and got[0].how == "sentence"
    assert got[0].why.startswith("the paper prints 0.34")


def test_a_check_that_lists_only_some_of_the_numbers_is_not_answered_with_edits() -> None:
    hit = "unsourced_number: the paper prints 0.34. A further 5 number(s) in this paper trace to nothing either — “x”"
    got, reason = paper_patch.plan_passages(PAPER, END, hits=[("unsourced_number", hit)], claims=[], captions=[])
    assert got == [] and "only some of the numbers" in reason
