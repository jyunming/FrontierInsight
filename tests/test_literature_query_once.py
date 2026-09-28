"""The literature step derives its search queries once per pass and input (the 2026-09-28 re-audit, F-09).

The idea step derives a seed query from the topic alone; the literature step derives its own from the topic AND the
chosen direction (and, on a later pass, the hypothesis): different inputs, both kept. What was redundant is the
literature step deriving again with the same inputs -- a resume after a pause runs the step again, spent a second
call, and could drift to other queries finding other papers. The query set is kept in ``.fi/literature_queries.json``
(the step's own state is not saved until it finishes) and reused when the derivation prompt and the pass match.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.passages as pmod
from tests.test_literature_facets import FACETS, _engine, _state


@pytest.fixture(autouse=True)
def _unscored(monkeypatch):
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)


def _counting(eng):
    calls: list[str] = []

    async def chat(prompt, node=""):  # noqa: ANN001
        calls.append(node)
        return json.dumps({"queries": FACETS})

    eng._chat = chat

    async def no_hits(query, **kw):  # noqa: ANN001
        return []

    eng.knowledge.asearch = no_hits  # type: ignore[method-assign]

    async def route(*a, **k):  # noqa: ANN001
        return ["crossref"]

    eng.knowledge.choose_sources = route  # type: ignore[method-assign]
    return calls


def _sets(eng) -> list[dict]:
    return json.loads((eng.fi_dir / "literature_queries.json").read_text(encoding="utf-8"))["entries"]


@pytest.mark.asyncio
async def test_the_same_pass_with_the_same_inputs_derives_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    first = await eng._node_literature(_state())
    again = await eng._node_literature(_state())  # a resume runs the step again, its state not yet saved
    assert calls.count("literature_query") == 1, calls
    assert first["literature_queries"] == again["literature_queries"] == FACETS
    assert first["literature_query_set"]["reused"] is False and first["literature_query_set"]["reason"] == "the first search"
    assert again["literature_query_set"]["reused"] is True
    (entry,) = _sets(eng)
    assert entry["queries"] == FACETS and entry["prompt_sha256"] and entry["iteration"] == 1


@pytest.mark.asyncio
async def test_a_new_pass_or_changed_inputs_derive_again_and_say_why(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    # A broaden-the-literature re-entry: pass 2, with the design's hypothesis folded in.
    second = await eng._node_literature(_state(literature_iter=1, design={"hypothesis": "h1"}))
    assert second["literature_query_set"]["reason"] == "a new search pass (2)"
    # The same pass again, but the hypothesis changed.
    third = await eng._node_literature(_state(literature_iter=1, design={"hypothesis": "h2"}))
    assert third["literature_query_set"]["reason"] == "the topic, the chosen direction, the hypothesis or the model changed"
    assert calls.count("literature_query") == 3
    # ... and that pass with h2 once more is reused.
    fourth = await eng._node_literature(_state(literature_iter=1, design={"hypothesis": "h2"}))
    assert fourth["literature_query_set"]["reused"] is True and calls.count("literature_query") == 3


@pytest.mark.asyncio
async def test_a_failed_derivation_is_not_kept(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)

    async def broken(prompt, node=""):  # noqa: ANN001
        calls.append(node)
        return "not json"

    eng._chat = broken
    await eng._node_literature(_state())
    assert not (eng.fi_dir / "literature_queries.json").exists()
    calls2 = _counting(eng)  # the model answers now: the next run tries again
    patch = await eng._node_literature(_state())
    assert calls2.count("literature_query") == 1 and patch["literature_queries"] == FACETS


@pytest.mark.asyncio
async def test_a_state_without_the_record_works(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _counting(eng)
    patch = await eng._node_literature(_state(literature_queries=["old"], literature_query="old"))
    assert patch["literature_queries"] == FACETS


@pytest.mark.asyncio
async def test_another_kind_of_quest_or_another_model_derives_again(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    books = await eng._node_literature(_state(no_simulation_resolved=True))  # the other vocabulary
    assert books["literature_query_set"]["reused"] is False and calls.count("literature_query") == 2
    eng.config.provider.node_models = {**(eng.config.provider.node_models or {}), "literature_query": "a-better-model"}
    other = await eng._node_literature(_state())
    assert other["literature_query_set"]["reused"] is False and calls.count("literature_query") == 3
    assert "model changed" in other["literature_query_set"]["reason"]


@pytest.mark.asyncio
async def test_the_ideate_record_is_never_used_by_the_literature_step(tmp_path: Path) -> None:
    from core.engine import _record_query_set

    eng = _engine(tmp_path)
    calls = _counting(eng)
    _record_query_set(eng.fi_dir, {"stage": "ideate", "queries": ["seed only"], "reason": "x"})
    patch = await eng._node_literature(_state())
    assert calls.count("literature_query") == 1 and patch["literature_queries"] == FACETS


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json", '{"entries": "oops"}', '[1, 2]'])
async def test_an_unreadable_record_is_set_aside_not_overwritten(tmp_path: Path, content: str, caplog) -> None:
    import logging

    eng = _engine(tmp_path)
    calls = _counting(eng)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "literature_queries.json").write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        patch = await eng._node_literature(_state())
    assert (eng.fi_dir / "literature_queries.json.bad").read_text(encoding="utf-8") == content
    assert calls.count("literature_query") == 1 and patch["literature_queries"] == FACETS
    # The quest's own log (run.log) says so; it does not propagate to the root logger.
    run_log = (eng.fi_dir / "run.log").read_text(encoding="utf-8") if (eng.fi_dir / "run.log").exists() else ""
    assert "could not be read" in run_log or any("could not be read" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_saved_set_of_the_wrong_type_is_not_reused(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    calls = _counting(eng)
    await eng._node_literature(_state())
    path = eng.fi_dir / "literature_queries.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["entries"][-1]["queries"] = "abc"  # a string: reusing it would search 'a', 'b', 'c'
    path.write_text(json.dumps(data), encoding="utf-8")
    patch = await eng._node_literature(_state())
    assert calls.count("literature_query") == 2 and patch["literature_queries"] == FACETS


@pytest.mark.asyncio
async def test_a_real_resume_with_a_new_engine_reuses_the_queries(tmp_path: Path) -> None:
    from core.engine import Engine

    first = _engine(tmp_path)
    calls = _counting(first)
    await first._node_literature(_state())
    resumed = Engine(first.config, resume_quest_id=first.quest_id)  # a new process on the same quest folder
    calls_after = _counting(resumed)
    patch = await resumed._node_literature(_state())
    assert calls.count("literature_query") == 1 and calls_after.count("literature_query") == 0
    assert patch["literature_query_set"]["reused"] is True and patch["literature_queries"] == FACETS
