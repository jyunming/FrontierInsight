"""The novelty note: after the literature step, one call says whether the retrieved sources already answer the
study's question. FI checks every quoted sentence against the source's text; the note is only disclosed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import novelty
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine

TOPIC = "How does the soil moisture of a terraced garden change with mulch depth?"
S1 = ("Mulch layers of five centimetres cut evaporation from garden soil by about a third in dry weeks. "
      "Deeper layers gave smaller further gains.")
S2 = "Terrace walls warm the soil on their sunny side, which dries it faster than the shaded side does."


def _engine(tmp_path: Path, **kn) -> Engine:
    eng = Engine(Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, web_search=False, **kn),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses=PausesConfig(papers=False),
    ))
    return eng


def _lit() -> list[dict]:
    return [
        {"content": f"Mulch and evaporation\n\n{S1}", "metadata": {"title": "Mulch and evaporation", "year": 2011,
                                                                     "doi": "10.1/a", "source": "crossref"}},
        {"content": f"Terrace walls\n\n{S2}", "metadata": {"title": "Terrace walls", "year": 2015,
                                                            "doi": "10.1/b", "source": "crossref"}},
    ]


STATE = {"topic": TOPIC, "chosen_idea": {"title": "Mulch depth and terrace moisture", "rationale": "dose response"}}


def _reply(**over) -> str:
    body = {
        "already_answered": "partly",
        "evidence": [{"source": "1", "quote": "Mulch layers of five centimetres cut evaporation from garden soil "
                                              "by about a third in dry weeks."}],
        "what_this_adds": "moisture for mulch depths of 0 to 15 cm on a terrace",
        "suggested_framing": "A dose-response extension of a known mulch effect.",
    }
    body.update(over)
    return json.dumps(body)


def _chat(monkeypatch, reply, seen=None):
    async def chat(self, messages, **kw):
        if seen is not None:
            seen.append((messages[0]["content"], kw.get("node")))
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr("core.engine.LLMClient.chat", chat)


async def _run(eng, monkeypatch, reply, seen=None):
    _chat(monkeypatch, reply, seen)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    await eng._connect_llm()
    return await eng._literature_novelty(dict(STATE), _lit(), 1)


@pytest.mark.asyncio
async def test_record_written_and_checked(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    seen: list = []
    rec = await _run(eng, monkeypatch, _reply(), seen)
    assert rec["already_answered"] == "partly" and rec["backed"] is True
    assert [e["source"] for e in rec["evidence"]] == ["1"]
    on_disk = json.loads((eng.quest_root / ".fi" / "literature_novelty.json").read_text(encoding="utf-8"))
    assert on_disk["evidence"][0]["source"] == "1" and on_disk["counts"]["kept"] == 1
    # one call, as its own step, so the cost row names it
    assert len(seen) == 1 and seen[0][1] == "literature_novelty"
    assert "Mulch depth and terrace moisture" in seen[0][0] and "[2] (2015) Terrace walls" in seen[0][0]
    cost = (eng.quest_root / ".fi" / "cost.jsonl").read_text(encoding="utf-8")
    assert '"literature_novelty"' in cost


@pytest.mark.asyncio
async def test_invented_source_id_is_dropped(tmp_path, monkeypatch):
    ev = [{"source": "7", "quote": "Mulch layers of five centimetres cut evaporation from garden soil."},
          {"source": "2", "quote": "Terrace walls warm the soil on their sunny side"}]
    rec = await _run(_engine(tmp_path), monkeypatch, _reply(evidence=ev))
    assert [e["source"] for e in rec["evidence"]] == ["2"]
    assert rec["counts"]["dropped_unknown_source"] == 1


@pytest.mark.asyncio
async def test_quote_not_in_source_is_dropped(tmp_path, monkeypatch):
    ev = [{"source": "1", "quote": "Mulch layers of ten centimetres doubled the soil moisture of every terrace."},
          {"source": "2", "quote": "Mulch layers of five centimetres cut evaporation from garden soil by a third."}]
    rec = await _run(_engine(tmp_path), monkeypatch, _reply(evidence=ev))
    assert rec["evidence"] == [] and rec["counts"]["dropped_quote_not_found"] == 2
    assert rec["backed"] is False  # a "partly" no passage backs is not passed on
    assert novelty.design_line(rec) == "" and novelty.write_block(rec) == ""


def test_quote_match_ignores_whitespace_entities_and_quotes():
    src = [{"id": "1", "title": "t", "text": "The soil\n  doesn&#x27;t dry out when the \u201cmulch\u201d is deep enough."}]
    rec, _ = novelty.check({"already_answered": "yes", "evidence": [
        {"source": "[1]", "quote": "The soil doesn't dry out when the \"mulch\" is deep enough."}]}, src)
    assert rec["counts"]["kept"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [RuntimeError("network down"), "not json at all", json.dumps({"x": 1}),
                                   json.dumps({"already_answered": "maybe"})])
async def test_failed_call_goes_on_without_a_record(tmp_path, monkeypatch, reply):
    eng = _engine(tmp_path)
    assert await _run(eng, monkeypatch, reply) is None
    assert not (eng.quest_root / ".fi" / "literature_novelty.json").exists()


@pytest.mark.asyncio
async def test_switch_off_makes_no_call(tmp_path, monkeypatch):
    seen: list = []
    eng = _engine(tmp_path, novelty_check=False)
    assert await _run(eng, monkeypatch, _reply(), seen) is None and seen == []


@pytest.mark.asyncio
async def test_design_and_write_prompts_get_the_line(tmp_path, monkeypatch):
    eng = _engine(tmp_path)
    rec = await _run(eng, monkeypatch, _reply())
    state = {**STATE, "iteration": 0, "literature": _lit(), "literature_novelty": rec}
    design = eng._design_prompt(state)
    assert "answer part of its question. This study adds: moisture for mulch depths" in design
    assert "What the prior work already answers" in design
    assert "$novelty_block" not in design
    block = novelty.write_block(rec)
    assert "A dose-response extension of a known mulch effect." in block and "[1]" in block
    assert "$novelty_block" in eng._prompts["write"].template
    # the instruction comes before the inputs in both prompts
    for name, head in (("design", "What the prior work already answers"), ("write", "How the introduction should frame")):
        t = eng._prompts[name].template
        assert t.index(head) < t.index("# Inputs")
    # no record: the prompts carry no heading for it
    assert "What the prior work already answers\n" not in eng._design_prompt({**state, "literature_novelty": None})
