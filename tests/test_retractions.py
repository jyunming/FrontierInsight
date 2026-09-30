"""A retracted source is marked and never counts as support.

After the literature search de-duplicates what it found, every DOI is looked up in Crossref, which carries the
Retraction Watch database as ``updated-by`` notices on the retracted work. A retracted source is marked
``[retracted]`` wherever the prior-work block names it, the claim check does not accept it as the basis of a claim,
the lookup is kept in ``.fi/literature_queries.json`` (the sealed record of the search) and ``run.log`` gets one plain
line. A lookup that did not get an answer is "could not be checked", never "not retracted", and it never stops a quest.
No test here reaches the network: Crossref's answers are an ``httpx.MockTransport``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from core import retractions, todo
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import (
    Engine, _claim_source_block, _format_lit_from_state, _format_lit_header, _query_set_digest,
    _query_set_standing,
)
from core.knowledge import RetrievedDoc

RETRACTED_DOI = "10.1016/S0140-6736(97)11096-0"
CLEAN_DOI = "10.1038/s41598-023-41032-5"
UNKNOWN_DOI = "10.5555/not-in-crossref"

# Crossref's answer for the retracted paper, trimmed to the fields the lookup asks for (``select=DOI,updated-by``);
# the shape is the real one (api.crossref.org/works?filter=doi:... on 2026-09-30).
RETRACTED_ITEM = {
    "DOI": RETRACTED_DOI.lower(),
    "updated-by": [
        {"DOI": "10.1016/s0140-6736(04)15715-2", "type": "correction", "label": "Correction",
         "source": "retraction-watch", "updated": {"date-parts": [[2004, 3, 6]],
                                                   "date-time": "2004-03-06T00:00:00Z"}},
        {"DOI": "10.1016/s0140-6736(10)60175-4", "type": "retraction", "label": "Retraction",
         "source": "retraction-watch", "updated": {"date-parts": [[2010, 2, 6]],
                                                   "date-time": "2010-02-06T00:00:00Z"}},
    ],
}
CLEAN_ITEM = {"DOI": CLEAN_DOI.lower()}


def _crossref(items: list[dict[str, Any]], seen: list[httpx.Request] | None = None, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        asked = {f.split(":", 1)[1] for f in request.url.params.get("filter", "").split(",") if f}
        found = [i for i in items if i["DOI"] in asked]
        if status != 200:
            return httpx.Response(status, text="busy")
        return httpx.Response(200, json={"status": "ok", "message": {"items": found}})
    return httpx.MockTransport(handler)


def _down():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)
    return httpx.MockTransport(handler)


def _entry(title: str, doi: str = "", **md: Any) -> dict[str, Any]:
    meta = {"title": title, "source": "crossref", "venue": "V", "year": 1998, **md}
    if doi:
        meta["doi"] = doi
    return {"content": f"{title}. " + "An abstract long enough to be a real one. " * 4, "metadata": meta}


# --- the lookup -----------------------------------------------------------------------------

def test_a_retracted_doi_a_clean_doi_and_one_crossref_does_not_hold() -> None:
    seen: list[httpx.Request] = []
    out = asyncio.run(retractions.check_dois(
        [RETRACTED_DOI, "https://doi.org/" + CLEAN_DOI, UNKNOWN_DOI],
        transport=_crossref([RETRACTED_ITEM, CLEAN_ITEM], seen)))
    assert len(seen) == 1, "one request for the whole batch"
    assert seen[0].url.params["select"] == "DOI,updated-by"
    assert "mailto" not in seen[0].url.params
    hit = out[RETRACTED_DOI.lower()]
    assert hit["status"] == retractions.RETRACTED
    assert {n["type"] for n in hit["notices"]} == {"correction", "retraction"}
    assert "2010-02-06" in hit["why"] and "10.1016/s0140-6736(10)60175-4" in hit["why"]
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_RETRACTED
    # Not in Crossref's answer: nothing is known, so it is not called clean.
    assert out[UNKNOWN_DOI]["status"] == retractions.NOT_CHECKED


@pytest.mark.parametrize("transport", [_down(), _crossref([RETRACTED_ITEM], status=500),
                                       _crossref([RETRACTED_ITEM], status=429)])
def test_no_answer_is_not_checked_never_not_retracted(transport, monkeypatch) -> None:
    async def no_wait(seconds):  # noqa: ANN001, ARG001
        return None

    monkeypatch.setattr(retractions, "_sleep", no_wait)
    out = asyncio.run(retractions.check_dois([RETRACTED_DOI, CLEAN_DOI], transport=transport))
    assert {r["status"] for r in out.values()} == {retractions.NOT_CHECKED}
    assert all(r["why"] for r in out.values())


def test_a_timeout_is_not_checked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=httpx.MockTransport(handler)))
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_CHECKED


def test_an_answer_that_is_not_crossrefs_shape_is_not_checked() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "ok", "message": "odd"}))
    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=transport))
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_CHECKED


def test_an_arxiv_doi_is_not_sent_to_crossref() -> None:
    seen: list[httpx.Request] = []
    out = asyncio.run(retractions.check_dois(["10.48550/arXiv.2409.13740"], transport=_crossref([], seen)))
    assert seen == []
    assert out["10.48550/arxiv.2409.13740"]["status"] == retractions.NOT_CHECKED


def test_many_dois_go_in_batches_one_after_another() -> None:
    seen: list[httpx.Request] = []
    dois = [f"10.1234/{i}" for i in range(retractions.BATCH_SIZE + 3)]
    out = asyncio.run(retractions.check_dois(dois, transport=_crossref([{"DOI": d} for d in dois], seen)))
    assert len(seen) == 2
    assert {r["status"] for r in out.values()} == {retractions.NOT_RETRACTED}


# --- the literature entries -------------------------------------------------------------------

def test_entries_get_their_status_and_one_row_each() -> None:
    lit = [_entry("The Lancet paper", RETRACTED_DOI), _entry("A clean paper", CLEAN_DOI),
           _entry("A web page", url="https://example.org/x", source="web_search"),
           _entry("An unanswered one", UNKNOWN_DOI)]
    out, rows = asyncio.run(retractions.check_literature(lit, transport=_crossref([RETRACTED_ITEM, CLEAN_ITEM])))
    status = {e["metadata"]["title"]: e["metadata"]["retraction"] for e in out}
    assert status == {"The Lancet paper": "retracted", "A clean paper": "not_retracted",
                      "A web page": "no_doi", "An unanswered one": "not_checked"}
    assert "retraction_note" in out[0]["metadata"] and "2010-02-06" in out[0]["metadata"]["retraction_note"]
    assert "retraction" not in lit[0]["metadata"], "the caller's entries are not changed in place"
    assert [r["status"] for r in rows] == ["retracted", "not_retracted", "no_doi", "not_checked"]
    assert rows[0]["source"] == "doi:" + RETRACTED_DOI.lower() and rows[0]["title"] == "The Lancet paper"


def test_an_entry_checked_before_is_not_asked_again_but_an_unanswered_one_is() -> None:
    first = _entry("A clean paper", CLEAN_DOI)
    first["metadata"]["retraction"] = "not_retracted"
    again = _entry("Asked before, no answer", RETRACTED_DOI)
    again["metadata"]["retraction"] = "not_checked"
    seen: list[httpx.Request] = []
    out, rows = asyncio.run(retractions.check_literature([first, again], transport=_crossref([RETRACTED_ITEM], seen)))
    assert out[0] is first
    assert out[1]["metadata"]["retraction"] == "retracted"
    assert [r["title"] for r in rows] == ["Asked before, no answer"]
    assert CLEAN_DOI.lower() not in seen[0].url.params["filter"]


def test_the_run_log_line_is_one_plain_sentence() -> None:
    rows = [
        {"title": "The Lancet paper", "status": "retracted"},
        {"title": "B", "status": "not_retracted"},
        {"title": "C", "status": "not_checked", "why": "Crossref could not be reached"},
        {"title": "D", "status": "not_checked", "why": "Crossref could not be reached"},
        {"title": "E", "status": "no_doi"},
    ]
    line = retractions.summary_line(rows)
    assert "\n" not in line
    assert line == ("Checked 4 sources for retractions: 1 retracted (The Lancet paper), "
                    "2 could not be checked (Crossref could not be reached), 1 not retracted; "
                    "1 without a DOI was not looked up")
    assert retractions.summary_line([{"title": "B", "status": "not_retracted"}]) == (
        "Checked 1 source for retractions: none retracted")
    # Nothing answered: the line does not say "checked".
    assert retractions.summary_line(rows[2:4]) == (
        "Could not check 2 sources for retractions (Crossref could not be reached)")
    # Two different reasons: the commoner one, said to be only the commoner one.
    mixed = rows[2:4] + [{"title": "F", "status": "not_checked", "why": "Crossref has no record of this DOI"}]
    assert retractions.summary_line(mixed) == (
        "Could not check 3 sources for retractions (mostly: Crossref could not be reached)")
    assert retractions.summary_line([]) == ""


def test_nothing_raises_whatever_the_entries_hold() -> None:
    odd = [{"content": "x", "metadata": None}, {"content": "y"}, "not a dict",
           RetrievedDoc(content="z", metadata={"title": "Doc", "doi": CLEAN_DOI})]
    out, rows = asyncio.run(retractions.check_literature(odd, transport=_down()))
    assert len(out) == 4
    assert [r["status"] for r in rows if r["title"] == "Doc"] == ["not_checked"]


# --- the prompt blocks ------------------------------------------------------------------------

def test_the_prior_work_block_marks_a_retracted_source() -> None:
    lit = [_entry("The Lancet paper", RETRACTED_DOI, retraction="retracted"),
           _entry("A clean paper", CLEAN_DOI, retraction="not_retracted"),
           _entry("Unanswered", UNKNOWN_DOI, retraction="not_checked")]
    block = _format_lit_from_state({"literature": lit})  # type: ignore[typeddict-item]
    lines = [ln for ln in block.splitlines() if ln.startswith("[")]
    assert lines[0].endswith("The Lancet paper [retracted]")
    assert "[retracted]" not in lines[1] and "[retracted]" not in lines[2]
    # The writer's block (with the thin marks) carries it too, after the title.
    assert "[1] (1998) The Lancet paper [retracted]" in _format_lit_from_state(
        {"literature": lit}, mark_thin=True)  # type: ignore[typeddict-item]
    assert _format_lit_header({"title": "T", "retraction": "retracted"}, 1, "title") == "[1] T [title only] [retracted]"


def test_the_claim_checks_source_block_marks_a_retracted_source() -> None:
    block = _claim_source_block("1", {"title": "The Lancet paper", "doi": RETRACTED_DOI, "retraction": "retracted"},
                                "text", [])
    assert block.startswith("[1] The Lancet paper [retracted]")


def test_the_prompts_say_a_retracted_source_supports_nothing() -> None:
    root = Path(__file__).resolve().parent.parent / "agents"
    for name in ("claim_check.md", "write.md", "write_patch.md"):
        assert "[retracted]" in (root / name).read_text(encoding="utf-8"), name


# --- the claim check --------------------------------------------------------------------------

PAPER = (
    "# T\n\n## Introduction\n\n"
    "Early exposure was linked to the later onset of the disorder in twelve children [1]. "
    "Large cohort studies found no such association across hundreds of thousands of children [2].\n\n"
    "## References\n\n1. Wakefield (1998).\n2. Madsen (2002).\n"
)


def test_apply_to_claims_marks_a_citation_of_a_retracted_source_unsupported() -> None:
    sources = {"1": ({"title": "The Lancet paper", "retraction": "retracted",
                      "retraction_note": "retraction on 2010-02-06, notice 10.1/x"}, "text"),
               "2": ({"title": "Cohort", "retraction": "not_retracted"}, "text"),
               "W1": ({"title": "Page", "retraction": "retracted"}, "text")}
    claims = [{"claim": "a", "basis": "citation", "citation_index": 1, "quote": "q", "evidence": "e"},
              {"claim": "b", "basis": "citation", "citation_index": 2, "quote": "q", "evidence": "e"},
              {"claim": "c", "basis": "citation", "citation_index": "W1", "quote": "q", "evidence": "e"},
              {"claim": "d", "basis": "experiment", "citation_index": None, "quote": "", "evidence": "e"}]
    out, changed = retractions.apply_to_claims(claims, sources)
    assert changed == 2
    assert [c["basis"] for c in out] == ["unsupported", "citation", "unsupported", "experiment"]
    assert out[0]["quote"] == "" and "[1] has been retracted" in out[0]["evidence"]
    assert "2010-02-06" in out[0]["evidence"]
    assert claims[0]["basis"] == "citation", "the caller's claims are not changed in place"


def _claim_engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="vaccines", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, claim_grounding=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


def test_the_claim_check_does_not_accept_a_retracted_source(tmp_path: Path) -> None:
    eng = _claim_engine(tmp_path)
    paper = tmp_path / "paper" / "paper.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(PAPER, encoding="utf-8")
    text1 = "Early exposure was linked to the later onset of the disorder in twelve children studied here."
    text2 = "Large cohort studies found no such association across hundreds of thousands of children."
    lit = [{"content": text1, "metadata": {"title": "Ileal-lymphoid-nodular hyperplasia", "doi": RETRACTED_DOI,
                                           "source": "crossref", "venue": "Lancet", "year": 1998,
                                           "retraction": "retracted",
                                           "retraction_note": "retraction on 2010-02-06, notice 10.1016/x"}},
           {"content": text2, "metadata": {"title": "A population-based study", "doi": CLEAN_DOI,
                                           "source": "crossref", "venue": "NEJM", "year": 2002,
                                           "retraction": "not_retracted"}}]
    prompts: list[str] = []

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001, ARG001 -- no real model is called
        prompts.append(prompt)
        return json.dumps({"claims": [
            {"claim": "Early exposure was linked to the later onset of the disorder in twelve children",
             "basis": "citation", "citation_index": 1, "quote": text1[:60], "evidence": "the case series"},
            {"claim": "Large cohort studies found no such association",
             "basis": "citation", "citation_index": 2, "quote": text2[:60], "evidence": "cohort"},
        ], "summary": "ok"})

    eng._chat = fake_chat  # type: ignore[method-assign]
    logged: list[str] = []
    eng._log = type("L", (), {  # type: ignore[assignment]
        "info": lambda self, m, *a: logged.append(m % a if a else m),
        "warning": lambda self, m, *a: logged.append(m % a if a else m),
        "debug": lambda self, m, *a: None,
        "error": lambda self, m, *a: logged.append(m % a if a else m),
        "exception": lambda self, m, *a: logged.append(m % a if a else m),
    })()
    out = asyncio.run(eng._node_claim_check({"topic": "t", "paper_md": str(paper), "literature": lit}))  # type: ignore[arg-type]
    grounding = out["claim_grounding"]
    assert grounding, (out, logged)
    assert grounding["unsupported"] == ["Early exposure was linked to the later onset of the disorder in twelve children"]
    assert grounding["grounded"] == 1
    flipped = [c for c in grounding["claims"] if c["citation_index"] == 1][0]
    assert "has been retracted" in flipped["evidence"]
    assert any("1 claim(s) rest on a retracted source" in m for m in logged), logged
    assert any("[1] Ileal-lymphoid-nodular hyperplasia [retracted]" in p for p in prompts)


# --- the literature node and its record -------------------------------------------------------

def _lit_engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="vaccines and autism", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False, web_search=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
        pauses=PausesConfig(papers=False),
    ))
    eng.config.knowledge.enabled = True
    eng.config.knowledge.source_routing = "manual"
    eng.config.knowledge.literature_screen = False
    eng.config.knowledge.foundational_works = False
    return eng


@pytest.mark.asyncio
async def test_the_literature_node_marks_records_and_logs_one_line(tmp_path: Path, monkeypatch) -> None:
    import core.passages as pmod

    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)
    eng = _lit_engine(tmp_path)

    async def no_model(prompt, node=""):  # noqa: ANN001, ARG001
        raise RuntimeError("no model")

    retrieved = [
        RetrievedDoc(content="The Lancet paper. " + "Abstract text. " * 20,
                     metadata={"title": "The Lancet paper", "doi": RETRACTED_DOI, "source": "crossref"}),
        RetrievedDoc(content="A clean paper. " + "Abstract text. " * 20,
                     metadata={"title": "A clean paper", "doi": CLEAN_DOI, "source": "crossref"}),
        RetrievedDoc(content="No answer. " + "Abstract text. " * 20,
                     metadata={"title": "No answer", "doi": UNKNOWN_DOI, "source": "crossref"}),
    ]

    async def fake_search(query, **kw):  # noqa: ANN001, ARG001
        return retrieved

    async def fetch(docs):  # noqa: ANN001
        return docs

    real = retractions.check_dois

    async def mocked(dois, **kw):  # noqa: ANN001, ARG001
        return await real(dois, transport=_crossref([RETRACTED_ITEM, CLEAN_ITEM]))

    monkeypatch.setattr(retractions, "check_dois", mocked)
    logged: list[str] = []
    real_info = eng._log.info

    def info(msg, *a, **kw):  # noqa: ANN001
        logged.append(msg % a if a else msg)
        return real_info(msg, *a, **kw)

    monkeypatch.setattr(eng._log, "info", info)
    eng._chat = no_model
    eng.knowledge.asearch = fake_search  # type: ignore[method-assign]
    eng.knowledge.fetch_full_text = fetch  # type: ignore[method-assign]
    patch = await eng._node_literature({"topic": "vaccines and autism", "chosen_idea": {"title": "T"},
                                        "no_simulation_resolved": True})
    status = {e["metadata"]["title"]: e["metadata"].get("retraction") for e in patch["literature"]}
    assert status == {"The Lancet paper": "retracted", "A clean paper": "not_retracted", "No answer": "not_checked"}
    lines = [m for m in logged if "for retractions" in m]
    assert len(lines) == 1, lines
    assert "1 retracted (The Lancet paper)" in lines[0] and "1 could not be checked" in lines[0]
    record = json.loads((eng.fi_dir / "literature_queries.json").read_text(encoding="utf-8"))
    (entry,) = [e for e in record["entries"] if e.get("stage") == "retractions"]
    assert entry["outcome"] == "checked"
    assert {r["title"]: r["status"] for r in entry["sources"]} == status
    assert entry["digest"] == _query_set_digest(entry) and _query_set_standing(entry) == "verified"
    # The to-do card every interface shows names it.
    items = [i for i in todo.waiting(eng.quest_root) if i.kind == "retracted"]
    assert len(items) == 1 and "The Lancet paper" in items[0].why
    # The list of papers to download never asks for a retracted one (the clean one shows the list is written).
    wanted = (eng.quest_root / "needs" / "WANTED_PAPERS.md").read_text(encoding="utf-8")
    assert "A clean paper" in wanted and "The Lancet paper" not in wanted


def test_the_todo_card_is_quiet_without_a_retracted_source(tmp_path: Path) -> None:
    fi = tmp_path / ".fi"
    fi.mkdir()
    (fi / "literature_queries.json").write_text(json.dumps({"schema": 2, "entries": [
        {"stage": "retractions", "sources": [{"source": "doi:10.1/a", "title": "A", "status": "not_checked"}]},
    ]}), encoding="utf-8")
    assert [i for i in todo.waiting(tmp_path) if i.kind == "retracted"] == []


# --- after the first review -------------------------------------------------------------------

def test_a_batch_asks_for_enough_rows_for_every_doi() -> None:
    seen: list[httpx.Request] = []
    dois = [f"10.1234/{i}" for i in range(retractions.BATCH_SIZE)]
    asyncio.run(retractions.check_dois(dois, transport=_crossref([], seen)))
    assert int(seen[0].url.params["rows"]) >= len(dois)  # Crossref's default is 20


def test_a_429_is_asked_once_more_and_then_answers(monkeypatch) -> None:
    waits: list[float] = []

    async def no_wait(seconds):  # noqa: ANN001
        waits.append(seconds)

    monkeypatch.setattr(retractions, "_sleep", no_wait)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return httpx.Response(200, json={"status": "ok", "message": {"items": [RETRACTED_ITEM]}})

    out = asyncio.run(retractions.check_dois([RETRACTED_DOI], transport=httpx.MockTransport(handler)))
    assert len(calls) == 2 and waits == [1.0]
    assert out[RETRACTED_DOI.lower()]["status"] == retractions.RETRACTED


def test_a_429_that_asks_for_a_long_wait_is_not_waited_for(monkeypatch) -> None:
    waits: list[float] = []

    async def no_wait(seconds):  # noqa: ANN001
        waits.append(seconds)

    monkeypatch.setattr(retractions, "_sleep", no_wait)
    transport = httpx.MockTransport(lambda r: httpx.Response(429, headers={"retry-after": "3600"}))
    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=transport))
    assert waits == []
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_CHECKED


def test_when_crossref_cannot_be_reached_the_later_batches_are_not_sent() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectTimeout("dropped", request=request)

    dois = [f"10.1234/{i}" for i in range(retractions.BATCH_SIZE * 2 + 1)]
    out = asyncio.run(retractions.check_dois(dois, transport=httpx.MockTransport(handler)))
    assert len(calls) == 1
    assert len(out) == len(dois) and {r["status"] for r in out.values()} == {retractions.NOT_CHECKED}


def test_an_error_status_on_one_batch_still_tries_the_next(monkeypatch) -> None:
    async def no_wait(seconds):  # noqa: ANN001, ARG001
        return None

    monkeypatch.setattr(retractions, "_sleep", no_wait)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500)
        asked = [f.split(":", 1)[1] for f in request.url.params["filter"].split(",")]
        return httpx.Response(200, json={"status": "ok", "message": {"items": [{"DOI": d} for d in asked]}})

    dois = [f"10.1234/{i}" for i in range(retractions.BATCH_SIZE + 2)]
    out = asyncio.run(retractions.check_dois(dois, transport=httpx.MockTransport(handler)))
    assert len(calls) == 2
    assert [out[d]["status"] for d in dois[:2]] == [retractions.NOT_CHECKED] * 2
    assert [out[d]["status"] for d in dois[-2:]] == [retractions.NOT_RETRACTED] * 2


@pytest.mark.parametrize("updated_by", [{"type": "retraction"}, "retraction", [["retraction"]]])
def test_an_updated_by_in_an_unknown_shape_is_not_checked(updated_by) -> None:
    item = {"DOI": CLEAN_DOI.lower(), "updated-by": updated_by}
    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=_crossref([item])))
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_CHECKED


def test_a_retraction_later_reinstated_is_not_retracted() -> None:
    item = {"DOI": CLEAN_DOI.lower(), "updated-by": [
        {"type": "reinstatement", "DOI": "10.1/r", "updated": {"date-time": "2021-05-01T00:00:00Z"}},
        {"type": "retraction", "DOI": "10.1/x", "updated": {"date-time": "2019-01-01T00:00:00Z"}},
    ]}
    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=_crossref([item])))
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_RETRACTED
    assert "reinstated on 2021-05-01" in out[CLEAN_DOI.lower()]["why"]


@pytest.mark.parametrize("raw", ["doi.org/10.1038/S41598-023-41032-5", "https://www.doi.org/10.1038/s41598-023-41032-5",
                                 "10.1038%2Fs41598-023-41032-5", "doi: 10.1038/s41598-023-41032-5"])
def test_the_usual_ways_of_writing_a_doi_are_read(raw) -> None:
    assert retractions.normalize_doi(raw) == CLEAN_DOI.lower()


def test_a_doi_that_cannot_be_read_is_not_checked_never_no_doi() -> None:
    lit = [_entry("Odd DOI", "not a doi at all")]
    out, rows = asyncio.run(retractions.check_literature(lit, transport=_crossref([])))
    assert out[0]["metadata"]["retraction"] == retractions.NOT_CHECKED
    assert rows[0]["status"] == retractions.NOT_CHECKED and "could not be read" in rows[0]["why"]


def test_a_sentence_that_cites_a_retracted_source_is_flagged_even_when_the_check_grounded_it_elsewhere() -> None:
    from core.engine import _citing_sentences, _same_statement

    paper = ("# T\n\n## Introduction\n\nThe incidence rose steadily among children over the whole decade studied [1, 2]."
             "\n\n## References\n\n1. A.\n2. B.\n")
    sources = {"1": ({"title": "A", "retraction": "retracted"}, "t"), "2": ({"title": "B"}, "t")}
    claims = [{"claim": "The incidence rose steadily among children over the whole decade studied",
               "basis": "citation", "citation_index": 2, "quote": "q", "evidence": "e"}]
    out, changed = retractions.apply_to_claims(claims, sources, _citing_sentences(paper), _same_statement)
    assert changed == 1
    added = out[-1]
    assert added["basis"] == "unsupported" and added["citation_index"] == 1
    assert "remove the citation" in added["evidence"]
    # Nothing is added twice: a sentence a claim already marks unsupported is left alone.
    again, n = retractions.apply_to_claims(out, sources, _citing_sentences(paper), _same_statement)
    assert n == 0 and len(again) == len(out)


def test_a_retracted_foundational_work_is_never_asked_for() -> None:
    from core.engine import _foundational_review_block, _foundational_write_block

    lit = [{"content": "A long abstract. " * 20, "metadata": {
        "title": "Withdrawn classic", "year": 1998, "authors": ["A. B"], "source": "openalex", "venue": "V",
        "doi": "10.1234/withdrawn", "foundational": "cited by 6 of the retrieved papers", "retraction": "retracted"}}]
    assert _foundational_write_block(lit) == ""
    assert _foundational_review_block(lit, "# T\n\nNothing cited.\n") == ""


# --- after the second review ------------------------------------------------------------------

def test_a_short_sentence_or_one_citing_two_retracted_sources_is_added_once() -> None:
    from core.engine import _citing_sentences, _same_statement

    paper = "# T\n\n## Introduction\n\nVaccines cause autism [1, 3].\n\n## References\n\n1. A.\n3. C.\n"
    sources = {"1": ({"title": "A", "retraction": "retracted"}, "t"), "3": ({"title": "C", "retraction": "retracted"}, "t")}
    claims = [{"claim": "Vaccines cause autism", "basis": "citation", "citation_index": 1, "quote": "q", "evidence": "e"}]
    out, changed = retractions.apply_to_claims(claims, sources, _citing_sentences(paper), _same_statement)
    assert changed == 1 and len(out) == 1 and out[0]["basis"] == "unsupported"
    out, changed = retractions.apply_to_claims([], sources, _citing_sentences(paper), _same_statement)
    assert changed == 1 and len(out) == 1


def test_an_undated_retraction_counts_as_the_latest() -> None:
    item = {"DOI": CLEAN_DOI.lower(), "updated-by": [
        {"type": "reinstatement", "DOI": "10.1/r", "updated": {"date-time": "2021-05-01T00:00:00Z"}},
        {"type": "retraction", "DOI": "10.1/x"},
    ]}
    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=_crossref([item])))
    assert out[CLEAN_DOI.lower()]["status"] == retractions.RETRACTED


def test_the_reason_names_the_kind_of_network_failure() -> None:
    out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=_down()))
    assert out[CLEAN_DOI.lower()]["why"] == "Crossref could not be reached: no connection"

    def proxy(request: httpx.Request) -> httpx.Response:
        raise httpx.ProxyError("407 Proxy Authentication Required", request=request)

    def cert(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed", request=request)

    for handler, words in ((proxy, "the proxy refused the request"), (cert, "a certificate problem")):
        out = asyncio.run(retractions.check_dois([CLEAN_DOI], transport=httpx.MockTransport(handler)))
        assert out[CLEAN_DOI.lower()]["why"] == f"Crossref could not be reached: {words}"


def test_arxiv_preprints_are_said_apart_and_never_hide_why_the_rest_failed() -> None:
    arxiv = [{"title": f"A{i}", "status": "not_checked", "why": retractions.ARXIV_WHY} for i in range(10)]
    down = [{"title": f"D{i}", "status": "not_checked", "why": "Crossref could not be reached: no connection"}
            for i in range(5)]
    assert retractions.summary_line(arxiv + down) == (
        "Could not check 5 sources for retractions (Crossref could not be reached: no connection); "
        "10 arXiv preprints not looked up (Crossref holds none)")
    assert retractions.summary_line(arxiv[:1]) == (
        "No source had a DOI that Crossref could check for retractions (1 source(s))")


def test_an_empty_answer_from_the_check_is_still_a_failure_when_a_rule_adds_claims(tmp_path: Path) -> None:
    eng = _claim_engine(tmp_path)
    paper = tmp_path / "paper" / "paper.md"
    paper.parent.mkdir(parents=True)
    body = "The incidence rose steadily among children over the whole decade studied here [1]. " * 40
    paper.write_text(f"# T\n\n## Introduction\n\n{body}\n\n## References\n\n1. A.\n", encoding="utf-8")
    lit = [{"content": "Some text of the source.", "metadata": {"title": "A", "doi": RETRACTED_DOI, "source": "crossref",
                                                               "retraction": "retracted"}}]

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001, ARG001
        return json.dumps({"claims": [], "summary": "nothing"})

    eng._chat = fake_chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_claim_check({"topic": "t", "paper_md": str(paper), "literature": lit}))  # type: ignore[arg-type]
    assert out["claim_grounding"]["status"] == "unknown"
    assert out["claim_grounding"]["total"] >= 1  # the rule's claim is still there


@pytest.mark.skipif(__import__("os").environ.get("FI_TEST_ALLOW_NETWORK") != "1",
                    reason="a live check against Crossref; set FI_TEST_ALLOW_NETWORK=1")
def test_live_crossref_still_reports_a_known_retraction() -> None:
    out = asyncio.run(retractions.check_dois([RETRACTED_DOI, CLEAN_DOI]))
    assert out[RETRACTED_DOI.lower()]["status"] == retractions.RETRACTED
    assert out[CLEAN_DOI.lower()]["status"] == retractions.NOT_RETRACTED
