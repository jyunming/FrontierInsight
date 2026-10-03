"""A pinned paper has no DOI: it is found by title in Crossref, then looked up for a retraction.

The match is strict (the same title once normalised, and the first author and the year wherever the paper states them);
no match or no answer is "could not be checked", never "not retracted". Crossref is an ``httpx.MockTransport``.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from core import retractions
from tests.test_retractions import RETRACTED_DOI, RETRACTED_ITEM, _entry

LOCAL_TITLE = "MMR vaccine and autism in children"


def _by_title(records: list[dict[str, Any]], seen: list[httpx.Request] | None = None):
    """Crossref for a title query (``query.bibliographic``) and for a DOI filter alike."""
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        q = request.url.params
        if q.get("query.bibliographic"):
            return httpx.Response(200, json={"message": {"items": records}})
        asked = {f.split(":", 1)[1] for f in q.get("filter", "").split(",") if f}
        return httpx.Response(200, json={"message": {"items": [r for r in records if r["DOI"] in asked]}})
    return httpx.MockTransport(handler)


def _local(title: str = LOCAL_TITLE, **md: Any) -> dict[str, Any]:
    return {"content": "body " * 20, "metadata": {"source": "local_paper", "title": title, **md}}


def _record(title: str, year: int = 1998, family: str = "Wakefield") -> dict[str, Any]:
    return {**RETRACTED_ITEM, "title": [title], "author": [{"family": family}], "issued": {"date-parts": [[year]]}}


@pytest.fixture(autouse=True)
def _real_title_lookup(monkeypatch):
    """tests/conftest.py leaves the lookups on for this file (a mock transport answers them); no real wait."""
    async def quick(_s):
        return None
    monkeypatch.setattr(retractions, "_sleep", quick)


def test_a_pinned_paper_without_a_doi_is_found_by_title_and_marked_retracted() -> None:
    seen: list[httpx.Request] = []
    lit = [_local(authors=["Wakefield, A. J."], year=1998)]
    out, rows = asyncio.run(retractions.check_literature(
        lit, transport=_by_title([_record("MMR vaccine and autism in children.")], seen)))
    meta = out[0]["metadata"]
    assert meta["retraction"] == "retracted" and meta["retraction_doi"] == RETRACTED_DOI.lower()
    assert "doi" not in meta, "the paper's own doi field is left alone"
    assert "2010-02-06" in meta["retraction_note"]
    assert rows[0]["status"] == "retracted" and rows[0]["doi"] == RETRACTED_DOI.lower()
    assert all(r.headers["user-agent"] == "FrontierInsight/1.0" for r in seen)


def test_a_near_miss_title_is_never_taken_for_the_paper() -> None:
    for other in ("MMR vaccine and autism in adults", "Measles vaccine and autism in children"):
        out, rows = asyncio.run(retractions.check_literature([_local()], transport=_by_title([_record(other)])))
        assert out[0]["metadata"]["retraction"] == "not_checked", other
        assert "retraction_doi" not in out[0]["metadata"]
        assert rows[0]["why"] == retractions.NO_TITLE_MATCH


def test_a_matching_title_with_another_first_author_or_year_is_rejected() -> None:
    for lit in (_local(authors="Smith, J."), _local(year=2015)):
        out, _rows = asyncio.run(retractions.check_literature([lit], transport=_by_title([_record(LOCAL_TITLE)])))
        assert out[0]["metadata"]["retraction"] == "not_checked"


def test_no_doi_found_means_not_checked_and_never_not_retracted() -> None:
    out, rows = asyncio.run(retractions.check_literature([_local()], transport=_by_title([])))
    assert out[0]["metadata"]["retraction"] == "not_checked"
    assert rows[0]["status"] == "not_checked" and rows[0]["why"] == retractions.NO_TITLE_MATCH
    assert retractions.summary_line(rows) == (
        "Could not check 1 source for retractions (no DOI found for this paper by its title)")


def test_a_network_error_looking_for_the_doi_is_not_checked_and_stops_asking() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raise httpx.ConnectError("no network", request=request)
    lit = [_local(), _local("Another pinned paper about vaccines")]
    out, rows = asyncio.run(retractions.check_literature(lit, transport=httpx.MockTransport(handler)))
    assert [e["metadata"]["retraction"] for e in out] == ["not_checked", "not_checked"]
    assert all("could not be reached" in r["why"] for r in rows)
    assert len(seen) == 1, "after Crossref cannot be reached no further request is sent"


def test_only_papers_the_person_supplied_are_looked_up_by_title() -> None:
    seen: list[httpx.Request] = []
    lit = [_entry("MMR vaccine and autism in children"), _local()]
    out, _rows = asyncio.run(retractions.check_literature(lit, transport=_by_title([_record(LOCAL_TITLE)], seen)))
    assert out[0]["metadata"]["retraction"] == "no_doi"
    assert out[1]["metadata"]["retraction"] == "retracted"
    assert sum(1 for r in seen if r.url.params.get("query.bibliographic")) == 1


def test_a_title_one_word_or_one_notice_prefix_apart_is_not_the_paper() -> None:
    long_title = "A long study of the effect of treatment on outcomes in type 2 diabetes patients"
    for other in (long_title.replace("type 2", "type 1"), "Correction: " + long_title):
        out, _rows = asyncio.run(retractions.check_literature(
            [_local(long_title)], transport=_by_title([_record(other)])))
        assert out[0]["metadata"]["retraction"] == "not_checked", other


def test_a_work_a_publisher_renamed_retracted_is_still_found() -> None:
    out, _rows = asyncio.run(retractions.check_literature(
        [_local()], transport=_by_title([_record("RETRACTED: " + LOCAL_TITLE)])))
    assert out[0]["metadata"]["retraction"] == "retracted"


def test_two_different_dois_for_one_title_is_not_a_guess_either() -> None:
    twin = {**_record(LOCAL_TITLE), "DOI": "10.1234/preprint"}
    out, rows = asyncio.run(retractions.check_literature(
        [_local()], transport=_by_title([_record(LOCAL_TITLE), twin])))
    assert out[0]["metadata"]["retraction"] == "not_checked"
    assert "several records" in rows[0]["why"]


def test_a_page_header_is_never_looked_up_as_a_title() -> None:
    seen: list[httpx.Request] = []
    lit = [{"content": "Contents lists available at ScienceDirect\nmore", "metadata": {"source": "user_supplied"}}]
    out, rows = asyncio.run(retractions.check_literature(lit, transport=_by_title([], seen)))
    assert out[0]["metadata"]["retraction"] == "not_checked" and not seen
    assert rows[0]["why"] == "this paper has no title to look up"


def test_a_paper_with_only_its_text_is_looked_up_by_its_first_line() -> None:
    lit = [{"content": "# MMR vaccine and autism in children\nbody", "metadata": {"source": "user_supplied"}}]
    out, _rows = asyncio.run(retractions.check_literature(lit, transport=_by_title([_record(LOCAL_TITLE)])))
    assert out[0]["metadata"]["retraction"] == "retracted"


def test_a_pinned_paper_found_retracted_cannot_support_a_claim() -> None:
    out, _rows = asyncio.run(retractions.check_literature([_local()], transport=_by_title([_record(LOCAL_TITLE)])))
    meta = out[0]["metadata"]
    claims, changed = retractions.apply_to_claims(
        [{"claim": "c", "basis": "citation", "citation_index": 1, "quote": "q"}], {"1": (meta, "text")})
    assert changed == 1 and claims[0]["basis"] == "unsupported"
