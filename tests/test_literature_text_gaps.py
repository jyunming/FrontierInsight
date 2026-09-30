"""Why a quest's sources came back without their full text, said plainly in run.log.

A real quest (VS Code, 2026-09-30) logged "full-text coverage: 0/16 sources have real full text; 0 abstract only, 0 a
preview or table of contents, 16 snippet only". Every one of the 16 had its abstract (OpenAlex rebuilds it from its word
index; needs/*.json held it): "snippet only" meant "nothing downloaded". Nothing downloaded because 13 were SPIE / IEEE /
Optica journals and books a person must fetch, and the Optics Express papers, free to read, came with no download
link, only their DOI. Semantic Scholar refused 4 searches and Crossref 1 (HTTP 429), each asked only once.

The OpenAlex records below are the fields of a live search made while diagnosing that quest (no network here).
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import core.knowledge as kn
import launch
from core import source_failures as sf
from core.engine import _literature_entry, _literature_text_gaps, _search_record_has_abstract

_ABSTRACT = ("Optical proximity correction (OPC) and phase-shifting mask (PSM) are resolution enhancement techniques "
             "(RET) used extensively in the semiconductor industry to improve the resolution and pattern fidelity of "
             "optical lithography. Traditional RETs, however, fix the source thus limiting the degrees of freedom.")


def _inverted(text: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i, word in enumerate(text.split()):
        index.setdefault(word, []).append(i)
    return index


# Recorded: https://api.openalex.org/works?search=source+mask+optimization+lithography (fields as returned).
_OPENALEX_PAGE = {"results": [
    {   # SPIE proceedings: closed.
        "id": "https://openalex.org/W1", "doi": "https://doi.org/10.1117/12.838701", "type": "article",
        "title": "Source-mask co-optimization: optimize design for imaging and impact of source complexity",
        "publication_year": 2009, "authorships": [{"author": {"display_name": "Stephen Hsu"}}],
        "abstract_inverted_index": _inverted(_ABSTRACT),
        "open_access": {"is_oa": False, "oa_status": "closed", "oa_url": None},
        "best_oa_location": None,
        "primary_location": {"pdf_url": None, "source": {"display_name": "Proceedings of SPIE"}},
    },
    {   # Optics Express: gold open access, but the only free address is the DOI.
        "id": "https://openalex.org/W1978743468", "doi": "https://doi.org/10.1364/oe.17.005783", "type": "article",
        "title": "Pixel-based simultaneous source and mask optimization for resolution enhancement",
        "publication_year": 2009, "authorships": [{"author": {"display_name": "Xu Ma"}}],
        "abstract_inverted_index": _inverted(_ABSTRACT),
        "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://doi.org/10.1364/oe.17.005783"},
        "best_oa_location": {"pdf_url": None, "landing_page_url": "https://doi.org/10.1364/oe.17.005783"},
        "primary_location": {"pdf_url": None, "source": {"display_name": "Optics Express"}},
    },
]}


class _Client:
    """httpx.Client stand-in answering from a list of (status, body, headers)."""

    answers: list[tuple[int, object, dict]] = []
    calls = 0

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get(self, url, params=None, **_):
        type(self).calls += 1
        status, body, headers = type(self).answers.pop(0)
        return httpx.Response(status, json=body, headers=headers, request=httpx.Request("GET", url, params=params))


@pytest.fixture
def client(monkeypatch):
    _Client.answers, _Client.calls = [], 0
    monkeypatch.setattr("core.knowledge.httpx.Client", _Client)
    yield _Client
    sf.reset("q-lit")


def _in_quest(fn, *args, **kwargs):
    token = sf.current_quest.set("q-lit")
    try:
        return fn(*args, **kwargs)
    finally:
        sf.current_quest.reset(token)


def test_openalex_records_carry_their_abstract(client) -> None:
    client.answers = [(200, _OPENALEX_PAGE, {})]
    docs = kn._openalex_search("source mask optimization lithography", 20)
    assert len(docs) == 2
    for d in docs:
        assert _ABSTRACT in d.content, "the abstract is rebuilt from OpenAlex's word index"
        assert _search_record_has_abstract(d.content)
    assert not _search_record_has_abstract("Just a title\n\nshort snippet")


def test_the_log_says_why_the_text_is_missing(client, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    client.answers = [(200, _OPENALEX_PAGE, {})]
    docs = kn._openalex_search("source mask optimization lithography", 20)
    entries = [_literature_entry(tmp_path, d.content, d.metadata) for d in docs]
    assert {e["metadata"]["content_quality"] for e in entries} == {"snippet_only"}, "nothing was downloaded"
    for _ in range(4):
        _in_quest(sf.record_failure, "semantic_scholar", "http_429", status=429,
                  url="https://api.semanticscholar.org/graph/v1/paper/search")
    _in_quest(sf.record_failure, "crossref", "http_429", status=429, url="https://api.crossref.org/works")
    why = _literature_text_gaps(entries, sf.snapshot("q-lit"))
    assert "1 are not marked free to read (most likely behind a subscription)" in why
    assert "1 are marked free to read but the index gave no download link" in why
    assert "semantic_scholar 4 too many requests at api.semanticscholar.org" in why
    assert "crossref 1 too many requests at api.crossref.org" in why
    assert "no SEMANTIC_SCHOLAR_API_KEY is set" in why
    assert "no OPENALEX_API_KEY is set" in why


def test_a_failed_free_download_names_the_host(client, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k")
    entry = _literature_entry(tmp_path, "T\n\n" + _ABSTRACT, {
        "source": "openalex", "title": "T", "open_access": True, "free_url": "https://www.mdpi.com/x.pdf"})
    _in_quest(sf.record_failure, "oa_copy", "http_4xx", status=403, url="https://www.mdpi.com/x.pdf")
    _in_quest(sf.record_failure, "oa_copy", "timeout", url="https://www.osti.gov/biblio/1")
    _in_quest(sf.record_failure, "duckduckgo", "http_429", status=429, url="https://html.duckduckgo.com/html")
    why = _literature_text_gaps([entry], sf.snapshot("q-lit"))
    assert "1 are free to read but FI did not get their text;" in why
    assert "free copies 1 refused, 1 timed out at www.mdpi.com, www.osti.gov" in why
    assert "duckduckgo" not in why, "a web search engine's failure is not about the papers"
    assert "OPENALEX_API_KEY" not in why
    off = _literature_text_gaps([entry], sf.snapshot("q-lit"), downloads_on=False)
    assert "downloading is off: knowledge.try_fetch_full_text" in off


def test_the_openalex_key_is_mentioned_only_when_openalex_was_used(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    crossref_only = _literature_entry(tmp_path, "T\n\n" + _ABSTRACT, {"source": "crossref", "title": "T"})
    assert "OPENALEX_API_KEY" not in _literature_text_gaps([crossref_only], {})
    openalex = _literature_entry(tmp_path, "T\n\n" + _ABSTRACT, {"source": "openalex", "title": "T"})
    assert "no OPENALEX_API_KEY is set" in _literature_text_gaps([openalex], {})


def test_nothing_to_explain_when_every_source_has_its_text(tmp_path: Path) -> None:
    entry = {"content": "x", "metadata": {"source": "openalex", "content_quality": "full_text"}}
    assert _literature_text_gaps([entry], {}) == ""


# --- a search refused with HTTP 429 is asked again -------------------------------------------------------------

def test_a_rate_limited_search_is_asked_again(client, monkeypatch) -> None:
    waited: list[float] = []
    monkeypatch.setattr(kn, "_rate_limit_sleep", waited.append)
    client.answers = [(429, {}, {}), (429, {}, {"Retry-After": "3"}), (200, {"results": []}, {})]
    out = _in_quest(kn._http_get_json, "https://api.semanticscholar.org/graph/v1/paper/search", {}, 5.0,
                    source="semantic_scholar")
    assert out == {"results": []}
    assert client.calls == 3 and waited == [kn._RATE_LIMIT_WAITS_S[0], 3.0]
    assert sf.snapshot("q-lit")["total"] == 0, "a search that got through in the end is not a failure"


def test_a_search_out_of_its_daily_budget_is_not_waited_for(client, monkeypatch) -> None:
    waited: list[float] = []
    monkeypatch.setattr(kn, "_rate_limit_sleep", waited.append)
    client.answers = [(429, {}, {"Retry-After": "3600"})]
    assert _in_quest(kn._http_get_json, "https://api.openalex.org/works", {}, 5.0, source="openalex") is None
    assert client.calls == 1 and waited == []
    assert sf.snapshot("q-lit")["by_source"] == {"openalex": {"http_429": 1}}


def test_a_spent_allowance_is_not_asked_again(client, monkeypatch) -> None:
    waited: list[float] = []
    monkeypatch.setattr(kn, "_rate_limit_sleep", waited.append)
    client.answers = [(429, {}, {"X-RateLimit-Remaining": "0"})]
    assert _in_quest(kn._http_get_json, "https://api.openalex.org/works", {}, 5.0, source="openalex") is None
    assert client.calls == 1 and waited == []
    # A short Retry-After is the server saying a retry will do, whatever the counter says.
    client.calls = 0
    client.answers = [(429, {}, {"X-RateLimit-Remaining": "0", "Retry-After": "1"}), (200, {"ok": 1}, {})]
    assert _in_quest(kn._http_get_json, "https://api.openalex.org/works", {}, 5.0, source="openalex") == {"ok": 1}
    assert waited == [1.0]


def test_retry_after_forms(client, monkeypatch) -> None:
    from email.utils import format_datetime
    import datetime as dt

    waited: list[float] = []
    monkeypatch.setattr(kn, "_rate_limit_sleep", waited.append)
    later = format_datetime(dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=2), usegmt=True)
    client.answers = [(429, {}, {"Retry-After": later})]
    assert _in_quest(kn._http_get_json, "https://api.crossref.org/works", {}, 5.0, source="crossref") is None
    assert client.calls == 1 and waited == [], "an HTTP-date two hours away is not waited for"
    client.calls = 0
    client.answers = [(429, {}, {"Retry-After": "0"}), (200, {"ok": 1}, {})]
    assert _in_quest(kn._http_get_json, "https://api.crossref.org/works", {}, 5.0, source="crossref") == {"ok": 1}
    assert waited == [0.5]


def test_other_errors_are_not_retried(client) -> None:
    client.answers = [(503, {}, {})]
    assert _in_quest(kn._http_get_json, "https://api.crossref.org/works", {}, 5.0, source="crossref") is None
    assert client.calls == 1


def test_retries_run_out(client, monkeypatch) -> None:
    client.answers = [(429, {}, {})] * (len(kn._RATE_LIMIT_WAITS_S) + 1)
    assert _in_quest(kn._http_get_json, "https://api.crossref.org/works", {}, 5.0, source="crossref") is None
    assert client.calls == len(kn._RATE_LIMIT_WAITS_S) + 1
    assert sf.snapshot("q-lit")["by_source"] == {"crossref": {"http_429": 1}}, "one failure per search, not per try"


# --- the .env in FI's own folder -------------------------------------------------------------------------------

def test_the_env_file_in_fis_folder_is_read_when_run_from_elsewhere(tmp_path: Path, monkeypatch) -> None:
    """The VS Code extension runs a quest in the workspace folder; the key sat in FI's own .env and was never read."""
    fi = tmp_path / "fi"
    work = tmp_path / "work"
    fi.mkdir()
    work.mkdir()
    (fi / ".env").write_text("OPENALEX_API_KEY=from-fi\nFI_TEST_BOTH=from-fi\n", encoding="utf-8")
    (work / ".env").write_text("FI_TEST_BOTH=from-work\n", encoding="utf-8")
    monkeypatch.setattr(launch, "__file__", str(fi / "launch.py"))
    monkeypatch.chdir(work)
    for name in ("OPENALEX_API_KEY", "FI_TEST_BOTH", "FI_TEST_REAL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FI_TEST_REAL", "real")
    (work / ".env").write_text("FI_TEST_BOTH=from-work\nOPENALEX_API_KEY=\n", encoding="utf-8")
    (fi / ".env").write_text("OPENALEX_API_KEY=from-fi\nFI_TEST_BOTH=from-fi\nFI_TEST_REAL=from-fi\n", encoding="utf-8")
    try:
        launch._load_dotenvs()
        import os
        assert os.environ["OPENALEX_API_KEY"] == "from-fi", "a blank line in the working folder's .env sets nothing"
        assert os.environ["FI_TEST_BOTH"] == "from-work", "the working folder's .env wins over FI's"
        assert os.environ["FI_TEST_REAL"] == "real", "a real environment variable wins over both"
    finally:
        import os
        for name in ("OPENALEX_API_KEY", "FI_TEST_BOTH"):
            os.environ.pop(name, None)
