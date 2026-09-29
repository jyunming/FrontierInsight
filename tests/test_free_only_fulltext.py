"""FI downloads only papers that are free to read; a paper not confirmed free is left for a person.

The full-text step used to GET whatever pdf_url or landing page a search hit carried, relying on the host's own
network access (a VPN or campus login) to get past a paywall. Now a hit is fetched only when it is confirmed free
(``is_open_access``), and then only from its free location: the copy its source named, or an address on a free
server, never a DOI resolver or publisher page. Anything else may be recovered only through the free-copy cascade
(Unpaywall, Europe PMC, Semantic Scholar, CORE); if that finds nothing the paper goes on
``needs/WANTED_PAPERS.md``. No network and no model is used here: every stand-in raises on a request.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import core.knowledge as kn
from core.knowledge import RetrievedDoc, is_open_access
from tests.test_literature_facets import _engine, _state

PUBLISHER_PDF = "https://publisher.example/paywalled.pdf"
PUBLISHER_PAGE = "https://publisher.example/article/123"
FREE_TEXT = "Free full text of the paper. " * 30


class _SpyClient:
    """Stands in for httpx.Client; records every URL asked for and raises, so no request succeeds."""

    urls: list[str] = []

    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        _SpyClient.urls.append(url)
        raise AssertionError(f"unexpected GET {url}")


@pytest.fixture()
def spy(monkeypatch):
    _SpyClient.urls = []
    monkeypatch.setattr(kn.httpx, "Client", _SpyClient)

    def module_get(url, *a, **k):  # the free-copy lookups call httpx.get, not httpx.Client
        _SpyClient.urls.append(url)
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(kn.httpx, "get", module_get)
    return _SpyClient


def _paywalled() -> RetrievedDoc:
    return RetrievedDoc(content="abstract", metadata={
        "source": "crossref", "title": "A paywalled paper", "doi": "10.1109/x.1",
        "pdf_url": PUBLISHER_PDF, "url": PUBLISHER_PAGE,
    })


def test_a_paywalled_hit_is_never_requested_from_the_publisher(monkeypatch, spy) -> None:
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    assert kn._fetch_full_text(_paywalled(), timeout_s=5, max_kb=64) is None
    assert spy.urls == []


def test_a_paywalled_page_is_neither_fetched_nor_rendered(monkeypatch, spy) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda url, **k: rendered.append(url) or "<html>x</html>")
    doc = RetrievedDoc(content="snippet", metadata={"url": "https://link.springer.com/article/10.1007/x"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True) is None
    assert spy.urls == [] and rendered == []


def test_a_confirmed_free_hit_is_still_fetched(monkeypatch, spy) -> None:
    seen: list[str] = []
    monkeypatch.setattr(kn, "_fetch_pdf_bytes", lambda url, *, timeout_s: seen.append(url) or b"%PDF-1.4 x")
    monkeypatch.setattr(kn, "_pdf_bytes_to_text", lambda body, *, cap: FREE_TEXT)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    doc = RetrievedDoc(content="abstract", metadata={"arxiv_id": "2401.00001", "pdf_url": "https://arxiv.org/pdf/2401.00001"})
    assert kn._fetch_full_text(doc, timeout_s=5, max_kb=64) == FREE_TEXT
    assert seen == ["https://arxiv.org/pdf/2401.00001"], "an arXiv paper is free, so its own address is still used"


@pytest.mark.parametrize("md,free", [
    ({"open_access": True}, True),
    ({"arxiv_id": "2401.1"}, True),
    ({"pmcid": "PMC1"}, True),
    ({"doi": "10.1101/2020.01.01.1"}, True),
    ({"url": "https://www.medrxiv.org/content/x"}, True),
    ({}, False),
    ({"open_access": None, "doi": "10.1016/j.x"}, False),
    ({"url": "https://ieeexplore.ieee.org/document/1"}, False),
    ({"url": "https://www.mdpi.com/2073-4441/12/3/456"}, True),
    ({"url": "https://journals.plos.org/plosone/article?id=10.1371/x"}, True),
    ({"url": "https://www.frontiersin.org/articles/10.3389/x/full"}, True),
    ({"url": "https://link.springer.com/article/10.1007/x"}, False),
    ({"url": "https://www.sciencedirect.com/science/article/pii/S1?ref=arxiv.org"}, False),
    ({"url": "https://notarxiv.org.evil.example/abs/1"}, False),
    ({"url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1/"}, True),
    ({"url": "https://www.ncbi.nlm.nih.gov/pubmed/1"}, False),
    (None, False),
])
def test_only_confirmed_free_sources_count_as_free(md, free) -> None:
    assert is_open_access(md) is free


def test_an_unknown_hit_is_confirmed_through_the_free_copy_cascade_then_fetched(monkeypatch, spy) -> None:
    asked: list[str] = []

    def cascade(doc, *, timeout_s, cap):
        asked.append(doc.metadata["doi"])
        return FREE_TEXT  # Unpaywall found a free copy

    monkeypatch.setattr(kn, "_fetch_via_open_apis", cascade)
    assert kn._fetch_full_text(_paywalled(), timeout_s=5, max_kb=64) == FREE_TEXT
    assert asked == ["10.1109/x.1"] and spy.urls == []


def test_an_unpaywall_failure_leaves_the_paper_for_a_person(monkeypatch) -> None:
    import httpx

    asked: list[str] = []

    class _Down(_SpyClient):
        def get(self, url, **kw):
            asked.append(url)
            raise httpx.ConnectError("unpaywall is down")

    def down_get(url, *a, **k):
        asked.append(url)
        raise httpx.ConnectError("unpaywall is down")

    monkeypatch.setattr(kn.httpx, "Client", _Down)
    monkeypatch.setattr(kn.httpx, "get", down_get)
    monkeypatch.setattr(kn, "_resolve_ids", lambda doc, timeout_s: {"doi": "10.1109/x.1", "pmcid": "", "arxiv_id": ""})
    assert kn._fetch_full_text(_paywalled(), timeout_s=5, max_kb=64) is None
    assert asked, "the free-copy lookup was tried"
    assert not [u for u in asked if "publisher.example" in u], asked


# --- the papers gate in the literature step -------------------------------------------------------------------

def _gate_engine(tmp_path: Path, monkeypatch, docs, *, pauses_papers: bool):
    eng = _engine(tmp_path)
    eng.config.pauses.papers = pauses_papers
    eng.config.knowledge.foundational_works = False

    async def chat(prompt, node=""):
        return '{"queries": ["a query"]}'

    eng._chat = chat

    async def hits(query, **kw):
        return list(docs)

    eng.knowledge.asearch = hits  # type: ignore[method-assign]

    async def route(*a, **k):
        return ["crossref"]

    eng.knowledge.choose_sources = route  # type: ignore[method-assign]

    async def no_fetch(found):
        return found

    eng.knowledge.fetch_full_text = no_fetch  # type: ignore[method-assign]
    paused: list[dict] = []
    monkeypatch.setattr(eng, "_pause_for_human", lambda **kw: paused.append(kw))
    lines: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    eng._log.addHandler(_H())
    eng._log.setLevel(logging.INFO)
    return eng, paused, lines


@pytest.mark.asyncio
async def test_an_unattended_run_logs_one_count_and_lists_the_papers(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=False)
    await eng._node_literature(_state())
    assert paused == []
    counted = [line for line in lines if "not confirmed free" in line]
    assert len(counted) == 1 and counted[0].startswith("[literature] 1 paper(s)")
    wanted = (eng.quest_root / "needs" / "WANTED_PAPERS.md").read_text(encoding="utf-8")
    assert "A paywalled paper" in wanted


@pytest.mark.asyncio
async def test_an_attended_run_pauses_for_the_paywalled_paper(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=True)
    await eng._node_literature(_state())
    assert len(paused) == 1 and paused[0]["kind"] == "papers"
    assert not [line for line in lines if "not confirmed free" in line]
    assert (eng.quest_root / "needs" / "WANTED_PAPERS.md").is_file()


@pytest.mark.asyncio
async def test_an_unattended_run_does_not_tell_the_person_to_resume(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=False)
    await eng._node_literature(_state())
    wanted = (eng.quest_root / "needs" / "WANTED_PAPERS.md").read_text(encoding="utf-8")
    readme = (eng.quest_root / "inputs" / "papers" / "README.md").read_text(encoding="utf-8")
    assert "resume" not in wanted.lower().replace("nothing to resume", "")
    assert "resume" not in readme.lower().replace("nothing to resume", "")
    assert "nothing to resume" in wanted and "nothing to resume" in readme


@pytest.mark.asyncio
async def test_an_attended_run_tells_the_person_to_resume(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=True)
    await eng._node_literature(_state())
    wanted = (eng.quest_root / "needs" / "WANTED_PAPERS.md").read_text(encoding="utf-8")
    readme = (eng.quest_root / "inputs" / "papers" / "README.md").read_text(encoding="utf-8")
    assert "resume" in wanted.lower() and "resume" in readme.lower()
    assert "nothing to resume" not in wanted


@pytest.mark.asyncio
async def test_the_count_is_logged_even_when_papers_were_already_dropped(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=False)
    drop = eng.quest_root / "inputs" / "papers"
    drop.mkdir(parents=True, exist_ok=True)
    (drop / "mine.md").write_text("my own paper " * 200, encoding="utf-8")
    await eng._node_literature(_state())
    assert [ln for ln in lines if "not confirmed free" in ln and "already has files" in ln]


# --- a web hit is opened only when it is confirmed free, whatever its host ------------------------------------

@pytest.mark.parametrize("url,doi", [
    ("https://www.science.org/doi/10.1126/science.abc1234", ""),
    ("https://www.nejm.org/doi/full/10.1056/NEJMoa1", ""),
    ("https://journal.unlisted.example/some/page", "10.5555/unlisted.1"),
    ("https://pub.unlisted.example/doi/10.5555/abc", ""),
    ("https://pub.unlisted.example/journals/lancet/article/PIIS0140/fulltext", ""),
])
def test_a_journal_page_on_any_host_is_neither_fetched_nor_rendered(monkeypatch, spy, url, doi) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda u, **k: rendered.append(u) or "<html>x</html>")
    md = {"url": url, **({"doi": doi} if doi else {})}
    doc = RetrievedDoc(content="snippet", metadata=md)
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True) is None
    assert spy.urls == [] and rendered == []


def test_a_fully_open_access_publisher_page_is_still_read(monkeypatch, spy) -> None:
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda u, **k: None)
    doc = RetrievedDoc(content="snippet", metadata={"url": "https://www.mdpi.com/2073-4441/12/3/456"})
    kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False)
    assert "https://www.mdpi.com/2073-4441/12/3/456" in spy.urls


def test_an_ordinary_web_page_is_still_read(monkeypatch, spy) -> None:
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    doc = RetrievedDoc(content="s", metadata={"url": "https://www.iea.org/reports/outlook"})
    kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False)
    assert "https://www.iea.org/reports/outlook" in spy.urls


# --- a record marked free is fetched only from its free copy --------------------------------------------------

def _openaire_payload(instances):
    return {"results": [{
        "mainTitle": "Green OA paper", "bestAccessRight": {"label": "OPEN"},
        "pids": [{"scheme": "doi", "value": "10.5555/green.1"}],
        "instances": [{"type": "Article", **i} for i in instances],
    }]}


def test_openaire_free_copy_is_the_open_instance_and_never_the_doi(monkeypatch, spy) -> None:
    payload = _openaire_payload([
        {"accessRight": {"label": "CLOSED"}, "urls": ["https://publisher.example/paywalled.pdf"]},
        {"accessRight": {"label": "OPEN"}, "urls": ["https://repo.example/handle/1.pdf"]},
    ])
    monkeypatch.setattr(kn, "_http_get_json", lambda *a, **k: payload)
    doc = kn._openaire_search("green", 5)[0]
    assert doc.metadata["open_access"] is True
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    kn._fetch_full_text(doc, timeout_s=5, max_kb=64)
    assert spy.urls and set(spy.urls) == {"https://repo.example/handle/1.pdf"}


def test_an_openaire_hit_with_no_free_copy_named_is_not_free(monkeypatch, spy) -> None:
    payload = _openaire_payload([{"accessRight": {"label": "OPEN"}, "urls": ["https://doi.org/10.5555/green.1"]}])
    monkeypatch.setattr(kn, "_http_get_json", lambda *a, **k: payload)
    doc = kn._openaire_search("green", 5)[0]
    assert not is_open_access(doc.metadata)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    assert kn._fetch_full_text(doc, timeout_s=5, max_kb=64) is None
    assert spy.urls == []


def test_a_core_hit_is_free_only_with_a_download_url(monkeypatch, spy) -> None:
    payload = {"results": [
        {"title": "No copy", "doi": "10.5555/c.1", "sourceFulltextUrls": ["https://publisher.example/x"]},
        {"title": "Has copy", "downloadUrl": "https://core.ac.uk/download/1.pdf"},
    ]}
    monkeypatch.setattr(kn, "_http_get_json", lambda *a, **k: payload)
    no_copy, has_copy = kn._core_search("q", 5)
    assert not is_open_access(no_copy.metadata) and is_open_access(has_copy.metadata)
    assert has_copy.metadata["free_url"] == "https://core.ac.uk/download/1.pdf"
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    kn._fetch_full_text(no_copy, timeout_s=5, max_kb=64)
    assert spy.urls == []


def test_a_semantic_scholar_hit_is_free_only_with_an_open_pdf(monkeypatch) -> None:
    payload = {"data": [
        {"title": "Closed", "externalIds": {"DOI": "10.5555/s.1"}, "url": "https://www.semanticscholar.org/paper/1"},
        {"title": "Open", "url": "https://www.semanticscholar.org/paper/2",
         "openAccessPdf": {"url": "https://repo.example/2.pdf"}},
    ]}
    monkeypatch.setattr(kn, "_http_get_json", lambda *a, **k: payload)
    closed, opened = kn._semantic_scholar_search("q", 5)
    assert not is_open_access(closed.metadata)
    assert opened.metadata["open_access"] is True and opened.metadata["free_url"] == "https://repo.example/2.pdf"


def test_an_openalex_free_hit_uses_its_best_open_location_not_the_publisher(monkeypatch, spy) -> None:
    work = {"id": "https://openalex.org/W1", "title": "T", "doi": "https://doi.org/10.5555/o.1",
            "open_access": {"is_oa": True},
            "primary_location": {"pdf_url": PUBLISHER_PDF, "landing_page_url": PUBLISHER_PAGE},
            "best_oa_location": {"pdf_url": "https://repo.example/o.pdf"}}
    doc = kn._openalex_work_doc(work)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    kn._fetch_full_text(doc, timeout_s=5, max_kb=64)
    assert spy.urls and set(spy.urls) == {"https://repo.example/o.pdf"}


def test_a_free_hit_with_only_a_doi_or_publisher_url_requests_nothing(monkeypatch, spy) -> None:
    doc = RetrievedDoc(content="a", metadata={"open_access": True, "doi": "10.5555/d.1",
                                              "url": "https://doi.org/10.5555/d.1", "pdf_url": PUBLISHER_PDF})
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    assert kn._fetch_full_text(doc, timeout_s=5, max_kb=64) is None
    assert spy.urls == []
