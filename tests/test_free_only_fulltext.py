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


# --- round 3: redirects, unlisted scholarly hosts, the page-metadata backstop, free-flag without a free address ---

import httpx  # noqa: E402


def _mock_client(monkeypatch, handler):
    """Every httpx.Client the fetchers open goes through ``handler`` (no network); returns the list of requested URLs."""
    asked: list[str] = []
    real = httpx.Client

    def wrapped(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return handler(request)

    def factory(*a, **k):
        k["transport"] = httpx.MockTransport(wrapped)
        return real(*a, **k)

    monkeypatch.setattr(kn.httpx, "Client", factory)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    return asked


def test_a_redirect_from_an_ordinary_page_to_a_journal_article_is_not_followed(monkeypatch) -> None:
    def handler(req):
        if req.url.host == "old.example":
            return httpx.Response(302, headers={"location": "https://opg.optica.org/abstract.cfm?doi=10.1364/x"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>" + "paper " * 500 + "</html>")

    asked = _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://old.example/some-page"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False) is None
    assert asked == ["https://old.example/some-page"]


def test_a_redirect_to_a_free_server_is_followed(monkeypatch) -> None:
    def handler(req):
        if req.url.host == "old.example":
            return httpx.Response(301, headers={"location": "https://arxiv.org/abs/2401.00001"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html><p>" + "real text " * 300 + "</p></html>")

    asked = _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://old.example/x"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False)
    assert asked[-1] == "https://arxiv.org/abs/2401.00001"


def test_a_pdf_redirect_to_a_publisher_is_refused(monkeypatch) -> None:
    def handler(req):
        return httpx.Response(302, headers={"location": "https://www.sciencedirect.com/science/article/pii/S1/pdf"})

    asked = _mock_client(monkeypatch, handler)
    assert kn._fetch_pdf_bytes("https://repo.example/paper.pdf", timeout_s=5) is None
    assert asked == ["https://repo.example/paper.pdf"]


def test_a_free_page_naming_a_publisher_pdf_does_not_reach_it(monkeypatch) -> None:
    page = '<html><head><meta name="citation_pdf_url" content="https://direct.mit.edu/x/pdf"></head></html>'

    def handler(req):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=page)

    asked = _mock_client(monkeypatch, handler)
    assert kn._pdf_from_free_page("https://repo.example/landing", timeout_s=5) is None
    assert asked == ["https://repo.example/landing"]


def test_a_redirect_chain_that_is_too_long_is_dropped(monkeypatch) -> None:
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(302, headers={"location": f"https://hop{n['i']}.example/"})

    _mock_client(monkeypatch, handler)
    assert kn._fetch_pdf_bytes("https://repo.example/p.pdf", timeout_s=5) is None
    assert n["i"] <= kn._MAX_HOPS + 1


def test_the_headless_render_is_told_to_refuse_a_navigation_to_a_journal(monkeypatch) -> None:
    seen: dict = {}

    def fake_render(url, *, timeout_s, allow=None):
        seen["allow"] = allow
        return None

    monkeypatch.setattr(kn, "_playwright_fetch_html", fake_render)
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)

    def handler(req):
        return httpx.Response(403, text="blocked")

    _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://news.example/story"})
    kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True)
    allow = seen["allow"]
    assert allow("https://news.example/other") is True
    assert allow("https://direct.mit.edu/article/10.1162/x") is False


@pytest.mark.parametrize("url", [
    "https://muse.jhu.edu/article/123456",
    "https://direct.mit.edu/daed/article/151/3/1/1234",
    "https://www.jneurosci.org/content/41/1/1",
    "https://www.igi-global.com/gateway/article/12345",
    "https://www.ingentaconnect.com/content/x/y/2020/1/1",
    "https://www.proquest.com/docview/123",
    "https://search.ebscohost.com/login.aspx?direct=true",
    "https://www.cairn.info/revue-x-2020-1-page-1.htm",
    "https://www.taylorfrancis.com/books/x",
    "https://pubs.geoscienceworld.org/gsa/article/1",
    "https://ashpublications.org/blood/article/1",
    "https://aacrjournals.org/cancerres/article/1",
    "https://rupress.org/jcb/article/1",
    "https://muse.jhu.edu./article/1",
])
def test_scholarly_hosts_seen_in_the_wild_are_not_fetched(url, monkeypatch, spy) -> None:
    monkeypatch.setattr(kn, "_fetch_via_open_apis", lambda *a, **k: None)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda *a, **k: "<html>x</html>")
    doc = RetrievedDoc(content="s", metadata={"url": url})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True) is None
    assert spy.urls == []


@pytest.mark.parametrize("url", [
    "https://www.example-news.com/article/climate-report-released",
    "https://blog.example.org/content/how-we-built-it",
    "https://www.bbc.com/news/articles/c1234abcd",
    "https://www.nationalgeographic.com/science/article/cave-art-found",
    "https://www.reuters.com/science/article/mars-rover-lands",
    "https://www.gov.uk/government/abs/foo",
    "https://spectrum.ieee.org/some-chip-story",
    "https://cen.acs.org/articles/98/i1/story.html",
    "https://www.endocrine.org/news-and-advocacy/news-room/2020/story",
    "https://www.physiology.org/news/story",
])
def test_news_style_article_paths_are_not_taken_for_scholarly(url) -> None:
    assert kn._looks_scholarly({"url": url}) is False and not kn._is_academic_source(url)


def test_a_page_that_says_it_is_a_journal_article_is_discarded_and_never_rendered(monkeypatch) -> None:
    page = ('<html><head><meta name="citation_journal_title" content="J"><meta name="citation_doi" content="10.1/x">'
            "</head><body>" + "paywalled article text " * 200 + "</body></html>")
    rendered: list[str] = []

    def handler(req):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=page)

    _mock_client(monkeypatch, handler)
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda u, **k: rendered.append(u) or "<html>x</html>")
    doc = RetrievedDoc(content="s", metadata={"url": "https://unlisted-publisher.example/read/77"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True) is None
    assert rendered == []
    assert doc.metadata["scholarly_page"] is True


def test_a_rendered_page_with_journal_metadata_is_discarded(monkeypatch) -> None:
    def handler(req):
        return httpx.Response(403, headers={"content-type": "text/html"}, text="blocked")

    _mock_client(monkeypatch, handler)
    rendered_html = ('<html><head><script type="application/ld+json">{"@type": "ScholarlyArticle"}</script></head>'
                     "<body>" + "text " * 500 + "</body></html>")
    monkeypatch.setattr(kn, "_playwright_fetch_html", lambda u, **k: rendered_html)
    doc = RetrievedDoc(content="s", metadata={"url": "https://unlisted-publisher.example/read/77"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=True) is None
    assert doc.metadata["scholarly_page"] is True


def test_a_page_with_journal_metadata_that_is_confirmed_free_is_kept(monkeypatch) -> None:
    page = ('<html><head><meta name="citation_doi" content="10.3390/x"></head><body>'
            + "<p>open access text here</p>" * 100 + "</body></html>")

    def handler(req):
        return httpx.Response(200, headers={"content-type": "text/html"}, text=page)

    _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://www.mdpi.com/2073-4441/12/3/456"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False)


def test_a_discarded_scholarly_page_is_asked_for_by_the_papers_gate() -> None:
    from core.engine import _is_abstract_only
    md = {"source": "web_search", "title": "T", "url": "https://x.example/1"}
    assert _is_abstract_only(RetrievedDoc(content="short", metadata={**md, "scholarly_page": True})) is True


def test_a_doi_resolver_is_never_a_free_location_for_any_source() -> None:
    assert kn._free_locations({"free_url": "https://doi.org/10.1/x", "url": "https://dx.doi.org/10.1/x"}) == []
    assert kn._free_locations({"free_url": "https://repo.example/p.pdf"}) == ["https://repo.example/p.pdf"]


@pytest.mark.parametrize("md,free", [
    ({"doi": "10.1101/2020.01.01.123456"}, True),
    ({"doi": "10.1101/gad.123456"}, False),
    ({"doi": "10.1101/sqb.2020.85.1"}, False),
    ({"url": "https://zenodo.org/records/1"}, True),
    ({"url": "https://osf.io/abcde"}, True),
    ({"url": "https://europepmc.org/article/MED/1"}, False),
    ({"url": "https://europepmc.org/article/MED/1", "open_access": True, "free_url": "https://europepmc.org/articles/PMC1"}, True),
    ({"url": "https://www.ncbi.nlm.nih.gov/pmcfoo"}, False),
    ({"url": "https://arxiv.org./abs/1"}, True),
])
def test_free_rules_after_review(md, free) -> None:
    assert is_open_access(md) is free


def test_a_free_flag_with_no_free_address_goes_to_the_person() -> None:
    from core.engine import _is_open_access
    flagged = RetrievedDoc(content="a", metadata={"open_access": True, "url": "https://doi.org/10.1/x"})
    assert _is_open_access(flagged) is False
    with_id = RetrievedDoc(content="a", metadata={"open_access": True, "arxiv_id": "2401.1", "url": "https://doi.org/10.1/x"})
    assert _is_open_access(with_id) is True
    named = RetrievedDoc(content="a", metadata={"open_access": True, "free_url": "https://repo.example/p.pdf"})
    assert _is_open_access(named) is True


def test_url_host_ignores_a_trailing_dot() -> None:
    assert kn._url_host("https://Muse.JHU.edu./article/1").lower() == "muse.jhu.edu"


@pytest.mark.asyncio
async def test_the_attended_pause_card_does_not_say_paywalled(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=True)
    await eng._node_literature(_state())
    card = paused[0]
    text = card["headline"] + " ".join(card["steps"])
    assert "paywalled" not in text and "not confirmed free" in text


@pytest.mark.asyncio
async def test_the_papers_readme_follows_the_current_run(tmp_path: Path, monkeypatch) -> None:
    eng, paused, lines = _gate_engine(tmp_path, monkeypatch, [_paywalled()], pauses_papers=False)
    await eng._node_literature(_state())
    readme = eng.quest_root / "inputs" / "papers" / "README.md"
    assert "nothing to resume" in readme.read_text(encoding="utf-8")
    eng.config.pauses.papers = True
    await eng._node_literature(_state())
    text = readme.read_text(encoding="utf-8")
    assert "nothing to resume" not in text and "--resume" in text


# --- round 4: headless redirect hops, OA-cascade redirects, PDF/content-type backstop, news pages, europepmc ---

def _serve(handler_cls):
    import threading
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_a_real_browser_never_sends_a_request_to_a_refused_redirect_target() -> None:
    pytest.importorskip("playwright.sync_api")
    from http.server import BaseHTTPRequestHandler
    hits: list[str] = []

    class B(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>publisher article</body></html>")

        def log_message(self, *a):
            pass

    srv_b = _serve(B)
    target = f"http://localhost:{srv_b.server_port}/doi/10.1234/x"

    class A(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("location", target)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv_a = _serve(A)
    start = f"http://127.0.0.1:{srv_a.server_port}/go"
    try:
        try:
            html = kn._playwright_fetch_html(start, timeout_s=15, allow=lambda u: kn._redirect_allowed(u, start))
        except Exception as e:  # no browser installed on this machine
            pytest.skip(f"headless browser unavailable: {e}")
        assert html is None or "publisher article" not in html
        assert hits == []
    finally:
        srv_a.shutdown()
        srv_b.shutdown()


def _pdf_like(monkeypatch):
    monkeypatch.setattr(kn, "_pdf_bytes_to_text", lambda b, cap=0: "word " * 400)


def test_unpaywall_does_not_follow_a_repository_link_into_a_publisher(monkeypatch) -> None:
    def handler(req):
        if req.url.host == "api.unpaywall.org":
            return httpx.Response(200, json={"oa_locations": [{"url": "https://hdl.example/1", "host_type": "repository"}]})
        if req.url.host == "hdl.example":
            return httpx.Response(302, headers={"location": "https://www.sciencedirect.com/science/article/pii/S1"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<html>" + "paper " * 500 + "</html>")

    asked = _mock_client(monkeypatch, handler)
    assert kn._unpaywall_fulltext("10.1/x", timeout_s=5, cap=10000) is None
    assert not any("sciencedirect" in u for u in asked)


def test_semantic_scholar_pdf_link_that_redirects_to_a_publisher_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(kn.httpx, "get", lambda *a, **k: httpx.Response(
        200, json={"openAccessPdf": {"url": "https://repo.example/x.pdf"}}, request=httpx.Request("GET", a[0])))

    def handler(req):
        if req.url.host == "repo.example":
            return httpx.Response(302, headers={"location": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 x")

    asked = _mock_client(monkeypatch, handler)
    _pdf_like(monkeypatch)
    assert kn._s2_oa_pdf("10.1/x", timeout_s=5, cap=10000) is None
    assert not any("ieee" in u for u in asked)


def test_the_arxiv_and_biorxiv_routes_recheck_every_redirect(monkeypatch) -> None:
    def handler(req):
        return httpx.Response(302, headers={"location": "https://www.sciencedirect.com/science/article/pii/S1"})

    asked = _mock_client(monkeypatch, handler)
    assert kn._preprint_fulltext({"doi": "https://doi.org/10.1101/068312"}, timeout_s=5, cap=1000) is None
    assert asked and all("sciencedirect" not in u for u in asked)


def test_the_oa_text_helper_rejects_a_paywall_stub_and_a_publisher_landing_page() -> None:
    stub = httpx.Response(200, headers={"content-type": "text/html"},
                          text="<html><body>Buy this article for $35. Subscribe to read the full text. "
                               + "x " * 300 + "</body></html>",
                          request=httpx.Request("GET", "https://repo.example/a"))
    assert kn._pdf_or_html_text(stub, cap=10000, origin="https://repo.example/a") is None
    landed = httpx.Response(200, headers={"content-type": "text/html"},
                            text='<html><head><meta name="citation_doi" content="10.1/x"></head><body>' + "text " * 500 + "</body></html>",
                            request=httpx.Request("GET", "https://www.sciencedirect.com/x"))
    assert kn._pdf_or_html_text(landed, cap=10000, origin="https://hdl.example/1") is None


def test_a_pdf_from_an_unlisted_site_that_names_a_doi_is_discarded(monkeypatch) -> None:
    monkeypatch.setattr(kn, "_pdf_bytes_to_text", lambda b, cap=0: "Journal of X. https://doi.org/10.5555/abc.123 " + "word " * 400)

    def handler(req):
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 x")

    _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://unlisted.example/files/view/8812.pdf"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False) is None
    assert doc.metadata["scholarly_page"] is True


def test_a_same_host_redirect_to_an_article_address_is_discarded(monkeypatch) -> None:
    _pdf_like(monkeypatch)

    def handler(req):
        if req.url.path == "/landing":
            return httpx.Response(302, headers={"location": "/doi/pdf/10.5555/abc"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 x")

    _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://unlisted.example/landing"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False) is None
    assert doc.metadata["scholarly_page"] is True


@pytest.mark.parametrize("ctype", ["text/plain", "", "application/octet-stream"])
def test_a_journal_page_cannot_hide_behind_its_content_type(ctype, monkeypatch) -> None:
    page = ('<html><head><meta name="citation_doi" content="10.1/x"></head><body>'
            + "paywalled article text " * 200 + "</body></html>")

    def handler(req):
        return httpx.Response(200, headers={"content-type": ctype} if ctype else {}, content=page.encode())

    _mock_client(monkeypatch, handler)
    doc = RetrievedDoc(content="s", metadata={"url": "https://unlisted-publisher.example/read/77"})
    assert kn._fetch_web_page_text(doc, timeout_s=5, max_kb=64, headless=False) is None
    assert doc.metadata["scholarly_page"] is True


@pytest.mark.parametrize("tag", [
    '<meta name="citation_doi" content="10.1/x">',
    "<meta property='citation_doi' content='10.1/x'>",
    '<meta content="10.1/x" name="citation_doi">',
    "<meta name=citation_doi content=10.1/x>",
    '<meta name="prism.doi" content="10.1/x">',
    '<meta content="10.1234/x" name="dc.identifier">',
    '<script type="application/ld+json">{"@type": ["Article", "MedicalScholarlyArticle"]}</script>',
])
def test_the_metadata_check_reads_the_common_spellings(tag) -> None:
    assert kn._has_scholarly_meta(f"<html><head>{tag}</head></html>") is True


def test_the_metadata_check_ignores_an_ordinary_page() -> None:
    assert kn._has_scholarly_meta('<html><head><meta name="description" content="hello"></head></html>') is False


@pytest.mark.parametrize("doi,expected", [
    ("10.1101/068312", True),
    ("https://doi.org/10.1101/2020.01.01.123456", True),
    ("10.1101/gad.123456", False),
])
def test_biorxiv_doi_shapes(doi, expected) -> None:
    assert bool(kn._BIORXIV_DOI_RE.match(doi)) is expected
    assert is_open_access({"doi": doi}) is expected
