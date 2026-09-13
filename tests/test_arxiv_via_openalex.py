"""arXiv is searched through OpenAlex, and scholarly API keys reach the adapters.

arXiv's own query API (export.arxiv.org/api/query) answered HTTP 429 to a
single request made hours after the last one, and from a second, unrelated IP
too — capacity throttling on arXiv's side, not something a client can fix.
Meanwhile its abstract/PDF pages answered 200. So the "arxiv" source now
searches OpenAlex's arXiv repository (S4306400194) and full text still comes
from arxiv.org. OpenAlex itself has required a key for its full daily budget
since February 2026; Semantic Scholar's keyless pool mostly 429s. Both keys
can be set in YAML or the environment.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

import core.knowledge as kn
from core.config import KnowledgeConfig
from core.knowledge import Knowledge


class _Recorder:
    """Fake httpx.Client that records every GET and returns a JSON payload."""

    calls: list[dict] = []

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        _Recorder.calls = []

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def get(self, url, params=None, headers=None, **_):
        _Recorder.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        r = MagicMock()
        r.json.return_value = self.payload
        r.raise_for_status = MagicMock()
        return r


def _work(**over) -> dict:
    w = {
        "id": "https://openalex.org/W1",
        "title": "Symplectic integrators\\n for damped oscillators",
        "doi": "https://doi.org/10.48550/arXiv.2401.01234",
        "publication_year": 2024,
        "publication_date": "2024-01-03",
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "abstract_inverted_index": {"energy": [0], "drift": [1]},
        "primary_location": {"landing_page_url": None, "pdf_url": None,
                             "source": {"display_name": "arXiv (Cornell University)"}},
    }
    w.update(over)
    return w


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "")


def test_arxiv_search_never_contacts_export_arxiv(monkeypatch) -> None:
    monkeypatch.setattr("core.knowledge.httpx.Client", _Recorder({"results": [_work()]}))
    docs = kn._arxiv_search("symplectic integrator", 5)
    assert docs
    hosts = [c["url"] for c in _Recorder.calls]
    assert hosts == ["https://api.openalex.org/works"]
    assert not any("arxiv.org" in h for h in hosts)


def test_arxiv_id_comes_from_a_doi_only_record(monkeypatch) -> None:
    """Some OpenAlex arXiv records carry the DOI but no landing or PDF URL.
    The id must still be recovered, or full text and BibTeX eprint are lost."""
    monkeypatch.setattr("core.knowledge.httpx.Client", _Recorder({"results": [_work()]}))
    (doc,) = kn._arxiv_search("x", 5)
    md = doc.metadata
    assert md["arxiv_id"] == "2401.01234"
    assert md["url"] == "https://arxiv.org/abs/2401.01234"
    assert md["pdf_url"] == "https://arxiv.org/pdf/2401.01234"
    assert md["venue"] == "arXiv" and md["open_access"] is True
    assert md["title"] == "Symplectic integrators for damped oscillators", "literal \\n cleaned"


@pytest.mark.parametrize("text,expected", [
    ("https://doi.org/10.48550/arxiv.hep-th/9701001", "hep-th/9701001"),
    ("http://arxiv.org/abs/2106.09685v3", "2106.09685"),
    ("https://arxiv.org/pdf/0910.3621", "0910.3621"),
])
def test_arxiv_id_forms(text: str, expected: str) -> None:
    assert kn._arxiv_id_from_openalex({"doi": text}) == expected


def test_openalex_sends_the_key_only_when_configured(monkeypatch) -> None:
    rec = _Recorder({"results": []})
    monkeypatch.setattr("core.knowledge.httpx.Client", rec)
    kn._openalex_search("q", 5)
    assert "api_key" not in _Recorder.calls[0]["params"]

    monkeypatch.setenv("OPENALEX_API_KEY", "OA-KEY")
    kn._openalex_search("q", 5)
    kn._arxiv_search("q", 5)
    assert _Recorder.calls[1]["params"]["api_key"] == "OA-KEY"
    assert _Recorder.calls[2]["params"]["api_key"] == "OA-KEY"


def test_semantic_scholar_sends_its_key_as_a_header(monkeypatch) -> None:
    monkeypatch.setattr("core.knowledge.httpx.Client", _Recorder({"data": []}))
    kn._semantic_scholar_search("q", 5)
    assert "x-api-key" not in _Recorder.calls[0]["headers"]
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "S2-KEY")
    kn._semantic_scholar_search("q", 5)
    assert _Recorder.calls[1]["headers"]["x-api-key"] == "S2-KEY"


def test_yaml_keys_reach_the_environment_without_overriding_it(monkeypatch) -> None:
    Knowledge(KnowledgeConfig(enabled=False, openalex_api_key="FROM-YAML",
                              semantic_scholar_api_key="S2-YAML"))
    assert os.environ["OPENALEX_API_KEY"] == "FROM-YAML"
    assert os.environ["SEMANTIC_SCHOLAR_API_KEY"] == "S2-YAML"

    monkeypatch.setenv("OPENALEX_API_KEY", "FROM-ENV")
    Knowledge(KnowledgeConfig(enabled=False, openalex_api_key="OTHER-YAML"))
    assert os.environ["OPENALEX_API_KEY"] == "FROM-ENV", "a real env var wins"


def test_config_reads_keys_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", " OA ")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "S2")
    cfg = KnowledgeConfig()
    assert cfg.openalex_api_key == "OA"
    assert cfg.semantic_scholar_api_key == "S2"


def test_the_key_never_reaches_a_failure_log(monkeypatch) -> None:
    import httpx
    from core import source_failures as sf

    class _Limited:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, params=None, **_):
            return httpx.Response(429, request=httpx.Request("GET", url, params=params))

    monkeypatch.setattr("core.knowledge.httpx.Client", _Limited)
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRET-OA-KEY")
    token = sf.current_quest.set("q-key")
    try:
        assert kn._arxiv_search("q", 5) == []
        snap = sf.snapshot("q-key")
    finally:
        sf.current_quest.reset(token)
        sf.reset("q-key")
    assert snap["by_source"] == {"arxiv": {"http_429": 1}}
    assert "SECRET-OA-KEY" not in str(snap)
