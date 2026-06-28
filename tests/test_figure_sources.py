"""Topic-figure enrichment acquisition (core/figure_sources.py).

Offline + deterministic: httpx is monkeypatched, so no network is hit. The
license gate, Commons-API parsing, and arXiv-figure extraction are covered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import core.figure_sources as fs


class _Resp:
    def __init__(self, status=200, text="", content=b"", headers=None, json_data=None):
        self.status_code = status
        self.text = text
        self.content = content or text.encode()
        self.headers = headers or {}
        self._json = json_data

    def json(self):
        return self._json


def test_is_free_license_allows_cc_and_pd_rejects_nc_nd() -> None:
    for ok in ("CC BY 4.0", "CC BY-SA 3.0", "cc0", "Public domain", "CC-BY 2.0"):
        assert fs._is_free_license(ok), ok
    for bad in ("CC BY-NC 4.0", "CC BY-ND 4.0", "CC BY-NC-SA 3.0", "", "All rights reserved"):
        assert not fs._is_free_license(bad), bad


def test_fetch_commons_filters_license_and_mime(monkeypatch, tmp_path: Path) -> None:
    payload = {"query": {"pages": {
        "1": {"title": "File:Free diagram.png", "imageinfo": [{
            "mime": "image/png", "thumburl": "https://up.example/free.png",
            "descriptionurl": "https://commons.example/free",
            "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "<a href=x>Jane Doe</a>"}},
        }]},
        "2": {"title": "File:Paywalled.png", "imageinfo": [{
            "mime": "image/png", "thumburl": "https://up.example/nc.png",
            "extmetadata": {"LicenseShortName": {"value": "CC BY-NC 4.0"}},
        }]},
        "3": {"title": "File:Doc.pdf", "imageinfo": [{
            "mime": "application/pdf", "thumburl": "https://up.example/d.png",
            "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
        }]},
    }}}

    def fake_get(url, **kw):
        if "api.php" in url:
            return _Resp(json_data=payload)
        return _Resp(content=b"\x89PNG\r\n\x1a\nrealbytes", headers={"content-type": "image/png"})

    monkeypatch.setattr(fs.httpx, "get", fake_get)
    figs = fs.fetch_commons_figures("diagram", out_dir=tmp_path, max_n=3)
    assert len(figs) == 1                       # only the CC BY-SA image kept
    f = figs[0]
    assert f.license == "CC BY-SA 4.0"
    assert f.attribution == "Jane Doe"          # HTML stripped
    assert f.kind == "commons" and f.local_path.is_file()


def test_fetch_arxiv_figure_requires_cc_license(monkeypatch, tmp_path: Path) -> None:
    abs_noncc = "<html>arXiv perpetual non-exclusive license</html>"

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            return _Resp(text=abs_noncc)

    monkeypatch.setattr(fs.httpx, "Client", _Client)
    # No CC license on the abstract page → skip (default arXiv license).
    assert fs.fetch_arxiv_figure("2404.09143", "topic", out_dir=tmp_path) is None


def test_fetch_arxiv_figure_picks_relevant_caption(monkeypatch, tmp_path: Path) -> None:
    abs_cc = '<a href="http://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>'
    html = (
        '<figure><img src="extracted/x/unrelated.png">'
        '<figcaption class="ltx_caption">An unrelated appendix table</figcaption></figure>'
        '<figure><img src="extracted/x/methods.png">'
        '<figcaption class="ltx_caption">Exoplanet detection methods overview</figcaption></figure>'
    )

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            return _Resp(text=(abs_cc if "/abs/" in url else html))

    monkeypatch.setattr(fs.httpx, "Client", _Client)
    monkeypatch.setattr(
        fs, "_download_image",
        lambda url, *, timeout_s: b"\x89PNG\r\n" if "methods.png" in url else b"\x89PNG\r\n",
    )
    f = fs.fetch_arxiv_figure("2404.09143", "exoplanet detection methods", out_dir=tmp_path)
    assert f is not None
    assert f.license == "CC BY 4.0"
    assert "detection methods" in f.caption          # the relevant figure won
    assert f.kind == "oa_paper" and f.attribution == "arXiv:2404.09143"


def test_collect_topic_figures_combines_sources(monkeypatch, tmp_path: Path) -> None:
    made = []

    def fake_arxiv(aid, query, *, out_dir, timeout_s=25.0):
        p = out_dir / "a.png"; p.write_bytes(b"x"); made.append("arxiv")
        return fs.WebFigure(p, "cap", "u", "CC BY 4.0", "arXiv:1", "oa_paper")

    def fake_commons(query, *, out_dir, max_n, timeout_s=25.0):
        made.append(f"commons:{max_n}")
        return [fs.WebFigure(out_dir / "c.png", "c", "u", "CC0", "x", "commons")]

    monkeypatch.setattr(fs, "fetch_arxiv_figure", fake_arxiv)
    monkeypatch.setattr(fs, "fetch_commons_figures", fake_commons)
    figs = fs.collect_topic_figures("t", ["2404.09143"], out_dir=tmp_path, max_n=3)
    assert [f.kind for f in figs] == ["oa_paper", "commons"]
    assert made == ["arxiv", "commons:2"]   # arXiv first, Commons fills remaining
