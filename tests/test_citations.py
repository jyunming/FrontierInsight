"""Unit tests for the machine-readable citation export (core/citations.py).

Drives the renderers off the same dict shape ``build_references()`` produces:
``title / authors / year / venue / doi / arxiv_id / url / site / source``.
"""

from __future__ import annotations

import json

from core.citations import to_bibtex, to_csl_json

# A representative reference list spanning the entry types we emit.
_REFS = [
    {  # journal article: DOI + venue → @article / article-journal
        "n": 1,
        "title": "Overlay metrology for the 7nm node",
        "authors": ["Smith, Jane A.", "Liu, Wei"],
        "year": "2021",
        "venue": "Proc. SPIE",
        "doi": "10.1117/12.2222",
        "url": "https://doi.org/10.1117/12.2222",
        "source": "crossref",
    },
    {  # bare arXiv preprint → @misc with eprint/archivePrefix
        "n": 2,
        "title": "A Transformer for Edge Placement Error",
        "authors": ["Alex Doe"],
        "year": "2023",
        "arxiv_id": "2301.01234",
        "url": "https://arxiv.org/abs/2301.01234",
        "source": "arxiv",
    },
    {  # web page → @misc / webpage
        "n": 3,
        "title": "Measuring accuracy & precision in lithography",
        "authors": [],
        "year": "",
        "url": "https://www.asml.com/overlay",
        "site": "asml.com",
        "source": "web_search",
    },
]


def test_bibtex_entry_types_and_fields() -> None:
    bib = to_bibtex(_REFS)
    # Journal article.
    assert "@article{" in bib
    assert "journal = {Proc. SPIE}" in bib
    assert "doi = {10.1117/12.2222}" in bib
    # arXiv preprint.
    assert "eprint = {2301.01234}" in bib
    assert "archivePrefix = {arXiv}" in bib
    # Web page.
    assert "@misc{" in bib
    assert "howpublished" in bib
    # Title case preserved via extra braces.
    assert "title = {{Overlay metrology for the 7nm node}}" in bib


def test_bibtex_keys_unique_and_stable() -> None:
    # Two refs that would collapse to the same base key get a/b suffixes.
    dupes = [
        {"title": "Overlay budget", "authors": ["Smith, J."], "year": "2020"},
        {"title": "Overlay budget", "authors": ["Smith, J."], "year": "2020"},
    ]
    bib = to_bibtex(dupes)
    keys = [line.split("{", 1)[1].rstrip(",")
            for line in bib.splitlines() if line.startswith("@")]
    assert len(keys) == len(set(keys)) == 2  # unique
    assert keys[0] == "smith2020overlay"
    assert keys[1] == "smith2020overlaya"


def test_bibtex_escapes_special_chars() -> None:
    bib = to_bibtex([_REFS[2]])  # title has a raw '&'
    assert r"\&" in bib
    assert " & " not in bib.replace(r"\&", "")  # no unescaped ampersand left


def test_bibtex_empty_is_empty_string() -> None:
    assert to_bibtex([]) == ""


def test_csl_json_parses_and_maps_fields() -> None:
    items = json.loads(to_csl_json(_REFS))
    assert isinstance(items, list) and len(items) == 3
    art = items[0]
    assert art["type"] == "article-journal"
    assert art["title"] == "Overlay metrology for the 7nm node"
    assert art["DOI"] == "10.1117/12.2222"
    assert art["issued"] == {"date-parts": [[2021]]}
    # Author name split into family/given.
    assert {"family": "Smith", "given": "Jane A."} in art["author"]
    # Web page maps to the webpage type.
    assert items[2]["type"] == "webpage"
    # arXiv-only preprint maps to article.
    assert items[1]["type"] == "article"


def test_csl_json_unique_ids_match_bibtex_keys() -> None:
    items = json.loads(to_csl_json(_REFS))
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids))
    # The id scheme matches the BibTeX cite key (firstauthor+year+word).
    assert ids[0] == "smith2021overlay"


def test_authors_accepts_delimited_string() -> None:
    # Some adapters store authors as one string — handle it.
    ref = {"title": "X", "authors": "Smith, J.; Liu, W.", "year": "2022"}
    items = json.loads(to_csl_json([ref]))
    assert len(items[0]["author"]) == 2


def test_authors_accepts_stringified_list() -> None:
    """A repr'd Python list (how on-disk literature serializes authors) is
    parsed back into separate authors, not rendered as one bogus name."""
    ref = {"title": "Overlay", "authors": "['Eren Canga', 'Victor M. Blanco']",
           "year": "2024", "doi": "10.1/x", "venue": "SPIE"}
    bib = to_bibtex([ref])
    assert "author = {Eren Canga and Victor M. Blanco}" in bib
    assert "['" not in bib  # no list-repr leakage
    items = json.loads(to_csl_json([ref]))
    assert len(items[0]["author"]) == 2
