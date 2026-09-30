"""scripts/wiki_sync_index.py: the index lists every page once, in English, and the repo's wiki is up to date."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wiki_sync_index", ROOT / "scripts" / "wiki_sync_index.py")
wsi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsi)


def _page(wiki: Path, name: str, text: str) -> None:
    (wiki / name).write_text(text, encoding="utf-8")


def test_every_page_is_listed_once_with_its_title_and_the_header_is_english(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _page(wiki, "a.md", "---\ntitle: Alpha page\n---\n# Alpha\n")
    _page(wiki, "b.md", "# Beta heading\n\nSee [[a|Alpha page]].\n")
    _page(wiki, "c.md", "---\ntitle: Gamma\ntype: concept\n---\nbody\n")
    _page(wiki, "log.md", "# Wiki Log\n")
    assert wsi.main(["--wiki-dir", str(wiki)]) == 0
    index = (wiki / "index.md").read_text(encoding="utf-8")
    assert index.startswith("# Wiki Index\n\n> Rebuilt by `scripts/wiki_sync_index.py`.")
    assert index.count("[[a|Alpha page]]") == 1 and index.count("[[b|Beta heading]]") == 1
    assert "## Concept (1)" in index and "[[c|Gamma]]" in index
    assert "log" not in index.split("pages.")[1]
    assert all(ord(ch) < 128 for ch in index)
    assert wsi.main(["--wiki-dir", str(wiki), "--check"]) == 0


def test_check_reports_a_stale_index_and_a_link_by_title(tmp_path: Path, capsys) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _page(wiki, "a.md", "# Alpha\n\n[[Some Title]]\n")
    assert wsi.main(["--wiki-dir", str(wiki), "--check"]) == 1
    out = capsys.readouterr().out
    assert "broken link: a.md: [[Some Title]]" in out and "out of date" in out


def test_a_link_in_a_table_cell_and_types_that_differ_by_case(tmp_path: Path, capsys) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    _page(wiki, "a.md", "---\ntype: Concept\n---\n# A\n\n| x | y |\n|---|---|\n| 1 | [[b\\|Bee]] |\n")
    _page(wiki, "b.md", "---\ntype: concept\n---\n# B\n")
    _page(wiki, "c.md", "---\ntype: CONCEPT\n---\n# C\n")
    assert wsi.broken_links(wiki) == []
    index = wsi.render(wiki)
    assert index.count("## Concept (3)") == 1 and "## Concept (" not in index.replace("## Concept (3)", "")


def test_a_bom_does_not_hide_the_front_matter(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "a.md").write_bytes(b"\xef\xbb\xbf---\ntitle: Alpha\n---\n# Other\n")
    assert wsi.pages(wiki) == [("a", "Alpha", "Pages")]


def test_crlf_pages_read_like_lf(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "a.md").write_bytes(b"---\r\ntitle: Alpha\r\n---\r\n# Other\r\n")
    assert wsi.pages(wiki) == [("a", "Alpha", "Pages")]


def test_the_repository_wiki_index_is_current_and_has_no_broken_links() -> None:
    assert wsi.main(["--wiki-dir", str(ROOT / "wiki"), "--check"]) == 0
