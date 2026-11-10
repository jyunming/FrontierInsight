"""Citations are written as ``{{cite:N}}`` (the label of the prior-work block) and rendered by FI; nothing is guessed from
the characters before a bracket in a paper written that way."""
from __future__ import annotations

from pathlib import Path

from core import quality_report
from core.cite_labels import REMOVED_RECORD, write_removed
from core.engine import _finalize_paper_sources, _renumber_citations


def _paper(title: str, doi: str) -> dict:
    return {"content": f"{title}. An abstract.", "metadata": {
        "source": "crossref", "title": title, "doi": doi, "authors": ["A. Author"], "year": 2020, "venue": "J. Tests"}}


def _web(title: str, url: str) -> dict:
    return {"content": f"{title} page text", "metadata": {
        "source": "web_search", "kind": "web_page", "title": title, "url": url, "site": url.split("/")[2]}}


LIT = [_paper("Alpha", "10.1/a"), _web("Page one", "https://one.example/p"),
       _paper("Beta", "10.1/b"), _paper("Gamma", "10.1/c")]


def test_placeholder_renders_to_the_right_number_and_renumbers_by_first_use() -> None:
    out, mapping, dropped = _renumber_citations(
        "First {{cite:3}}, then {{cite:1, 3}}, then {{cite: 2}} and again {{cite:3}}.", 3)
    assert out == "First [1], then [1, 2], then [3] and again [1]."
    assert mapping == {3: 1, 1: 2, 2: 3} and dropped == []


def test_name_bracket_in_a_new_paper_is_not_guessed() -> None:
    """Old code read ``Smith[2]`` as an index by the letter before it and kept a number that no longer meant anything
    once the others were renumbered; a paper that cites by placeholder owns its brackets, and FI renders its own
    citation with a space so no later pass reads it as an index either."""
    out, mapping, dropped = _renumber_citations("Smith[2] showed it{{cite:3}}, and {{cite:2}}.", 3)
    assert out == "Smith[2] showed it [1], and [2]."
    assert mapping == {3: 1, 2: 2} and dropped == []


def test_unknown_label_is_removed_and_listed_not_numbered() -> None:
    out, mapping, dropped = _renumber_citations("A {{cite:9}} B {{cite:1, 7}} C {{cite:abc}} D {{cite:W5}}.", 3, {"W1"})
    assert out == "A B [1] C D."
    assert mapping == {1: 1}
    assert dropped == [9, 7, "abc", "W5"]


def test_array_indexing_in_text_is_untouched() -> None:
    body = "The value cell_values[0] and x[0] and m[1][4] hold {{cite:2}}; `y{{cite:1}}` and $a_{{cite:1}}$ stay."
    out, mapping, _ = _renumber_citations(body, 3)
    assert out == "The value cell_values[0] and x[0] and m[1][4] hold [1]; `y{{cite:1}}` and $a_{{cite:1}}$ stay."
    assert mapping == {2: 1}


def test_a_paper_with_no_placeholder_is_read_by_the_old_rules() -> None:
    out, mapping, dropped = _renumber_citations("As shown [3] and [1, 2]; weights[2,3] keeps its index.", 3)
    assert out == "As shown [1] and [2, 3]; weights[2,3] keeps its index."
    assert mapping == {3: 1, 1: 2, 2: 3} and dropped == []


def test_finalize_lists_the_cited_sources_in_order_of_first_use() -> None:
    md = "# T\n\n## Intro\nBeta {{cite:2}}, alpha {{cite:1}}, a page {{cite:W1}}, none {{cite:8}}.\n"
    text, ordered, dropped = _finalize_paper_sources(md, LIT, "researcher")
    assert "Beta [1], alpha [2], a page [W1], none." in text
    assert dropped == [8]
    assert ordered[0]["metadata"]["title"] == "Beta"
    refs = text.split("## References")[1]
    assert refs.index("Beta") < refs.index("Alpha") and "Gamma" not in refs


def test_removed_labels_are_listed_in_the_quality_report(tmp_path: Path) -> None:
    write_removed(tmp_path, ["9", "ABC"])
    q = quality_report._Quest(tmp_path)
    assert q.citations_removed == ["9", "ABC"]
    texts = [p["text"] for p in quality_report._paper_problems(q)]
    assert any("[9]" in t and "taken out" in t for t in texts) and any("[ABC]" in t for t in texts)
    write_removed(tmp_path, [])
    assert not (tmp_path / REMOVED_RECORD).exists()
