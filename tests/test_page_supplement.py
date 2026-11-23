"""FI's own parts go to the supplement, in a fixed order, without a model call.

Every paper, with or without a page limit, keeps FI's tables of every setting (a pointer stays) and its Further reading
(the References keep one line saying how many pages it lists and where) in ``paper/supplement.md``; the References stay.
FI writes no Limitations block, so nothing more of FI's moves for a page limit (the writer's own Limitations stay in the
paper). The decision is a record FI's blocks follow, so a later write of the blocks keeps it; a new draft gets
everything back and is decided again, the parts every paper keeps in the supplement going there at once. A record an
older FI wrote, whose Limitations moved with the writer's own paragraphs, gives those paragraphs back.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core import paper_facts as pf
from core import supplement as sup
from core.engine import Engine, QuestArtifacts
from tests.test_page_limit import _engine, _quiet_log, _review

TABLE = "| setting | value |\n|---|---|\n| a = 1 | 0.51 ± 0.02 |\n"
TOPIC_PART = "the second factor the topic asked to vary"
DRAFT = (
    "# Outbreaks in a Small Population\n\n## Abstract\n\nOne paragraph.\n\n## Results\n\nText [1].\n\n"
    "{{results_table}}\n\n## Discussion\n\nWhat it means.\n\n## Limitations\n\nThe study held the second factor fixed.\n\n"
    "## References\n\n1. Kermack, W. O. (1927). A contribution.\n\n"
    "## Further reading\n\n- [W1] Page 1. https://example.org/1\n- [W2] Page 2. https://example.org/2\n"
)


def _records(root: Path) -> None:
    (root / ".fi").mkdir(parents=True, exist_ok=True)
    (root / ".fi" / "results_table.md").write_text(TABLE, encoding="utf-8")
    (root / ".fi" / "topic_covered.json").write_text(json.dumps({"not_done": [
        {"kind": "other", "name": TOPIC_PART, "sentence": "The study held the second factor fixed."}]}), encoding="utf-8")


def _put(eng: Engine, root: Path, text: str = DRAFT) -> Path:
    _records(root)
    paper = root / "paper" / "paper.md"
    paper.write_text(text, encoding="utf-8")
    eng._rewrite_fi_blocks({}, "test", path=paper)  # FI writes its blocks, as after any write
    return paper


def _pages(monkeypatch, base: int, *, fail_at: int | None = None) -> list[str]:  # noqa: ANN001
    """Stand in for the render: ``base`` pages, +2 with FI's full tables, +1 with the writer's Limitations (they never
    move), +1 with the Further reading. Returns the text each render was given."""
    seen: list[str] = []

    async def fake(self, paper_path, state):  # noqa: ANN001, ARG001
        text = Path(paper_path).read_text(encoding="utf-8")
        seen.append(text)
        if fail_at is not None and len(seen) == fail_at:
            return None
        pages = base + 2 * ("| a = 1 |" in text) + ("held the second factor fixed" in text) + ("## Further reading" in text)
        return {"pages": pages, "words": 1000, "last_page_lines": 3, "last_page_empty": 0.5}

    monkeypatch.setattr(Engine, "_measure_draft_pages", fake)
    return seen


def _model_calls(eng: Engine) -> list[str]:
    calls: list[str] = []
    answer = eng._chat

    async def spy(prompt, *, node=None):  # noqa: ANN001
        calls.append(str(node))
        return await answer(prompt, node=node)

    eng._chat = spy  # type: ignore[method-assign]
    return calls


def test_the_parts_every_paper_keeps_in_the_supplement_move_before_the_page_check(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    calls = _model_calls(eng)
    seen = _pages(monkeypatch, 3)  # 3 + 2 + 1 + 1 = 7 with everything in; tables and Further reading out: 4, the limit
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert patch["review"]["page_limit"] == {"pages": 4, "limit": 4}, "the page check moved nothing more"
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    assert "| a = 1 |" not in text and sup.table_pointer() in text
    assert "## Further reading" not in text and sup.further_pointer(2) in text
    assert "held the second factor fixed" in text, "the Limitations stay: the paper fits without moving them"
    assert "## References\n\n1. Kermack, W. O. (1927). A contribution." in text
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert "| a = 1 |" in shown and "- [W1] Page 1. https://example.org/1" in shown
    assert len(seen) == 1, "measured once, with the two parts already out"
    assert [c for c in calls if c != "review"] == [], "no model call for the move"


def test_a_paper_still_over_with_everything_moved_keeps_them_moved(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 6)  # tables and Further reading out: 7, still over, and nothing more of FI's to move
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert "supplement" not in patch["review"]["page_limit"]
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    assert patch["review"]["page_limit"]["pages"] == 7
    assert "held the second factor fixed" in text, "the writer's Limitations stay"
    assert "## Further reading" not in text and "## References" in text
    supplement = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert "- [W1] Page 1. https://example.org/1" in supplement and "- [W2] Page 2." in supplement
    assert sup.further_labels(tmp_path) == ["W1", "W2"]
    said = pf.page_limit_text(tmp_path)
    assert "over the 4-page limit" in said and "its Further reading in the supplement" in said


def test_a_paper_within_the_limit_moves_only_what_every_paper_moves(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    seen = _pages(monkeypatch, 0)  # 1 page with the two parts out
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert len(seen) == 1 and patch["review"]["page_limit"] == {"pages": 1, "limit": 4}
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    assert "held the second factor fixed" in text and "| a = 1 |" not in text


NO_LIMIT = "Outbreaks in a small population, written up as a paper."


def test_without_a_page_limit_the_tables_and_further_reading_go_to_the_supplement(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path)
    calls = _model_calls(eng)
    seen = _pages(monkeypatch, 0)
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert "page_limit" not in patch["review"] and seen == [], "no limit: nothing is rendered"
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    assert sup.read(tmp_path)["limit"] is None
    assert "| a = 1 |" not in text and sup.table_pointer() in text
    assert "## Further reading" not in text and "[W1]" not in text and sup.further_pointer(2) in text
    assert "held the second factor fixed" in text, "the Limitations move only for a page limit"
    assert "## References\n\n1. Kermack, W. O. (1927). A contribution." in text
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert "| a = 1 |" in shown and "- [W1] Page 1." in shown and "- [W2] Page 2." in shown
    assert "page limit" not in shown and "always keeps" in shown
    assert "page limit" not in pf.page_limit_text(tmp_path)
    assert "FI keeps its tables of every setting and its Further reading" in pf.page_limit_text(tmp_path)
    assert [c for c in calls if c != "review"] == [], "no model call for the move"


def test_without_a_page_limit_a_new_draft_after_a_reset_keeps_them_out(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    # the write step's reset gives the writer everything back...
    eng._supplement_reset({}, paper)
    text = paper.read_text(encoding="utf-8")
    assert "| a = 1 |" in text and "- [W2] Page 2." in text and not sup.FURTHER_POINTER.search(text)
    assert not (tmp_path / sup.RECORD).exists()
    # ...and the new draft it writes goes through the review: both parts go out again
    _put(eng, tmp_path, DRAFT.replace("What it means.", "What the new draft says it means."))
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert "What the new draft says it means." in text
    assert "| a = 1 |" not in text and "## Further reading" not in text and sup.further_pointer(2) in text
    assert text.count("Further reading: ") == 1


def test_the_paper_as_delivered_keeps_them_out_without_a_page_limit(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    eng.config.engine.claim_grounding = False  # the citation reading itself is not what this checks
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    # a later step (the citation check's renumbering) rebuilt the References and started a new Further reading list
    text = paper.read_text(encoding="utf-8")
    text = sup.FURTHER_POINTER.sub("", text).rstrip() + "\n\n## Further reading\n\n- [W3] Page 3. https://example.org/3\n"
    paper.write_text(text, encoding="utf-8")
    asyncio.run(eng._node_citation_check({"topic": eng.config.topic, "paper_md": str(paper), "review": {}}))  # type: ignore[arg-type]
    text = paper.read_text(encoding="utf-8")
    assert "## Further reading" not in text and "| a = 1 |" not in text
    assert text.count("Further reading: ") == 1 and sup.further_pointer(3) in text
    assert sup.further_labels(tmp_path) == ["W1", "W2", "W3"]
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert all(f"- [W{n}] Page {n}." in shown for n in (1, 2, 3))


def test_a_stop_between_the_move_and_the_supplement_write_still_delivers_the_file(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path)
    # FI stopped after it moved the parts (the record and the pointers are there) and before it wrote the file
    for part in sup.ALWAYS:
        moved = [p for p in sup.ALWAYS if p == part or p in sup.moved(tmp_path)]
        eng._supplement_move({}, paper, part, paper.read_text(encoding="utf-8"), moved, None)
    (tmp_path / sup.FILE).unlink(missing_ok=True)
    text = paper.read_text(encoding="utf-8")
    assert sup.table_pointer() in text and sup.further_pointer(2) in text
    # the resumed step finds nothing new to move, and still leaves the file the pointers name
    assert eng._supplement_always({}, paper, None) == []
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert "| a = 1 |" in shown and "- [W1] Page 1." in shown and "- [W2] Page 2." in shown
    assert paper.read_text(encoding="utf-8") == text
    # written as it should be, it is not written again
    before = (tmp_path / sup.FILE).stat().st_mtime_ns
    real = Path.write_text

    def no_rewrite(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        assert self != tmp_path / sup.FILE, "the supplement was written again with the same text"
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", no_rewrite)
    eng._supplement_always({}, paper, None)
    assert (tmp_path / sup.FILE).stat().st_mtime_ns == before


def test_a_supplement_page_the_text_comes_to_cite_leaves_the_supplement(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    assert sup.further_labels(tmp_path) == ["W1", "W2"]
    # a fix of the citation check cites W2 instead of a source that did not say it; the References now list it
    text = paper.read_text(encoding="utf-8").replace("Text [1].", "Text [W2].")
    text = text.replace("1. Kermack, W. O. (1927). A contribution.", "- [W2] Page 2. https://example.org/2")
    paper.write_text(text, encoding="utf-8")
    eng._supplement_always({}, paper, None)
    text = paper.read_text(encoding="utf-8")
    assert sup.further_labels(tmp_path) == ["W1"] and sup.further_pointer(1) in text and text.count("Further reading: ") == 1
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert "- [W1] Page 1." in shown and "[W2]" not in shown


def test_the_citation_check_may_cite_a_page_that_is_in_the_supplement(tmp_path: Path, monkeypatch) -> None:
    from core import citation_support as cs

    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    assert "[W1]" not in paper.read_text(encoding="utf-8")
    seen: dict[str, Any] = {}
    real = cs.Source

    def spy(*a, **k):  # noqa: ANN002, ANN003
        src = real(*a, **k)
        seen[src.label] = src.usable
        return src

    monkeypatch.setattr(cs, "Source", spy)
    web = {"content": "Page 1 text.", "metadata": {"source": "web_search", "kind": "web_page", "title": "Page 1",
                                                   "url": "https://example.org/1", "site": "example.org"}}
    state = {"topic": eng.config.topic, "paper_md": str(paper), "review": {}, "literature": [web]}
    asyncio.run(eng._check_citations(state))  # type: ignore[arg-type]
    assert seen.get("W1") is True, "a page the paper lists in its supplement stays usable for a fix"


def test_a_tables_block_without_a_table_stays_where_it_is(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path)
    note = "Results for all 40 settings are in the supplementary table `paper/results_by_setting.csv`."
    (tmp_path / ".fi" / "results_table.md").write_text(note + "\n", encoding="utf-8")
    eng._rewrite_fi_blocks({}, "test", path=paper)
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert sup.moved(tmp_path) == ["further_reading"]
    assert note in text and sup.table_pointer() not in text, "the sentence already points to the files"


def test_the_supplement_says_why_each_part_is_there() -> None:
    both = sup.text(None, {sup.TABLES: "| a |", sup.FURTHER_READING: "- [W1] A page."})
    assert "always keeps" in both and "page limit" not in both
    limited = sup.text(4, {sup.TABLES: "| a |", sup.LIMITATIONS: "- One thing."})
    assert "always keeps the paper's tables of every setting here" in limited
    assert "The Limitations in full moved here to keep the paper within its page limit of 4 pages." in limited
    assert sup.further_pointer(1) == ("Further reading: 1 web page the text does not cite is listed in the supplement "
                                      "(`paper/supplement.md`).")
    assert sup.FURTHER_POINTER.fullmatch(sup.further_pointer(12)) and sup.further_pointer(0) == ""


def test_a_new_draft_is_decided_again_with_everything_back(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    _review(eng, tmp_path)
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    # A shorter draft: it fits; the parts every paper keeps in the supplement go there again.
    _pages(monkeypatch, 0)
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert patch["review"]["page_limit"] == {"pages": 1, "limit": 4}
    assert "held the second factor fixed" in text
    assert "| a = 1 |" not in text and "## Further reading" not in text and sup.further_pointer(2) in text
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]


def test_a_later_write_of_fis_blocks_keeps_the_move(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 4)
    _review(eng, tmp_path)
    # A record changes after the review (FI's tables are written again) and FI's blocks are written again.
    (tmp_path / ".fi" / "results_table.md").write_text(TABLE.replace("0.51", "0.73"), encoding="utf-8")
    eng._rewrite_fi_blocks({}, "test", path=paper)
    eng._write_supplement({}, 4)
    text = paper.read_text(encoding="utf-8")
    assert sup.table_pointer() in text and "| a = 1 |" not in text
    assert "0.73" in (tmp_path / sup.FILE).read_text(encoding="utf-8")


def test_further_reading_whose_page_the_text_cites_stays(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, DRAFT.replace("Text [1].", "Text [1, W2]."))
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    patch = _review(eng, tmp_path)
    assert "supplement" not in patch["review"]["page_limit"]
    assert sup.moved(tmp_path) == ["results_table"]
    text = paper.read_text(encoding="utf-8")
    assert "## Further reading" in text and "- [W1] Page 1." in text and not sup.FURTHER_POINTER.search(text)


def test_further_reading_whose_page_the_text_cites_stays_without_a_page_limit(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, topic=NO_LIMIT)
    paper = _put(eng, tmp_path, DRAFT.replace("Text [1].", "Text [1, W2]."))
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    assert sup.moved(tmp_path) == ["results_table"]
    text = paper.read_text(encoding="utf-8")
    assert "## Further reading" in text and "- [W1] Page 1." in text and not sup.FURTHER_POINTER.search(text)


def test_a_render_that_cannot_be_counted_leaves_everything_as_it_was(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    # what the page check starts from: the parts every paper keeps in the supplement already there
    assert eng._supplement_always({}, paper, 4) == ["results_table", "further_reading"]
    before = paper.read_bytes()
    record = sup.read(tmp_path)
    _quiet_log(eng)
    _pages(monkeypatch, 4, fail_at=2)  # a second render, if any, fails
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    assert paper.read_bytes() == before
    assert sup.read(tmp_path) == record and (tmp_path / sup.FILE).exists()
    assert "supplement" not in patch["review"]["page_limit"]


def test_the_exports_and_the_reviewer_see_what_moved(tmp_path: Path) -> None:
    from generation.paper import listed_further_reading

    further = [{"label": "W1"}, {"label": "W2"}, {"label": "W3"}]
    body = "# T\n\nBody [1].\n\n## References\n\n1. A paper.\n"
    assert [w["label"] for w in listed_further_reading(further, body + "- [W3] Cited page.\n", None)] == ["W3"]
    sup.save(tmp_path, [sup.FURTHER_READING], limit=4, further=["W1", "W2"], further_lines="- [W1] a\n- [W2] b")
    listed = listed_further_reading(further, body + "- [W3] Cited page.\n", None, tmp_path)
    assert [w["label"] for w in listed] == ["W1", "W2", "W3"]
    eng = _engine(tmp_path / "q")
    assert eng._supplement_for_review() == ""
    sup.write(eng.quest_root, 4, {sup.LIMITATIONS: "- One thing."})
    shown = eng._supplement_for_review()
    assert "- One thing." in shown and "unchanged" in shown


def test_the_supplement_ships_beside_the_paper(tmp_path: Path) -> None:
    from core.config import Config
    from generation.paper import PaperGenerator

    quest = tmp_path / "quest"
    (quest / "paper").mkdir(parents=True)
    md = quest / "paper" / "paper.md"
    md.write_text("# T\n\nBody.\n", encoding="utf-8")
    sup.write(quest, 4, {sup.LIMITATIONS: "- One <!-- fi:fact-seed -->7<!-- /fi:fact-seed --> thing."})
    cfg = Config.model_validate({"topic": "t", "title": "tt", "output": {"kinds": ["paper_md"],
                                                                          "output_dir": str(tmp_path / "o")}})
    art = QuestArtifacts(quest_id="q1", quest_root=quest, paper_md=md)
    art.raw_state = {"literature": []}
    result: dict[str, Any] = PaperGenerator(cfg).generate(art, tmp_path / "out")
    shipped = (tmp_path / "out" / "paper" / "supplement.md").read_text(encoding="utf-8")
    assert result["supplement_md"] == tmp_path / "out" / "paper" / "supplement.md"
    assert "- One 7 thing." in shipped and "fi:fact" not in shipped


WRITER = "Our own caveat about the small sample."


def test_an_older_record_gives_the_writers_own_limitations_paragraph_back(tmp_path: Path) -> None:
    """An older FI moved the writer's own Limitations paragraphs with its block for a page limit; the record it wrote is
    still read, so the next draft gets them back where they stood."""
    eng = _engine(tmp_path)
    _records(tmp_path)
    old_block = "<!-- fi:fact-limitations -->\nFI recorded 1 limitation; it is stated in the supplement.\n<!-- /fi:fact-limitations -->"
    left = DRAFT.replace("## Limitations\n\nThe study held the second factor fixed.\n\n", f"## Limitations\n\n{old_block}\n\n")
    whole = left.replace(f"## Limitations\n\n{old_block}", f"## Limitations\n\n{WRITER}\n\n{old_block}")
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text(left, encoding="utf-8")  # the paper as the older FI left it, its own block in the Limitations
    sup.save(tmp_path, [sup.LIMITATIONS], limit=4, writer=WRITER,
             section=(sup.section_body(whole) or "", sup.section_body(left) or ""))
    assert sup.writer_text(tmp_path) == WRITER
    assert eng._supplement_reset({}, paper) is True
    text = paper.read_text(encoding="utf-8")
    assert text.count(WRITER) == 1 and "## Limitations\n\n" + WRITER in text
    assert "fi:fact-limitations" not in text, "the older FI's block goes; the writer's paragraph stays"
    assert not (tmp_path / sup.RECORD).exists()


def _older_paper_with_moved_paragraphs(tmp_path: Path) -> Path:
    """A paper as an older FI left it for a page limit: its Limitations section holds only FI's pointer to the supplement,
    and the supplement's record holds the writer's own paragraphs that moved from there."""
    _records(tmp_path)
    old_block = ("<!-- fi:fact-limitations -->\nFI recorded 1 limitation of this study; each is stated in full in the "
                 "supplement (`paper/supplement.md`), with the paper's own discussion of them.\n<!-- /fi:fact-limitations -->")
    left = DRAFT.replace("## Limitations\n\nThe study held the second factor fixed.\n\n", f"## Limitations\n\n{old_block}\n\n")
    whole = left.replace(f"## Limitations\n\n{old_block}", f"## Limitations\n\n{WRITER}\n\n{old_block}")
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text(left, encoding="utf-8")
    sup.save(tmp_path, [sup.LIMITATIONS], limit=4, writer=WRITER,
             section=(sup.section_body(whole) or "", sup.section_body(left) or ""))
    return paper


def test_an_older_papers_moved_paragraphs_survive_the_strip_and_then_the_reset(tmp_path: Path) -> None:
    """The review path on a resumed paper takes FI's old block out first and resets the supplement after: the writer's
    own paragraphs, which moved with that block, are the writer's text and come back; no Limitations heading is left
    empty (replay of 14 stored papers: 7 of 8 resumed ones ended with an empty Limitations heading)."""
    eng = _engine(tmp_path)
    paper = _older_paper_with_moved_paragraphs(tmp_path)
    state: dict = {"paper_md": str(paper)}
    eng._refresh_paper_facts(state)  # the strip, as the review does first on a resumed paper
    eng._supplement_reset(state, paper)
    text = paper.read_text(encoding="utf-8")
    assert text.count(WRITER) == 1 and "## Limitations\n\n" + WRITER + "\n\n## References" in text
    assert "fi:fact-limitations" not in text and "stated in full in the supplement" not in text


def test_an_older_papers_moved_paragraphs_come_back_on_delivery_and_leave_the_supplement(tmp_path: Path) -> None:
    """Delivered without a reset (FI writes its blocks again and the supplement from its records): the paragraphs are in
    the paper once, the supplement no longer shows them, and the paper keeps no empty Limitations heading."""
    eng = _engine(tmp_path)
    paper = _older_paper_with_moved_paragraphs(tmp_path)
    assert eng._rewrite_fi_blocks({"paper_md": str(paper)}, "test", path=paper) is True
    text = paper.read_text(encoding="utf-8")
    assert text.count(WRITER) == 1 and "## Limitations\n\n" + WRITER + "\n\n## References" in text
    supplement = tmp_path / sup.FILE
    assert not supplement.is_file() or WRITER not in supplement.read_text(encoding="utf-8")


def test_moved_paragraphs_come_back_when_the_further_reading_pointer_follows_the_limitations(tmp_path: Path) -> None:
    """The Limitations is the paper's last section and FI's line pointing to the Further reading came after it: that line is
    FI's, no part of the section, so the paragraphs still come back (replay: the one resumed paper of seven whose
    paragraphs did not come back had this line there)."""
    eng = _engine(tmp_path)
    _records(tmp_path)
    old_block = "<!-- fi:fact-limitations -->\nFI recorded 1 limitation; it is stated in the supplement.\n<!-- /fi:fact-limitations -->"
    pointer = "Further reading: 2 web pages the text does not cite are listed in the supplement (`paper/supplement.md`)."
    head = "# T\n\n## Results\n\nText [1].\n\n## References\n\n1. A.\n\n## Limitations\n\n"
    left = f"{head}{old_block}\n"
    whole = f"{head}{WRITER}\n\n{old_block}\n"
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text(f"{head}{old_block}\n\n{pointer}\n", encoding="utf-8")  # the pointer came after the move
    sup.save(tmp_path, [sup.LIMITATIONS], limit=4, writer=WRITER,
             section=(sup.section_body(whole) or "", sup.section_body(left) or ""))
    state: dict = {"paper_md": str(paper)}
    eng._refresh_paper_facts(state)
    text = paper.read_text(encoding="utf-8")
    assert "## Limitations\n\n" + WRITER in text and "fi:fact-limitations" not in text and text.count(WRITER) == 1
    eng._supplement_reset(state, paper)
    text = paper.read_text(encoding="utf-8")
    assert text.count(WRITER) == 1 and "## Limitations\n\n" + WRITER in text


def test_an_older_records_paragraphs_go_back_only_into_the_section_they_came_from() -> None:
    block = "<!-- fi:fact-limitations -->\n- One.\n\n- Two.\n<!-- /fi:fact-limitations -->"
    text = f"# T\n\n## 5. Limitations\n\nOur caveat.\n\n{block}\n\n## References\n\n1. A.\n"
    left = text.replace("Our caveat.\n\n", "", 1)
    assert sup.put_back_writer_limitations(left, sup.section_body(text), sup.section_body(left)) == text
    assert sup.put_back_writer_limitations(left.replace("## References", "More.\n\n## References"),
                                           sup.section_body(text), sup.section_body(left)) is None


def test_the_knowledge_base_stores_the_supplement_with_the_paper(tmp_path: Path) -> None:
    from tests.test_knowledge import _BrainAddText, _calls_by_kind, _enabled_knowledge_with

    brain = _BrainAddText()
    k = _enabled_knowledge_with(brain)
    (tmp_path / "paper").mkdir()
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text("# paper body", encoding="utf-8")
    sup.write(tmp_path, 4, {sup.LIMITATIONS: "- A limitation the paper points to."})
    assert k.add_quest_artifacts(quest_id="q42", paper_md_path=paper, summary="key finding") is True
    body, _meta = _calls_by_kind(brain)["fi_quest_paper"]
    assert "# paper body" in body and "- A limitation the paper points to." in body


def test_a_supplement_that_cannot_be_written_moves_nothing(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    before = paper.read_bytes()
    _quiet_log(eng)
    _pages(monkeypatch, 3)

    def refuse(*a, **k):  # noqa: ANN002, ANN003
        raise OSError("disk full")

    monkeypatch.setattr(sup, "write", refuse)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    assert paper.read_bytes() == before, "the paper points to no supplement that is not there"
    assert not (tmp_path / sup.RECORD).exists() and "supplement" not in patch["review"]["page_limit"]


def test_a_failed_render_puts_the_receipts_back_too(tmp_path: Path, monkeypatch) -> None:
    from core import receipts

    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    receipts.write(tmp_path, "claim_check", status="pass", producer="claim_check", started_at=receipts.now(),
                   inputs={"paper": paper.read_bytes()}, output="ok")
    assert receipts.read(tmp_path, "claim_check")[0] == "pass", "a receipt FI would carry over"
    _quiet_log(eng)
    _pages(monkeypatch, 4, fail_at=2)  # a second render, if any, fails
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert "held the second factor fixed" in text and sup.LIMITATIONS not in sup.moved(tmp_path)
    assert receipts.read(tmp_path, "claim_check")[0] == "pass"
    assert receipts.read(tmp_path, "claim_check")[1]["input_hashes"]["paper"] == receipts.sha256(paper.read_bytes())


def test_a_further_reading_list_that_came_back_after_the_move_moves_too(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    state = {"topic": eng.config.topic, "paper_md": str(paper)}
    assert asyncio.run(eng._fit_with_supplement(state, paper, 4, {"pages": 10}))  # type: ignore[arg-type]
    assert sup.moved(tmp_path) == ["results_table", "further_reading"]
    # the citation check's renumbering starts a new list with a page the text no longer cites
    paper.write_text(paper.read_text(encoding="utf-8").rstrip() + "\n\n## Further reading\n\n- [W3] Page 3. "
                     "https://example.org/3\n", encoding="utf-8")
    assert asyncio.run(eng._fit_with_supplement(state, paper, 4, {"pages": 7}))  # type: ignore[arg-type]
    assert "## Further reading" not in paper.read_text(encoding="utf-8")
    assert sup.further_labels(tmp_path) == ["W1", "W2", "W3"]
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert all(f"- [W{n}] Page {n}." in shown for n in (1, 2, 3))


def test_a_reset_into_a_paper_with_a_new_further_reading_keeps_the_moved_pages(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    _review(eng, tmp_path)
    assert sup.further_labels(tmp_path) == ["W1", "W2"]
    paper.write_text(paper.read_text(encoding="utf-8").rstrip() + "\n\n## Further reading\n\n- [W3] Page 3. "
                     "https://example.org/3\n", encoding="utf-8")
    eng._supplement_reset({}, paper)
    text = paper.read_text(encoding="utf-8")
    assert text.count("## Further reading") == 1
    assert all(f"- [W{n}] Page {n}." in text for n in (1, 2, 3))


def test_merged_further_reading_lists_each_page_once(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    state = {"topic": eng.config.topic, "paper_md": str(paper)}
    assert asyncio.run(eng._fit_with_supplement(state, paper, 4, {"pages": 10}))  # type: ignore[arg-type]
    paper.write_text(paper.read_text(encoding="utf-8").rstrip() + "\n\n## Further reading\n\n- [W2] Page 2, as "
                     "listed now. https://example.org/2\n", encoding="utf-8")
    assert asyncio.run(eng._fit_with_supplement(state, paper, 4, {"pages": 7}))  # type: ignore[arg-type]
    assert sup.further_labels(tmp_path) == ["W1", "W2"]
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert shown.count("[W2]") == 1 and "Page 2, as listed now." in shown and "- [W1] Page 1." in shown


def test_a_reset_that_cannot_write_fis_blocks_keeps_what_they_point_to(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)  # no writer paragraph and nothing else to put back first: only FI's blocks
    _pages(monkeypatch, 4)  # 8; tables out 6; Further reading out 5
    _review(eng, tmp_path)
    held = sup.read(tmp_path)
    assert sup.TABLES in held["moved"] and (tmp_path / sup.FILE).is_file()
    paper.write_text(paper.read_text(encoding="utf-8").rstrip() + "\n\n## Further reading\n\n- [W1] Page 1. "
                     "https://example.org/1\n- [W2] Page 2. https://example.org/2\n", encoding="utf-8")
    _quiet_log(eng)
    real = Path.write_text

    def refuse(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        if self == paper:
            raise OSError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", refuse)
    eng._supplement_reset({}, paper)
    monkeypatch.setattr(Path, "write_text", real)
    assert sup.table_pointer() in paper.read_text(encoding="utf-8")
    assert sup.read(tmp_path) == held and (tmp_path / sup.FILE).is_file(), "the pointer still has its supplement"
