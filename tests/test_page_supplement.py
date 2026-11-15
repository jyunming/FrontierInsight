"""A paper over its page limit moves FI's own parts to the supplement, in a fixed order, without a model call.

On 16 stored papers (limit 4) every paper was over it (5-9 pages, mean 6.9); moving FI's own tables of every setting,
its Limitations with the writer's own paragraphs of that section, and its Further reading gave 4.75 (6 of 16 within). The review now moves them, in that order, to
``paper/supplement.md`` until the paper fits: the tables leave a pointer, the Limitations keep one line (how many things
FI recorded, each part of the topic the study did not do by name, and where they all are), the References stay. The
decision is a record FI's blocks follow, so a later write of the blocks keeps it, and a new draft is decided again.
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
    "{{results_table}}\n\n## Discussion\n\nWhat it means.\n\n## Limitations\n\n{{limitations}}\n\n"
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
    eng._restate_limitations({}, "test", path=paper)  # FI writes its blocks, as after any write
    return paper


def _pages(monkeypatch, base: int, *, fail_at: int | None = None) -> list[str]:  # noqa: ANN001
    """Stand in for the render: ``base`` pages, +2 with FI's full tables, +1 with its full Limitations, +1 with the
    Further reading. Returns the text each render was given."""
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


def test_the_parts_move_in_order_and_stop_as_soon_as_the_paper_fits(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    calls = _model_calls(eng)
    seen = _pages(monkeypatch, 2)  # 2 + 2 + 1 + 1 = 6; tables out 4: within the limit of 4
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert patch["review"]["page_limit"] == {"pages": 4, "limit": 4, "supplement": ["results_table"]}
    assert sup.moved(tmp_path) == ["results_table"]
    assert "| a = 1 |" not in text and sup.table_pointer() in text
    assert "held the second factor fixed" in text, "the Limitations stay: the paper fits without moving them"
    assert "## Further reading" in text and "## References" in text
    assert "| a = 1 |" in (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert len(seen) == 2
    assert [c for c in calls if c != "review"] == [], "no model call for the move"


def test_limitations_keep_one_line_that_names_the_part_of_the_topic_not_done(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 3)  # 7; tables out 5; Limitations out 4
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert patch["review"]["page_limit"]["supplement"] == ["results_table", "limitations"]
    assert "## Limitations" in text
    line = sup.limitations_pointer(1, [TOPIC_PART])
    assert line in text and TOPIC_PART in line and sup.SHOWN in line
    assert "## Further reading" in text and "## References" in text
    supplement = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    # A known disclosure is in the supplement word for word.
    assert "The study held the second factor fixed." in supplement
    assert supplement.index("## Results for every setting") < supplement.index("## Limitations")
    # QUALITY.md says what moved.
    assert "its tables of every setting and its Limitations" in pf.page_limit_text(tmp_path)


def test_a_paper_still_over_with_everything_moved_keeps_them_moved(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 6)  # 10; with all three out 6, still over
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert patch["review"]["page_limit"]["supplement"] == ["results_table", "limitations", "further_reading"]
    assert patch["review"]["page_limit"]["pages"] == 6
    assert "## Further reading" not in text and "## References" in text
    supplement = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert "- [W1] Page 1. https://example.org/1" in supplement and "- [W2] Page 2." in supplement
    assert sup.further_labels(tmp_path) == ["W1", "W2"]
    said = pf.page_limit_text(tmp_path)
    assert "over the 4-page limit" in said and "its Further reading to the supplement" in said


def test_a_paper_within_the_limit_is_not_touched(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    before = paper.read_bytes()
    seen = _pages(monkeypatch, 0)  # 4 pages as written
    patch = _review(eng, tmp_path)
    assert paper.read_bytes() == before and len(seen) == 1
    assert patch["review"]["page_limit"] == {"pages": 4, "limit": 4}
    assert not (tmp_path / sup.RECORD).exists() and not (tmp_path / sup.FILE).exists()


def test_a_new_draft_is_decided_again_with_everything_back(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    _review(eng, tmp_path)
    assert sup.moved(tmp_path) == ["results_table", "limitations", "further_reading"]
    # A shorter draft: it fits with all of FI's parts in it.
    _pages(monkeypatch, 0)
    patch = _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert patch["review"]["page_limit"] == {"pages": 4, "limit": 4}
    assert "| a = 1 |" in text and "held the second factor fixed" in text
    assert "- [W1] Page 1. https://example.org/1\n- [W2] Page 2. https://example.org/2" in text
    assert not (tmp_path / sup.RECORD).exists() and not (tmp_path / sup.FILE).exists()


def test_a_later_write_of_fis_blocks_keeps_the_move(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    _pages(monkeypatch, 3)
    _review(eng, tmp_path)
    # A record changes after the review (a new thing FI states) and the Limitations are written again.
    covered = json.loads((tmp_path / ".fi" / "topic_covered.json").read_text(encoding="utf-8"))
    covered["not_done"].append({"kind": "other", "name": "a third part", "sentence": "The study left out a third part."})
    (tmp_path / ".fi" / "topic_covered.json").write_text(json.dumps(covered), encoding="utf-8")
    eng._restate_limitations({}, "test", path=paper)
    eng._write_supplement({}, 4)
    text = paper.read_text(encoding="utf-8")
    assert sup.limitations_pointer(2, [TOPIC_PART, "a third part"]) in text
    assert "The study left out a third part." not in text
    assert "The study left out a third part." in (tmp_path / sup.FILE).read_text(encoding="utf-8")


def test_further_reading_whose_page_the_text_cites_stays(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, DRAFT.replace("Text [1].", "Text [1, W2]."))
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    patch = _review(eng, tmp_path)
    assert patch["review"]["page_limit"]["supplement"] == ["results_table", "limitations"]
    assert "## Further reading" in paper.read_text(encoding="utf-8")


def test_a_render_that_cannot_be_counted_leaves_everything_as_it_was(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    before = paper.read_bytes()
    _quiet_log(eng)
    _pages(monkeypatch, 3, fail_at=3)  # the render after the Limitations moved fails
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    patch = _review(eng, tmp_path)
    assert paper.read_bytes() == before
    assert not (tmp_path / sup.RECORD).exists() and not (tmp_path / sup.FILE).exists()
    assert "supplement" not in patch["review"]["page_limit"]


def test_a_figure_statement_left_out_counts_as_stated_in_the_supplement(tmp_path: Path, monkeypatch) -> None:
    item = pf.Item(pf.FIGURE_LEFT_OUT, "left-out-fig", "figure 1 statements", "Two statements about Figure 1 were left out.")
    monkeypatch.setattr(pf, "required_items", lambda root: [item])
    (tmp_path / ".fi").mkdir()
    paper = "# T\n\n## Limitations\n\n<!-- fi:fact-limitations -->\nFI recorded 1 limitation.\n<!-- /fi:fact-limitations -->\n"
    pf.keep_check(tmp_path, [], paper)
    unmet = json.loads((tmp_path / pf.RECORD).read_text(encoding="utf-8"))["unmet_items"]
    assert [u["id"] for u in unmet] == ["left-out-fig"], "not in the paper and nothing moved: unmet"
    sup.save(tmp_path, [sup.LIMITATIONS], limit=4)
    pf.keep_check(tmp_path, [], paper)
    assert not (tmp_path / pf.RECORD).exists(), "moved with the Limitations: stated"


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
CITED = "A caveat that a source backs [1]."
FIGURE = "![Figure 3: the spread](figures/fig3.png)"
WRITER_DRAFT = DRAFT.replace("## Limitations\n\n{{limitations}}",
                             f"## Limitations\n\n{WRITER}\n\n{CITED}\n\n{FIGURE}\n\n{{{{limitations}}}}")


def test_the_writers_own_limitations_paragraph_moves_with_fis(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, WRITER_DRAFT)
    _pages(monkeypatch, 3)  # 7; tables out 5; Limitations out 4
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert sup.moved(tmp_path) == ["results_table", "limitations"]
    assert WRITER not in text
    # a paragraph that cites a source and a figure stay where they were, with FI's one line
    assert CITED in text and FIGURE in text and "## Limitations" in text
    assert sup.limitations_pointer(1, [TOPIC_PART], discussion=True) in text
    supplement = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    lim = supplement[supplement.index("## Limitations"):]
    assert WRITER in lim and lim.index(WRITER) < lim.index("The study held the second factor fixed.")
    assert CITED not in supplement


def test_the_same_draft_gets_its_paragraph_back_once_and_a_new_one_keeps_its_own(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, WRITER_DRAFT)
    _pages(monkeypatch, 3)
    _review(eng, tmp_path)
    assert WRITER not in paper.read_text(encoding="utf-8")
    _pages(monkeypatch, 0)  # fits with everything in it: the paragraph comes back, once, before FI's list
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert text.count(WRITER) == 1 and text.index(WRITER) < text.index("held the second factor fixed")
    assert not (tmp_path / sup.RECORD).exists()
    # moved again, then a new draft writes its own paragraph there: the old one is not added
    _pages(monkeypatch, 3)
    _review(eng, tmp_path)
    assert WRITER not in paper.read_text(encoding="utf-8")
    own = "A new caveat the second draft wrote."
    text = paper.read_text(encoding="utf-8").replace("## Limitations\n\n", f"## Limitations\n\n{own}\n\n", 1)
    paper.write_text(text, encoding="utf-8")
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert own in text and WRITER not in text


def test_a_limitations_paragraph_outside_a_limitations_section_stays() -> None:
    block = "<!-- fi:fact-limitations -->\n- One.\n\n- Two.\n<!-- /fi:fact-limitations -->"
    for text in (f"# T\n\n## Discussion and limitations\n\nWhat it means.\n\n{block}\n",
                 f"# T\n\n**Limitations.** Our caveat.\n\n{block}\n"):
        assert sup.take_writer_limitations(text) == (text, "")
    text = f"# T\n\n## 5. Limitations\n\nOur caveat.\n\n{block}\n\n## References\n\n1. A.\n"
    left, taken = sup.take_writer_limitations(text)
    assert taken == "Our caveat." and block in left and "## References" in left and "Our caveat" not in left
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
    held = receipts.path(tmp_path, "claim_check").read_bytes()
    _quiet_log(eng)
    _pages(monkeypatch, 3, fail_at=3)  # the render after the Limitations moved fails
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _review(eng, tmp_path)
    assert receipts.read(tmp_path, "claim_check")[1]["input_hashes"]["paper"] == receipts.sha256(paper.read_bytes())
    assert receipts.path(tmp_path, "claim_check").read_bytes() == held


def test_a_further_reading_list_that_came_back_after_the_move_moves_too(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path)
    eng._trim_body_to_fit = lambda *a, **k: asyncio.sleep(0, None)  # type: ignore[method-assign]
    _pages(monkeypatch, 6)
    state = {"topic": eng.config.topic, "paper_md": str(paper)}
    assert asyncio.run(eng._fit_with_supplement(state, paper, 4, {"pages": 10}))  # type: ignore[arg-type]
    assert sup.moved(tmp_path) == ["results_table", "limitations", "further_reading"]
    # the citation check's renumbering starts a new list with a page the text no longer cites
    paper.write_text(paper.read_text(encoding="utf-8").rstrip() + "\n\n## Further reading\n\n- [W3] Page 3. "
                     "https://example.org/3\n", encoding="utf-8")
    assert asyncio.run(eng._fit_with_supplement(state, paper, 4, {"pages": 7}))  # type: ignore[arg-type]
    assert "## Further reading" not in paper.read_text(encoding="utf-8")
    assert sup.further_labels(tmp_path) == ["W1", "W2", "W3"]
    shown = (tmp_path / sup.FILE).read_text(encoding="utf-8")
    assert all(f"- [W{n}] Page {n}." in shown for n in (1, 2, 3))


def test_the_pointer_names_every_part_of_the_topic_not_done() -> None:
    parts = ["part A", "part B", "part C", "part D", "part E"]
    line = sup.limitations_pointer(7, parts)
    assert all(p in line for p in parts) and "more" not in line


def test_a_limitations_section_with_sub_headings_loses_no_text() -> None:
    block = "<!-- fi:fact-limitations -->\n- One.\n<!-- /fi:fact-limitations -->"
    for text in (f"# T\n\n## Limitations\n\nOur caveat.\n\n### Data\n\nA data caveat.\n\n{block}\n",
                 f"# T\n\n## Limitations\n\n{block}\n\n### Data\n\nA data caveat.\n"):
        left, taken = sup.take_writer_limitations(text)
        assert all(x in left or x in taken for x in ("Our caveat.", "A data caveat.") if x in text)
        assert "A data caveat." in left, "a paragraph under a sub-heading stays with it"


def test_only_the_draft_the_paragraphs_came_from_gets_them_back(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, WRITER_DRAFT)
    _pages(monkeypatch, 3)
    _review(eng, tmp_path)
    assert WRITER not in paper.read_text(encoding="utf-8")
    # a new draft whose Limitations hold only a cited paragraph of its own: the old paragraph is not added to it
    text = paper.read_text(encoding="utf-8").replace(CITED, "A new caveat a source backs [1].")
    paper.write_text(text, encoding="utf-8")
    _pages(monkeypatch, 0)
    _review(eng, tmp_path)
    text = paper.read_text(encoding="utf-8")
    assert WRITER not in text and "A new caveat a source backs [1]." in text


def test_a_reset_that_cannot_write_the_paper_keeps_the_record(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, WRITER_DRAFT)
    _pages(monkeypatch, 3)
    _review(eng, tmp_path)
    held = sup.read(tmp_path)
    assert held.get("limitations_writer") == WRITER
    _quiet_log(eng)
    real = Path.write_text

    def refuse(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        if self == paper:
            raise OSError("locked")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "write_text", refuse)
    assert eng._supplement_reset({}, paper) is False
    monkeypatch.setattr(Path, "write_text", real)
    assert sup.read(tmp_path) == held, "the paragraphs are still kept in the record and the supplement"


def test_the_write_step_gives_the_writer_its_paragraphs_back(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(eng, tmp_path, WRITER_DRAFT)
    _pages(monkeypatch, 3)
    _review(eng, tmp_path)
    assert WRITER not in paper.read_text(encoding="utf-8")
    seen: list[str] = []

    async def stop(self):  # noqa: ANN001 -- the write step's first call; what the writer reads is on disk by then
        seen.append(paper.read_text(encoding="utf-8"))
        raise RuntimeError("stop here")

    monkeypatch.setattr(Engine, "_resolve_write_persona", lambda self, state: (_ for _ in ()).throw(RuntimeError("x")))

    async def code_ok(self):  # noqa: ANN001
        return None

    monkeypatch.setattr(Engine, "_check_code_project", code_ok)
    try:
        asyncio.run(eng._node_write({"topic": eng.config.topic, "paper_md": str(paper)}))  # type: ignore[arg-type]
    except RuntimeError:
        pass
    text = paper.read_text(encoding="utf-8")
    assert WRITER in text and "held the second factor fixed" in text and "| a = 1 |" in text
    assert not (tmp_path / sup.RECORD).exists()


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
    _pages(monkeypatch, 4)  # 8; tables out 6; Limitations out 5; Further reading out 4
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
