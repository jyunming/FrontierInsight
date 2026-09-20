"""A Further reading entry that would push a paper past its page limit is dropped instead.

FI writes ``## Further reading`` itself and sets it after the body. On a graded
SIR quest (limit 4 pages) the paper ended on 5 pages, page 5 holding two entries
of that list. ``over_page_limit`` fired twice, the writer was sent into whole-
paper rewrites, and they changed a correct citation and added unsupported ones.

Now the review, having counted the draft's pages, first tries the draft with the
list's last entries dropped: only when that alone brings it within the limit is
the paper written to disk that way, no model is called and the body is not
rewritten. A draft whose body is over the limit is left to the existing shortening.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

from core.config import Config, resolve_page_limit
from core.engine import (
    Engine,
    QuestArtifacts,
    _further_reading_block,
    _page_limit_hit,
    _trim_further_reading,
    _web_labels_cited,
    further_reading_listed,
)
from generation.paper import PaperGenerator
from tests.test_page_limit import _engine, _quiet_log, _review

ENTRY = re.compile(r"^- \[(W\d+)\] ", re.MULTILINE)
# What the review forces for a 5-page draft over the limit of 4 whose last page
# is 90% empty (the stand-in render below): about 100 words to cut.
OVER = _page_limit_hit(5, 4, 100)


def _entries(n: int) -> str:
    return "\n".join(f"- [W{i}] Page {i}. https://example.org/{i}" for i in range(1, n + 1))


def _paper(n: int, *, cites: str = "") -> str:
    return (
        "# Outbreaks in a Small Population\n\n## Abstract\n\nOne paragraph.\n\n## Introduction\n\n"
        f"Text [1].{cites}\n\n## References\n\n1. Kermack, W. O. (1927). A contribution.\n\n"
        + (f"## Further reading\n\n{_entries(n)}\n" if n else "")
    )


def _labels(text: str) -> list[str]:
    return ENTRY.findall(text)


def _put(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "paper" / "paper.md"
    path.write_text(text, encoding="utf-8")
    return path


def _by_entries(monkeypatch, *, spills_at: int | None, over: int = 5, base: int = 4) -> list[tuple[Path, str]]:  # noqa: ANN001
    """Stand in for the render: a draft with ``spills_at`` or more Further reading
    entries is ``over`` pages, one with fewer is ``base`` (never, with ``None``).
    Returns what each render was given: the file and the text it held then."""
    seen: list[tuple[Path, str]] = []

    async def fake(self, paper_path, state):  # noqa: ANN001, ARG001
        text = Path(paper_path).read_text(encoding="utf-8")
        seen.append((Path(paper_path), text))
        spilled = spills_at is None or len(_labels(text)) >= spills_at
        pages = over if spilled else base
        return {"pages": pages, "words": 1200, "last_page_lines": 3, "last_page_empty": 0.9}

    monkeypatch.setattr(Engine, "_measure_draft_pages", fake)
    return seen


# --- the trim itself -------------------------------------------------------------

def test_trim_keeps_the_first_entries_and_drops_the_rest_from_the_end() -> None:
    text = _paper(4)
    assert _labels(_trim_further_reading(text, 4)) == ["W1", "W2", "W3", "W4"]
    assert _trim_further_reading(text, 9) == text
    assert _labels(_trim_further_reading(text, 2)) == ["W1", "W2"]
    assert _trim_further_reading(text, 2) == text.replace(
        "- [W3] Page 3. https://example.org/3\n- [W4] Page 4. https://example.org/4\n", "",
    )
    # Nothing else changes: everything before the list is the same text.
    assert _trim_further_reading(text, 1).startswith(text[:text.index("## Further reading")])


def test_trim_to_none_removes_the_heading_and_leaves_what_follows() -> None:
    text = _paper(3)
    out = _trim_further_reading(text, 0)
    assert "Further reading" not in out and out.endswith("A contribution.\n")
    # A section after the list, at the same level, is not part of it.
    with_appendix = text.rstrip() + "\n\n## Appendix\n\nKept.\n"
    trimmed = _trim_further_reading(with_appendix, 0)
    assert "Further reading" not in trimmed and trimmed.endswith("## Appendix\n\nKept.\n")
    assert "- [W" not in trimmed
    assert _labels(_trim_further_reading(with_appendix, 1)) == ["W1"]
    assert _trim_further_reading(with_appendix, 1).endswith("## Appendix\n\nKept.\n")


def test_a_paper_without_the_engines_list_is_left_alone() -> None:
    assert _trim_further_reading(_paper(0), 0) == _paper(0)
    prose = "# T\n\nBody [1].\n\n## Further reading\n\nSome pages I liked.\n\n- [W1] A page. https://x.org\n"
    assert _further_reading_block(prose) is None
    assert _trim_further_reading(prose, 0) == prose
    assert further_reading_listed(prose) is None
    assert further_reading_listed(_paper(3)) == ["W1", "W2", "W3"]
    assert further_reading_listed(_paper(0)) is None


def test_the_web_pages_the_text_cites() -> None:
    text = _paper(4, cites=" More [2, W2], see [W4](https://x.org) and `[W1]` and $[W1]$.")
    assert _web_labels_cited(text) == {"W2"}
    assert _web_labels_cited(_paper(4)) == set()
    # The lists themselves cite nothing: "[W3]" there is an entry's label.
    assert _web_labels_cited(_paper(4, cites=" And [W3].")) == {"W3"}


# --- the review -----------------------------------------------------------------

@pytest.mark.parametrize("panel", [False, True])
def test_a_list_that_alone_overflows_is_trimmed_and_nothing_is_sent_back(
    tmp_path: Path, monkeypatch, panel: bool,
) -> None:
    """The graded quest's shape: 3 entries, the last two on page 5."""
    eng = _engine(tmp_path, panel=panel)
    written = _paper(3)
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=2)
    prompts: list[str] = []
    warnings = _quiet_log(eng)
    accepting = eng._chat

    async def spy(prompt, *, node=None):  # noqa: ANN001
        prompts.append(prompt)
        return await accepting(prompt, node=node)

    eng._chat = spy  # type: ignore[method-assign]

    patch = _review(eng, tmp_path)

    # No finding, no rewrite counted, no iteration used.
    assert patch["review"]["must_flag_hits"] == []
    assert "page_limit_rewrites" not in patch and "iteration" not in patch
    assert patch["review"]["page_limit"] == {
        "pages": 4, "limit": 4, "rewrites": 0, "further_reading_dropped": ["W2", "W3"],
    }
    assert eng._route_after_review({"review": patch["review"], "iteration": 0}) == "done"  # type: ignore[arg-type]
    # The paper on disk lost those two entries and nothing else.
    now = paper.read_text(encoding="utf-8")
    assert _labels(now) == ["W1"]
    assert now == written.replace("- [W2] Page 2. https://example.org/2\n- [W3] Page 3. https://example.org/3\n", "")
    # One line says how many were dropped and which.
    dropped = [w for w in warnings if "Further reading" in w]
    assert len(dropped) == 1, warnings
    assert "dropped 2 of its 3 entries" in dropped[0] and "[W2], [W3]" in dropped[0]
    assert "5 pages" in dropped[0] and "renders to 4 pages" in dropped[0]
    # The reviewer reads the paper as it now is.
    reviewed = [p for p in prompts if "Text [1]." in p]
    assert reviewed and all("Page 1. https://example.org/1" in p and "example.org/2" not in p for p in reviewed)
    # Only the review's render and the trial's leave nothing behind.
    assert [p.name for p, _ in seen] == ["paper.md", "page_check_trial.md", "page_check_trial.md", "page_check_trial.md"]
    assert not (tmp_path / ".fi" / "page_check_trial.md").exists()


def test_the_entries_are_dropped_from_the_end_and_the_first_fit_is_kept(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(tmp_path, _paper(5))
    seen = _by_entries(monkeypatch, spills_at=3)  # two or fewer entries fit
    patch = _review(eng, tmp_path)
    assert patch["review"]["page_limit"]["further_reading_dropped"] == ["W3", "W4", "W5"]
    text = paper.read_text(encoding="utf-8")
    assert _labels(text) == ["W1", "W2"]
    # The list emptied first, as the test that it can fit at all; then one more
    # entry put back each time, from the last, until the draft fits.
    assert [len(_labels(t)) for _, t in seen] == [5, 0, 4, 3, 2]
    # What was written is what the render that fit was given: the PDF lists what paper.md does.
    assert seen[-1][1] == text


def test_a_list_that_fits_only_with_none_of_it_is_removed_whole(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    paper = _put(tmp_path, _paper(2))
    _by_entries(monkeypatch, spills_at=1)
    patch = _review(eng, tmp_path)
    assert patch["review"]["must_flag_hits"] == []
    assert patch["review"]["page_limit"]["further_reading_dropped"] == ["W1", "W2"]
    assert "Further reading" not in paper.read_text(encoding="utf-8")


@pytest.mark.parametrize("panel", [False, True])
def test_a_body_that_alone_is_over_the_limit_is_left_to_the_shortening(
    tmp_path: Path, monkeypatch, panel: bool,
) -> None:
    eng = _engine(tmp_path, panel=panel)
    written = _paper(3)
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=0)  # 5 pages however few entries it lists
    patch = _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written
    assert patch["review"]["must_flag_hits"] == [OVER]
    assert patch["page_limit_rewrites"] == 1
    assert "further_reading_dropped" not in patch["review"]["page_limit"]
    assert [len(_labels(t)) for _, t in seen] == [3, 0], "one render to find that out, and no more"


@pytest.mark.parametrize("panel", [False, True])
def test_a_draft_within_the_limit_is_not_touched_and_rendered_once(tmp_path: Path, monkeypatch, panel: bool) -> None:
    eng = _engine(tmp_path, panel=panel)
    written = _paper(3)
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=None, over=4)  # 4 pages whatever it lists
    patch = _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written
    assert len(seen) == 1
    assert patch["review"]["page_limit"] == {"pages": 4, "limit": 4, "rewrites": 0}
    assert patch["review"]["must_flag_hits"] == []


def test_a_paper_without_further_reading_costs_no_extra_render(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    written = _paper(0)
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=0)
    patch = _review(eng, tmp_path)
    assert len(seen) == 1 and paper.read_text(encoding="utf-8") == written
    assert patch["review"]["must_flag_hits"] == [OVER]


def test_an_entry_the_text_cites_is_never_dropped(tmp_path: Path, monkeypatch) -> None:
    """Dropping it would leave a citation naming nothing, and the body is not rewritten."""
    eng = _engine(tmp_path)
    # W3, the last entry, is cited: nothing may go, so nothing is even tried.
    written = _paper(3, cites=" It rests on [W3].")
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=2)
    patch = _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written and len(seen) == 1
    assert patch["review"]["must_flag_hits"] == [OVER]

    # W2 is cited: only W3, after it, may go.
    written = _paper(3, cites=" As in [1, W2].")
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=3)
    patch = _review(eng, tmp_path)
    assert _labels(paper.read_text(encoding="utf-8")) == ["W1", "W2"]
    assert patch["review"]["page_limit"]["further_reading_dropped"] == ["W3"]
    assert [len(_labels(t)) for _, t in seen] == [3, 2], "the one entry that may go is the whole search"

    # W2 cited, and dropping W3 is not enough: nothing changes.
    written = _paper(3, cites=" As in [W2].")
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=2)
    _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written


def test_a_list_the_engine_did_not_write_is_not_touched(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    written = _paper(0) + "## Further reading\n\nA few pages I liked.\n\n- [W1] A page. https://x.org\n"
    paper = _put(tmp_path, written)
    seen = _by_entries(monkeypatch, spills_at=0)
    _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written and len(seen) == 1


def test_a_render_that_fails_midway_leaves_the_paper_as_written(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    written = _paper(4)
    paper = _put(tmp_path, written)
    renders: list[int] = []

    async def flaky(self, paper_path, state):  # noqa: ANN001, ARG001
        n = len(_labels(Path(paper_path).read_text(encoding="utf-8")))
        renders.append(n)
        if len(renders) == 3:
            return None  # the render could not be counted
        return {"pages": 5 if n >= 3 else 4, "words": 1, "last_page_lines": 3, "last_page_empty": 0.9}

    monkeypatch.setattr(Engine, "_measure_draft_pages", flaky)
    patch = _review(eng, tmp_path)
    assert paper.read_text(encoding="utf-8") == written
    assert not (tmp_path / ".fi" / "page_check_trial.md").exists()
    assert patch["review"]["must_flag_hits"] == [OVER]


# --- the exports follow the paper -------------------------------------------------

def _web(title: str, url: str) -> dict[str, Any]:
    return {"content": f"{title} page text", "metadata": {
        "source": "web_search", "kind": "web_page", "title": title, "url": url, "site": url.split("/")[2]}}


def _bib(tmp_path: Path, paper: str, review: dict[str, Any] | None = None) -> str:
    cfg = Config.model_validate({"topic": "t", "title": "tt", "output": {
        "kinds": ["paper_md"], "output_dir": str(tmp_path / "outputs")}})
    quest = tmp_path / "quest"
    (quest / "paper").mkdir(parents=True, exist_ok=True)
    md = quest / "paper" / "paper.md"
    md.write_text(paper, encoding="utf-8")
    art = QuestArtifacts(quest_id="q1", quest_root=quest, paper_md=md)
    art.raw_state = {
        "literature": [_web("First page", "https://one.example/a"), _web("Second page", "https://two.example/b")],
        **({"review": review} if review else {}),
    }
    out = tmp_path / "out"
    PaperGenerator(cfg).generate(art, out)
    path = out / "paper" / "further_reading.bib"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_the_further_reading_bib_lists_what_the_paper_lists(tmp_path: Path) -> None:
    both = "# T\n\nBody.\n\n## Further reading\n\n- [W1] First page. https://one.example/a\n- [W2] Second page. https://two.example/b\n"
    only_first = both.replace("- [W2] Second page. https://two.example/b\n", "")
    assert "one.example" in _bib(tmp_path / "a", both) and "two.example" in _bib(tmp_path / "a2", both)
    bib = _bib(tmp_path / "b", only_first)
    assert "one.example" in bib and "two.example" not in bib
    # A paper with no such section and no record of a drop: every page, as before.
    none_written = "# T\n\nBody [1].\n"
    assert "two.example" in _bib(tmp_path / "c", none_written)
    # The whole list was dropped to fit the page limit: the review says which pages.
    record = {"page_limit": {"pages": 4, "limit": 4, "rewrites": 0, "further_reading_dropped": ["W1", "W2"]}}
    assert _bib(tmp_path / "d", none_written, record) == ""
    partly = {"page_limit": {"further_reading_dropped": ["W2"]}}
    bib = _bib(tmp_path / "e", none_written, partly)
    assert "one.example" in bib and "two.example" not in bib


# --- a real render ----------------------------------------------------------------

@pytest.mark.slow
def test_a_real_render_drops_only_entries_and_the_pdf_lists_what_paper_md_does(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine

    repo = Path(__file__).resolve().parent.parent
    if find_pandoc(repo) is None or find_pdf_engine(repo) is None:
        pytest.skip("needs pandoc and a LaTeX engine")
    eng = _engine(tmp_path, topic="A short paper (at most 2 pages).")
    body = " ".join(f"Sentence {k} reports one more result of the outbreak study." for k in range(75))
    entries = "\n".join(
        f"- [W{i}] A long page title about stochastic epidemic models and their extinction times, part {i}. "
        f"https://example.org/pages/{i}" for i in range(1, 27)
    )
    written = f"# A Draft\n\n## Abstract\n\nShort.\n\n## Results\n\n{body}\n\n## Further reading\n\n{entries}\n"
    paper = _put(tmp_path, written)
    hits, record = asyncio.run(eng._page_limit_review({}, paper))  # type: ignore[arg-type]
    assert record is not None and record.get("further_reading_dropped"), record
    assert hits == [] and record["pages"] <= 2 and resolve_page_limit(eng.config) == 2
    now = paper.read_text(encoding="utf-8")
    kept = _labels(now)
    assert kept == [f"W{i}" for i in range(1, len(kept) + 1)]  # a prefix: dropped from the end
    assert now.startswith(written[:written.index("## Further reading")])  # the body is as written
    # The final render of what is now on disk lists exactly those entries.
    out = tmp_path / "final"
    out.mkdir()
    pdf, skip = PaperGenerator(eng.config)._compile_pdf(paper, out)
    assert pdf is not None, skip
    doc = pdfium.PdfDocument(str(pdf))
    text = "\n".join(doc[i].get_textpage().get_text_range() for i in range(len(doc)))
    assert len(doc) <= 2
    assert re.findall(r"\[W(\d+)\]", text) == [w[1:] for w in kept]
