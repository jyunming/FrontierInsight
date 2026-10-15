"""The LaTeX path of ``paper.pdf`` does not stop on a glyph, numbers a section once,
and says so when the paper is not typeset.

Replays the two ``paper_pdf_source.md`` files of the s1 suite round whose PDF fell
back to the browser: pdflatex stopped on U+0304 (the macron of an x-bar) in one and
on U+208D (a subscript parenthesis) in the other, and the page limit was then never
checked. Both must now be typeset: with XeLaTeX when it is installed (every glyph set as
itself or as its exact LaTeX form), and with pdflatex when it is the only engine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.config import Config
from generation import paper as paper_mod
from generation._headings import strip_heading_numbers
from generation.paper import PaperGenerator, _sanitize_unicode_for_latex as sanitize

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "tests" / "fixtures" / "paper_latex"


def _config(tmp_path: Path, **output) -> Config:
    return Config.model_validate({
        "topic": "t", "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"], "output_dir": str(tmp_path / "outputs"),
            "html_pdf_fallback": False, **output,
        },
    })


def _need_latex() -> None:
    pytest.importorskip("pypdfium2")
    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine

    if find_pandoc(REPO) is None or find_pdf_engine(REPO) is None:
        pytest.skip("needs pandoc and a LaTeX engine")


def _producer(pdf: Path) -> str:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf.read_bytes())
    try:
        return str(doc.get_metadata_dict().get("Producer", ""))
    finally:
        doc.close()


def _figures(folder: Path, source_md: str) -> None:
    """A small PNG for every figure the markdown names, so the replay needs no run folder."""
    from PIL import Image

    for name in re.findall(r"\]\((figures/[^)]+\.png)\)", source_md):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 400), "white").save(target)


# ---------------------------------------------------------------------------
# The sanitizer (pdflatex form): exact LaTeX or the glyph as it is, never another one


def test_a_combining_macron_becomes_a_math_accent() -> None:
    assert sanitize("the mean x̄ and σ̂") == r"the mean \ensuremath{\bar{x}} and \ensuremath{\hat{\sigma}}"


def test_a_letter_with_a_mark_unicode_has_one_character_for_is_that_character() -> None:
    # NFC first: y + U+0302 is ŷ, e + U+0301 is é, = + U+0338 is ≠.
    assert sanitize("ŷ and café") == "ŷ and café"
    assert sanitize("a ≠ b") == r"a \ensuremath{\neq} b"


def test_a_combining_accent_on_a_greek_letter_wraps_the_letters_command() -> None:
    assert sanitize("σ̂") == r"\ensuremath{\hat{\sigma}}"


def test_a_letter_unicode_has_one_character_for_is_that_character() -> None:
    assert sanitize("café") == "café"


def test_a_mark_with_no_latex_accent_is_kept_never_dropped() -> None:
    # A stray mark, and a strike-through on a letter: kept as written (XeLaTeX sets them;
    # pdflatex stops on them and the paper is marked not typeset).
    assert sanitize("a ̄ b") == "a ̄ b"
    assert sanitize("x̸") == "x̸"


def test_a_subscript_parenthesis_is_a_subscript() -> None:
    assert sanitize("E₍a₎") == r"E\ensuremath{_{(}}a\ensuremath{_{)}}"
    assert sanitize("Eₐ") == r"E\ensuremath{_{a}}"


def test_a_glyph_in_no_map_is_left_as_it_is_never_replaced() -> None:
    # U+2B50 (a star), an emoji and U+2A01 have no exact LaTeX form here: they stay, so
    # XeLaTeX sets them or pdflatex stops and the paper is marked. Nothing becomes "?".
    out = sanitize("a ⭐ b \U0001d465 c \U0001f600 d ⨁")
    assert out == "a ⭐ b \\ensuremath{x} c \U0001f600 d ⨁"
    assert "?" not in out


def test_latin_letters_and_other_scripts_are_left_alone() -> None:
    text = "Müller, Łódź, Привет, 中文"
    assert sanitize(text) == text


# ---------------------------------------------------------------------------
# A heading's own number


def _doc(*headings: str) -> str:
    return "# A Title\n\n" + "\n\nText.\n\n".join(headings) + "\n"


@pytest.mark.parametrize("headings, expected", [
    # The document's own numbering comes off, in every form the writer uses.
    (["## 1. Introduction", "## 2. Methods"], ["## Introduction", "## Methods"]),
    (["## 1 Introduction", "## 2 Methods"], ["## Introduction", "## Methods"]),
    (["## 1) Introduction", "## 2) Methods"], ["## Introduction", "## Methods"]),
    (["## 1: Introduction", "## 2: Methods"], ["## Introduction", "## Methods"]),
    (["## I. Introduction", "## II. Methods", "## III. Results"], ["## Introduction", "## Methods", "## Results"]),
    (["## 1. Intro", "## 2. Body", "## 3. Results", "### 3.1 Setup", "### 3.2. Coverage", "#### 3.2.1 Detail"],
     ["## Intro", "## Body", "## Results", "### Setup", "### Coverage", "#### Detail"]),
    (["## Abstract", "## 1. Introduction"], ["## Abstract", "## Introduction"]),
    # A count that is part of the title is not a section number: it does not continue the
    # document's numbering.
    (["## 3 Body Problem Dynamics"], ["## 3 Body Problem Dynamics"]),
    (["## 10 Years Later"], ["## 10 Years Later"]),
    (["## 3 Key Findings"], ["## 3 Key Findings"]),
    (["## 2.5 Million Cells"], ["## 2.5 Million Cells"]),
    (["## 2 Sample Tests Compared"], ["## 2 Sample Tests Compared"]),
    (["## 1. Introduction", "## 3 Body Problem", "## 2. Methods"], ["## Introduction", "## 3 Body Problem", "## Methods"]),
    (["## 2024 outlook"], ["## 2024 outlook"]),
    (["## 3.5% growth"], ["## 3.5% growth"]),
    (["## 4 × 4 grids"], ["## 4 × 4 grids"]),
    # A subsection number belongs to its section.
    (["## 1. Intro", "### 2.1 Stray"], ["## Intro", "### 2.1 Stray"]),
    # Below the levels LaTeX numbers, the writer's number is the only one and stays.
    (["## 1. A", "### 1.1 B", "#### 1.1.1 C", "##### 1.1.1.1 D"], ["## A", "### B", "#### C", "##### 1.1.1.1 D"]),
])
def test_a_headings_own_section_number_comes_off(headings: list[str], expected: list[str]) -> None:
    assert strip_heading_numbers(_doc(*headings)) == _doc(*expected)


def test_the_title_is_never_touched() -> None:
    assert strip_heading_numbers("# 3 Body Problem Dynamics\n\n## 1. Introduction\n") == (
        "# 3 Body Problem Dynamics\n\n## Introduction\n"
    )


def test_where_nothing_numbers_headings_the_writers_number_stays() -> None:
    text = _doc("## 1. Introduction", "### 1.1 Aim")
    assert strip_heading_numbers(text, numbered_depth=0) == text
    # The browser's LaTeX-like theme numbers two levels; its briefing theme none.
    from generation._html_pdf import _numbered_depth, theme_css_path

    assert _numbered_depth(theme_css_path("latexlike")) == 2
    assert _numbered_depth(theme_css_path("briefing")) == 0


def test_numbers_in_code_and_in_text_stay() -> None:
    text = "1. First item\n\n```\n## 1. not a heading\n```\n\n## 1. Real\n"
    assert strip_heading_numbers(text) == "1. First item\n\n```\n## 1. not a heading\n```\n\n## Real\n"


def test_the_pdf_source_has_no_doubled_number(tmp_path: Path) -> None:
    _need_latex()
    md = tmp_path / "paper.md"
    md.write_text(
        "# T\n\n## Abstract\n\nA.\n\n## 1. Introduction\n\nText.\n\n### 1.1 Aim\n\nMore.\n\n"
        "#### 1.1.1 Deep\n\nDeeper.\n", encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    pdf, skip = PaperGenerator(_config(tmp_path))._compile_pdf(md, out)
    assert skip is None
    source = (out / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert "## Introduction" in source and "1. Introduction" not in source
    from generation._pdf_measure import measure_pdf

    text = " ".join(line.text for page in measure_pdf(pdf).pages for line in page.lines)
    assert re.search(r"\b1 Introduction\b", text) and "1 1." not in text and "1.1 1.1" not in text
    # A level-4 heading is numbered by the template too (secnumdepth 3), once.
    assert "1.1.1 Deep" in text and "1.1.1 1.1.1" not in text


# ---------------------------------------------------------------------------
# The replay of the two papers whose PDF fell back to the browser


def _unicode_engine():
    from generation import _cjk
    from generation._pdf_engine import find_pdf_engine

    return _cjk.find_xelatex(find_pdf_engine(REPO))


@pytest.mark.parametrize("name, glyph", [("kinetics", "̄"), ("fdr", "₍")])
def test_the_s1_papers_are_typeset(tmp_path: Path, name: str, glyph: str) -> None:
    """Under the Unicode engine (XeLaTeX) when there is one, which is what FI uses."""
    _need_latex()
    if _unicode_engine() is None:
        pytest.skip("needs XeLaTeX")
    source = (FIXTURES / f"s1_{name}_paper_pdf_source.md").read_text(encoding="utf-8")
    assert glyph in source, "the fixture must hold the glyph pdflatex stopped on"
    md = tmp_path / "in.md"
    md.write_text(source, encoding="utf-8")
    _figures(tmp_path, source)
    gen = PaperGenerator(_config(tmp_path))
    pdf, skip = gen._compile_pdf(md, tmp_path)
    assert skip is None, skip.summary if skip else ""
    assert "dvipdfmx" in _producer(pdf).lower(), _producer(pdf)   # XeLaTeX's PDF writer
    record = json.loads((tmp_path / ".fi" / "paper_render.json").read_text(encoding="utf-8"))
    assert record["typeset"] is True and record["rc"] == 0 and record["engine"] == "xelatex"
    assert record["pages"] and record["pages"] >= 3
    assert record["glyphs_not_set"] == []
    assert record["glyphs_replaced"], "each replacement is recorded"


@pytest.mark.parametrize("name, glyph", [("kinetics", "̄"), ("fdr", "₍")])
def test_the_s1_papers_compile_with_pdflatex_when_there_is_no_unicode_engine(
    tmp_path: Path, monkeypatch, name: str, glyph: str,
) -> None:
    _need_latex()
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("needs pdflatex")
    monkeypatch.setattr(PaperGenerator, "_find_unicode_engine", lambda self, engine: None)
    source = (FIXTURES / f"s1_{name}_paper_pdf_source.md").read_text(encoding="utf-8")
    md = tmp_path / "in.md"
    md.write_text(source, encoding="utf-8")
    _figures(tmp_path, source)
    gen = PaperGenerator(_config(tmp_path))
    pdf, skip = gen._compile_pdf(md, tmp_path)
    assert skip is None, skip.summary if skip else ""
    assert "pdfTeX" in _producer(pdf), _producer(pdf)
    record = json.loads((tmp_path / ".fi" / "paper_render.json").read_text(encoding="utf-8"))
    assert record["typeset"] is True and record["engine"] == "pdflatex"


def test_a_glyph_pdflatex_cannot_set_marks_the_paper_not_typeset(tmp_path: Path, monkeypatch, caplog) -> None:
    """U+0250 has no LaTeX form and inputenc does not know it. Without a Unicode engine
    pdflatex stops on it; the glyph is never replaced by another one: the paper is not
    typeset, and the record and the log name it."""
    _need_latex()
    import shutil

    if shutil.which("pdflatex") is None:
        pytest.skip("needs pdflatex")
    monkeypatch.setattr(PaperGenerator, "_find_unicode_engine", lambda self, engine: None)
    md = tmp_path / "paper.md"
    md.write_text("# T\n\n## Abstract\n\nA.\n\n## Body\n\nThe symbol ɐ appears here.\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    gen = PaperGenerator(_config(tmp_path))
    with caplog.at_level("WARNING", logger="frontier_insight.paper"):
        pdf, skip = gen._compile_pdf(md, out)
    assert pdf is None and skip is not None
    assert "U+0250" in caplog.text
    assert "ɐ" in (out / "paper_pdf_source.md").read_text(encoding="utf-8")
    record = paper_mod.read_render_record(out)
    assert record["typeset"] is False and "U+0250" in record["reason"]
    assert record["glyphs_not_set"] == [{"glyph": "ɐ", "code_point": "U+0250"}]


def test_the_unicode_engine_sets_a_glyph_no_map_knows_as_itself(tmp_path: Path) -> None:
    _need_latex()
    if _unicode_engine() is None:
        pytest.skip("needs XeLaTeX")
    from generation._pdf_measure import measure_pdf

    md = tmp_path / "paper.md"
    md.write_text("# T\n\n## Abstract\n\nA.\n\n## Body\n\nThe symbol ɐ appears here.\n", encoding="utf-8")
    gen = PaperGenerator(_config(tmp_path))
    pdf, skip = gen._compile_pdf(md, tmp_path)
    assert skip is None and pdf is not None
    text = " ".join(line.text for page in measure_pdf(pdf).pages for line in page.lines)
    assert "symbol ɐ appears" in text
    assert gen.last_render["typeset"] is True


def test_the_glyphs_a_log_names_are_read() -> None:
    log = "! LaTeX Error: Unicode character ɐ (U+0250)\n(inputenc) not set up for use with LaTeX.\n"
    assert paper_mod._glyphs_pdflatex_stopped_on(log) == ["ɐ"]
    assert paper_mod._glyphs_pdflatex_stopped_on("no such thing") == []
    xe = "[WARNING] Missing character: There is no ᵀ (U+1D40) in font [lmroman10-regular]:mapping=tex-text;!\n"
    assert paper_mod._glyphs_left_off_the_page(xe) == ["ᵀ"]
    assert paper_mod._glyphs_left_off_the_page(log) == []


# ---------------------------------------------------------------------------
# A paper that is not typeset says so


def test_the_fallback_pdf_is_marked_and_recorded(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("pypdfium2")
    from generation import _html_pdf

    if paper_mod.find_pandoc(REPO) is None:
        pytest.skip("needs pandoc")
    md = tmp_path / "paper.md"
    md.write_text("# T\n\n## Abstract\n\nA.\n\n## Body\n\nText.\n", encoding="utf-8")
    seen: dict = {}

    def fake_render(paper_md, out_pdf, **kw):
        seen.update(kw)
        out_pdf.write_bytes(b"%PDF-1.4\n")
        return out_pdf, ""

    monkeypatch.setattr(_html_pdf, "render_paper_html_pdf", fake_render)
    monkeypatch.setattr(_html_pdf, "find_html_browser", lambda: ("chrome", "chrome"))
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: None)
    gen = PaperGenerator(_config(tmp_path, html_pdf_fallback=True))
    pdf, skip = gen._compile_pdf(md, tmp_path)
    assert pdf is not None and skip is None
    assert seen["not_typeset"] == "no LaTeX engine found"
    record = paper_mod.read_render_record(tmp_path)
    assert record["engine"] == "html_fallback" and record["typeset"] is False
    assert record["reason"] == "no LaTeX engine found"


def test_no_pdf_at_all_is_recorded_too(tmp_path: Path, monkeypatch) -> None:
    md = tmp_path / "paper.md"
    md.write_text("# T\n\nText.\n", encoding="utf-8")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: None)
    monkeypatch.setattr(PaperGenerator, "_try_html_pdf_fallback", lambda self, *a, **k: None)
    if paper_mod.find_pandoc(REPO) is None:
        pytest.skip("needs pandoc")
    pdf, skip = PaperGenerator(_config(tmp_path))._compile_pdf(md, tmp_path)
    assert pdf is None and skip is not None
    record = paper_mod.read_render_record(tmp_path)
    assert record["typeset"] is False and record["engine"] == "none" and record["pages"] is None


def test_the_real_browser_fallback_prints_a_not_typeset_line(tmp_path: Path) -> None:
    pytest.importorskip("pypdfium2")
    from generation._html_pdf import NOT_TYPESET_TEXT, find_html_browser
    from generation._pdf_measure import measure_pdf

    browser = find_html_browser()
    if browser is None or paper_mod.find_pandoc(REPO) is None:
        pytest.skip("needs pandoc and a Chromium-family browser")
    md = tmp_path / "paper.md"
    md.write_text("# T\n\n## Abstract\n\nA.\n\n## 1. Introduction\n\nText here.\n", encoding="utf-8")
    gen = PaperGenerator(_config(tmp_path, html_pdf_fallback=True))
    pdf = gen._try_html_pdf_fallback(md, tmp_path, paper_mod.find_pandoc(REPO), why="pdflatex errored (rc=43)", rc=43)
    assert pdf is not None
    text = " ".join(line.text for line in measure_pdf(pdf).pages[0].lines)
    assert NOT_TYPESET_TEXT.split(":")[0] in text
    assert "1 1." not in text            # numbered once (by the theme's CSS counter)
    assert gen.last_render["rc"] == 43 and gen.last_render["typeset"] is False


def test_compile_for_page_count_falls_back_even_when_the_config_has_it_off(tmp_path: Path, monkeypatch) -> None:
    """The page-limit step counts the pages of whatever PDF there is; with LaTeX down it
    must not skip the count just because ``html_pdf_fallback`` is off."""
    from generation import _html_pdf

    if paper_mod.find_pandoc(REPO) is None:
        pytest.skip("needs pandoc")
    md = tmp_path / "paper.md"
    md.write_text("# T\n\nText.\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def fake_render(paper_md, out_pdf, **kw):
        out_pdf.write_bytes(b"%PDF-1.4\n")
        return out_pdf, ""

    monkeypatch.setattr(_html_pdf, "render_paper_html_pdf", fake_render)
    monkeypatch.setattr(_html_pdf, "find_html_browser", lambda: ("chrome", "chrome"))
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: None)
    gen = PaperGenerator(_config(tmp_path, html_pdf_fallback=False))
    pdf, why, record = gen.compile_for_page_count(md, scratch)
    assert pdf is not None and why == ""
    assert record["typeset"] is False and record["engine"] == "html_fallback"


# ---------------------------------------------------------------------------
# The neutral template


def test_the_neutral_article_template_typesets_a_real_paper(tmp_path: Path) -> None:
    _need_latex()
    from generation._pdf_measure import measure_pdf

    source = (FIXTURES / "s1_kinetics_paper_pdf_source.md").read_text(encoding="utf-8")
    md = tmp_path / "in.md"
    md.write_text(source, encoding="utf-8")
    _figures(tmp_path, source)
    gen = PaperGenerator(_config(tmp_path, paper_format="article", page_limit=4))
    pdf, skip = gen._compile_pdf(md, tmp_path)
    assert skip is None, skip.summary if skip else ""
    assert (tmp_path / "fi_paper.sty").is_file()          # copied next to the source
    assert "skia" not in _producer(pdf).lower() and "chrom" not in _producer(pdf).lower()
    doc = measure_pdf(pdf)
    first = " ".join(line.text for line in doc.pages[0].lines)
    assert "Frontier Insight" in first                    # the neutral byline
    assert not doc.pages[0].images                        # no logo in the page foot
    everything = " ".join(line.text for page in doc.pages for line in page.lines)
    for venue in ("ICLR", "NeurIPS", "Conference", "Under review", "IEEE"):
        assert venue not in everything
    assert gen.last_render["typeset"] is True


def test_the_article_format_is_a_choice_everywhere_the_formats_are_listed() -> None:
    import typing

    from core.config import PaperFormat, SCIENTIFIC_PAPER_FORMATS
    from core.interview import PAPER_FORMATS

    assert "article" in typing.get_args(PaperFormat)
    assert "article" in SCIENTIFIC_PAPER_FORMATS
    assert "article" in {c.value for c in PAPER_FORMATS}
