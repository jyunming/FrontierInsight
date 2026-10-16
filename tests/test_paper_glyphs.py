"""Every glyph of a paper is printed with its meaning: set as itself (XeLaTeX), set by
its exact LaTeX form, or the paper is marked not typeset. Never a ``?``, never another
letter, never dropped, and code is never rewritten.

The real-compile tests replay the review's failing inputs (``Xᵀ`` printed ``XT``, ``⅓``
printed ``13``, ``a ≠ b`` printed ``a = b``, ``∆G`` printed ``?G``, a Cyrillic name that
stopped pdflatex, code that printed ``\\ensuremath{\\bar{x}}``) and read the PDF's own
text layer (pypdfium2) for the meaning, not just the exit code.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from core.config import Config
from generation import _cjk
from generation import paper as paper_mod
from generation._unicode_fonts import Font, FontPlan, find_fonts, unsupported_script
from generation.paper import PaperGenerator, _prepare_unicode, _sanitize_unicode_for_latex as sanitize

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# No TeX needed


def _plan(text_covers: str = "", **fallbacks: str) -> FontPlan:
    """A font plan whose text font holds ASCII plus ``text_covers`` and whose fallback
    fonts hold what each keyword gives."""
    text = Font("text", Path("/f/text.otf"), frozenset(range(0x80)) | {ord(c) for c in text_covers})
    return FontPlan(
        text=(text,),
        fallbacks=tuple(Font(name, Path(f"/f/{name}.ttf"), frozenset(ord(c) for c in chars))
                        for name, chars in fallbacks.items()),
        monos=(Font("Mono", Path("/f/mono.otf"), frozenset(range(0x80))),
               Font("WideMono", Path("/f/widemono.ttf"), frozenset(range(0x80)) | set(map(ord, "≤┌─┐x̄Δ")))),
    )


def test_glyphs_the_text_font_lacks_are_wrapped_in_the_fallback_font_that_has_them() -> None:
    prepared = _prepare_unicode("Kolmogorov (Колмогоров, 1933).", fonts=_plan(Cyr="Колмгрв"))
    assert prepared.text == "Kolmogorov (\\FIfontA{Колмогоров}, 1933)."
    assert "\\newfontfamily\\FIfamilyA{Cyr.ttf}" in prepared.header
    assert prepared.not_set == []
    assert ("о", "itself, in Cyr", "text") in prepared.replacements


def test_a_glyph_no_font_sets_is_listed_not_replaced() -> None:
    prepared = _prepare_unicode("a ⭐ b", fonts=_plan())
    assert prepared.not_set == ["⭐"]
    assert "⭐" in prepared.text and "?" not in prepared.text


def test_a_right_to_left_script_is_not_typeset() -> None:
    assert unsupported_script(ord("ש")) and unsupported_script(ord("ع")) and not unsupported_script(ord("Ж"))
    prepared = _prepare_unicode("Shalom (שלום).", fonts=_plan(Heb="שלום"))
    assert prepared.not_set == ["ש", "ל", "ו", "ם"]


def test_code_is_never_rewritten_and_gets_a_code_font_with_its_glyphs() -> None:
    md = "Inline `x̄ ≤ 3` here.\n\n```\nmean = x̄  # ≤ 3, Δ\n┌──┐\n```\n"
    prepared = _prepare_unicode(md, fonts=_plan())
    assert prepared.text == md
    assert prepared.code_font == "WideMono"
    assert "\\setmonofont{widemono.ttf}" in prepared.header
    assert prepared.not_set == []


def test_a_code_glyph_no_code_font_has_is_listed() -> None:
    prepared = _prepare_unicode("`a ★ b`", fonts=_plan())
    assert prepared.not_set == ["★"]


def test_under_pdflatex_the_codes_glyphs_are_declared_not_rewritten() -> None:
    prepared = _prepare_unicode("Use `x ≤ 3` and ≤ in text.", fonts=None)
    assert prepared.text == "Use `x ≤ 3` and \\ensuremath{\\leq} in text."
    assert "\\DeclareUnicodeCharacter{2264}{\\ensuremath{\\leq}}" in prepared.header


@pytest.mark.parametrize("glyph, latex", [
    ("ᵀ", r"\ensuremath{^{\mathsf{T}}}"),        # the transpose
    ("ᴴ", r"\ensuremath{^{\mathsf{H}}}"),        # the conjugate transpose
    ("ʰ", r"\ensuremath{^{h}}"),
    ("ᵝ", r"\ensuremath{^{\beta}}"),
    ("⁻", r"\ensuremath{^{-}}"),
    ("∆", r"\ensuremath{\Delta}"),               # U+2206, the Mac's Option-J delta
    ("⊤", r"\ensuremath{\top}"),
    ("⩽", r"\ensuremath{\leqslant}"),
    ("⩾", r"\ensuremath{\geqslant}"),
    ("≠", r"\ensuremath{\neq}"),
    ("≮", r"\ensuremath{\nless}"),
    ("⊄", r"\ensuremath{\not\subset}"),
    ("⨯", r"\ensuremath{\times}"),
    ("⋆", r"\ensuremath{\star}"),
    ("ε", r"\ensuremath{\varepsilon}"),           # U+03B5 is the open epsilon
    ("ϵ", r"\ensuremath{\epsilon}"),              # U+03F5 is the lunate one
    ("Α", r"\ensuremath{\mathrm{A}}"),            # capital alpha, drawn like A
    ("\U0001d431", r"\ensuremath{\mathbf{x}}"),   # bold x
])
def test_each_glyph_maps_to_its_exact_latex(glyph: str, latex: str) -> None:
    assert sanitize(glyph) == latex


@pytest.mark.parametrize("glyph, latex", [
    ("⅓", r"\ensuremath{\tfrac{1}{3}}"),
    ("⅔", r"\ensuremath{\tfrac{2}{3}}"),
    ("½", r"\ensuremath{\tfrac{1}{2}}"),
    ("𝟙", r"\ensuremath{\mathbf{1}}"),
    ("★", r"\ensuremath{\bigstar}"),
])
def test_pdflatex_gets_an_exact_form_for_what_xelatex_sets_as_itself(glyph: str, latex: str) -> None:
    assert sanitize(glyph) == latex
    prepared = _prepare_unicode(glyph, fonts=_plan(Sym=glyph))
    assert prepared.text == "\\FIfontA{" + glyph + "}"


def test_a_decomposed_negation_is_the_negated_relation() -> None:
    # NFC first: = + U+0338 is ≠ and < + U+0338 is ≮; the overlay is never dropped.
    assert sanitize("a ≠ b and p ≮ 0.05") == r"a \ensuremath{\neq} b and p \ensuremath{\nless} 0.05"


def test_unit_signs_are_their_letters() -> None:
    assert sanitize("Å and 5 Ω") == "Å and 5 \\ensuremath{\\Omega}"


def test_a_brace_after_a_glyph_stays_a_brace() -> None:
    # Pandoc read the braces of "𝟙{X>0}" as the argument of the LaTeX the glyph became.
    assert sanitize("𝟙{X>0}") == r"\ensuremath{\mathbf{1}}\{X>0}"


def test_no_glyph_ever_becomes_a_question_mark() -> None:
    sample = "".join(chr(c) for c in [*range(0x2000, 0x2BFF, 7), *range(0x0370, 0x03FF), *range(0x1D400, 0x1D7FF, 5)])
    assert "?" not in sanitize(sample)


def test_the_record_lists_every_replacement() -> None:
    prepared = _prepare_unicode("a ≠ b ≠ c, x̄", fonts=None)
    record = prepared.record()
    assert {"glyph": "≠", "code_point": "U+2260", "became": r"\ensuremath{\neq}", "where": "text", "count": 2} in record
    assert any(r["glyph"] == "x̄" and r["became"] == r"\ensuremath{\bar{x}}" for r in record)


def test_the_poster_keeps_rewriting_its_code() -> None:
    assert sanitize("`≤`") == r"`\ensuremath{\leq}`"
    assert _prepare_unicode("`≤`", fonts=None).text == "`≤`"


# ---------------------------------------------------------------------------
# A real compile with XeLaTeX, read back from the PDF


def _xelatex_or_skip() -> None:
    pytest.importorskip("pypdfium2")
    from generation._pdf_engine import find_pdf_engine

    if paper_mod.find_pandoc(REPO) is None:
        pytest.skip("needs pandoc")
    engine = find_pdf_engine(REPO)
    if engine is None or _cjk.find_xelatex(engine) is None:
        pytest.skip("needs XeLaTeX")
    if not find_fonts(_cjk.find_xelatex(engine)[1]).fallbacks:
        pytest.skip("needs a fallback font (DejaVu, Noto or Latin Modern Math)")


def _compile(tmp_path: Path, body: str, **output) -> tuple[PaperGenerator, Path | None, object]:
    md = tmp_path / "paper.md"
    md.write_text("# A Test Paper\n\n## Abstract\n\nShort.\n\n## Introduction\n\n" + body, encoding="utf-8")
    cfg = Config.model_validate({"topic": "t", "title": "t", "output": {
        "kinds": ["paper_md", "paper_pdf"], "output_dir": str(tmp_path / "outputs"),
        "html_pdf_fallback": False, **output,
    }})
    gen = PaperGenerator(cfg)
    pdf, skip = gen._compile_pdf(md, tmp_path)
    return gen, pdf, skip


def _text(pdf: Path) -> str:
    from generation._pdf_measure import measure_pdf

    text = "\n".join(line.text for page in measure_pdf(pdf).pages for line in page.lines)
    # pdfium hands a character beyond U+FFFF (𝟙) back as its two UTF-16 halves.
    return text.encode("utf-16", "surrogatepass").decode("utf-16")


def _typeset(gen: PaperGenerator, pdf, skip) -> None:
    assert skip is None and pdf is not None, skip.summary if skip else ""
    assert gen.last_render["typeset"] is True and gen.last_render["engine"] == "xelatex", gen.last_render
    assert gen.last_render["glyphs_not_set"] == []


def test_a_transpose_prints_as_a_raised_t(tmp_path: Path) -> None:
    _xelatex_or_skip()
    import pypdfium2 as pdfium

    gen, pdf, skip = _compile(tmp_path, "The estimator is (Xᵀ X)⁻¹ Xᵀ y and the Hermitian Aᴴ.\n")
    _typeset(gen, pdf, skip)
    doc = pdfium.PdfDocument(pdf.read_bytes())
    try:
        page = doc[0].get_textpage()
        chars = [(page.get_text_range(i, 1), page.get_charbox(i)) for i in range(page.count_chars())]
    finally:
        doc.close()
    raised = 0
    for (ch, box), (nxt, nbox) in zip(chars, chars[1:]):
        if ch in "XA" and nxt in "TH":
            # The script letter sits higher and is smaller than its base: (left, bottom, right, top).
            assert nbox[1] > box[1] + 1 and (nbox[3] - nbox[1]) < (box[3] - box[1]), (ch, nxt, box, nbox)
            raised += 1
    assert raised == 3


def test_a_vulgar_fraction_is_printed_as_itself(tmp_path: Path) -> None:
    _xelatex_or_skip()
    gen, pdf, skip = _compile(tmp_path, "About ⅓ of tests and ⅔ of the rest; one ⅛ share.\n")
    _typeset(gen, pdf, skip)
    assert "About ⅓ of tests and ⅔ of the rest; one ⅛ share." in _text(pdf)


def test_a_decomposed_negation_prints_negated(tmp_path: Path) -> None:
    _xelatex_or_skip()
    gen, pdf, skip = _compile(tmp_path, "We find a ≠ b and p ≮ 0.05.\n")
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    # \neq is drawn as the negation slash over "=", which the text layer reads as U+0338 "=".
    assert ("a ̸= b" in text or "a ≠ b" in text) and "p ≮ 0.05" in text, text
    assert "a = b" not in text


def test_math_symbols_print_as_their_symbols(tmp_path: Path) -> None:
    _xelatex_or_skip()
    gen, pdf, skip = _compile(tmp_path, "The ∆G of the Α state and Ω ohm; Å spacing; slanted p ⩽ 0.05; w⊤x; 𝟙{X>0}.\n")
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    assert ("∆G" in text or "ΔG" in text) and "p ⩽ 0.05" in text and "w⊤x" in text, text
    assert "𝟙{X>0}" in text and "Å spacing" in text and "?" not in text


def test_a_cyrillic_name_and_many_modifier_letters_typeset(tmp_path: Path) -> None:
    """The typesetter never stops on a glyph: pdflatex stopped on a Cyrillic name and on
    four or more modifier letters (one glyph learnt per rerun, three reruns)."""
    _xelatex_or_skip()
    gen, pdf, skip = _compile(
        tmp_path, "Results follow Kolmogorov (Колмогоров, 1933) closely.\n\nGlyphs: ᵀ ᴴ ʰ ᵝ ᵞ ᴺ ˡ ɐ ʔ and done.\n",
    )
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    assert "Kolmogorov (Колмогоров, 1933)" in text
    assert "ɐ ʔ and done" in text
    record = json.loads((tmp_path / ".fi" / "paper_render.json").read_text(encoding="utf-8"))
    became = {r["glyph"]: r["became"] for r in record["glyphs_replaced"]}
    assert became["о"].startswith("itself, in ") and became["ᵀ"] == r"\ensuremath{^{\mathsf{T}}}"


def test_code_prints_exactly_as_written(tmp_path: Path) -> None:
    _xelatex_or_skip()
    gen, pdf, skip = _compile(tmp_path, "Inline `x̄ ≤ 3` here.\n\n```\nmean = x̄  # ≤ 3, Δ\n┌──┐\n```\n")
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    assert "x̄ ≤ 3" in text and "mean = x̄ # ≤ 3, Δ" in text.replace("  ", " ") and "┌──┐" in text, text
    assert "ensuremath" not in text


def test_a_star_is_a_star_not_markdown(tmp_path: Path) -> None:
    _xelatex_or_skip()
    gen, pdf, skip = _compile(tmp_path, "★ Significant at the 5% level.\n\nA ★best★ value.\n")
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    assert "★ Significant at the 5% level." in text and "A ★best★ value." in text


def test_cjk_symbols_and_a_decomposed_kana_keep_their_marks(tmp_path: Path) -> None:
    _xelatex_or_skip()
    body = "活化能 50 kJ/mol，※ 註：結果見表 1。● 重點一。が (が)。\n"
    if _cjk.find_cjk_font(body) is None:
        pytest.skip("needs a CJK font")
    gen, pdf, skip = _compile(tmp_path, body)
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    assert "※" in text and "●" in text and text.count("が") >= 2, text


def test_the_s1_fixtures_are_typeset_with_xelatex_in_the_article_format(tmp_path: Path) -> None:
    _xelatex_or_skip()
    import re

    from PIL import Image

    source = (REPO / "tests" / "fixtures" / "paper_latex" / "s1_fdr_paper_pdf_source.md").read_text(encoding="utf-8")
    for name in re.findall(r"\]\((figures/[^)]+\.png)\)", source):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 400), "white").save(tmp_path / name)
    md = tmp_path / "in.md"
    md.write_text(source, encoding="utf-8")
    cfg = Config.model_validate({"topic": "t", "title": "t", "output": {
        "kinds": ["paper_md", "paper_pdf"], "output_dir": str(tmp_path / "o"), "html_pdf_fallback": False,
        "paper_format": "article",
    }})
    gen = PaperGenerator(cfg)
    pdf, skip = gen._compile_pdf(md, tmp_path)
    _typeset(gen, pdf, skip)
    counts = Counter(r["glyph"] for r in gen.last_render["glyphs_replaced"])
    assert "₍" in counts
