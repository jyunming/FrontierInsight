r"""Two super- or subscript glyphs in a row inside math are one script.

The PDF's unicode rewrite (``generation/paper.py``) writes ``⁻`` as
``\ensuremath{^{-}}`` and ``³`` as ``\ensuremath{^{3}}``. In prose each is its
own ``$^{-}$``, and ``cm⁻³`` compiles. Inside ``$...$`` the two are a bare
``^{-}^{3}``: "Double superscript", and pdflatex stops, so a paper with
``$10⁻³$`` in it had no PDF. Inside math a run of them is set as one script.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.config import Config
from generation import paper as paper_mod
from generation.paper import PaperGenerator, _sanitize_unicode_for_latex as sanitize

REPO = Path(__file__).resolve().parent.parent
FORMATS = sorted(p.parent.name for p in (REPO / "templates" / "paper").glob("*/template.tex"))

SUP = "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b\u207c\u207d\u207e\u207f"
SUB = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u208a\u208b"


@pytest.mark.parametrize("source, expected", [
    ("$10\u207b\u00b3$", r"$10\ensuremath{^{-3}}$"),
    ("$x\u2081\u2082$", r"$x\ensuremath{_{12}}$"),
    ("$\\mathrm{cm}\u207b\u00b3$", r"$\mathrm{cm}\ensuremath{^{-3}}$"),
    ("$m s\u207b\u00b9$", r"$m s\ensuremath{^{-1}}$"),
    ("$x\u207d\u207f\u207e$", r"$x\ensuremath{^{(n)}}$"),
    ("$10\u2074\u00b0$", r"$10\ensuremath{^{4\circ}}$"),                 # the degree sign is a superscript
    ("\\(x\u207b\u00b9y\u207b\u00b9\\)", r"\(x\ensuremath{^{-1}}y\ensuremath{^{-1}}\)"),   # two runs, two scripts
    ("$$x\u00b2\u207a\u00b9$$", r"$$x\ensuremath{^{2+1}}$$"),
])
def test_a_run_of_script_glyphs_in_math_is_one_script(source: str, expected: str) -> None:
    assert sanitize(source) == expected


@pytest.mark.parametrize("source, expected", [
    ("$N\u2080$", r"$N\ensuremath{_{0}}$"),
    ("$x\u00b2$", r"$x\ensuremath{^{2}}$"),
    ("$x\u00b2\u2081$", r"$x\ensuremath{^{2}}\ensuremath{_{1}}$"),       # a sup then a sub is legal
])
def test_a_single_script_glyph_in_math_is_as_it_was(source: str, expected: str) -> None:
    assert sanitize(source) == expected


def test_script_glyphs_in_prose_and_in_code_are_as_they_were() -> None:
    assert sanitize("cm\u207b\u00b3 and x\u2081\u2082") == (
        r"cm\ensuremath{^{-}}\ensuremath{^{3}} and x\ensuremath{_{1}}\ensuremath{_{2}}"
    )
    assert sanitize("see `$10\u207b\u00b3$` here") == (
        r"see `$10\ensuremath{^{-}}\ensuremath{^{3}}$` here"
    )


def test_every_script_glyph_of_the_map_merges_with_its_neighbour() -> None:
    glyphs = paper_mod._SCRIPT_GLYPHS
    assert set(SUP + SUB + "\u00b0") == set(glyphs)
    for glyph in glyphs:
        out = sanitize(f"$x{glyph}{glyph}y$")
        assert out.count("\\ensuremath{") == 1, (glyph, out)


# ---------------------------------------------------------------------------
# A real compile, under every template


PAPER = (
    "# Script Probe\n\n## Body\n\n"
    "Prose: n = 1e18 cm\u207b\u00b3 and H\u2082O.\n\n"
    "Math: $10\u207b\u00b3$ and $\\mathrm{cm}\u207b\u00b3$ and $x\u2081\u2082$ and $N\u2080$ and $10\u2074\u00b0$.\n"
)


@pytest.mark.slow
@pytest.mark.parametrize("fmt", FORMATS)
def test_a_run_of_script_glyphs_in_math_compiles_under_every_template(tmp_path: Path, fmt: str) -> None:
    pytest.importorskip("pypdfium2")
    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine
    from generation._pdf_measure import measure_pdf

    if find_pandoc(REPO) is None or find_pdf_engine(REPO) is None:
        pytest.skip("needs pandoc and a LaTeX engine")
    md = tmp_path / "paper.md"
    md.write_text(PAPER, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    config = Config.model_validate({
        "topic": "t", "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"], "output_dir": str(tmp_path / "outputs"),
            "paper_format": fmt, "html_pdf_fallback": False,
        },
    })
    pdf, skip = PaperGenerator(config)._compile_pdf(md, out)
    assert skip is None, skip.summary if skip else ""
    text = re.sub(r"\s+", "", " ".join(line.text for page in measure_pdf(pdf).pages for line in page.lines))
    for expected in ("10\u22123", "cm\u22123", "x12", "N0", "H2O"):
        assert expected in text, (fmt, expected, text)
