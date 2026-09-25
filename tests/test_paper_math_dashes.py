r"""A dash inside a math span prints as a dash, not as minus signs.

The PDF's unicode rewrite (``generation/paper.py``) turned an en dash into
``--`` and an em dash into ``---`` everywhere. In prose those are LaTeX's dash
ligatures; inside ``$...$`` they are two and three minus signs, so a range the
writer typed as ``$0.322–0.338$`` printed as ``0.322 - -0.338``. Inside a math
span the dashes are set with ``\text{}`` instead.

The math is found by pandoc's own rules (``generation/_pandoc.py``): a dollar
amount is not math, code is not math, ``\(..\)``, ``\[..\]`` and ``$$..$$`` are.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config
from generation import _pptx_math
from generation import paper as paper_mod
from generation._pandoc import MATH_SPAN_RE, TEX_MATH_DOLLARS
from generation.paper import PaperGenerator, _sanitize_unicode_for_latex as sanitize

REPO = Path(__file__).resolve().parent.parent
FORMATS = sorted(p.parent.name for p in (REPO / "templates" / "paper").glob("*/template.tex"))

EN = "–"
EM = "—"


# ---------------------------------------------------------------------------
# What is rewritten, and where


def test_a_range_inside_dollar_math_is_set_with_a_text_dash() -> None:
    assert sanitize(f"(95% CI $0.322{EN}0.338$)") == r"(95% CI $0.322\text{--}0.338$)"


def test_an_em_dash_inside_math_is_set_with_a_text_em_dash() -> None:
    assert sanitize(f"$a{EM}b$") == r"$a\text{---}b$"


@pytest.mark.parametrize("span", [
    f"$0.322{EN}0.338$",
    f"$$0.322{EN}0.338$$",
    f"$$\n0.322{EN}0.338\n$$",                # display math over lines
    f"\\(0.322{EN}0.338\\)",
    f"\\[0.322{EN}0.338\\]",
    f"$x = 1 {EN}\n 2$",                     # a wrapped paragraph breaks the line inside inline math
])
def test_every_spelling_of_math_gets_the_text_dash(span: str) -> None:
    out = sanitize(f"Before {span} after {EN} 1{EN}2.")
    assert r"\text{--}" in out
    assert out.startswith("Before ") and out.endswith(" after -- 1--2.")   # prose is as it was
    assert EN not in out


def test_prose_keeps_the_ligature_dashes() -> None:
    assert sanitize(f"1{EN}2 and a {EM} b") == "1--2 and a --- b"


@pytest.mark.parametrize("text", [
    f"It costs $5 and $10 {EN} a range.",
    f"Prices run $5{EN}6 and $7{EN}8 a seat.",   # a digit right after a closing $ is not a closing $
    f"Between $ 5 and 7 $ {EN} units.",           # a space inside each $
    f"An escaped \\$9{EN}10 stays text.",
    f"Only one $ sign {EN} here.",
])
def test_dollar_signs_that_are_not_math_are_prose(text: str) -> None:
    out = sanitize(text)
    assert r"\text{" not in out
    assert out == text.replace(EN, "--")


def test_math_does_not_run_across_a_blank_line() -> None:
    """A stray ``$`` in one paragraph must not pair with a ``$`` in the next:
    pandoc ends the search at the paragraph."""
    out = sanitize(f"It cost $5 {EN} or so.\n\nLater $a{EN}b$ agrees.")
    assert out == "It cost $5 -- or so.\n\nLater $a\\text{--}b$ agrees."


def test_a_digit_right_after_the_closing_dollar_ends_it_as_math() -> None:
    """``$\\pm$1nm`` is not math to pandoc; the tightening rule must not make it
    one either (it has no digit rule of its own)."""
    assert sanitize(f"$\\pm {EN} x$1 nm") == "$\\pm -- x$1 nm"


def test_code_keeps_its_dash_as_it_always_did() -> None:
    src = f"```\nx = '$a{EN}b$'\n```\nsee `$c{EM}d$` and $e{EN}f$"
    out = sanitize(src)
    # A dollar sign in code is a dollar sign: it is not math, so no \text{}.
    assert "```\nx = '$a--b$'\n```" in out
    assert "`$c---d$`" in out
    assert r"$e\text{--}f$" in out


def test_a_dash_the_writer_typed_inside_math_is_left_alone() -> None:
    assert sanitize(f"$a--b$ and $c---d$ and $e{EN}f$") == r"$a--b$ and $c---d$ and $e\text{--}f$"


def test_the_spaced_math_the_tightening_promotes_is_math_too() -> None:
    r"""``$ \alpha – \beta $`` is text to pandoc but the PDF pipeline rewrites it
    to ``$\alpha \text{--} \beta$`` before pandoc reads it (the unicode rewrite
    runs first, the tightening after)."""
    spaced = f"$ \\alpha {EN} \\beta $"
    out = paper_mod._tighten_inline_math(sanitize(spaced))
    assert out == r"$\alpha \text{--} \beta$"


def test_a_dash_the_writer_already_put_inside_a_text_command_still_nests() -> None:
    r"""``\text{95\% CI 0.32–0.34}`` printed a dash before; ``\text`` inside
    ``\text`` is legal, so it still does."""
    out = sanitize("$\\text{95\\% CI 0.32" + EN + "0.34}$")
    assert out == "$\\text{95\\% CI 0.32\\text{--}0.34}$"


def test_only_the_dashes_differ_between_math_and_prose() -> None:
    """Every other glyph in the map means the same in both modes (probed with
    a real pdflatex: ``\\ensuremath`` commands, the empty strings and the
    spaces). A curly quote is left as the map has it: bare in math it is a
    prime, which is what ``f’(x)`` wants."""
    prose, math = paper_mod._LATEX_UNICODE_REPLACEMENTS, {
        chr(code): replacement
        for code, replacement in paper_mod._LATEX_MATH_TRANSLATOR.items()
    }
    assert {ch for ch in prose if prose[ch] != math[ch]} == {EN, EM}


def test_the_glyphs_the_map_does_not_change_are_the_same_in_math() -> None:
    for glyph, replacement in paper_mod._LATEX_UNICODE_REPLACEMENTS.items():
        if glyph in (EN, EM):
            continue
        assert sanitize(f"$x{glyph}y$") == f"$x{replacement}y$", repr(glyph)


def test_the_result_is_the_same_when_it_is_rewritten_again() -> None:
    once = sanitize(f"$0.322{EN}0.338$ and {EN} $a{EM}b$")
    assert sanitize(once) == once


def test_the_deck_and_the_paper_read_the_dollar_rule_from_one_definition() -> None:
    assert _pptx_math._MATH_RE.pattern == TEX_MATH_DOLLARS
    assert TEX_MATH_DOLLARS in MATH_SPAN_RE.pattern


# ---------------------------------------------------------------------------
# The PDF pipeline


def _config(tmp_path: Path, paper_format: str = "generic") -> Config:
    return Config.model_validate({
        "topic": "t", "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"], "output_dir": str(tmp_path / "outputs"),
            "paper_format": paper_format, "html_pdf_fallback": False,
        },
    })


def test_the_pdf_source_carries_the_text_dash_and_paper_md_is_left_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: ("pdflatex", "/fake/pdflatex"))

    def fake_run(cmd, **_kw):  # noqa: ANN001
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)
    md = tmp_path / "paper.md"
    written = f"# Title\n\n## Abstract\n\nSurvival was 0.33 (95% CI $0.322{EN}0.338$).\n\n## Results\n\nAlso 1{EN}2 runs.\n"
    md.write_text(written, encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    PaperGenerator(_config(tmp_path))._compile_pdf(md, out)
    source = (out / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert r"$0.322\text{--}0.338$" in source
    assert "1--2 runs" in source
    assert md.read_text(encoding="utf-8") == written


# ---------------------------------------------------------------------------
# A real compile, under every paper template


PAPER = (
    "# Range Probe\n\n## Body\n\n"
    f"Inline: survival was 0.33 (95% CI $0.322{EN}0.338$) and $a{EM}b$ too.\n\n"
    f"Paren: \\(0.411{EN}0.417\\).\n\n"
    f"Display:\n\n$$0.5{EN}0.6$$\n\n"
    f"Prose: 10{EN}20 runs, and it costs $5 and $10.\n"
)


@pytest.mark.slow
@pytest.mark.parametrize("fmt", FORMATS)
def test_a_range_in_math_prints_a_dash_under_every_template(tmp_path: Path, fmt: str) -> None:
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
    pdf, skip = PaperGenerator(_config(tmp_path, fmt))._compile_pdf(md, out)
    assert skip is None, skip.summary if skip else ""
    text = " ".join(line.text for page in measure_pdf(pdf).pages for line in page.lines)
    text = re.sub(r"\s+", " ", text)
    for expected in ("0.322–0.338", "0.411–0.417", "0.5–0.6", "10–20", "a—b"):
        assert expected in text, (fmt, expected, text)
    # Two minus signs would print as U+2212 with spaces around them.
    assert "−" not in text, (fmt, text)


def test_set_and_logic_symbols_in_prose_become_latex() -> None:
    """A model writes "R0 ∈ {0.9, 1.5}" in prose; pdflatex stopped on the first "∈" in every run of one model, so the
    PDF fell back to another renderer and the page-limit check was skipped. Each glyph is spelt as a LaTeX command."""
    from generation import paper

    text = "R0 ∈ A ∉ B ⊆ C ∪ D ∩ E = ∅, ∀x ∃y, a ≡ b ∼ c, x ∈ ℝ"
    out = text.translate(paper._LATEX_UNICODE_TRANSLATOR)
    assert not any(ord(ch) > 127 for ch in out), out
    assert r"\ensuremath{\in}" in out and r"\ensuremath{\mathbb{R}}" in out
