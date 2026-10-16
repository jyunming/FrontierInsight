"""A combining mark after a LaTeX command, or on a letter in math, is the accent meant.

Unicode NFC joins a letter and a following mark. Run over the source as written, it joined
the mark to the last letter of a command: ``$\\Pi`` + U+0302 ``$`` became ``\\P`` + ``î`` and
printed the pilcrow "¶î", typeset and unmarked. And ``$y`` + U+0302 ``$`` became an upright
``ŷ`` (pdflatex stopped on it; XeLaTeX set it in a text font) where ``\\hat{y}`` was meant.
"""

from __future__ import annotations

from pathlib import Path

from generation import paper as paper_mod
from tests.test_paper_glyphs import _compile, _text, _typeset, _xelatex_or_skip

HAT, TILDE, TIE = "̂", "̃", "͡"
B = "\\"


def _prep(text: str, **kw) -> str:
    return paper_mod._prepare_unicode(text, fonts=None, **kw).text


def test_a_command_with_a_mark_becomes_the_accent_over_the_command() -> None:
    assert _prep(f"${B}Pi{HAT}$ and ${B}Pi{TILDE}k$") == (
        f"${B}ensuremath{{{B}hat{{{B}Pi}}}}$ and ${B}ensuremath{{{B}tilde{{{B}Pi}}}}k$"
    )
    assert _prep(f"${B}beta{HAT}$") == f"${B}ensuremath{{{B}hat{{{B}beta}}}}$"
    assert _prep(f"in text {B}alpha{HAT} too") == f"in text {B}ensuremath{{{B}hat{{{B}alpha}}}} too"


def test_a_command_with_a_mark_latex_has_no_accent_for_is_left_as_written() -> None:
    # Never merged into the command name, never patched: the engine stops on it or reports
    # it missing, and the paper is marked not typeset.
    source = f"${B}Pi{TIE}$ x"
    assert _prep(source) == source


def test_nfc_never_merges_a_mark_into_a_command_name() -> None:
    out = _prep(f"${B}Pi{TIE}$ and ${B}Pi{HAT}$")
    assert f"{B}Pi{TIE}" in out and "î" not in out


def test_a_letter_with_a_hat_in_math_is_the_math_accent() -> None:
    assert _prep(f"$y{HAT} = x$") == f"${B}ensuremath{{{B}hat{{y}}}} = x$"


def test_a_letter_with_a_hat_in_prose_is_still_the_composed_letter() -> None:
    assert _prep(f"fête and e{HAT}") == "fête and ê"


def test_a_command_with_a_hat_prints_the_accented_command_not_a_pilcrow(tmp_path: Path) -> None:
    _xelatex_or_skip()
    gen, pdf, skip = _compile(tmp_path, f"The projection estimate ${B}Pi{HAT}$ and $y{HAT}$ differ.\n")
    _typeset(gen, pdf, skip)
    text = _text(pdf)
    assert "¶" not in text and "î" not in text, text
    assert "Π" in text, text
