"""Native PowerPoint equations for the deck's $...$ math
(``generation/_pptx_math.py``)."""

from __future__ import annotations

import pytest
from lxml import etree

from generation._pptx_math import fallback_runs, math_xml, plain_text, split_math

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def _one(latex: str, **kw) -> etree._Element:
    parts = math_xml(latex, size_pt=kw.get("size_pt", 16.5), color=kw.get("color", "16222B"),
                     font="Segoe UI", bold=kw.get("bold", False))
    assert len(parts) == 1
    return etree.fromstring(parts[0])


def test_the_decks_math_is_split_from_its_text_by_pandocs_rule() -> None:
    assert split_math("At $h = 0.01$, RK4 error is $5.297 \\times 10^{-9}$.") == [
        ("At ", False), ("h = 0.01", True), (", RK4 error is ", False),
        ("5.297 \\times 10^{-9}", True), (".", False),
    ]
    assert split_math("$$E = mc^2$$") == [("E = mc^2", True)]


@pytest.mark.parametrize("text", [
    "Licences cost $5-$10 a seat.",      # a digit right after the closing $
    "Licences cost $5 and $7 a seat.",
    "Between $ 5 and 7 $ units.",        # a space inside each $
    "An escaped \\$9 stays text.",
    "Only one $ sign here.",
])
def test_dollar_signs_that_are_not_math_stay_text(text: str) -> None:
    assert split_math(text) == [(text, False)]


def test_a_power_of_ten_becomes_a_superscript_equation_with_its_fallback() -> None:
    el = _one("5.297 \\times 10^{-9}\\text{m}")
    assert el.tag == f"{{{NS['mc']}}}AlternateContent"
    choice = el.find("mc:Choice", NS)
    assert choice.get("Requires") == "a14"
    math = choice.find("a14:m/m:oMath", NS)
    sup = math.find("m:sSup", NS)
    assert sup.findtext("m:e/m:r/m:t", namespaces=NS) == "10"
    assert "".join(sup.find("m:sup", NS).itertext()) == "−9"          # U+2212, not a hyphen
    assert [t.text for t in math.findall("m:r/m:t", NS)] == ["5.297", "×", "m"]
    # The unit is upright text; letters of the formula are italic.
    unit = math.findall("m:r", NS)[-1]
    assert unit.find("m:rPr/m:sty", NS).get(f"{{{NS['m']}}}val") == "p"
    rpr = math.find("m:r/a:rPr", NS)
    assert rpr.get("sz") == "1650" and rpr.find("a:solidFill/a:srgbClr", NS).get("val") == "16222B"
    assert rpr.find("a:latin", NS).get("typeface") == "Cambria Math"
    # What LibreOffice and the visual check see.
    fallback = el.findall("mc:Fallback/a:r", NS)
    assert [(r.findtext("a:t", namespaces=NS), r.find("a:rPr", NS).get("baseline")) for r in fallback] == [
        ("5.297 × 10", None), ("−9", "30000"), (" m", None),
    ]


def test_subscripts_fractions_and_roots_have_their_own_equation_parts() -> None:
    math = _one("E_{analytical} = \\frac{a}{b} + \\sqrt{x}").find("mc:Choice/a14:m/m:oMath", NS)
    assert "".join(math.find("m:sSub/m:sub", NS).itertext()) == "analytical"
    assert math.findtext("m:f/m:num/m:r/m:t", namespaces=NS) == "a"
    assert math.findtext("m:f/m:den/m:r/m:t", namespaces=NS) == "b"
    assert math.find("m:rad/m:radPr/m:degHide", NS) is not None


def test_bold_math_keeps_the_bold_and_colour_of_its_line() -> None:
    el = _one("c x'", color="0A4F4D", bold=True)
    for rpr in el.iter(f"{{{NS['a']}}}rPr"):
        assert rpr.get("b") == "1"
        assert rpr.find("a:solidFill/a:srgbClr", NS).get("val") == "0A4F4D"


def test_the_fallback_reads_like_the_formula() -> None:
    assert fallback_runs("E_{analytical}(t) = E_0 \\exp(-(c/m)t)") == [
        ("E", ""), ("analytical", "sub"), ("(t) = E", ""), ("0", "sub"), (" exp(−(c/m)t)", ""),
    ]
    assert fallback_runs("c x'") == [("cx′", "")]
    assert fallback_runs("\\frac{a+b}{c}") == [("(a + b)/c", "")]


def test_math_outside_the_equation_subset_is_written_as_text_only() -> None:
    parts = math_xml("\\begin{matrix} a & b \\end{matrix}", size_pt=16, color="16222B", font="Segoe UI")
    assert parts and all(etree.fromstring(p).tag == f"{{{NS['a']}}}r" for p in parts)


@pytest.mark.parametrize("latex", ["\\frac{", "}{", "^^^", "\\undefinedcommand{x}", ""])
def test_unreadable_math_never_raises(latex: str) -> None:
    runs = fallback_runs(latex)
    assert isinstance(runs, list)
    for part in math_xml(latex, size_pt=16, color="16222B", font="Segoe UI"):
        etree.fromstring(part)


def test_plain_text_measures_a_formula_by_what_the_slide_shows() -> None:
    text = "At $h = 0.01$, RK4 error is $5.297 \\times 10^{-9}\\text{m}$."
    assert plain_text(text) == "At h = 0.01, RK4 error is 5.297 × 10−9 m."
