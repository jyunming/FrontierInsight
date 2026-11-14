"""A shortened text never ends inside a LaTeX math span."""
from __future__ import annotations

from core import figure_facts as ff
from core import paper_facts as pf
from core.math_safe_cut import head_before_math

TITLE = "Relative error of $\hat x$, $(\hat x - x)/x$"


def _unescaped_dollars(s: str) -> int:
    return len(s.replace("\$", "")) - len(s.replace("\$", "").replace("$", ""))


def test_title_cut_inside_math_is_balanced():
    # the old code cut at 47 characters and left one `$` open
    out = ff._one(TITLE, 40)
    assert _unescaped_dollars(out) % 2 == 0
    assert out == "Relative error of $\hat x$,..."


def test_cut_of_a_span_with_nothing_before_keeps_the_span():
    out = ff._one("$\alpha_{long} + \beta_{long}$ is the error of the model", 20)
    assert out.startswith("$\alpha_{long} + \beta_{long}$")
    assert _unescaped_dollars(out) % 2 == 0


def test_open_paren_math_is_not_cut():
    out = ff._one(r"Error of \(\hat x - x\) over the grid of settings", 22)
    assert out == "Error of..."


def test_escaped_dollar_is_not_a_delimiter():
    text = r"Cost in \$ per run for the larger and the smaller study"
    assert ff._one(text, 30) == text[:27].rstrip() + "..."


def test_short_and_math_free_text_unchanged():
    assert ff._one(TITLE, 80) == TITLE
    plain = "A plain title with no formula that is rather long indeed, yes"
    assert ff._one(plain) == plain[:47].rstrip() + "..."
    assert head_before_math(plain, 20) == plain[:20]


def test_quoted_claim_keeps_math_balanced():
    claim = "word " * 59 + "and $a + b$ is large"
    out = pf._quoted_claim(claim)
    assert _unescaped_dollars(out) % 2 == 0
    claim = "word " * 58 + "and $a + b + c + d$ is large"
    assert _unescaped_dollars(pf._quoted_claim(claim)) % 2 == 0


def test_caption_words_keeps_math_balanced():
    from core.engine import _caption_words
    text = r"Error of the estimate of $\hat x$, $(\hat x - x)/x$ against noise level of the data"
    out = _caption_words(text, 40)
    assert _unescaped_dollars(out) % 2 == 0
    plain = "a plain label " * 12
    assert _caption_words(plain, 40) == plain[:40].rsplit(" ", 1)[0]



def test_caption_words_closes_a_bracket_the_cut_left_open():
    """A label cut inside "(plan formula)" left "(plan." in a caption; the bracket is now closed after " ..."."""
    from core.engine import _caption_words
    label = "Expected doubling time of the colony at the highest nutrient level (plan formula)"
    out = _caption_words(label, 78)
    assert out == "Expected doubling time of the colony at the highest nutrient level (plan ...)" and len(out) <= 78
    closed = "Doubling time (hours) of the colony at the highest nutrient level of the plan"
    assert _caption_words(closed, 60) == "Doubling time (hours) of the colony at the highest nutrient"
    assert _caption_words(label, 200) == label, "a label that fits is not touched"
    assert _caption_words("(long label with a bracket and many more words to follow", 30) == "(long label with a ...)"


def test_caption_words_keeps_two_labels_that_differ_inside_a_bracket_apart():
    from core.engine import _caption_words
    a = _caption_words("Rate (control group of the colony at the highest nutrient level here)", 30)
    b = _caption_words("Rate (treated group of the colony at the highest nutrient level here)", 30)
    assert a != b and a.endswith("...)") and b.endswith("...)") and max(len(a), len(b)) <= 30


def test_caption_words_reads_no_bracket_of_a_formula_or_an_escape():
    from core.engine import _caption_words
    out = _caption_words(r"Value $\left(x$ followed by many extra words for the label", 28)
    assert _unescaped_dollars(out) % 2 == 0 and not out.endswith(")")
    assert not _caption_words(r"Value \{set followed by many extra words for the label", 30).endswith(")")


def test_caption_words_drops_a_trailing_comma_left_by_the_cut():
    """A cut after a comma left ",," once the caption joined the next name."""
    from core.engine import _caption_words
    label = "relative deviation of the colony size from the logistic size, in percent"
    assert _caption_words(label, 62) == "relative deviation of the colony size from the logistic size"
    assert len(_caption_words("a long label of words " * 10, 60)) <= 60, "a long label is still cut"


def test_closers_left_open_skips_math_and_escapes():
    from core.math_safe_cut import closers_left_open
    assert closers_left_open("a (b [c") == "])" and closers_left_open("a (b) c") == ""
    assert closers_left_open(r"$\left(x$ y") == "" and closers_left_open(r"a \{b") == ""
    assert closers_left_open("x $(a$ (b") == ")"
