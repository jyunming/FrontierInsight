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
