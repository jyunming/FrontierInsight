"""Text FI writes into a paper is complete prose for a reader: no cut list, no remark about FI's own bookkeeping."""
from __future__ import annotations

from core import paper_facts as pf


def test_a_difference_without_a_plain_sentence_is_never_cut_with_a_count_of_more() -> None:
    problems = [f"the metric `m{i}` has {i} trials in FI's record but the analysis lists none of them " + "x" * 80 for i in range(7)]
    text = pf.run_difference_text({"status": "warned", "problems": problems})
    assert "more)" not in text and "..." not in text
    assert "7 places" in text and problems[0] in text, "the count in words and the first finding whole"


def test_a_short_set_of_differences_names_every_one() -> None:
    text = pf.run_difference_text({"status": "warned", "problems": ["the first differs.", "the second differs"]})
    assert "the first differs; the second differs" in text


def test_one_difference_is_quoted_whole() -> None:
    long = "the analysis lists a thing " + "word " * 120
    assert long.strip() in pf.run_difference_text({"status": "warned", "problems": [long]})
