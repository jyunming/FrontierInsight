"""Text FI writes into a paper is complete prose for a reader: no cut list, no remark about FI's own bookkeeping."""
from __future__ import annotations

from core import paper_facts as pf


def test_a_difference_is_counted_in_words_and_never_cut_with_a_count_of_more() -> None:
    problems = [f"the metric `m{i}` has {i} trials in FI's record but the analysis lists none of them " + "x" * 80 for i in range(7)]
    text = pf.run_difference_text({"status": "warned", "problems": problems})
    assert "more)" not in text and "..." not in text
    assert "7 places" in text and "FI" not in text and "`" not in text


def test_one_difference_is_one_place() -> None:
    text = pf.run_difference_text({"status": "warned", "problems": ["the first differs."]})
    assert "differ in 1 place." in text and text.endswith("rest on that analysis")


def test_a_record_with_nothing_to_state_gives_no_sentence() -> None:
    assert pf.run_difference_text({"status": "warned", "problems": []}) == ""
