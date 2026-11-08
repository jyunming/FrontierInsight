"""Two edges of the paper's placeholders. (1) A bold run-in heading followed only by a key FI cannot fill left
"**Large m.** [number removed]." in a delivered paper; the heading has nothing to head, so both go. (2) A placeholder of FI's
own right after ``_``, ``^`` or ``}`` was read as LaTeX and left raw; only a LaTeX command makes it LaTeX. No model is called."""

from __future__ import annotations

from pathlib import Path

from core import paper_facts as pf


def _mark(key: str) -> str:
    return f"<!-- fi:v {{{{v:{key}}}}} -->not recorded<!-- /fi:v -->"


def test_a_run_in_heading_over_only_an_unfilled_key_goes_with_it() -> None:
    md = ("Intro sentence.\n\n**Large m.** " + _mark("x:mean @ m=2000") + ".\n\n**Power.** BH power is high. "
          "**Late.** " + _mark("y:mean @ m=2") + ".\n\nEnd.\n")
    text, taken = pf.take_out_unfilled(md)
    assert "Large m" not in text and "[number removed]" not in text
    assert "**Power.** BH power is high." in text and "Intro sentence." in text and text.rstrip().endswith("End.")
    assert {t["how"] for t in taken} == {"sentence removed"}


def test_a_heading_with_more_text_after_the_key_keeps_the_heading() -> None:
    md = "**Large m.** " + _mark("x:mean @ m=2000") + ". The next sentence is true.\n"
    text, _taken = pf.take_out_unfilled(md)
    assert text.startswith("**Large m.**") and "The next sentence is true." in text and "[number removed]" not in text


def test_a_sentence_with_a_real_body_is_still_handled_as_before() -> None:
    md = "**Power.** The estimate was " + _mark("x:mean @ m=5") + " in this run.\n"
    text, taken = pf.take_out_unfilled(md)
    assert "**Power.**" in text and taken and taken[0]["how"] == "sentence removed"


def test_fi_name_after_underscore_caret_or_brace_is_filled(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    for before in ("snake_case_", "x^", "{{trials_per_setting}}"):
        out = pf.render(f"See {before}{{{{seed}}}} here.", tmp_path)
        assert "{{seed}}" not in out.text and "fi:fact-seed" in out.text


def test_a_command_or_a_name_not_fi_s_stays_latex(tmp_path: Path) -> None:
    (tmp_path / ".fi").mkdir()
    assert pf.render(r"\textbf{{seed}} stays", tmp_path).text == r"\textbf{{seed}} stays"
    assert pf.render("a_{{n}} stays", tmp_path).text == "a_{{n}} stays"
