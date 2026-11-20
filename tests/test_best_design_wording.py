"""The paper's *Best design found* section is written for the paper's reader: no file path, no FI identifier of a
search method, and no negative zero (``core/best_design_report.py``). Neutral made-up records."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core import best_design_report as report


def _best(method: str = "bounded_local", why: str = "", improvement: float = 0.5) -> dict:
    return {
        "objective": {"quantity": "yield", "unit": "kg", "direction": "maximise"},
        "baseline": {"design": {"a": 1.0}, "objective": 10.0},
        "best": {"design": {"a": 1.5}, "objective": 10.5},
        "improvement": {"value": improvement},
        "evaluations": {"search": 40, "budget": 100, "stopped_because": "converged"},
        "method": {"requested": method, "used": method, "why": why},
    }


def _section(monkeypatch, best: dict, check: dict | None = None) -> str:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": best, "optimum_check": check})
    return report.section(Path("."))


def test_the_section_names_no_file(monkeypatch) -> None:
    text = _section(monkeypatch, _best())
    assert "FI's own record of the search" in text
    for path in ("results/", "raw/", "needs/", ".json", ".jsonl"):
        assert path not in text


@pytest.mark.parametrize("method_id", ["bounded_local", "global_then_local", "exhaustive",
                                       "scipy:differential_evolution", "optuna:tpe", "something_new"])
def test_the_section_names_no_method_id(monkeypatch, method_id: str) -> None:
    text = _section(monkeypatch, _best(method=method_id))
    assert method_id not in text
    assert re.search(r"Method: \S", text)


def test_a_known_method_reads_as_what_it_does() -> None:
    assert report.method_name("bounded_local").startswith("a local search")
    assert "SciPy" in report.method_name("scipy:differential_evolution")
    assert report.method_name("") == report.method_name("never_seen_before")


def test_an_id_inside_the_reason_is_replaced(monkeypatch) -> None:
    text = _section(monkeypatch, _best(method="exhaustive", why="too many values; the built-in bounded_local search was used"))
    assert "bounded_local" not in text and "exhaustive" not in text
    assert "a local search inside the ranges" in text


def test_a_negative_zero_prints_as_zero(monkeypatch) -> None:
    assert report._fmt(-0.0) == "0"
    assert report._fmt(-0.00001) == "-1e-05"
    assert report._fmt(-0.5) == "-0.5"
    text = _section(monkeypatch, _best(improvement=-0.0))
    assert not re.search(r"(?<![\d.])-0(?:\.0+)?(?![\d.e])", text)


def test_the_one_line_for_a_zero_improvement(monkeypatch) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(improvement=-0.0), "optimum_check": None})
    line = report.summary_line(Path("."))
    assert "-0" not in line and "no better than the baseline" in line


def test_the_starting_points_said_are_those_that_ran_not_the_plans_count(monkeypatch) -> None:
    best = _best(method="global_then_local")
    best["evaluations"].update(starts=6, per_start=10)
    best["starts"] = [{"start": 0, "evaluations": 30, "stopped_because": "budget"}]
    text = _section(monkeypatch, best)
    assert "for each of 6 starting points; the local search ran from 1 starting point)" in text
    best["starts"] = [{"start": k} for k in range(6)]
    assert "the local search ran from" not in _section(monkeypatch, best), "as planned: nothing more to say"


# ---- one "Best design found" heading in the paper -----------------------------------------------------------------

_PAPER = ("# A study\n\n## Abstract\n\nA.\n\n## Results\n\nThe search ran.\n\n## Best design found\n\n"
          "The writer's reading of the best design.\n\n## Discussion\n\nIt worked.\n\n---\n\n## References\n\n1. x\n")


def _headings_named(text: str, name: str) -> int:
    return sum(1 for ln in text.split("\n") if re.match(r"^#{1,6}[ \t]+(?:\d+(?:\.\d+)*\.?\s+)?" + name + r"[ \t]*$", ln))


def test_a_writer_section_of_fis_heading_keeps_one_heading_with_fis_record_at_its_end(monkeypatch) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(), "optimum_check": None})
    text = report.for_paper(Path("."), None, _PAPER)
    out = report.mark_paper(_PAPER, text)
    assert _headings_named(out, report.HEADING) == 1, out
    # The writer's text stays under its heading, outside the markers; FI's record ends the section, before the discussion.
    assert (out.index("## Best design found") < out.index("The writer's reading") < out.index(report.BEGIN)
            < out.index("The numbers from here to the end of this section are FI's own record") < out.index(report.END)
            < out.index("## Discussion"))
    # Every later draft: FI's record goes in again, once, and the writer's section is kept as it is.
    assert report.mark_paper(out, report.for_paper(Path("."), None, out)) == out
    # The claim check leaves FI's part out when it is exactly FI's, as for a section with its own heading.
    stripped = report.strip_for_checks(out, report.for_paper(Path("."), None, out))
    assert report.BEGIN not in stripped and "The writer's reading" in stripped
    # The writer's section taken out in a later draft: FI's own section, with its heading, before the discussion.
    gone = out.replace("## Best design found\n\nThe writer's reading of the best design.\n\n", "")
    again = report.mark_paper(gone, report.for_paper(Path("."), None, gone))
    assert _headings_named(again, report.HEADING) == 1
    assert again.index(report.BEGIN) < again.index("## Best design found") < again.index("## Discussion")
    assert "The numbers in this section are FI's own record" in again


@pytest.mark.parametrize("heading", ["## 4. Best design found", "### Best design found", "## Best Design Found"])
def test_the_writers_heading_is_found_by_fis_heading_whatever_its_number_level_or_case(monkeypatch, heading: str) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(), "optimum_check": None})
    paper = _PAPER.replace("## Best design found", heading)
    out = report.mark_paper(paper, report.for_paper(Path("."), None, paper))
    assert sum(1 for ln in out.split("\n") if re.match(r"^#+\s+(?:\d+\.\s+)?best design found\s*$", ln, re.I)) == 1, out
    assert out.index(heading) < out.index(report.BEGIN) < out.index("## Discussion")


def test_a_heading_in_a_code_block_or_in_other_words_is_not_the_writers_section(monkeypatch) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(), "optimum_check": None})
    fenced = _PAPER.replace("## Best design found\n\nThe writer's reading of the best design.\n\n",
                            "```\n## Best design found\n```\n\n")
    out = report.mark_paper(fenced, report.for_paper(Path("."), None, fenced))
    assert "The numbers in this section are FI's own record" in out and out.count("## Best design found") == 2
    other = _PAPER.replace("## Best design found", "## The best design found by the search")
    out = report.mark_paper(other, report.for_paper(Path("."), None, other))
    assert "The numbers in this section are FI's own record" in out
    assert out.index("## The best design found by the search") < out.index(report.BEGIN) < out.index("## Discussion")


def test_a_writer_section_of_fis_heading_after_the_discussion_takes_fis_record_there(monkeypatch) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(), "optimum_check": None})
    paper = ("# A study\n\n## Results\n\nR.\n\n## Discussion\n\nD.\n\n## Best design found\n\nW.\n\n---\n\n"
             "## References\n\n1. x\n")
    out = report.mark_paper(paper, report.for_paper(Path("."), None, paper))
    assert _headings_named(out, report.HEADING) == 1
    rule = out.index("\n---\n")
    assert out.index("W.") < out.index(report.BEGIN) < out.index(report.END) < rule < out.index("## References")


def test_the_writer_is_told_the_heading_is_fis(monkeypatch) -> None:
    monkeypatch.setattr(report, "records", lambda _root: {"best_design": _best(), "optimum_check": None})
    note = report.write_note(Path("."))
    assert f'"{report.HEADING}"' in note and "do not give a section of your own that heading" in note
