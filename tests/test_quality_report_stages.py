"""Every problem the quality report lists names one of its own stages, so ordering them never fails."""
import ast
from pathlib import Path

from core import quality_report


def test_every_problem_names_a_known_stage():
    tree = ast.parse(Path(quality_report.__file__).read_text(encoding="utf-8"))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_problem" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                used.add(first.value)
    assert used, "no _problem call found"
    assert used <= set(quality_report.STAGES), sorted(used - set(quality_report.STAGES))


def test_a_figure_finding_is_ordered_with_the_paper_problems():
    problems = [quality_report._problem("paper", "fig.png: a legend entry the record does not draw", "FI", "not fixed", "r"),
                quality_report._problem("code", "a code finding", "reader", "not fixed", "c")]
    ordered = sorted(problems, key=lambda p: quality_report.STAGES.index(p["stage"]))
    assert [p["stage"] for p in ordered] == ["code", "paper"]
