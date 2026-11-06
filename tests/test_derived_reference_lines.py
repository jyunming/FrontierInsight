"""A quantity the plan declares as a reference with a formula of the plan's numbers (a theory value, a nominal level) is one of
the references a figure may draw by its id, evaluated by FI at every setting, and named by the plan's words."""

from __future__ import annotations

from pathlib import Path

from tests.test_analysis_frame import _derived, _design, _quest, _run

_BODY = """
def figure_fig1(fig, table):
    keys = [k for k in table.keys() if table.setting(k)["dose"] == 1]
    xs = [table.setting(k)["rate"] for k in keys]
    fig.line(0, 1, xs, [table.mean("growth", k) for k in keys], "growth")
    fig.ref(0, "{ref}")
"""


def _with_reference(tmp_path: Path, formula: str, name: str = "law_line") -> tuple[int, dict, str]:
    design = _design([_derived(name, role="reference", formula=formula, inputs=())])
    code, _printed, record, err = _run(_quest(tmp_path / name, design, _BODY.replace("{ref}", name)))
    return code, record, err


def test_a_declared_reference_of_the_plan_s_numbers_is_drawn_by_its_id(tmp_path: Path) -> None:
    code, record, err = _with_reference(tmp_path, "base*rate")
    assert code == 0, err
    assert [f["status"] for f in record["figures"]] == ["ok"], record["figures"]


def test_a_declared_reference_that_reads_a_result_is_not_a_line_the_figure_may_draw(tmp_path: Path) -> None:
    # a formula naming a measured result is not theory: it never becomes a reference line
    code, record, err = _with_reference(tmp_path, "growth*2", "from_data")
    assert [f["status"] for f in record["figures"]] != ["ok"], record["figures"]
