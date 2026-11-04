"""The slice of a long table of every setting (core/results_by_setting.py ``_slice``): a study run as sub-studies is cut
part by part, and a fully crossed grid keeps its old picks. No model is called."""

from __future__ import annotations

from core import results_by_setting as rbs

X = list(range(1, 35))
Y = [1, 2, 3, 4]


def _cell(x: int, y: int) -> dict:
    summary = {"n": 3, "mean": x * 10.0 + y, "sd": 0.5, "binary": False, "successes": None, "index": False,
               "constant": False, "digest": f"d{x}-{y}"}
    return {"cell": {"x": x, "y": y}, "key": f"x={x},y={y}", "level": None, "metrics": {"out": summary}}


def _two_part_built(sub_studies: bool = True) -> rbs.Built:
    # part A varies x and holds y = 2; part B varies y and holds x = 17
    cells = [_cell(x, 2) for x in X] + [_cell(17, y) for y in Y if y != 2]
    protocol: dict = {"grid": {"x": X, "y": Y}}
    if sub_studies:
        protocol["sub_studies"] = [{"name": "sweep of x", "vary": ["x"], "hold": {"y": 2}},
                                   {"name": "sweep of y", "vary": ["y"], "hold": {"x": 17}}]
    return rbs.build(cells, protocol, {})


def _rows(table: rbs.Table) -> list[tuple[str, str]]:
    return [(r[table.columns.index("x")], r[table.columns.index("y")]) for r in table.rows]


def test_a_part_of_the_study_is_not_cut_away_and_its_held_axis_is_a_dash() -> None:
    built = _two_part_built()
    assert len(built.tables[0].rows) == 37
    cut = rbs._slice(built)[0]
    rows = _rows(cut)
    # part A at the first, middle and last x; y is held there, so it shows a dash and not 2
    assert [r for r in rows if r[1] == rbs.HELD_AXIS] == [("1", "–"), ("18", "–"), ("34", "–")]
    # part B at the first, middle and last y it varies, x held
    assert [r for r in rows if r[0] == rbs.HELD_AXIS] == [("–", "1"), ("–", "3"), ("–", "4")]
    assert len(rows) == 6
    assert "marks a setting a part of the study holds fixed" in cut.caption


def test_a_slice_over_the_cap_keeps_the_first_and_last_value_of_each_part() -> None:
    cut = rbs._slice(_two_part_built(), cap=4)[0]
    assert sorted(_rows(cut)) == sorted([("1", "–"), ("34", "–"), ("–", "1"), ("–", "4")])
    assert "first and last value" in cut.caption


def test_a_fully_crossed_grid_keeps_its_old_picks() -> None:
    cells = [_cell(x, y) for x in X for y in Y]
    plan = {"grid": {"x": X, "y": Y}}
    cut = rbs._slice(rbs.build(cells, plan, {}))[0]
    assert sorted(_rows(cut)) == sorted((str(x), str(y)) for x in (1, 18, 34) for y in (1, 3, 4))
    assert all("–" not in r for r in cut.rows)
    assert cut.caption.endswith("Only the first, middle and last value of each setting are shown here.")
    # sub_studies left empty in the plan: the same table
    again = rbs._slice(rbs.build(cells, {**plan, "sub_studies": []}, {}))[0]
    assert again.rows == cut.rows and again.caption == cut.caption


def test_the_dash_of_a_held_axis_is_written_for_latex() -> None:
    cut = rbs._slice(_two_part_built())[0]
    assert "–" in "".join(r[1] for r in cut.rows) and "–" not in rbs.latex(cut)
    assert " -- " in rbs.latex(cut)
