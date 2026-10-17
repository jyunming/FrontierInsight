"""A value compared inside each run (``in_each_run``) is never pooled by a paper key: its values come from the same trials."""

from core import paper_values as pv

PROTOCOL = {"in_each_run": {"estimator": ["weighted", "plain"]}}


def _cells(with_level_field=True):
    cells = []
    for sigma in (0.1, 0.2):
        values = [1.0, 2.0, 3.0, 4.0]
        for est in ("weighted", "plain"):
            c = {"cell": {"sigma": sigma, "estimator": est}, "metrics": {"k": {"values": values}}}
            if with_level_field:
                c["level"] = {"estimator": est}
            cells.append(c)
        cells.append({"cell": {"sigma": sigma}, "metrics": {"ea": {"values": values}}})
    return cells


def test_all_over_in_run_axis_is_not_filled_and_says_why():
    rec = pv.from_cells(_cells(), PROTOCOL)
    v = pv.resolve("k:mean_ci @ sigma=0.1, estimator=all", rec)
    assert not v.filled
    assert "estimator" in v.hit.reason and "weighted" in v.hit.reason and "same trials" in v.hit.reason
    assert "estimator=weighted" in v.hit.reason


def test_one_level_counts_the_trials_once():
    rec = pv.from_cells(_cells(), PROTOCOL)
    v = pv.resolve("k:mean_ci @ sigma=all, estimator=weighted", rec)
    assert v.filled and v.inputs["trials"] == 8 and v.inputs["settings"] == 2


def test_record_alone_names_the_axis_without_the_plan():
    rec = pv.from_cells(_cells(), None)
    assert not pv.resolve("k @ sigma=0.1, estimator=all", rec).filled


def test_grid_axis_all_still_pools_and_per_run_metric_is_unchanged():
    rec = pv.from_cells(_cells(), PROTOCOL)
    assert pv.resolve("k:mean_ci @ sigma=all, estimator=plain", rec).filled
    v = pv.resolve("ea:mean_ci @ sigma=all, estimator=all", rec)  # one number per run: no estimator in its cells
    assert v.filled and v.inputs["trials"] == 8


def test_no_in_run_axis_pools_as_before():
    cells = [{"cell": {"sigma": s, "n": n}, "metrics": {"k": {"values": [1.0, 2.0, 3.0]}}}
             for s in (0.1, 0.2) for n in (5, 10)]
    v = pv.resolve("k:mean_ci @ sigma=all, n=all", pv.from_cells(cells))
    assert v.filled and v.inputs["trials"] == 12


def test_sheet_offers_no_all_line_over_the_axis_but_pools_the_grid_per_level():
    sheet = pv.sheet(pv.from_cells(_cells(), PROTOCOL))
    lines = [x for x in sheet.splitlines() if x.startswith("- {{v:k:mean_ci")]
    assert not any("estimator=all" in x.split(" -> ")[0] for x in lines)
    assert any("sigma=all, estimator=weighted" in x for x in lines)
    assert any("sigma=all, estimator=plain" in x for x in lines)
