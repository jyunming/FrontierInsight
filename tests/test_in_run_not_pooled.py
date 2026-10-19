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


def test_sheet_still_offers_the_whole_pool_of_a_one_number_per_run_metric_and_no_line_twice():
    sheet = pv.sheet(pv.from_cells(_cells(), PROTOCOL))
    lines = [x for x in sheet.splitlines() if x.startswith("- ")]
    assert any(x.startswith("- {{v:ea:mean_ci @ sigma=all, estimator=all}}") for x in lines)
    assert len(lines) == len(set(lines))


def _many(settings: int, levels: int) -> list[dict]:
    cells = []
    for s in range(settings):
        for lv in range(levels):
            cells.append({"cell": {"size": float(s), "shift": lv * 0.1, "route": "Direct"}, "level": {"shift": lv * 0.1},
                          "metrics": {"k": {"values": [float(s) + lv]}, "w": {"values": [1.0, 2.0, float(lv)]}}})
        cells.append({"cell": {"size": float(s), "route": "Direct"}, "metrics": {"ea": {"values": [1.0, 2.0, 3.0]}}})
    return cells


def test_a_setting_is_found_by_its_values_as_the_comparison_finds_it():
    rec = pv.from_cells(_many(3, 4), {"in_each_run": {"shift": [0.0, 0.1, 0.2, 0.30000000000000004]}})
    by_text = pv.resolve("w @ size=2, shift=0.3, route=direct", rec)  # 0.30000000000000004 and "Direct", as compared
    assert by_text.filled and by_text.inputs["trials"] == 3
    assert pv.resolve("w @ size=2, shift=0.30000000000000004, route=Direct", rec).text == by_text.text
    assert not pv.resolve("w @ size=7, shift=0.3, route=Direct", rec).filled
    assert pv.resolve("w @ size=all, shift=0.1, route=Direct", rec).inputs["trials"] == 9


def test_the_sheet_of_a_record_per_value_is_built_without_rescanning_it(monkeypatch):
    """A real study compared 102 values inside each of 4452 runs: 455,292 records, and the writer's sheet looked up
    every key by scanning all of them (each axis list again too), for hours. Each lookup reads only its setting now."""
    rec = pv.from_cells(_many(60, 50), {"in_each_run": {"shift": [lv * 0.1 for lv in range(50)]}})
    scanned = []
    real = pv._same
    monkeypatch.setattr(pv, "_same", lambda a, b: scanned.append(1) or real(a, b))
    text = pv.sheet(rec)
    assert "{{v:w:mean_ci @ size=0.0, shift=0.0, route=Direct}}" in text or "Too many settings" in text
    assert len(scanned) < 1000 * len(rec.cells), len(scanned)  # it was 23,000 per record
