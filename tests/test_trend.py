"""FI words the direction and size of a change along a setting, the pairwise orders of an error series, and a result against
another inside each setting, from its own trial record (core/trend.py). Neutral made-up study: a score measured at several
sizes for two groups. No model is called."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core import paper_values as pv
from core import trend as tr

NAN = float("nan")


def _metric(values: list[float]) -> dict:
    return {"values": values, "trials": list(range(len(values))), "count": len(values), "total": 0.0, "non_finite": 0}


def _cell(cell: dict, **metrics: list[float]) -> dict:
    key = ",".join(f"{k}={v}" for k, v in cell.items())
    return {"cell": cell, "key": key, "planned": 1, "ok": 1, "failed": 0, "metrics": {m: _metric(v) for m, v in metrics.items()}}


def _grid(level, sizes=(10, 20, 40), groups=("a", "b"), n=200, seed=1) -> pv.Records:
    """``level(size, group)`` is the true mean of ``score``; noise sd 0.1, ``n`` trials per setting."""
    rng = np.random.default_rng(seed)
    cells = [_cell({"size": s, "group": g}, score=list(level(s, g) + 0.1 * rng.standard_normal(n))) for g in groups for s in sizes]
    return pv.from_cells(cells)


def _say(rec: pv.Records, key: str) -> pv.Value:
    return pv.resolve("trend:" + key, rec)


# --- direction and size, from the 95% interval of last minus first ------------------------------------------------------


def test_a_series_that_falls_is_said_to_fall_with_its_size_and_ratio() -> None:
    rec = _grid(lambda s, g: 1.0 - 0.4 * (s - 10) / 30)
    value = _say(rec, "score over size @ group=a")
    assert value.filled, value.hit
    assert value.text.startswith("score over size at group = a: falls, from ")
    assert "95% interval" in value.text and "the last is 0.6" in value.text
    assert "no change" not in value.text and "flat" not in value.text
    assert value.numbers[0] == pytest.approx(0.6, abs=0.02)


def test_a_series_that_rises_is_said_to_rise() -> None:
    rec = _grid(lambda s, g: 1.0 + 0.5 * (s - 10) / 30)
    assert _say(rec, "score over size @ group=a").text.split(": ", 1)[1].startswith("rises, from ")


def test_a_series_with_no_change_is_no_change_beyond_its_interval() -> None:
    """The known negative: nothing changes along size, so FI never says it rises or falls."""
    rec = _grid(lambda s, g: 1.0)
    text = _say(rec, "score over size @ group=each").text
    assert "shows no change beyond its 95% interval at 2 of 2 values of group" in text
    assert " falls" not in text and " rises" not in text


def test_the_two_settings_of_another_axis_are_told_apart() -> None:
    """One group falls and the other does not: a sentence about both says so (a scope lost is the old fault)."""
    rec = _grid(lambda s, g: 1.0 - (0.5 * (s - 10) / 30 if g == "a" else 0.0))
    text = _say(rec, "score over size @ group=each").text
    assert "falls at 1 of 2 values of group" in text and "shows no change beyond its 95% interval at 1" in text
    assert "At group = a: falls" in text and "at group = b: no change beyond its 95% interval" in text


def test_a_series_with_one_trial_per_setting_has_no_interval_and_says_so() -> None:
    cells = [_cell({"size": s}, score=[1.0 / s]) for s in (10, 20, 40)]
    value = _say(pv.from_cells(cells), "score over size @ ")
    assert value.filled, value.hit
    assert "changes from 0.1000 at size = 10 to 0.02500 at size = 40" in value.text
    assert "no 95% interval" in value.text
    assert " falls" not in value.text and "no change beyond" not in value.text


def test_a_setting_with_no_number_is_left_out_and_said() -> None:
    cells = [_cell({"size": s, "group": "a"}, score=list(np.linspace(0.9, 1.1, 40) * (2.0 if s == 40 else 1.0)))
             for s in (10, 20, 40)]
    cells += [_cell({"size": s, "group": "b"}, score=[NAN] * 5) for s in (10, 20, 40)]
    text = _say(pv.from_cells(cells), "score over size @ group=each").text
    assert "score has no numbers at group = b" in text and "group = a" in text


def test_an_interval_that_is_paired_by_draw_sees_a_small_shift_the_unpaired_one_cannot() -> None:
    """Settings that ran on the same draws are compared draw by draw: a shift of 0.02 under a draw effect of sd 1 shows."""
    rng = np.random.default_rng(3)
    draw = rng.standard_normal(100)
    cells = [_cell({"size": 10}, score=list(draw)), _cell({"size": 20}, score=list(draw + 0.05 + 0.01 * rng.standard_normal(100)))]
    rec = pv.from_cells(cells)
    unpaired = _say(rec, "score over size @ ").text
    assert "no change beyond its 95% interval" in unpaired
    rec.seeds = {c["key"]: {i: i for i in range(100)} for c in cells}
    assert _say(rec, "score over size @ ").text.startswith("score over size: rises")


def test_the_shape_names_where_the_means_are_lowest_and_highest() -> None:
    rec = _grid(lambda s, g: {10: 1.0, 20: 3.0, 40: 2.0}[s], groups=("a",))
    text = _say(rec, "score over size @ group=a").text
    assert "the means are not monotone (lowest at size = 10, highest at size = 20)" in text


# --- refusals ------------------------------------------------------------------------------------------------------------


def test_a_pooled_other_setting_is_refused_with_a_plain_reason() -> None:
    rec = _grid(lambda s, g: 1.0)
    value = _say(rec, "score over size @ group=all")
    assert value.text == pv.NOT_RECORDED and "name each setting" in value.hit.reason


def test_an_axis_the_results_do_not_carry_can_be_pooled() -> None:
    cells = [_cell({"size": s, "group": "a"}, score=[1.0, 2.0, 3.0 + s]) for s in (10, 20)]
    cells += [_cell({"size": s, "group": "b"}, other=[1.0, 2.0]) for s in (10, 20)]  # group varies, but not for ``score``
    value = _say(pv.from_cells(cells), "score over size @ group=all")
    assert value.filled, value.hit


def test_an_unknown_result_a_missing_axis_and_a_setting_of_the_series_are_refused() -> None:
    rec = _grid(lambda s, g: 1.0)
    assert "not a value the trials recorded" in _say(rec, "nothing over size @ group=a").hit.reason
    assert "not a setting of the record" in _say(rec, "score over speed @ group=a").hit.reason
    assert "leave it out" in _say(rec, "score over size @ group=a, size=10").hit.reason
    assert "every axis the study varies" in _say(rec, "score over size @ ").hit.reason


def test_a_setting_that_is_not_numeric_has_no_ordered_change() -> None:
    cells = [_cell({"kind": k}, score=[1.0, 2.0, 3.0]) for k in ("x", "y")]
    assert "no ordered numeric values" in _say(pv.from_cells(cells), "score over kind @ ").hit.reason


# --- pairwise orders ---------------------------------------------------------------------------------------------------


def _errors(errs: list[float], sizes: list[int]) -> pv.Records:
    return pv.from_cells([_cell({"steps": n}, error=[e]) for n, e in zip(sizes, errs)])


STEPS = [16, 32, 64, 128, 256, 512]


def test_an_irregular_pair_at_the_coarsest_steps_is_named_the_coarsest_not_the_finest() -> None:
    errs = [8.0e-4, 8.1e-4, 8.1e-4 / 4, 8.1e-4 / 16, 8.1e-4 / 64, 8.1e-4 / 256]
    text = _say(_errors(errs, STEPS), "error over steps @ | order").text
    assert "steps = 16 to 32: -0.0" in text
    assert "the error does not fall at steps = 16 to 32 (the coarsest pair)" in text
    assert "finest pair)" not in text.split("the error does not fall")[1]


def test_an_irregular_pair_at_the_finest_steps_is_named_the_finest() -> None:
    errs = [1e-2, 1e-2 / 4, 1e-2 / 16, 1e-2 / 64, 1e-2 / 256, 1e-2 / 256 * 1.2]
    text = _say(_errors(errs, STEPS), "error over steps @ | order").text
    assert "the error does not fall at steps = 256 to 512 (the finest pair)" in text


def test_clean_second_order_gives_two_at_every_pair_and_no_pair_that_fails_to_fall() -> None:
    errs = [1e-2 / 4 ** i for i in range(6)]
    value = _say(_errors(errs, STEPS), "error over steps @ | order")
    assert value.text.count(": 2.00") == 5 and "does not fall" not in value.text
    assert value.numbers == [pytest.approx(2.0), pytest.approx(2.0)]


def test_an_axis_of_step_sizes_is_read_the_other_way_round() -> None:
    sizes = [0.5, 0.25, 0.125, 0.0625]
    errs = [(h ** 3) for h in sizes]
    text = _say(_errors(errs, sizes), "error over steps @ | order_size").text
    assert text.count(": 3.00") == 3 and "(the coarsest pair)" not in text


def test_a_count_of_steps_the_writer_called_a_step_size_is_still_read_by_where_the_error_falls() -> None:
    sizes = [2 ** k for k in range(2, 13)]
    errs = [1e-2 * 16.0 ** -k for k in range(11)]
    value = _say(_errors(errs, sizes), "error over steps @ | order_size")
    assert value.numbers == [pytest.approx(4.0), pytest.approx(4.0)]
    assert "-4" not in value.text and "does not fall" not in value.text
    assert "coarsest" not in value.text and "finest" not in value.text
    assert "the pair at the high-error end" in value.text


def test_when_the_mean_error_is_the_same_at_both_ends_the_orders_are_absolute_and_no_verdict_is_given() -> None:
    sizes = [2 ** k for k in range(2, 13)]
    errs = [1e-3, 5e-4, 2e-4, 5e-4, 1e-3, 5e-4, 2e-4, 5e-4, 1e-3, 5e-4, 1e-3]
    text = _say(_errors(errs, sizes), "error over steps @ | order").text
    assert "does not fall" not in text and "coarsest" not in text and "the record does not say" in text


def test_an_order_with_an_error_of_zero_is_not_defined() -> None:
    text = _say(_errors([1e-3, 2.5e-4, 0.0, 0.0], [16, 32, 64, 128]), "error over steps @ | order").text
    assert "steps = 16 to 32: 2.00" in text and "steps = 32 to 64: not defined" in text


def test_many_pairs_give_the_range_and_where_the_lowest_and_highest_are() -> None:
    sizes = [2 ** k for k in range(2, 14)]
    errs = [4.0 ** -k for k in range(12)]
    errs[1] = errs[0] * 1.5
    text = _say(_errors(errs, sizes), "error over steps @ | order").text
    assert "orders between 11 neighbouring pairs" in text
    assert "lowest" in text and "steps = 4 to 8 (the coarsest pair)" in text


# --- a result against another inside each setting --------------------------------------------------------------------------


def _study(sigmas=(0.1, 1.0, 10.0), sizes=(5, 20), n=300, seed=7) -> tuple[pv.Records, dict]:
    rng = np.random.default_rng(seed)
    cells, truth = [], {}
    for sg in sigmas:
        for k in sizes:
            est = sg / math.sqrt(k) * rng.standard_normal(n)
            err = sg / math.sqrt(k) * (1 + 0.05 * rng.standard_normal(n))
            cells.append(_cell({"noise": sg, "points": k}, estimate=list(est), reported_error=list(err)))
            truth[(sg, k)] = (float(np.std(est, ddof=1)), float(np.sqrt(np.mean(err ** 2))))
    return pv.from_cells(cells), truth


def test_the_ratio_of_an_sd_to_the_rms_of_the_reported_error_is_worked_out_in_each_setting() -> None:
    rec, truth = _study()
    value = pv.resolve("comparison:estimate sd vs reported_error rms @ noise=each, points=each", rec)
    assert value.filled, value.hit
    ratios = sorted(a / b for a, b in truth.values())
    assert value.numbers == [pytest.approx(ratios[0]), pytest.approx(ratios[-1])]
    assert "over 6 settings" in value.text and "lowest at noise = " in value.text
    assert "lies inside the 95% interval of the SD of estimate at" in value.text


def test_a_spread_pooled_over_settings_of_different_noise_is_refused_by_name() -> None:
    """The old fault: a pooled SD (RMS over settings of different noise) set beside a per-setting error. FI refuses it."""
    rec, _ = _study()
    value = pv.resolve("comparison:estimate sd vs reported_error rms @ noise=all, points=5", rec)
    assert value.text == pv.NOT_RECORDED
    assert "noise=all pools settings that differ in noise" in value.hit.reason and "name each setting" in value.hit.reason


def test_an_sd_is_not_compared_with_the_mean_of_an_error() -> None:
    rec, _ = _study()
    value = pv.resolve("comparison:estimate sd vs reported_error mean @ noise=each, points=each", rec)
    assert "not like for like" in value.hit.reason and "root-mean-square" in value.hit.reason


def test_a_comparison_needs_a_setting_kept_apart() -> None:
    rec, _ = _study(sigmas=(1.0,), sizes=(5,))
    assert "mark at least one setting =each" in pv.resolve("comparison:estimate sd vs reported_error rms @ noise=1.0, points=5", rec).hit.reason


# --- the key as the paper holds it ---------------------------------------------------------------------------------------


def test_a_typed_trend_key_is_filled_between_markers_and_turned_back_for_the_next_round() -> None:
    rec = _grid(lambda s, g: 1.0 - 0.4 * (s - 10) / 30)
    typed = "The score drops: {{trend:score over size @ group=a}} as stated."
    filled, hits = pv.fill(typed, rec)
    assert not hits and "<!-- fi:v {{v:trend:score over size @ group=a}} -->score over size at group = a: falls" in filled
    assert pv.to_placeholders(filled) == typed
    assert pv.keys_in(typed) == ["trend:score over size @ group=a"]
    assert pv.without_marks(filled).startswith("The score drops: score over size at group = a: falls")


def test_a_trend_key_places_no_quantity_for_the_check_of_asked_answers() -> None:
    rec = _grid(lambda s, g: 1.0 - 0.4 * (s - 10) / 30)
    filled, _ = pv.fill("{{trend:score over size @ group=a}}", rec)
    assert pv.placed_metrics(filled) == set()


def test_the_writer_is_told_the_keys_exist() -> None:
    rec = _grid(lambda s, g: 1.0)
    block = pv.writer_block(rec)
    assert "{{trend:" in block and "{{comparison:" in block


def test_a_range_over_settings_leaves_out_a_setting_without_a_number_and_says_so() -> None:
    """Before: one all-NaN setting made the whole ``| range`` key fail."""
    cells = [_cell({"size": 10, "group": "a"}, score=[1.0, 2.0]), _cell({"size": 20, "group": "a"}, score=[3.0, 5.0]),
             _cell({"size": 10, "group": "b"}, score=[NAN, NAN])]
    value = pv.resolve("{{v:score @ size=each, group=each | range}}", pv.from_cells(cells))
    assert value.filled, value.hit
    assert value.text == "1.500 to 4.000 (no number at size = 10, group = b)"


def _empty_cells(missing, sizes=(10, 20, 40), groups=("a", "b"), kinds=("x", "y")):
    return [_cell({"size": s, "group": g, "kind": k}, score=[NAN, NAN] if missing(s, g, k) else [1.0, 2.0])
            for s in sizes for g in groups for k in kinds]


def _empty_note(missing) -> str:
    value = pv.resolve("{{v:score @ size=each, group=each, kind=each | range}}", pv.from_cells(_empty_cells(missing)))
    assert value.filled, value.hit
    return value.text[value.text.index("(no number"):]


def test_settings_without_a_number_are_all_named_never_cut_off_with_a_count() -> None:
    """Before: '(no number at size = 10, group = a, kind = x; ... and 5 more)' left the reader unable to tell which."""
    whole = _empty_note(lambda s, g, k: g == "a")
    assert whole == "(no number where group = a, for every size and every kind)"
    part = _empty_note(lambda s, g, k: g == "a" and s in (10, 20))
    assert part == "(no number where size = 10 or 20, group = a, for every kind)"
    scattered = _empty_note(lambda s, g, k: (s, g, k) in {(10, "a", "x"), (20, "b", "y"), (40, "a", "y"), (40, "b", "y")})
    assert "more" not in scattered
    for setting in ("size = 10, group = a, kind = x", "size = 20, group = b, kind = y"):
        assert setting.split(", ")[0] in scattered
    assert scattered.count(";") == 2 and "size = 40, kind = y" in scattered


def test_a_single_setting_without_a_number_is_still_named_by_itself() -> None:
    assert _empty_note(lambda s, g, k: (s, g, k) == (20, "b", "y")) == "(no number at size = 20, group = b, kind = y)"


# --- review fixes: ratio of a signed value, plateau, exact values, paired subset, kept failures ---------------------------


def test_a_ratio_is_printed_only_when_both_ends_are_above_zero() -> None:
    """Before: with a negative first value 'the last is 0.3 times the first' was printed, which means nothing."""
    negative = _say(_grid(lambda s, g: -0.2 - 0.4 * (s - 10) / 30, groups=("a",)), "score over size @ group=a")
    assert negative.filled and "falls" in negative.text and "times the first" not in negative.text
    crossing = _say(_grid(lambda s, g: 0.5 - 1.0 * (s - 10) / 30, groups=("a",)), "score over size @ group=a")
    assert "falls" in crossing.text and "times the first" not in crossing.text
    assert crossing.numbers == []
    one_trial = _say(pv.from_cells([_cell({"size": s}, score=[v]) for s, v in ((10, -1.0), (20, -2.0))]), "score over size @ ")
    assert "times the first" not in one_trial.text
    positive = _say(_grid(lambda s, g: 1.0 - 0.4 * (s - 10) / 30, groups=("a",)), "score over size @ group=a")
    assert "the last is 0.6" in positive.text  # a known positive: both above zero keeps the ratio


def test_equal_neighbouring_means_are_a_plateau_not_non_monotone() -> None:
    cells = [_cell({"size": 10}, score=[0.9, 1.1] * 20), _cell({"size": 20}, score=[0.4, 0.6] * 20),
             _cell({"size": 40}, score=[0.4, 0.6] * 20)]
    text = _say(pv.from_cells(cells), "score over size @ ").text
    assert "not monotone" not in text and "never rise" in text
    bumpy = [_cell({"size": s}, score=[v - 0.1, v + 0.1] * 20) for s, v in ((10, 1.0), (20, 0.5), (40, 0.8))]
    assert "not monotone" in _say(pv.from_cells(bumpy), "score over size @ ").text  # a real bump is still named


def test_trials_with_no_spread_on_both_sides_are_exact_not_a_zero_width_interval() -> None:
    cells = [_cell({"size": 10}, score=[1.0, 1.0, 1.0]), _cell({"size": 20}, score=[0.5, 0.5, 0.5])]
    text = _say(pv.from_cells(cells), "score over size @ ").text
    assert "falls" in text and "identical, so the values are exact" in text
    assert "95% interval, " not in text and "(change -0.5000, 95%" not in text
    same = [_cell({"size": 10}, score=[1.0, 1.0]), _cell({"size": 20}, score=[1.0, 1.0])]
    assert "no change" in _say(pv.from_cells(same), "score over size @ ").text


def test_a_paired_interval_over_fewer_draws_says_how_many_were_paired() -> None:
    """Before: the means printed were over every trial and the change over the paired draws, with no word of it."""
    rng = np.random.default_rng(5)
    draw = rng.standard_normal(100)
    cells = [_cell({"size": 10}, score=list(draw)), _cell({"size": 20}, score=list(draw[:60] + 0.5 + 0.01 * rng.standard_normal(60)))]
    rec = pv.from_cells(cells)
    rec.seeds = {cells[0]["key"]: {i: i for i in range(100)}, cells[1]["key"]: {i: i for i in range(60)}}
    text = _say(rec, "score over size @ ").text
    assert "the means are over all 100 and 60 trials, the change and its interval over the 60 paired draws" in text


def test_the_same_shift_on_every_shared_draw_is_not_called_identical_trials() -> None:
    """The second review's input: draws that vary, each shifted by the same 0.5 on the same seeds, were said to have
    "identical" trials; and a shift that is not a binary fraction printed a 17-digit interval at the rounding."""
    for shift in (0.5, 0.1):
        draws = [0.25, 0.75, 0.5, 1.0, 0.125]
        cells = [_cell({"size": 10}, score=draws), _cell({"size": 20}, score=[d + shift for d in draws])]
        rec = pv.from_cells(cells)
        rec.seeds = {c["key"]: {i: i for i in range(5)} for c in cells}
        text = _say(rec, "score over size @ ").text
        assert "rises" in text and "changed by the same amount" in text, text
        assert "identical" not in text and "0.0000000000" not in text


def test_a_refinement_ratio_of_three_gives_the_order_from_the_real_ratio() -> None:
    errs = [1e-2 / 9 ** i for i in range(4)]  # second order, the step shrinks by 3 each time
    value = _say(_errors(errs, [9, 27, 81, 243]), "error over steps @ | order")
    assert value.text.count(": 2.00") == 3 and "does not fall" not in value.text


def test_a_diverged_setting_makes_a_largest_value_fail_but_a_range_names_it() -> None:
    """Before: ``max_abs`` over the settings that survived read as the largest over all of them."""
    cells = [_cell({"size": 10, "group": "a"}, score=[1.0, 2.0]), _cell({"size": 20, "group": "a"}, score=[3.0, -5.0]),
             _cell({"size": 10, "group": "b"}, score=[NAN, NAN])]
    rec = pv.from_cells(cells)
    for reducer in ("max_abs", "max", "min"):
        value = pv.resolve("{{v:score @ size=each, group=each | %s}}" % reducer, rec)
        assert not value.filled and "not computable at size = 10, group = b" in value.hit.reason, reducer
    assert pv.resolve("{{v:score @ size=each, group=each | range}}", rec).filled


def test_a_setting_without_a_number_is_told_by_its_exception_not_by_its_wording(monkeypatch) -> None:
    cells = [_cell({"size": 10}, score=[1.0, 2.0]), _cell({"size": 20}, score=[3.0, 5.0])]
    rec = pv.from_cells(cells)
    real = pv._one

    def one(r, key, fixed):
        if fixed["size"].value == 10:
            raise pv._NoNumbers("a reworded reason")
        return real(r, key, fixed)

    monkeypatch.setattr(pv, "_one", one)
    value = pv.resolve("{{v:score @ size=each | range}}", rec)
    assert value.filled and "no number at size = 10" in value.text


def test_a_statistic_that_cannot_be_worked_out_is_not_recorded_not_a_broken_write(monkeypatch) -> None:
    def boom(rec, body):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(tr, "trend_text", boom)
    value = _say(_grid(lambda s, g: 1.0), "score over size @ group=a")
    assert not value.filled and "ZeroDivisionError" in value.hit.reason


def test_a_change_between_single_runs_says_there_is_no_interval_and_no_verdict_clause() -> None:
    """Before: the sentence ended 'and FI does not say whether the change is more than noise', a fact already in the run facts
    (one run per setting). Now it says there is no interval and stops."""
    cells = [_cell({"size": s, "group": "a"}, score=[1.0 + 0.1 * s]) for s in (10, 20, 40)]
    text = _say(pv.from_cells(cells), "score over size @ group=a").text
    assert text.startswith("score over size at group = a: changes from ")
    assert text.endswith("so there is no 95% interval")
    assert "more than noise" not in text
    assert "95% interval" in _say(_grid(lambda s, g: 1.0 - 0.4 * (s - 10) / 30), "score over size @ group=a").text  # an interval is still said when there is one
