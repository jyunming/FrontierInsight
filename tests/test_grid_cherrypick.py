"""The best results picked out of every setting cannot pass as the mean over a subset of the trials.

Reporting, for each setting of the grid, only the ten largest values as the "subset", with `p_count` = 10 and the true
`p_total`, kept every per-cell check happy: the values are real trial values, the count matches the list, the total
matches the trials run, and all settings are covered. Where FI cannot count the subset itself (the trials do not
return the proportion as 1 or 0), values that are exactly the largest (or smallest) of a setting's trials are now
sent back, saying which end and the one way out (return the proportion per trial), unless a cut-off the protocol fixes
on that quantity explains them, and unless the setting has too few trials for it to mean anything.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from core import trial_runner as tr

GRID = {"R0": [1.5, 3.0], "N": [100]}
PROTOCOL = {"grid": GRID, "runs_per_setting": 100, "metrics": [
    {"id": "p", "kind": "proportion", "estimand": "P(major)", "unit": "run"},
    {"id": "m", "kind": "mean", "estimand": "E[m | major]", "unit": "run", "given": "p"},
]}
CELLS = ["R0=1.5,N=100", "R0=3.0,N=100"]


def _m(t: int) -> float:
    """A value unrelated to membership, so the subset is not the top of it."""
    return float((t * 37) % 101 + 1)


def _rows(membership: bool) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key in CELLS:
        rows = []
        for t in range(100):
            values: dict[str, float] = {"m": _m(t)}
            if membership:
                values["p"] = 1.0 if t % 3 == 0 else 0.0
            rows.append({"cell": key, "trial": t, "seed": t, "values": values})
        out[key] = rows
    return out


def _per_cell(rows: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Counter]]:
    return {key: {name: Counter(str(r["values"][name]) for r in rs if name in r["values"])
                  for name in ("m", "p") if any(name in r["values"] for r in rs)} for key, rs in rows.items()}


def _stratum(values: list[float], count: int, total: int = 100) -> dict[str, Any]:
    return {"p_count": count, "p_total": total, "m_values": values}


TOP10 = sorted((_m(t) for t in range(100)), reverse=True)[:10]
BOTTOM10 = sorted(_m(t) for t in range(100))[:10]
MEMBERS = [_m(t) for t in range(100) if t % 3 == 0]


def _cherry(values: list[float]) -> dict[str, Any]:
    return {key: _stratum(values, 10) for key in CELLS}


def test_the_ten_best_of_every_setting_are_sent_back_when_fi_cannot_count_the_subset() -> None:
    per_cell = _per_cell(_rows(membership=False))
    for picked in (TOP10, BOTTOM10):
        problems = tr.given_values_not_run(PROTOCOL, per_cell, _cherry(picked))
        assert len(problems) == len(CELLS) and all("picked by hand" in p for p in problems)
        assert all("have run_trial return `p` as 1 or 0" in p for p in problems)
    side = lambda picked: tr.given_values_not_run(PROTOCOL, per_cell, _cherry(picked))[0]  # noqa: E731
    assert "10 largest" in side(TOP10) and "10 smallest" in side(BOTTOM10)


def test_the_same_pick_is_sent_back_as_fractions_of_n() -> None:
    per_cell = _per_cell(_rows(membership=False))
    problems = tr.given_values_not_run(PROTOCOL, per_cell, _cherry([v / 100 for v in TOP10]))
    assert len(problems) == len(CELLS)


def test_the_same_pick_is_still_caught_when_the_trials_return_membership() -> None:
    rows = _rows(membership=True)
    result = _cherry(TOP10)
    analysis, _ = tr.given_rows_problems(PROTOCOL, rows, result)
    assert analysis, "the per-trial check still blocks it"
    assert tr.given_values_not_run(PROTOCOL, _per_cell(rows), result)


def test_a_genuine_subset_and_the_whole_strata_pass() -> None:
    per_cell = _per_cell(_rows(membership=False))
    subset = {key: _stratum(MEMBERS, len(MEMBERS)) for key in CELLS}
    assert tr.given_values_not_run(PROTOCOL, per_cell, subset) == []
    every = {key: _stratum([_m(t) for t in range(100)], 100) for key in CELLS}
    assert tr.given_values_not_run(PROTOCOL, per_cell, every) == []
    fractions = {key: _stratum([_m(t) / 100 for t in range(100) if t % 3 == 0], len(MEMBERS)) for key in CELLS}
    assert tr.given_values_not_run(PROTOCOL, per_cell, fractions) == []


def test_a_short_list_and_a_membership_the_trials_record_are_not_flagged() -> None:
    per_cell = _per_cell(_rows(membership=False))
    two = {key: _stratum(TOP10[:2], 2) for key in CELLS}
    assert tr.given_values_not_run(PROTOCOL, per_cell, two) == []
    with_p = _per_cell(_rows(membership=True))
    honest = {key: _stratum(MEMBERS, len(MEMBERS)) for key in CELLS}
    assert tr.given_values_not_run(PROTOCOL, with_p, honest) == []


def _with_cutoff(cutoff: float) -> dict[str, Any]:
    return {**PROTOCOL, "thresholds": {"major": cutoff}}


def test_a_subset_defined_by_a_declared_cutoff_on_the_same_quantity_passes() -> None:
    per_cell = _per_cell(_rows(membership=False))
    above = [_m(t) for t in range(100) if _m(t) > 90.5]
    below = [_m(t) for t in range(100) if _m(t) <= 10.5]
    assert len(above) >= 3 and len(below) >= 3
    for cutoff, picked in ((90.5, above), (10.5, below)):
        result = {key: _stratum(picked, len(picked)) for key in CELLS}
        assert tr.given_values_not_run(_with_cutoff(cutoff), per_cell, result) == []
        assert tr.given_values_not_run(PROTOCOL, per_cell, result), "without the declared cut-off it is still refused"
    fractions = {key: _stratum([v / 100 for v in above], len(above)) for key in CELLS}
    assert tr.given_values_not_run(_with_cutoff(0.905), per_cell, fractions) == []


def test_a_cutoff_that_does_not_separate_the_pick_does_not_excuse_it() -> None:
    per_cell = _per_cell(_rows(membership=False))
    assert tr.given_values_not_run(_with_cutoff(50.0), per_cell, _cherry(TOP10))


def test_a_setting_with_too_few_trials_to_tell_is_not_judged() -> None:
    small = {key: [{"cell": key, "trial": t, "seed": t, "values": {"m": _m(t)}} for t in range(8)] for key in CELLS}
    per_cell = _per_cell(small)
    top3 = sorted((_m(t) for t in range(8)), reverse=True)[:3]
    assert tr.given_values_not_run(PROTOCOL, per_cell, _cherry(top3)) == []


def test_the_message_counts_the_values_it_judged_not_the_nulls_beside_them() -> None:
    per_cell = _per_cell(_rows(membership=False))
    padded = {key: _stratum(TOP10 + [None, None], 10) for key in CELLS}
    problems = tr.given_values_not_run(PROTOCOL, per_cell, padded)
    assert problems and all("your 10 values" in p and "10 largest" in p for p in problems)
