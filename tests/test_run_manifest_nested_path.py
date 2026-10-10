"""A per-trial list the analysis prints under a nested path (``by_a`` / ``5`` / ``by_b`` / ``0.01``) belongs to the setting the
WHOLE path names, not to its last key: the same ``0.01`` under four different ``a`` is four settings, not four copies of one.
A live quest (first-order kinetics, 20 settings x 500 trials) stopped as "accounts for 2500" of 10000 on this shape. No model
is called."""
from __future__ import annotations

from collections import Counter
from typing import Any

from core import run_manifest as rm

A = [5, 10, 20, 50]
B = [0.01, 0.02, 0.05, 0.1, 0.2]
RUNS = 500
GRID: dict[str, list[Any]] = {"n_pts": A, "nsr": B}
PROTOCOL: dict[str, Any] = {"grid": GRID, "runs_per_setting": RUNS, "metrics": [{"id": "err", "kind": "mean"}]}


def _key(a: Any, b: Any) -> str:
    return f"nsr={b},n_pts={a}"


MANIFEST: dict[str, Any] = {"successful_per_cell": {_key(a, b): RUNS for a in A for b in B}}


def _recorded_by_cell() -> dict[str, dict[str, Counter]]:
    out: dict[str, dict[str, Counter]] = {}
    n = 0
    for a in A:
        for b in B:
            counts: Counter = Counter()
            for _ in range(RUNS):
                n += 1
                counts[f"{n}.5"] += 1
            out[_key(a, b)] = {"err": counts}
    return out


def _recorded() -> dict[str, Counter]:
    total: Counter = Counter()
    for by_name in _recorded_by_cell().values():
        total.update(by_name["err"])
    return {"err": total}


def _nested(short: tuple[Any, Any] | None = None, extra: tuple[Any, Any] | None = None) -> dict[str, Any]:
    """``by_n_pts`` / <a> / ``by_nsr`` / <b> / ``{n_runs, err_values}``."""
    out: dict[str, Any] = {"by_n_pts": {}}
    for a in A:
        by_nsr: dict[str, Any] = {}
        for b in B:
            n = RUNS
            if short == (a, b):
                n -= 100
            if extra == (a, b):
                n += 1
            by_nsr[str(b)] = {"n_runs": n, "err_values": [0.5] * n}
        out["by_n_pts"][str(a)] = {"by_nsr": by_nsr}
    return out


def _find(analysis: dict[str, Any]) -> list[tuple[str, bool]]:
    return rm._value_count_findings(PROTOCOL, MANIFEST, analysis, trial_mode=True, recorded=_recorded(),
                                    recorded_by_cell=_recorded_by_cell())


def test_the_whole_path_names_the_setting() -> None:
    assert rm._path_cells(("by_n_pts", "5", "by_nsr", "0.01"), GRID) == frozenset({("n_pts", 5), ("nsr", 0.01)})
    assert rm._path_cells(("n_pts=5", "nsr=0.01"), GRID) == frozenset({("n_pts", 5), ("nsr", 0.01)})
    assert rm._path_cells(("nsrs", "0.1"), GRID) == frozenset({("nsr", 0.1)})
    assert rm._path_cells(("per_nsr", "0.1", "n_pts", "10"), GRID) == frozenset({("nsr", 0.1), ("n_pts", 10)})
    # no setting named, an unknown value, a bare value, or one axis with two values: nothing is read
    assert rm._path_cells(("results", "0.01"), GRID) is None
    assert rm._path_cells(("by_nsr", "0.3"), GRID) is None
    assert rm._path_cells(("by_nsr", "0.1", "by_nsr", "0.2"), GRID) is None


def test_nested_lists_account_for_every_setting() -> None:
    assert _find(_nested()) == []
    groups, _ = rm._listed_by_group(_nested(), "err", GRID)
    assert len(groups) == 20
    defined, undefined, excesses, acc_defined, acc_undefined = rm._listed_counts(
        _nested(), "err", GRID, (20 * RUNS, 0), _recorded_by_cell())
    assert (acc_defined, acc_undefined, excesses) == (20 * RUNS, 0, [])


def test_without_the_grid_the_last_key_is_still_all_that_is_read() -> None:
    groups, _ = rm._listed_by_group(_nested(), "err")
    assert sorted(groups, key=float) == [str(b) for b in B]


def test_a_short_setting_lowers_what_is_accounted() -> None:
    _, _, _, acc_defined, _ = rm._listed_counts(
        _nested(short=(10, 0.05)), "err", GRID, (20 * RUNS, 0), _recorded_by_cell())
    assert acc_defined == 20 * RUNS - 100
    # a few missing stay within the tolerance, but a half-empty report does not
    half = _nested()
    for a in A[:2]:
        for b in B:
            half["by_n_pts"][str(a)]["by_nsr"][str(b)]["err_values"] = [0.5] * 100
    found = _find(half)
    assert found and "accounts for 6000" in found[0][0], found


def test_a_setting_listing_more_than_FIs_record_is_reported() -> None:
    found = _find(_nested(extra=(20, 0.2)))
    assert found and "reports more than that" in found[0][0], found
    assert "n_pts=20,nsr=0.2" in found[0][0], found[0][0]


def test_a_copy_of_the_same_setting_under_a_second_path_counts_once() -> None:
    analysis = _nested()
    swapped: dict[str, Any] = {"by_nsr": {}}
    for b in B:
        swapped["by_nsr"][str(b)] = {"by_n_pts": {
            str(a): analysis["by_n_pts"][str(a)]["by_nsr"][str(b)] for a in A}}
    both = {**analysis, **swapped}
    assert _find(both) == []
    _, _, excesses, acc_defined, _ = rm._listed_counts(both, "err", GRID, (20 * RUNS, 0), _recorded_by_cell())
    assert acc_defined == 20 * RUNS and excesses == []
    # and the copy cannot make up for a short original
    short = _nested(short=(5, 0.01))
    _, _, _, acc_defined, _ = rm._listed_counts({**short, **swapped}, "err", GRID, (20 * RUNS, 0), _recorded_by_cell())
    assert acc_defined == 20 * RUNS  # the copy under the second path holds the full list, the larger one stands
    _, _, _, acc_short, _ = rm._listed_counts(short, "err", GRID, (20 * RUNS, 0), _recorded_by_cell())
    assert acc_short == 20 * RUNS - 100
