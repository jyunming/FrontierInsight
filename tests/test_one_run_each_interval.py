"""A mean over settings that each ran once has no interval: its values are different settings, not repeated draws."""
from __future__ import annotations

from core import paper_values as pv


def _cell(a: int, values: list[float]) -> dict:
    return {"cell": {"a": a}, "key": f"a={a}", "planned": len(values), "ok": len(values), "failed": 0,
            "metrics": {"y": {"values": values, "trials": list(range(len(values))), "count": len(values),
                              "total": sum(values), "non_finite": 0}}}


def _text(cells: list[dict], key: str) -> str:
    return pv.resolve(key, pv.Records(cells=cells)).text


def test_one_run_per_setting_gives_the_mean_and_no_interval():
    cells = [_cell(a, [0.1 * a]) for a in range(1, 6)]
    out = _text(cells, "y:mean_ci @ a=all")
    assert "mean over 5 settings, one run each; no interval" in out
    assert "trials" not in out and " to " not in out


def test_one_run_per_setting_gives_no_ci_or_se():
    cells = [_cell(a, [0.1 * a]) for a in range(1, 6)]
    for form in ("ci", "pm", "se"):
        assert _text(cells, f"y:{form} @ a=all") == "not recorded"


def test_repeated_runs_keep_their_interval():
    cells = [_cell(a, [0.1 * a + d for d in (0.0, 0.02, -0.02, 0.01)]) for a in range(1, 4)]
    out = _text(cells, "y:mean_ci @ a=all")
    assert "95% t interval" in out and "12 trials" in out


def test_a_setting_that_gave_no_value_is_not_counted_as_a_run():
    cells = [_cell(a, [0.1 * a] if a <= 3 else []) for a in range(1, 6)]
    assert "mean over 3 settings, one run each; no interval" in _text(cells, "y:mean_ci @ a=all")


def test_a_comparison_with_the_reference_over_settings_that_each_ran_once_is_left_unfilled_not_raised():
    """``vs ref`` judges above / within / below by an interval; settings that each ran once have none (it used to
    raise an AssertionError, and before the one-run rule it pooled the settings into an interval)."""
    protocol = {"grid": {"m": [20, 200], "r": [1, 2, 3], "pi1": [0.2]}, "thresholds": {"alpha": 0.05},
                "metrics": [{"id": "y", "reference": {"formula": "(m - floor(pi1*m + 0.5))/m*alpha", "kind": "value"}}]}
    cells = []
    for m in (20, 200):
        for r in (1, 2, 3):
            ref = (m - int(0.2 * m + 0.5)) / m * 0.05
            cells.append({"cell": {"m": m, "r": r, "pi1": 0.2}, "key": f"m={m},r={r},pi1=0.2", "planned": 1, "ok": 1,
                          "failed": 0, "metrics": {"y": {"values": [ref * (1 + 0.01 * r)]}}})
    got = pv.resolve("y:mean @ m=each, r=all, pi1=0.2 | vs ref", pv.from_cells(cells, protocol))
    assert not got.filled and "each setting ran once" in got.hit.reason
