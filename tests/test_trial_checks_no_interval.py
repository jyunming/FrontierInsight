"""A trial whose fit gave no estimate at all, counted apart from the coverage of the reported intervals
(core/trial_checks.py): the verdict's coverage is counted over the trials that gave an estimate, and those trials are judged
by the count of trials with no estimate (a fault above half of a setting's trials). A trial that gave an estimate and no
interval is still a miss. The coverage counting every trial without an interval as a miss is still measured, and where it
is grossly low while the verdict's is not, FI states it in the paper's Limitations and in QUALITY.md, never as a repair or
a stop. Made-up settings, no model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from core import frame_table, frozen_protocol, paper_facts, quality_report, trial_checks as tc

NAN = float("nan")
TRUE = 2.0
N = 200


def _spec() -> dict:
    return {"id": "rate", "estimand": "rate", "kind": "mean", "unit": "one trial", "label": "Recovered rate",
            "known_value": {"true_value": TRUE, "estimate": "rate_hat", "interval": ["rate_lo", "rate_hi"], "level": 0.95}}


def _trials(half_width: float = 0.05, gone: int = 0, ends_only: int = 0) -> dict[str, list[float]]:
    """``N`` trials of one setting: estimates near the truth with intervals ``half_width`` either side; the first ``gone``
    trials give no number at all (a fit that failed), the next ``ends_only`` keep their estimate and lose the interval."""
    est = [TRUE + 0.01 * ((i % 7) - 3) for i in range(N)]
    lo, hi = [e - half_width for e in est], [e + half_width for e in est]
    for i in range(gone):
        est[i] = lo[i] = hi[i] = NAN
    for i in range(gone, gone + ends_only):
        lo[i] = hi[i] = NAN
    return {"rate_hat": est, "rate_lo": lo, "rate_hi": hi}


def _cells(per: dict[str, dict[str, list[float]]]) -> list[dict]:
    return [{"key": key, "cell": {"s": i}, "planned": N, "failed": 0,
             "metrics": {m: {"values": v, "trials": list(range(len(v)))} for m, v in metrics.items()}}
            for i, (key, metrics) in enumerate(per.items())]


def _honest() -> list[dict]:
    """Two of six settings where 35% of the fits failed (no estimate, no interval); every interval given covers."""
    return _cells({f"s={i}": _trials(gone=70 if i < 2 else 0) for i in range(6)})


def _verdicts(cells: list[dict]) -> tuple[tc.Verdict, tc.Verdict]:
    proto = {"metrics": [_spec()]}
    return (tc.judge(cells, proto, None, {}, repairs_left=True),
            tc.judge(cells, proto, None, {}, repairs_left=False))


def test_fits_that_gave_no_estimate_neither_ask_for_a_repair_nor_stop() -> None:
    left, spent = _verdicts(_honest())
    # Counting the 70 failed fits as misses the coverage is 130 of 200 (grossly low); over the 130 that gave an estimate it
    # is 1.0. No repair of the code can give a failed fit an interval.
    assert left.action == tc.OK, left.directive
    assert spent.action == tc.OK, spent.say
    assert not spent.calibration.faults
    cover = next(q for q in spent.calibration.quantities if q.kind == "coverage")
    assert {s.key: s.n for s in cover.settings}["s=0"] == 130 and not cover.gross
    empty = next(q for q in spent.calibration.quantities if q.kind == "no_estimate")
    assert [round(s.value, 2) for s in empty.settings] == [0.35, 0.35] and not empty.fault


def test_the_count_over_every_trial_is_kept_apart_and_never_a_fault() -> None:
    _left, spent = _verdicts(_honest())
    [apart] = spent.calibration.counted_as_misses
    assert apart.kind == tc.ALL_TRIALS and apart not in spent.calibration.quantities
    marked = {s.key: (round(s.value, 3), s.n) for s in apart.settings if s.gross}
    assert marked == {"s=0": (0.65, 200), "s=1": (0.65, 200)}
    assert apart.given["s=0"] == 1.0
    assert "counting each trial that gave no interval as a miss" in apart.says()
    rec = spent.record()["calibration"]
    [kept] = rec["counted_as_misses"]
    assert kept["fault"] is False and kept["gross"] == 2 and kept["withheld"] == []
    assert not any(q["kind"] == tc.ALL_TRIALS for q in rec["quantities"])
    # The analysis's own count (core/frame_table.py hits) is unchanged: every trial with no interval is a miss there.
    m = _honest()[0]["metrics"]
    assert sum(frame_table.hits(m, ["rate_lo", "rate_hi"], TRUE, estimate=m["rate_hat"])) == 130.0


def test_the_paper_and_quality_report_state_the_count_over_every_trial(tmp_path: Path) -> None:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    frozen_protocol.freeze(root, {"metrics": [_spec()], "runs_per_setting": N}, approved_by="test", source="test")
    _left, spent = _verdicts(_honest())
    tc.write_record(root, spent.record())
    said = [i.sentence for i in paper_facts.required_items(root) if i.id.startswith("calibration-counted-misses")]
    assert said == [
        'Counting each trial that gave no interval as a miss, the 95% intervals reported for "Recovered rate" contain the '
        "true value in far fewer trials than 95% in 2 of 6 settings (s=0: 0.65, 130 of 200 trials, 1.00 of those that "
        "gave an estimate; s=1: 0.65, 130 of 200 trials, 1.00 of those that gave an estimate). FI judged the intervals on "
        "the trials that gave an estimate; the trials that gave none are stated apart."]
    assert said[0] in paper_facts.limitations_text(root)  # the Limitations block FI writes into the paper
    summary = quality_report._trial_checks_summary(SimpleNamespace(trial_checks=tc.read_record(root)))  # type: ignore[arg-type]
    assert summary is not None and summary["faults"] == 0
    assert any(line.startswith("counting each trial with no interval as a miss, grossly low at: s=0: 0.650 from 200 trials "
                               "(1.000 over the trials that gave an estimate)") for line in summary["lines"]), summary["lines"]


def test_a_trial_that_kept_its_estimate_and_lost_its_interval_is_still_a_miss() -> None:
    # a code that drops the interval where it would miss must not lift the coverage: still a fault (sent to repair; once
    # the repairs are spent, grossly low at two settings while right over all of them together, the existing disclosure)
    cells = _cells({f"s={i}": _trials(ends_only=70 if i < 2 else 0) for i in range(6)})
    left, spent = _verdicts(cells)
    assert left.action == tc.REPAIR and spent.action == tc.DISCLOSE
    [fault] = spent.calibration.faults
    assert fault.kind == "coverage" and fault.gross == 2
    assert not spent.calibration.counted_as_misses  # the two counts agree there: nothing apart to state


def test_a_too_narrow_interval_still_stops_where_some_fits_gave_no_estimate() -> None:
    cells = _cells({f"s={i}": _trials(half_width=0.005, gone=70 if i < 2 else 0) for i in range(6)})
    left, spent = _verdicts(cells)
    assert left.action == tc.REPAIR and spent.action == tc.STOP
    assert [q.kind for q in spent.calibration.faults] == ["coverage"]
    assert "grossly wrong" in spent.say


def test_most_fits_with_no_estimate_is_still_a_fault() -> None:
    cells = _cells({f"s={i}": _trials(gone=120 if i < 2 else 0) for i in range(6)})
    left, spent = _verdicts(cells)
    assert left.action == tc.REPAIR and spent.action == tc.STOP
    assert "no_estimate" in {q.kind for q in spent.calibration.faults}
