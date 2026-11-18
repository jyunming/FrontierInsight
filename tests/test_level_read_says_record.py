"""A coverage or bias read at a value of a factor compared inside each run, where FI's record holds nothing there, gives the
plain answer it gave before (no trials: ``[]``, ``None``, ``0``), and FI keeps its own sentence of what the record holds
(``table.unheld``): a result recorded once for the setting, a factor the runs did not compare, factors compared together, a
value given as text where the record holds a number, a value they did not record, or trials at that value that recorded no
interval or no value. The analysis writes that sentence as the reason of a result the study's function then leaves without
a value (it said "gave None here and no reason"). The read itself never raises for it: a study's code that caught a raise
and fell back to another read delivered a number the record contradicts. Where the record does hold the trials at that value,
the read is as before. Neutral made-up plans (the fixtures of tests/test_fx339_contract.py and tests/test_analysis_frame.py)."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest

from tests import test_analysis_frame as taf
from tests.test_fx339_contract import _cell, _metric, _table


def _at(noise: float, channel: Any, est_a: list[float], *, interval: bool = True) -> dict[str, Any]:
    """The trials of setting ``noise`` at ``channel`` (a factor compared inside each run)."""
    m: dict[str, Any] = {"est_a": _metric(est_a)}
    if interval:
        m |= {"lo_a": _metric([v - 1.0 for v in est_a]), "hi_a": _metric([v + 1.0 for v in est_a])}
    return {"cell": {"noise": noise, "channel": channel}, "level": {"channel": channel}, "key": f"noise={noise},channel={channel}",
            "planned": len(est_a), "ok": len(est_a), "failed": 0, "metrics": m}


def _says(t: Any, pattern: str) -> None:
    assert any(re.search(pattern, x) for x in t.unheld), t.unheld


def test_where_the_record_holds_the_trials_at_that_value_the_read_is_as_before() -> None:
    t = _table([_cell(0.1, [2.0, 2.0], [5.0, 5.0]), _at(0.1, 0, [1.5, 2.5, 4.0]), _at(0.1, 1, [2.0, 2.0, 2.0])])
    assert t.coverage("est_a", "noise=0.1", channel=0) == [1.0, 1.0, 0.0]
    assert t.coverage("est_a", "noise=0.1", channel=1) == [1.0, 1.0, 1.0]
    assert t.bias("est_a", "noise=0.1", channel=0) == pytest.approx(8 / 3 - 2.0)
    assert t.bias_dropped("est_a", "noise=0.1", channel=0) == 0
    assert t.unheld == []


def test_a_result_recorded_once_for_the_setting_gives_the_plain_answer_and_fi_keeps_why() -> None:
    """Old code: the same ``[]`` / ``None`` / ``0``, and no word of why."""
    t = _table([_cell(0.1, [1.5, 2.5, 4.0], [5.0, 5.0, 5.0])])
    assert t.coverage("est_a", "noise=0.1", channel=0) == []
    assert t.bias("est_a", "noise=0.1", channel=0) is None
    assert t.bias_dropped("est_a", "noise=0.1", channel=0) == 0
    _says(t, r"`est_a` is recorded once for setting noise=0.1, not for each value of channel.*holds no `est_a` at channel=0")
    assert len(t.unheld) == 1  # one sentence, kept once
    assert t.coverage("est_a", "noise=0.1") == [1.0, 1.0, 0.0]  # without the value: as before


@pytest.mark.parametrize("cells, read, pattern", [
    ([_at(0.1, 0, [2.0]), _at(0.1, 1, [2.0])], lambda t: t.coverage("est_a", "noise=0.1", factor="0"),
     r"did not compare `factor`.*the factors they compared are: `channel`"),
    ([_at(0.1, 0, [2.0]), _at(0.1, 1, [2.0])], lambda t: t.bias("est_a", "noise=0.1", channel=5),
     r"no trials at channel=5; the values they recorded are: channel=0, channel=1"),
    ([_at(0.1, 2, [2.0])], lambda t: t.coverage("est_a", "noise=0.1", channel="2"),
     r"recorded channel=2 as a number, and it was given as text: pass it as the record holds it \(channel=2\)"),
    ([_at(0.1, 0, [2.0, 2.1], interval=False)], lambda t: t.coverage("est_a", "noise=0.1", channel=0),
     r"at channel=0 recorded no interval \(lo_a, hi_a\) for `est_a`.*they recorded: est_a"),
    ([_at(0.1, 0, [2.0, 2.1], interval=False)], lambda t: t.bias("est_b", "noise=0.1", channel=0),
     r"at channel=0 recorded no value of `est_b`.*they recorded: est_a"),
], ids=["factor_not_compared", "value_not_recorded", "text_for_a_number", "no_interval", "no_value"])
def test_what_the_record_does_not_hold_there_is_said_and_the_read_gives_no_trials(cells: list[dict[str, Any]], read: Any,
                                                                                 pattern: str) -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), *cells])
    got = read(t)
    assert got in ([], None, 0)
    _says(t, pattern)


def test_trials_at_that_value_that_hold_the_value_but_no_interval_still_give_the_bias() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0, 2.1], interval=False)])
    assert t.bias("est_a", "noise=0.1", channel=0) == pytest.approx(0.05)
    assert t.unheld == []


def test_trials_that_recorded_the_result_but_no_finite_value_keep_the_plain_answer_and_say_nothing() -> None:
    """A value recorded as not finite is a trial left out of the mean (``bias_dropped`` counts it), not a missing record."""
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [float("nan"), float("nan")])])
    assert t.bias("est_a", "noise=0.1", channel=0) is None
    assert t.bias_dropped("est_a", "noise=0.1", channel=0) == 2
    assert t.unheld == []


def _two(noise: float, channel: int, arm: str, est_a: list[float]) -> dict[str, Any]:
    c = _at(noise, channel, est_a)
    c["cell"]["arm"], c["level"]["arm"] = arm, arm
    return c


def test_runs_that_compare_two_factors_together_ask_for_a_value_of_each() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _two(0.1, 0, "x", [2.0]), _two(0.1, 1, "x", [2.0])])
    assert t.coverage("est_a", "noise=0.1", channel=0) == []
    _says(t, r"compare `arm`, `channel` together: give a value of each \(channel=0\)")
    assert t.coverage("est_a", "noise=0.1", channel=0, arm="x") == [1.0]


def test_an_interval_recorded_where_the_true_value_is_not_finite_is_no_missing_interval() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0])])
    t._known["est_a"] = {"interval": ["lo_a", "hi_a"], "truth": float("nan")}
    assert t.coverage("est_a", "noise=0.1", channel=0) == []
    assert t.unheld == []


def test_a_result_the_setting_never_recorded_is_not_said_to_be_recorded_once() -> None:
    t = _table([_cell(0.1, [2.0], [5.0])])
    t._settings["noise=0.1"]["metrics"].pop("est_b")
    assert t.bias("est_b", "noise=0.1", channel=0) is None
    _says(t, r"recorded no value compared inside them, and no `est_b`; they recorded: est_a")


# ---- the analysis writes FI's sentence as the reason ---------------------------------------------------------------------------

def _known_design() -> dict[str, Any]:
    design = taf._design()
    growth = design["protocol"]["metrics"][0]
    growth["known_value"] = {"true_value": "base", "estimate": "growth", "standard_error": "wobble", "interval": ["lo", "hi"],
                             "level": 0.95}
    design["protocol"]["derived"] = [taf._derived("cover_a")]
    return design


def _known_record() -> dict[str, Any]:
    rec = copy.deepcopy(taf.RECORD)
    for c in rec["cells"]:
        vals = c["metrics"]["growth"]["values"]
        c["metrics"]["lo"] = {"values": [v - 1.0 for v in vals], "trials": list(range(len(vals)))}
        c["metrics"]["hi"] = {"values": [v + 1.0 for v in vals], "trials": list(range(len(vals)))}
    return rec


BODY = """
def derive_cover_a(table):
    out = {}
    for k in table.keys():
        v = table.coverage('growth', k, mode='a')
        out[k] = sum(v) / len(v) if v else None
    return out
"""


def test_a_result_left_without_a_value_after_such_a_read_gives_fi_s_sentence_as_its_reason(tmp_path: Path) -> None:
    """Old code: "`derive_cover_a` gave None here and no reason" at every setting."""
    _code, _printed, record, _err = taf._run(taf._quest(tmp_path, _known_design(), BODY, record=_known_record()))
    cover = record["derived"]["cover_a"]
    assert set(cover["missing"]) == set(taf.KEYS), cover
    for key in taf.KEYS:
        why = cover["missing"][key]
        assert "no reason" not in why and ("FI's record: `growth` is recorded once for setting %s," % key) in why, why


def test_bias_dropped_keeps_the_same_sentence_as_bias_where_the_trials_hold_no_value_of_the_result() -> None:
    t = _table([_cell(0.1, [2.0], [5.0]), _at(0.1, 0, [2.0, 2.1], interval=False)])
    assert t.bias_dropped("est_b", "noise=0.1", channel=0) == 0
    _says(t, r"at channel=0 recorded no value of `est_b`")


def test_each_sentence_is_kept_with_its_setting_and_a_new_read_starts_clean() -> None:
    t = _table([_cell(0.1, [1.5], [5.0]), _cell(0.2, [1.5], [5.0])])
    for key in ("noise=0.1", "noise=0.2"):
        assert t.coverage("est_a", key, channel=0) == []
        t.coverage("est_a", key, channel=0)  # the same read again keeps one sentence
    assert set(t.unheld_at) == {"noise=0.1", "noise=0.2"}
    for key, said in t.unheld_at.items():
        assert len(said) == 1 and ("setting %s," % key) in said[0], said
    t.reset_reads()
    assert t.unheld == [] and t.unheld_at == {}


def _two_settings_record() -> dict[str, Any]:
    rec = _known_record()
    rec["cells"] = [c for c in rec["cells"] if c["cell"]["dose"] == 1]  # rate=0.1 and rate=0.2 at dose 1
    return rec


def test_each_setting_s_reason_is_fi_s_sentence_about_that_setting(tmp_path: Path) -> None:
    """The first build matched the sentence by text and gave every setting the first setting's sentence."""
    design = _known_design()
    design["protocol"]["grid"] = {"rate": [0.1, 0.2], "dose": [1]}
    _code, _printed, record, _err = taf._run(taf._quest(tmp_path, design, BODY, record=_two_settings_record()))
    missing = record["derived"]["cover_a"]["missing"]
    keys = [c["key"] for c in _two_settings_record()["cells"]]
    assert set(missing) == set(keys), missing
    for key in keys:
        other = next(k for k in keys if k != key)
        assert ("setting %s," % key) in missing[key] and ("setting %s," % other) not in missing[key], missing[key]
